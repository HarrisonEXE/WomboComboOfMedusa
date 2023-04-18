import random
import time
import numpy as np
# from queue import Queue
# from threading import Thread
# import atexit
import csv

# from pythonosc import udp_client
# from pythonosc import dispatcher
# from pythonosc import osc_server

home = [0, 0, 0, 90, 0, 0, 0]
time_to_home = 8

hold = [0,0,0,0,0,0,0]
hold_t = [2,2,2,2,2,2,2]

gesture_info = []

def get_gesture_info():
    with open("vision_response.csv") as csv_db:
        csv_reader = csv.DictReader(csv_db, delimiter=',')
        for dict in csv_reader:
            temp_dict = {}
            #add properly formatted dict to gesture_info
            #dict is {'Gesture_num': , name: , poscell: , timecell: , bpm: ,tense_value: }
            temp_dict['Gesture_num'] = int(dict['Gesture_num'])
            temp_dict['Gesture_name'] = dict['Gesture_name']
            #remove the {}, split on ;
            temp_dict['Position_cell'] = dict['Position_cell'][1:-1].split(";")
            temp_dict['Time_cell'] = dict['Time_cell'][1:-1].split(";")
            angles = []
            times = []
            # if temp_dict['Gesture_name'] == "canon_lr" or temp_dict['Gesture_name'] == "canon_rl":
            #     print(temp_dict)
            for i in range(len(temp_dict['Time_cell'])):
                #print(dict['Position_cell'])
                joint_angles = temp_dict['Position_cell'][i].split(",")
                joint_times = temp_dict['Time_cell'][i].split(",")

                if (len(joint_angles) > 1):
                    joint_angles[0] = joint_angles[0][1:]
                    joint_angles[-1] = joint_angles[-1][:-1]
                    joint_times[0] = joint_times[0][1:]
                    joint_times[-1] = joint_times[-1][:-1]
                    
                joint_angles = [float(j) for j in joint_angles]
                joint_times = [float(j) for j in joint_times]

                angles.append(joint_angles)
                times.append(joint_times)
            
            temp_dict['Position_cell'] = angles
            temp_dict['Time_cell'] = times

            temp_dict['Bpm'] = int(dict['Bpm'])
            temp_dict['Tense_value'] = int(dict['Tense_value'])
      
            gesture_info.append(temp_dict)
            
def fifth_poly(q_i, q_f, t):
    # time/0.005
    traj_t = np.arange(0, t, 0.005)
    #print("shape is " + str(np.shape(traj_t)) + " time is " + str(t) + "qi is " + str(q_i))
    dq_i = 0
    dq_f = 0
    ddq_i = 0
    ddq_f = 0
    a0 = q_i
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * t ** 3) * (20 * (q_f - q_i) - (8 * dq_f + 12 * dq_i) * t - (3 * ddq_f - ddq_i) * t ** 2)
    a4 = 1 / (2 * t ** 4) * (30 * (q_i - q_f) + (14 * dq_f + 16 * dq_i) * t + (3 * ddq_f - 2 * ddq_i) * t ** 2)
    a5 = 1 / (2 * t ** 5) * (12 * (q_f - q_i) - (6 * dq_f + 6 * dq_i) * t - (ddq_f - ddq_i) * t ** 2)
    traj_pos = a0 + a1 * traj_t + a2 * traj_t ** 2 + a3 * traj_t ** 3 + a4 * traj_t ** 4 + a5 * traj_t ** 5
    return traj_pos

def make_traj(gesture_num):

    gesture_dict = gesture_info[gesture_num]
    
    
    angle_list = gesture_dict['Position_cell']
    time_list = gesture_dict['Time_cell']
    bpm = gesture_dict['Bpm']
    
    #use for the nathalie vision gestures as home value
    alt_home = [0,-25,0,9,0,39,0]
    
    step_values = []
    for j in range(len(angle_list)):
        #print("J IS " + str(j))
        #angle_list is list of 7 lists, one for each joint
        #calculate 5th poly for each pair of points for each joint
        joint_a = angle_list[j]
        #print(joint_a)
        joint_t = time_list[j]
        #print(joint_t)
    
        #where ALL gestures should start
        #abs_joint_pos = wave_ip[arm_num][j]
        # if gesture_num >=6 and gesture_num <=11:
        #     # if recoil/come gesture, make sure to use this starting point that the gestures were defined at
        #     home_point= alt_home[j]
        # else:
        #     # otherwise, this was the home point the gestures were defined at (0 0 0 90 0 0 0)
        home_point = home[j]
        abs_joint_pos = home[j]
        traj_pos_total = []
        #for delta in the range of changes in angle of THIS joint
        for d in range(len(joint_a)):
            if d == 0:
                #abs_joint_pos is the wave home point here
                #NEW second point should be new home + (old home - new home) + my delta value
                traj_pos = fifth_poly(abs_joint_pos, abs_joint_pos + (home_point - abs_joint_pos) + joint_a[d], joint_t[d]*(60/bpm))
                abs_joint_pos += (home_point - abs_joint_pos) + joint_a[d]
                traj_pos_total = np.copy(traj_pos)
            else:
                #current position is abs_joint_pos, final position is current + change[d], time index is d
                #stays the same because based off of the current point
                traj_pos = fifth_poly(abs_joint_pos, joint_a[d] + abs_joint_pos, joint_t[d]*(60/bpm))
                abs_joint_pos += joint_a[d]
                
                traj_pos_total = np.concatenate((traj_pos_total, traj_pos), axis=None)

        step_values.append(traj_pos_total)
            
    return step_values



def init():
    #get the gesture info
    get_gesture_info()


    # WAVE0 = [-0.25, 35.5, -2, 126.5, 101, 80.9, -45]
    # WAVE1 = [2.62, 33.5, 0, 127.1, 237.6, 72.6, -57.3]
    # WAVE2 = [-1.4, 29.4, 0, 120, -15, 23.1, -45]
    # WAVE3 = [-1.4, 30.9, 0, 120, 48.9, 44.6, -45]
    # WAVE4 = [-1.8, 30.9, 0, 120, -78.6, 44.6, -45]
    # wave_ip = [WAVE0, WAVE1, WAVE2, WAVE3, WAVE4]

    # strumD = 30
    # SIP0 = [-0.25, 87.38, -2, 126.5, -strumD / 2, 51.73, -45]
    # SIP1 = [2.62, 86.2, 0, 127.1, -strumD / 2, 50.13, -45]
    # SIP2 = [1.3, 81.68, 0.0, 120, -strumD / 2, 54.2, -45]
    # SIP3 = [-1.4, 83.8, 0, 120, -strumD / 2, 50.75, -45]
    # SIP4 = [-1.8, 81.8, 0, 120, -strumD / 2, 50.65, -45]
    # sip = [SIP0, SIP1, SIP2, SIP3, SIP4]