import atexit
import csv
import os
import time
import numpy as np
import math

from queue import Queue
from threading import Thread
from xarm.wrapper import XArmAPI
from pythonosc import dispatcher
from pythonosc import osc_server

import positions


def robomove(numarm, trajectory):
    track_time = time.time()
    initial_time = time.time()
    for j_angles in trajectory:
        # run command
        start_time = time.time()
        arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


def poseToPose(poseI, poseF, t):
    traj = []
    for p in range(len(poseI)):
        traj.append(fifth_poly(poseI[p], poseF[p], t))
        print(p)
    return traj


def gotoPose(numarm, traj):
    track_time = time.time()
    initial_time = time.time()
    for ang in range(len(traj[0])):
        angles = [traj[0][ang], traj[1][ang], traj[2][ang], traj[3][ang], traj[4][ang], traj[5][ang], traj[6][ang]]
        start_time = time.time()
        arms[numarm].set_servo_angle_j(angles=angles, is_radian=False)
        # print(angles)
        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.001)
        initial_time += 0.004


def robomove(numarm, trajectory):
    track_time = time.time()
    initial_time = time.time()
    for j_angles in trajectory:
        # run command
        start_time = time.time()
        # print(j_angles)
        arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


def setup():
    for a in range(len(arms)):
        arms[a].set_simulation_robot(on_off=False)
        arms[a].motion_enable(enable=True)
        arms[a].clean_warn()
        arms[a].clean_error()
        arms[a].set_mode(0)
        arms[a].set_state(0)
        curIP = IP[a]
        arms[a].set_servo_angle(angle=curIP, wait=True, speed=10, acceleration=0.25, is_radian=False)

        # arms[a].set_servo_angle(angle=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], wait=False, speed=10, acceleration=0.25, is_radian=False)


def setup2():
    for a in range(len(arms)):
        curIP = IP[a]
        arms[a].set_servo_angle(angle=curIP, wait=False, speed=10, acceleration=0.25, is_radian=False)

        # print(curIP)


def fifth_poly(q_i, q_f, t):
    # time/0.005
    traj_t = np.arange(0, t, 0.004)
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


def spline_poly(q_i, q_f, q_in, ta, tt, ttopstop, tbotstop):
    # qi is initial pos, qf is final pos (strike), qin is new initial (return pos)

    # initial accel (using first half of a 5th order poly)
    # ta is double the time till max acceleration (time doing 5th order poly)

    ########### this code calculates the trajectory for the first half (the way down) #############

    traj_ta = np.arange(0, ta, 0.004)
    dq_i = 0
    dq_f = 0
    ddq_i = 0
    ddq_f = 0
    a0 = q_i
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * ta ** 3) * (20 * (q_f - q_i) / 2 - (8 * dq_f + 12 * dq_i) * ta - (3 * ddq_f - ddq_i) * ta ** 2)
    a4 = 1 / (2 * ta ** 4) * (30 * (q_i - q_f) / 2 + (14 * dq_f + 16 * dq_i) * ta + (3 * ddq_f - 2 * ddq_i) * ta ** 2)
    a5 = 1 / (2 * ta ** 5) * (12 * (q_f - q_i) / 2 - (6 * dq_f + 6 * dq_i) * ta - (ddq_f - ddq_i) * ta ** 2)
    fifth_pos = a0 + a1 * traj_ta + a2 * traj_ta ** 2 + a3 * traj_ta ** 3 + a4 * traj_ta ** 4 + a5 * traj_ta ** 5
    fifth_vel = a1 + 2 * a2 * traj_ta + 3 * a3 * traj_ta ** 2 + 4 * a4 * traj_ta ** 3 + 5 * a5 * traj_ta ** 4

    # halfway point of acceleration array (hp)
    hp = math.floor(len(fifth_pos) / 2)
    delta1 = abs(fifth_pos[0] - fifth_pos[hp])
    # speed halfway (max speed)
    hv = fifth_vel[hp]

    # 5th order turnaround
    # tt is time for turning around
    traj_tt = np.arange(0, tt, 0.004)
    dq_i = hv
    dq_f = -hv
    ddq_i = 0
    ddq_f = 0
    # nq_i = pc[len(pc)-1] # new initial pos is the end of constant velocity part
    a0 = 0
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * ta ** 3) * (20 * (0) - (8 * dq_f + 12 * dq_i) * ta - (3 * ddq_f - ddq_i) * ta ** 2)
    a4 = 1 / (2 * ta ** 4) * (30 * (0) + (14 * dq_f + 16 * dq_i) * ta + (3 * ddq_f - 2 * ddq_i) * ta ** 2)
    a5 = 1 / (2 * ta ** 5) * (12 * (0) - (6 * dq_f + 6 * dq_i) * ta - (ddq_f - ddq_i) * ta ** 2)
    tfifth_pos = a0 + a1 * traj_ta + a2 * traj_ta ** 2 + a3 * traj_ta ** 3 + a4 * traj_ta ** 4 + a5 * traj_ta ** 5

    thp = math.floor(len(tfifth_pos) / 2)  # halfway point of turnaround traj
    delta2 = abs(tfifth_pos[0] - tfifth_pos[thp])

    # constant speed
    # tc is time at constant speed
    delta3 = abs(q_i - q_f) - delta1 - delta2
    if (delta3 < 0):
        print("accel time and turnaround time too big")

    tc = delta3 / abs(hv)

    traj_tc = np.arange(0, tc, 0.004)
    pc = fifth_pos[hp] + traj_tc * hv

    # stall time at top / bottom
    traj_top = np.ones(int(ttopstop / 0.004)) * q_i  # time stopped at top of trajectory, before strike
    traj_bot = np.ones(
        int(tbotstop / 0.004)) * q_f  # time stopped at bottom of trajectory, after strike (half of the total time)

    ########### this code calculates the trajectory for the second half (the way up) #############

    traj_ta = np.arange(0, ta, 0.004)
    dq_i = 0
    dq_f = 0
    ddq_i = 0
    ddq_f = 0
    a0 = q_in
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * ta ** 3) * (20 * (q_f - q_in) / 2 - (8 * dq_f + 12 * dq_i) * ta - (3 * ddq_f - ddq_i) * ta ** 2)
    a4 = 1 / (2 * ta ** 4) * (30 * (q_in - q_f) / 2 + (14 * dq_f + 16 * dq_i) * ta + (3 * ddq_f - 2 * ddq_i) * ta ** 2)
    a5 = 1 / (2 * ta ** 5) * (12 * (q_f - q_in) / 2 - (6 * dq_f + 6 * dq_i) * ta - (ddq_f - ddq_i) * ta ** 2)
    fifth_pos2 = a0 + a1 * traj_ta + a2 * traj_ta ** 2 + a3 * traj_ta ** 3 + a4 * traj_ta ** 4 + a5 * traj_ta ** 5
    fifth_vel2 = a1 + 2 * a2 * traj_ta + 3 * a3 * traj_ta ** 2 + 4 * a4 * traj_ta ** 3 + 5 * a5 * traj_ta ** 4

    # halfway point of acceleration array (hp)
    hp2 = math.floor(len(fifth_pos2) / 2)
    delta1 = abs(fifth_pos2[0] - fifth_pos2[hp2])
    # speed halfway (max speed)
    hv = fifth_vel2[hp2]

    # 5th order turnaround
    # tt is time for turning around
    traj_tt = np.arange(0, tt, 0.004)
    dq_i = hv
    dq_f = -hv
    ddq_i = 0
    ddq_f = 0
    # nq_i = pc[len(pc)-1] # new initial pos is the end of constant velocity part
    a0 = 0
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * ta ** 3) * (20 * (0) - (8 * dq_f + 12 * dq_i) * ta - (3 * ddq_f - ddq_i) * ta ** 2)
    a4 = 1 / (2 * ta ** 4) * (30 * (0) + (14 * dq_f + 16 * dq_i) * ta + (3 * ddq_f - 2 * ddq_i) * ta ** 2)
    a5 = 1 / (2 * ta ** 5) * (12 * (0) - (6 * dq_f + 6 * dq_i) * ta - (ddq_f - ddq_i) * ta ** 2)
    tfifth_pos2 = a0 + a1 * traj_ta + a2 * traj_ta ** 2 + a3 * traj_ta ** 3 + a4 * traj_ta ** 4 + a5 * traj_ta ** 5

    thp2 = math.floor(len(tfifth_pos2) / 2)  # halfway point of turnaround traj
    delta2 = abs(tfifth_pos2[0] - tfifth_pos2[thp2])

    # constant speed
    # tc is time at constant speed
    delta3 = abs(q_in - q_f) - delta1 - delta2
    if (delta3 < 0):
        print("accel time and turnaround time too big")

    tc = delta3 / abs(hv)

    traj_tc = np.arange(0, tc, 0.004)
    pc2 = fifth_pos2[hp2] + traj_tc * hv

    # stall time at top / bottom
    traj_top2 = np.ones(int(ttopstop / 0.004)) * q_in  # time stopped at top of trajectory, before strike
    traj_bot2 = np.ones(
        int(tbotstop / 0.004)) * q_f  # time stopped at bottom of trajectory, after strike (half of the total time)

    half_traj1 = np.concatenate((traj_top, fifth_pos[0:hp], pc, pc[len(pc) - 1] + tfifth_pos[0:thp], traj_bot))
    half_traj2 = np.flip(
        np.concatenate((traj_top2, fifth_pos2[0:hp2], pc2, pc2[len(pc2) - 1] + tfifth_pos2[0:thp2], traj_bot2)))

    full_traj = np.append(half_traj1, half_traj2)

    return full_traj


def drumbot(traj1, traj2, traj3, traj4, traj5, traj6, traj7, arm):
    # j_angles = pos
    track_time = time.time()
    initial_time = time.time()

    # print(len(traj1))
    print(len(traj2))
    # print(len(traj3))
    print(len(traj4))
    # print(len(traj5))
    print(len(traj6))
    # print(len(traj7))

    for i in range(min(len(traj2), len(traj4), len(traj6))):
        # for i in range(len(traj2)):
        # run command
        # start_time = time.time()
        # j_angles[4] = traj[i]
        # arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
        jointangles = [traj1[i], traj2[i], traj3[i], traj4[i], traj5[i], traj6[i], traj7[i]]
        # print(traj2[i])
        arms[arm].set_servo_angle_j(angles=jointangles, is_radian=False)
        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


def strumbot(numarm, traj):
    pos = IP[numarm]
    j_angles = pos
    track_time = time.time()
    initial_time = time.time()
    for i in range(len(traj)):
        # run command
        start_time = time.time()
        j_angles[4] = traj[i]
        arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


def prepGesture(numarm, traj):
    pos = IP[numarm]
    j_angles = pos.copy()
    track_time = time.time()
    initial_time = time.time()
    for i in range(len(traj)):
        # run command
        j_angles[1] = pos[1] + traj[i]
        j_angles[3] = pos[3] + traj[i]
        arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
        # print(j_angles)
        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


def drummer(inq, num):
    global CP
    global CPpass
    # for some reason have to declare CP and CPpass again lmao

    while True:
        [pnote, pvel] = inq.get()
        # play note, play velocity
        print("got!")

        # test what num is, then set trajectory depending on num

        # when num is 5 (snare)
        if (num == 5):
            # get velocity as 1,2,3,4 to determine the next IP

            # if velocity is 1 (middle snare):
            if (pvel == 1):
                IPN = IP1
            # if velocity is 2 (pure rim):
            elif (pvel == 2):
                IPN = IP2
            # if velocity is 3 (pure wood):
            elif (pvel == 3):
                IPN = IP3
            # if velocity is 4 (rimshot):
            elif (pvel == 4):
                IPN = IP4
            # if velocity is 6 (least soft)
            elif (pvel == 6):
                IPN = IP6
            # if velocity is 7
            elif (pvel == 7):
                IPN = IP7
            # if velocity is 8 (softest)
            elif (pvel == 8):
                IPN = IP8
            # if else, default to 1
            else:
                IPN = IP1

            ###these are hits with veloctiy 6,7,8 representing the IP!###
            # if note is 0, 1, 2 (lighter strikes)
            if (pnote == 0):
                traj1 = spline_poly(IP6[0], FP6[0], IPN[0], .3, .08, 0, .7)
                traj2 = spline_poly(IP6[1], FP6[1], IPN[1], .3, .08, .2, 0)
                traj3 = spline_poly(IP6[2], FP6[2], IPN[2], .3, .08, 0, .7)
                traj4 = spline_poly(IP6[3], FP6[3], IPN[3], .3, .08, .2, 0)
                traj5 = spline_poly(IP6[4], FP6[4], IPN[4], .3, .08, 0, 0.7)
                traj6 = spline_poly(IP6[5], FP6[5], IPN[5], .3, .08, .2, 0)
                traj7 = spline_poly(IP6[6], FP6[6], IPN[6], .3, .08, 0, 0.7)

                if (IP6 == CP):
                    CPpass = 1
                    CP = IPN

            if (pnote == 1):
                traj1 = spline_poly(IP7[0], FP7[0], IPN[0], .38, .08, 0, .7)
                traj2 = spline_poly(IP7[1], FP7[1], IPN[1], .38, .08, 0.2, 0)
                traj3 = spline_poly(IP7[2], FP7[2], IPN[2], .38, .08, 0, .7)
                traj4 = spline_poly(IP7[3], FP7[3], IPN[3], .38, .08, .2, 0)
                traj5 = spline_poly(IP7[4], FP7[4], IPN[4], .38, .08, 0, 0.7)
                traj6 = spline_poly(IP7[5], FP7[5], IPN[5], .38, .08, .2, 0)
                traj7 = spline_poly(IP7[6], FP7[6], IPN[6], .38, .08, 0, 0.7)

                if (IP7 == CP):
                    CPpass = 1
                    CP = IPN

            if (pnote == 2):
                traj1 = spline_poly(IP8[0], FP8[0], IPN[0], .46, .08, 0, .7)
                traj2 = spline_poly(IP8[1], FP8[1], IPN[1], .46, .08, 0.2, 0)
                traj3 = spline_poly(IP8[2], FP8[2], IPN[2], .46, .08, 0, .7)
                traj4 = spline_poly(IP8[3], FP8[3], IPN[3], .46, .08, .2, 0)
                traj5 = spline_poly(IP8[4], FP8[4], IPN[4], .46, .08, 0, 0.7)
                traj6 = spline_poly(IP8[5], FP8[5], IPN[5], .46, .08, .2, 0)
                traj7 = spline_poly(IP8[6], FP8[6], IPN[6], .46, .08, 0, 0.7)

                if (IP8 == CP):
                    CPpass = 1
                    CP = IPN
            ###these are hits with veloctiy 1 representing the IP!###

            # if note is 3 (midi 60) (normal strike)
            if (pnote == 3):
                traj1 = spline_poly(IP1[0], FP1[0], IPN[0], .2, .08, 0, .7)
                traj2 = spline_poly(IP1[1], FP1[1], IPN[1], .4, .08, 0.1, 0)
                traj3 = spline_poly(IP1[2], FP1[2], IPN[2], .2, .08, 0, .7)
                traj4 = spline_poly(IP1[3], FP1[3], IPN[3], .32, .08, .23, 0)
                traj5 = spline_poly(IP1[4], FP1[4], IPN[4], .2, .08, 0, 0.7)
                traj6 = spline_poly(IP1[5], FP1[5], IPN[5], .18, .08, .45, 0)
                traj7 = spline_poly(IP1[6], FP1[6], IPN[6], .2, .08, 0, 0.7)

                if (IP1 == CP):
                    CPpass = 1
                    CP = IPN

            # if note is 4 (double strike)
            elif (pnote == 4):
                # added .1 to all stopbots
                traj1 = spline_poly(IP1[0], FP5[0], IPN[0], .2, .08, 0, .8)
                traj2 = spline_poly(IP1[1], FP5[1], IPN[1], .5, .08, 0, 0.02)
                traj3 = spline_poly(IP1[2], FP5[2], IPN[2], .2, .08, 0, .8)
                traj4 = spline_poly(IP1[3], FP5[3], IPN[3], .32, .08, .13, 0.156)
                traj5 = spline_poly(IP1[4], FP5[4], IPN[4], .2, .08, 0, 0.8)
                traj6 = spline_poly(IP1[5], FP5[5], IPN[5], .2, .08, .35, 0.116)
                traj7 = spline_poly(IP1[6], FP5[6], IPN[6], .2, .08, 0, 0.8)

                if (IP1 == CP):
                    CPpass = 1
                    CP = IPN

            # if note is 5 (triple strike)
            elif (pnote == 5):
                # added .2 to all stopbots
                traj1 = spline_poly(IP1[0], FP5[0], IPN[0], .2, .08, 0, .8)
                traj2 = spline_poly(IP1[1], FP5[1], IPN[1], .5, .08, 0, 0.08)
                traj3 = spline_poly(IP1[2], FP5[2], IPN[2], .2, .08, 0, .8)
                traj4 = spline_poly(IP1[3], FP5[3], IPN[3], .32, .08, .13, 0.216)
                traj5 = spline_poly(IP1[4], FP5[4], IPN[4], .2, .08, 0, 0.8)
                traj6 = spline_poly(IP1[5], FP5[5], IPN[5], .2, .08, .35, 0.176)
                traj7 = spline_poly(IP1[6], FP5[6], IPN[6], .2, .08, 0, 0.8)

                if (IP1 == CP):
                    CPpass = 1
                    CP = IPN

            # if note is 6 (pure rim)
            elif (pnote == 6):
                traj1 = spline_poly(IP2[0], FP2[0], IPN[0], .2, .08, 0.05, .32)
                traj2 = spline_poly(IP2[1], FP2[1], IPN[1], .4, .08, 0.05, 0)
                traj3 = spline_poly(IP2[2], FP2[2], IPN[2], .2, .08, 0.05, .32)
                traj4 = spline_poly(IP2[3], FP2[3], IPN[3], .32, .08, 0.05, 0.18)
                traj5 = spline_poly(IP2[4], FP2[4], IPN[4], .2, .08, 0.05, 0.32)
                traj6 = spline_poly(IP2[5], FP2[5], IPN[5], .2, .08, 0.05, 0.32)
                traj7 = spline_poly(IP2[6], FP2[6], IPN[6], .2, .08, 0.05, 0.32)

                if (IP2 == CP):
                    CPpass = 1
                    CP = IPN

            ###these are hits with veloctiy 3 representing the IP!###

            # if note is 7 (pure wood)
            elif (pnote == 7):
                traj1 = spline_poly(IP3[0], FP3[0], IPN[0], .2, .08, 0, .32)
                traj2 = spline_poly(IP3[1], FP3[1], IPN[1], .4, .08, 0, 0)
                traj3 = spline_poly(IP3[2], FP3[2], IPN[2], .2, .08, 0, .32)
                traj4 = spline_poly(IP3[3], FP3[3], IPN[3], .32, .08, 0, 0.18)
                traj5 = spline_poly(IP3[4], FP3[4], IPN[4], .2, .08, 0, 0.32)
                traj6 = spline_poly(IP3[5], FP3[5], IPN[5], .2, .08, 0, 0.32)
                traj7 = spline_poly(IP3[6], FP3[6], IPN[6], .2, .08, 0, 0.32)

                if (IP3 == CP):
                    CPpass = 1
                    CP = IPN

            ###these are hits with veloctiy 4 representing the IP!###

            # if note is 8 (rimshot rim + skin)
            elif (pnote == 8):
                traj1 = spline_poly(IP4[0], FP4[0], IPN[0], .2, .08, 0, .32)
                traj2 = spline_poly(IP4[1], FP4[1], IPN[1], .4, .08, 0, 0)
                traj3 = spline_poly(IP4[2], FP4[2], IPN[2], .2, .08, 0, .32)
                traj4 = spline_poly(IP4[3], FP4[3], IPN[3], .32, .08, 0, 0.18)
                traj5 = spline_poly(IP4[4], FP4[4], IPN[4], .2, .08, 0, 0.32)
                traj6 = spline_poly(IP4[5], FP4[5], IPN[5], .2, .08, 0, 0.32)
                traj7 = spline_poly(IP4[6], FP4[6], IPN[6], .2, .08, 0, 0.32)

                if (IP4 == CP):
                    CPpass = 1
                    CP = IPN

        # when num is 6 (bodhron)
        elif (num == 6):
            # only one IP
            CPpass = 1
            # if note is 3 (midi 60) (normal strike)
            if (pnote == 3):
                traj1 = spline_poly(BIP1[0], BFP1[0], BIP1[0], .2, .08, 0, .7)
                traj2 = spline_poly(BIP1[1], BFP1[1], BIP1[1], .4, .08, 0.1, 0)
                traj3 = spline_poly(BIP1[2], BFP1[2], BIP1[2], .2, .08, 0, .7)
                traj4 = spline_poly(BIP1[3], BFP1[3], BIP1[3], .32, .08, .23, 0)
                traj5 = spline_poly(BIP1[4], BFP1[4], BIP1[4], .2, .08, 0, 0.7)
                traj6 = spline_poly(BIP1[5], BFP1[5], BIP1[5], .18, .08, .45, 0)
                traj7 = spline_poly(BIP1[6], BFP1[6], BIP1[6], .2, .08, 0, 0.7)

            # if note is 4 (double strike)
            elif (pnote == 4):
                # added .1 to all stopbots
                CPpass = 0
                traj1 = spline_poly(BIP1[0], BFP1[0], BIP1[0], .2, .08, 0, .8)
                traj2 = spline_poly(BIP1[1], BFP1[1], BIP1[1], .5, .08, 0, 0.02)
                traj3 = spline_poly(BIP1[2], BFP1[2], BIP1[2], .2, .08, 0, .8)
                traj4 = spline_poly(BIP1[3], BFP1[3], BIP1[3], .32, .08, .13, 0.156)
                traj5 = spline_poly(BIP1[4], BFP1[4], BIP1[4], .2, .08, 0, 0.8)
                traj6 = spline_poly(BIP1[5], BFP1[5], BIP1[5], .2, .08, .35, 0.116)
                traj7 = spline_poly(BIP1[6], BFP1[6], BIP1[6], .2, .08, 0, 0.8)

            # if note is 5 (triple strike)
            elif (pnote == 5):
                CPpass = 0
                # added .2 to all stopbots
                traj1 = spline_poly(BIP1[0], BFP1[0], BIP1[0], .2, .08, 0, .8)
                traj2 = spline_poly(BIP1[1], BFP1[1], BIP1[1], .5, .08, 0, 0.08)
                traj3 = spline_poly(BIP1[2], BFP1[2], BIP1[2], .2, .08, 0, .8)
                traj4 = spline_poly(BIP1[3], BFP1[3], BIP1[3], .32, .08, .13, 0.216)
                traj5 = spline_poly(BIP1[4], BFP1[4], BIP1[4], .2, .08, 0, 0.8)
                traj6 = spline_poly(BIP1[5], BFP1[5], BIP1[5], .2, .08, .35, 0.176)
                traj7 = spline_poly(BIP1[6], BFP1[6], BIP1[6], .2, .08, 0, 0.8)

            # outside rim of bodhron
            elif (pnote == 6):
                traj1 = spline_poly(BIP1[0], BFP2[0], BIP1[0], .2, .08, 0.05, .32)
                traj2 = spline_poly(BIP1[1], BFP2[1], BIP1[1], .4, .08, 0.05, 0)
                traj3 = spline_poly(BIP1[2], BFP2[2], BIP1[2], .2, .08, 0.05, .32)
                traj4 = spline_poly(BIP1[3], BFP2[3], BIP1[3], .32, .08, 0.15, 0.08)
                traj5 = spline_poly(BIP1[4], BFP2[4], BIP1[4], .2, .08, 0.05, 0.32)
                traj6 = spline_poly(BIP1[5], BFP2[5], BIP1[5], .2, .08, 0.35, 0.02)
                traj7 = spline_poly(BIP1[6], BFP2[6], BIP1[6], .2, .08, 0.05, 0.32)

        # send trajectories to drumbot to perform (unless CP is not met)
        if CPpass == 1:
            drumbot(traj1, traj2, traj3, traj4, traj5, traj6, traj7, num)
            CPpass = 0
        else:
            print("WARNING!!! Discontinuity")


def strummer(inq, num):
    i = 0
    uptraj = fifth_poly(-strumD / 2, strumD / 2, speed)
    downtraj = fifth_poly(strumD / 2, -strumD / 2, speed)
    bothnorm = [uptraj, downtraj]

    utrajfirst = positions.utraj[num]

    utrajsecond = utrajfirst[::-1]
    bothu = [utrajfirst, utrajsecond]
    tension = fifth_poly(0, -20, 0.5)
    release = fifth_poly(-20, 0, 0.75)
    strumMode = 1
    wave = positions.sintraj[num]
    circ = positions.circletraj[num]
    spintrajfirst = positions.spintraj[num]

    while True:
        play = inq.get()  # WHERE I AM GETTING A PLAY A NOT COMMAND
        newmode = int(play)

        # if someone waves:
        #     poseI = arms[num].angles
        #     newPos = poseToPose(poseI, [-0.25, 35.5, -2, 126.5, 101, 80.9, -45], 5)
        #     gotoPose(num, newPos)
        #     print("good")
        # if newmode != strumMode:
        #     i = 0
        #     if newmode > 5:
        #         print("switching to", newmode)
        #         newmode = newmode - 5
        #     poseI = arms[num].angles
        #     poseF = AllIP[newmode-1][num]
        #     setup = poseToPose(poseI, poseF, 1)
        #     gotoPose(num, setup)
        #     strumMode = newmode

        # posetoPose(,)
        # print("got!")

        if play == 1:  # waving HI
            poseI = arms[num].angles
            poseF = WAVE[num]
            print(poseI)
            print(poseF)
            newPos = poseToPose(poseI, poseF, 5)
            gotoPose(num, newPos)

        if play == 2:  # waving BYE
            poseI = arms[num].angles
            poseF = IP[num]
            print(poseI)
            print(poseF)
            newPos = poseToPose(poseI, poseF, 5)
            gotoPose(num, newPos)

        if play == 3:  # twirl
            poseI = arms[num].angles
            poseF = IPus[num]
            print(poseI)
            print(poseF)
            newPos = poseToPose(poseI, poseF, 4)
            gotoPose(num, newPos)
            robomove(num, spintrajfirst)


BUFFER_SIZE = 5


def buffered_smooth(buffer_x, buffer_y, buffer_z, coordinates):
    buffer_x.append(coordinates['x'])
    buffer_y.append(coordinates['y'])
    buffer_z.append(coordinates['z'])

    if len(buffer_x) >= BUFFER_SIZE:
        x = exponential_moving_average(buffer_x, 0.1)
        y = exponential_moving_average(buffer_y, 0.1)
        z = exponential_moving_average(buffer_z, 0.1)
        return x, y, z
    else:
        return None


def exponential_moving_average(values, alpha):
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema[-1]


def save_vision_data(filename, timestamp, raw_values, smoothed_values):
    row_data = [timestamp] + raw_values + smoothed_values
    columns = ['timestamp', 'raw_head_x', 'raw_head_y', 'raw_head_z', 'raw_shoulder_x', 'raw_shoulder_y',
               'raw_shoulder_z', 'smoothed_head_x', 'smoothed_head_y', 'smoothed_head_z', 'smoothed_shoulder_x',
               'smoothed_shoulder_y', 'smoothed_shoulder_z']

    # Check if the file exists
    if not os.path.isfile(filename):
        # Create a new file and write the header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

    # Append the data to the existing file
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)


def save_joint_data(filename, timestamp, joint_angles):
    if len(joint_angles) != 7:
        raise ValueError("Joint angles list must have exactly 7 elements.")

    with open(filename, 'a', newline='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        data_writer.writerow([timestamp,
                              joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3],
                              joint_angles[4], joint_angles[5], joint_angles[6]])


global tracking_offset
tracking_offset = 0


def respond_to_gesture(address, *args):
    if args[0] == "wave_hello":
        poseI = arms[0].angles
        poseF = [0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
        newPos = poseToPose(poseI, poseF, 5)
        gotoPose(0, newPos)
        xArm1_Play.start()
    elif args[0] == "wave_bye":
        q0.put(2)
    elif args[0] == "twirl":
        q0.put(3)


def respond_to_head(address, *args):
    if address == "/head":
        head = {'x': args[0], 'y': args[1], 'z': args[2]}
        shoulder = {'x': args[3], 'y': args[4], 'z': args[5]}

        smoothed_head = buffered_smooth(head_x, head_y, head_z, head)
        smoothed_shoulder = buffered_smooth(shoulder_x, shoulder_y, shoulder_z, shoulder)

        if smoothed_head is not None and smoothed_shoulder is not None:
            timestamp = time.time()
            save_vision_data('vision_data.csv', timestamp, [head, shoulder], [smoothed_head, smoothed_shoulder])

            map_angle_q.put([smoothed_head, smoothed_shoulder])

def play_arm(num, que):
    global tracking_offset
    while True:
        data = que.get()
        smoothed_head = data[0]
        smoothed_shoulder = data[1]

        if tracking_offset <= 300:
            offset0 = smoothed_head['x']
            offset1 = smoothed_head['y']
            offset3 = smoothed_shoulder['y']
            offset4 = smoothed_shoulder['x']

        j3 = np.interp(smoothed_shoulder['x'] - offset4, [-0.5, 0.5], [-30, 30])
        j4 = np.interp(smoothed_shoulder['y'] - offset3, [-0.5, 0.5], [70, 120])
        j5 = np.interp(smoothed_head['x'] - offset0, [-0.5, 0.5], [-60, 60])
        j6 = np.interp(smoothed_head['y'] - offset1, [-0.5, 0.5], [-70, 70])

        p = arms[num].angles
        p[2] = j3
        p[3] = j4
        p[4] = j5
        p[5] = j6

        save_joint_data('joint_data.csv', time.time(), p)
        arms[num].set_servo_angle_j(angles=p, is_radian=False)
        tracking_offset += 1


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ROBOT = "xArms"
    PORT = 5004
    global IP
    global arms
    global strumD
    global speed
    global notes
    # global positions
    global IPN
    global IP1
    global FP1
    global IP2
    global FP2
    global IP3
    global FP3
    global IP4
    global FP4
    global BIP1
    global BFP1
    global BFP2
    # CP is current position, pass is check to see if continuity is met
    global CP
    global CPpass

    strumD = 30
    speed = 0.25
    # SIP are strings initial positions

    # for the following IP's, make sure the FP is different in every element than the IP, even if its by .1

    # IPS for different drum strikes on snare

    # IP1 is middle snare
    IP1 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
    FP1 = [0.1, 48, 0.1, 60, 0.1, -12, 0.1]
    # IP2 is pure rim
    IP2 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
    FP2 = [0.1, 65, 0.1, 88.1, 0.1, -8, 0.1]
    # IP3 is pure wood
    IP3 = [0, 41.6, -16.1, 101.7, 0, 6, 0]
    FP3 = [0.1, 60, -16.0, 81.3, 0.1, 8.2, 0.1]
    # IP4 is rimshot (rim and skin)
    IP4 = [30.0, 67.2, 21.8, 109.1, 94.7, -94.9, -31.4]
    FP4 = [30.1, 75.8, 21.9, 90.4, 94.8, -80.1, -31.5]
    # IP5 for doubles and triples
    IP5 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
    FP5 = [0.1, 48, 0.1, 60, 0.1, -12, 0.1]
    # IP6 7 and 8 are for dynamics
    IP6 = [0, 33.1, 0, 53.4, 0, -55.8, 0]
    FP6 = [0.1, 47, 0.1, 60, 0.1, -12, 0.1]
    IP7 = [0, 38.1, 0, 55.4, 0, -50.8, 0]
    FP7 = [0.1, 46, 0.1, 60, 0.1, -10, 0.1]
    IP8 = [0, 43.1, 0, 57.4, 0, -45.8, 0]
    FP8 = [0.1, 45, 0.1, 60, 0.1, -8, 0.1]
    # current position variables
    CP = [0.0, 23.1, 0.0, 51.4, 0.0, -60.8, 0.0]  # DRUMMMING
    CPpass = 0;  # 0 is no go, 1 is good to go

    # IPS for strikes on Bodhron

    BIP1 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
    BFP1 = [0.1, 53, 0.1, 60, 0.1, -12, 0.1]
    BFP2 = [0.1, 70, 0.1, 88.1, 0.1, -8, 0.1]

    SIP0 = [-0.25, 87.38, -2, 126.5, -strumD / 2, 51.73, -45]
    SIP1 = [2.62, 86.2, 0, 127.1, -strumD / 2, 50.13, -45]
    SIP2 = [1.3, 81.68, 0.0, 120, -strumD / 2, 54.2, -45]
    SIP3 = [-1.4, 83.8, 0, 120, -strumD / 2, 50.75, -45]
    SIP4 = [-1.8, 81.8, 0, 120, -strumD / 2, 50.65, -45]  # [-3.9, 65, 3.5, 100.3, -strumD/2, 42.7, 101.1]
    DRUM1 = [0.0, 23.1, 0.0, 51.4, 0.0, -60.8, 0.0]  # DRUMMMING
    DRUM2 = [0.0, 23.1, 0.0, 51.4, 0.0, -60.8, 0.0]  # DRUMMMING

    WAVE0 = [-0.25, 35.5, -2, 126.5, 101, 80.9, -45]
    WAVE1 = [2.62, 33.5, 0, 127.1, 237.6, 72.6, -57.3]
    WAVE2 = [-1.4, 29.4, 0, 120, -15, 23.1, -45]
    WAVE3 = [-1.4, 30.9, 0, 120, 48.9, 44.6, -45]
    WAVE4 = [-1.8, 30.9, 0, 120, -78.6, 44.6, -45]

    # notes for strings
    notes = np.array([64, 60, 69, 55, 62])
    # drumnote nparray
    drumnotes = np.array([58, 59, 60, 61, 62, 63, 64, 65, 66, 67])

    IP = [SIP0, SIP1, SIP2, SIP3, SIP4, DRUM1, DRUM2]
    WAVE = [WAVE0, WAVE1, WAVE2, WAVE3, WAVE4, DRUM1, DRUM2]

    wavePos = []
    global AllIP

    print(positions.IPu)
    AllIP = [IP, positions.IPu, positions.IPs, positions.IPc]

    arm0 = XArmAPI('192.168.1.206')
    # arm0 = XArmAPI('192.168.1.208')
    arm1 = XArmAPI('192.168.1.226')
    arm2 = XArmAPI('192.168.1.244')
    arm3 = XArmAPI('192.168.1.203')
    arm4 = XArmAPI('192.168.1.237')
    drumarm1 = XArmAPI('192.168.1.236')
    drumarm2 = XArmAPI('192.168.1.204')
    arms = [arm0, arm1, arm2, arm3, arm4, drumarm1, drumarm2]
    # arms = [arm1]
    totalArms = len(arms)
    setup()
    # input("lets go")
    # setup2()
    # input("letsgo again")
    for a in arms:
        a.set_mode(1)
        a.set_state(0)

    q0 = Queue()
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    dq1 = Queue()
    dq2 = Queue()
    qList = [q0, q1, q2, q3, q4, dq1, dq2]
    delayarray = np.array([[0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0], [0.1, 0.15, 0.1, 0.15, 0.125, 0.0, 0.0]])

    # nums left to right

    xArm0 = Thread(target=strummer, args=(q0, 0,))  # num 2
    xArm1 = Thread(target=strummer, args=(q1, 1,))  # num 4
    xArm2 = Thread(target=strummer, args=(q2, 2,))  # num 1
    xArm3 = Thread(target=strummer, args=(q3, 3,))  # num 3
    xArm4 = Thread(target=strummer, args=(q4, 4,))  # num 5
    drumArm1 = Thread(target=drummer, args=(dq1, 5,))  # snare
    drumArm2 = Thread(target=drummer, args=(dq2, 6,))  # bodhron

    xArm0.start()
    xArm1.start()
    xArm2.start()
    xArm3.start()
    xArm4.start()
    drumArm1.start()
    drumArm2.start()

    global basesamp
    global usamp
    basesamp = 40
    usamp = 30
    IP0us = [-0.25 - basesamp / 2, 87.5 - usamp, -2, 126.5, 0, 51.7, -45]
    IP1us = [2.67 - basesamp / 2, 86.32 - usamp, 0, 127.1, 0, 50.1,
             -45]  # [2.67 , 86.1, 0, 127.1, -strumD / 2, 50.1, -45]
    IP2us = [1.3 - basesamp / 2, 81.8 - usamp, 0, 120, 0, 54.2, -45]
    IP3us = [-1.4 - basesamp / 2, 83.95 - usamp, 0, 120, 0, 50.75, -45]  # [-0.2, 83.8, 0, 120, -strumD/2, 50.75, -45]
    IP4us = [-1.8 - basesamp / 2, 81.88 - usamp, 0, 120, 0, 50.75, -45]

    global IPus
    IPus = [IP0us, IP1us, IP2us, IP3us, IP4us]
    # global spintraj
    # spintraj = []
    # for robots in IPus:
    #     # trajmaker(robots)
    #     spintraj.append(positions.spintrajmaker(robots))
    #     # input("NEX BOT")

    UDP_IP = "0.0.0.0"  # local IP
    UDP_PORT = 12346  # port to retrieve data from Max

    dispatcher = dispatcher.Dispatcher()  # dispatcher to send
    dispatcher.map("/gesture", respond_to_gesture)
    dispatcher.map("/head", respond_to_head)

    ######################################################
    head_x = []
    head_y = []
    head_z = []
    shoulder_x = []
    shoulder_y = []
    shoulder_z = []
    map_angle_q = Queue()
    xArm1_Play = Thread(target=play_arm, args=(0, map_angle_q))


    def server():
        server = osc_server.ThreadingOSCUDPServer((UDP_IP, UDP_PORT), dispatcher)
        print("Serving on {}".format(server.server_address))
        server.serve_forever()
        atexit.register(server.server_close())


    server()
