from xarm.wrapper import XArmAPI
import numpy as np
import time
import math
import positions


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
    if delta3 < 0:
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
    if delta3 < 0:
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


class XArmController:
    def __init__(self):
        self.ROBOT = "xArms"
        self.PORT = 5004

        # Connect to the arms
        self.arm0 = XArmAPI('192.168.1.208')
        self.arm1 = XArmAPI('192.168.1.226')
        self.arm2 = XArmAPI('192.168.1.244')
        self.arm3 = XArmAPI('192.168.1.203')
        self.arm4 = XArmAPI('192.168.1.237')
        self.drumarm1 = XArmAPI('192.168.1.236')
        self.drumarm2 = XArmAPI('192.168.1.204')
        self.arms = [self.arm0, self.arm1, self.arm2, self.arm3, self.arm4, self.drumarm1, self.drumarm2]

        self.strumD = 30
        self.speed = 0.25
        self.notes = np.array([64, 60, 69, 55, 62])

        # CP is current position, pass is check to see if continuity is met

        # SIP are strings initial positions
        # for the following IP's, make sure the FP is different in every element than the IP, even if its by .1
        # IPS for different drum strikes on snare

        # self. IP1 is middle snare
        self.IP1 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        self.FP1 = [0.1, 48, 0.1, 60, 0.1, -12, 0.1]
        # IP2 is pure rim
        self.IP2 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        self.FP2 = [0.1, 65, 0.1, 88.1, 0.1, -8, 0.1]
        # IP3 is pure wood
        self.IP3 = [0, 41.6, -16.1, 101.7, 0, 6, 0]
        self.FP3 = [0.1, 60, -16.0, 81.3, 0.1, 8.2, 0.1]
        # IP4 is rimshot (rim and skin)
        self.IP4 = [30.0, 67.2, 21.8, 109.1, 94.7, -94.9, -31.4]
        self.FP4 = [30.1, 75.8, 21.9, 90.4, 94.8, -80.1, -31.5]
        # IP5 for doubles and triples
        self.IP5 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        self.self.FP5 = [0.1, 48, 0.1, 60, 0.1, -12, 0.1]
        # IP6 7 and 8 are for dynamics
        self.IP6 = [0, 33.1, 0, 53.4, 0, -55.8, 0]
        self.FP6 = [0.1, 47, 0.1, 60, 0.1, -12, 0.1]
        self.IP7 = [0, 38.1, 0, 55.4, 0, -50.8, 0]
        self.FP7 = [0.1, 46, 0.1, 60, 0.1, -10, 0.1]
        self.IP8 = [0, 43.1, 0, 57.4, 0, -45.8, 0]
        self.FP8 = [0.1, 45, 0.1, 60, 0.1, -8, 0.1]
        # current position variables
        self.CP = [0.0, 23.1, 0.0, 51.4, 0.0, -60.8, 0.0]  # DRUMMMING
        self.CPpass = 0  # 0 is no go, 1 is good to go

        # IPS for strikes on Bodhron

        self.BIP1 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        self.BFP1 = [0.1, 53, 0.1, 60, 0.1, -12, 0.1]
        self.BFP2 = [0.1, 70, 0.1, 88.1, 0.1, -8, 0.1]

        self.SIP0 = [-0.25, 87.38, -2, 126.5, -self.strumD / 2, 51.73, -45]
        self.SIP1 = [2.62, 86.2, 0, 127.1, -self.strumD / 2, 50.13, -45]
        self.SIP2 = [1.3, 81.68, 0.0, 120, -self.strumD / 2, 54.2, -45]
        self.SIP3 = [-1.4, 83.8, 0, 120, -self.strumD / 2, 50.75, -45]
        self.SIP4 = [-1.8, 81.8, 0, 120, -self.strumD / 2, 50.65, -45]  # [-3.9, 65, 3.5, 100.3, -strumD/2, 42.7, 101.1]
        self.DRUM1 = [0.0, 23.1, 0.0, 51.4, 0.0, -60.8, 0.0]  # DRUMMING
        self.DRUM2 = [0.0, 23.1, 0.0, 51.4, 0.0, -60.8, 0.0]  # DRUMMING

        self.WAVE0 = [-0.25, 35.5, -2, 126.5, 101, 80.9, -45]
        self.WAVE1 = [2.62, 33.5, 0, 127.1, 237.6, 72.6, -57.3]
        self.WAVE2 = [-1.4, 29.4, 0, 120, -15, 23.1, -45]
        self.WAVE3 = [-1.4, 30.9, 0, 120, 48.9, 44.6, -45]
        self.WAVE4 = [-1.8, 30.9, 0, 120, -78.6, 44.6, -45]

        # notes for strings
        self.notes = np.array([64, 60, 69, 55, 62])

        # drumnote nparray
        self.drumnotes = np.array([58, 59, 60, 61, 62, 63, 64, 65, 66, 67])

        self.IP = [self.SIP0, self.SIP1, self.SIP2, self.SIP3, self.SIP4, self.DRUM1, self.DRUM2]
        self.WAVE = [self.WAVE0, self.WAVE1, self.WAVE2, self.WAVE3, self.WAVE4, self.DRUM1, self.DRUM2]

        self.wavePos = []

        self.AllIP = [self.IP, positions.IPu, positions.IPs, positions.IPc]
        self.IP = [self.IP0, self.IP1, self.IP2, self.IP3, self.IP4, self.DRUM1, self.DRUM2]

        self.delayarray = np.array([[0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0], [0.1, 0.15, 0.1, 0.15, 0.125, 0.0, 0.0]])

    def setup(self):
        '''
        Enable the arms and set them to the initial position.
        '''
        for a in range(len(self.arms)):
            self.arms[a].set_simulation_robot(on_off=False)
            self.arms[a].motion_enable(enable=True)
            self.arms[a].clean_warn()
            self.arms[a].clean_error()
            self.arms[a].set_mode(0)
            self.arms[a].set_state(0)
            self.currentInitiaPos = self.IP[a]
            self.arms[a].set_servo_angle(angle=self.currentInitiaPos, wait=False, speed=10, acceleration=0.25,
                                         is_radian=False)

    def setup_pos(self):
        '''
        Setups the arms to the initial position.
        '''
        for a in range(len(self.arms)):
            self.currentInitiaPos = self.IP[a]
            self.arms[a].set_servo_angle(angle=self.currentInitiaPos, wait=False, speed=10, acceleration=0.25,
                                         is_radian=False)

    def fifth_poly(self, q_i, q_f, t):
        '''
        Fifth order polynomial trajectory.
        :param q_i: Initial position
        :param q_f: Final position
        :param t: Time of trajectory
        :return: Incremental trajectory angles
        '''
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

    def move(self, queue, arm):
        '''
        Moves the arm to the desired position.
        :param queue: Queue with the desired positions
        :param arm: Arm to move
        '''
        while True:
            if not queue.empty():
                pos = queue.get()
                arm.set_servo_angle(angle=pos, wait=False, speed=10, acceleration=0.25, is_radian=False)

    def drummer(self, inq, num):
        global CP
        global CPpass
        # for some reason have to declare CP and CPpass again lmao

        while True:
            [pnote, pvel] = inq.get()
            # play note, play velocity
            print("got!")

            # test what num is, then set trajectory depending on num

            # when num is 5 (snare)
            if num == 5:
                # get velocity as 1,2,3,4 to determine the next IP

                # if velocity is 1 (middle snare):
                if pvel == 1:
                    IPN = self.IP1
                # if velocity is 2 (pure rim):
                elif pvel == 2:
                    IPN = self.IP2
                # if velocity is 3 (pure wood):
                elif pvel == 3:
                    IPN = self.IP3
                # if velocity is 4 (rimshot):
                elif pvel == 4:
                    IPN = self.IP4
                # if velocity is 6 (least soft)
                elif (pvel == 6):
                    IPN = self.IP6
                # if velocity is 7
                elif (pvel == 7):
                    IPN = self.IP7
                # if velocity is 8 (softest)
                elif (pvel == 8):
                    IPN = self.IP8
                # if else, default to 1
                else:
                    IPN = self.IP1

                # These are hits with veloctiy 6,7,8 representing the IP!###
                # if note is 0, 1, 2 (lighter strikes)
                if pnote == 0:
                    traj1 = spline_poly(self.IP6[0], self.FP6[0], IPN[0], .3, .08, 0, .7)
                    traj2 = spline_poly(self.IP6[1], self.FP6[1], IPN[1], .3, .08, .2, 0)
                    traj3 = spline_poly(self.IP6[2], self.FP6[2], IPN[2], .3, .08, 0, .7)
                    traj4 = spline_poly(self.IP6[3], self.FP6[3], IPN[3], .3, .08, .2, 0)
                    traj5 = spline_poly(self.IP6[4], self.FP6[4], IPN[4], .3, .08, 0, 0.7)
                    traj6 = spline_poly(self.IP6[5], self.FP6[5], IPN[5], .3, .08, .2, 0)
                    traj7 = spline_poly(self.IP6[6], self.FP6[6], IPN[6], .3, .08, 0, 0.7)

                    if self.IP6 == CP:
                        CPpass = 1
                        CP = IPN

                if (pnote == 1):
                    traj1 = spline_poly(self.IP7[0], self.FP7[0], IPN[0], .38, .08, 0, .7)
                    traj2 = spline_poly(self.IP7[1], self.FP7[1], IPN[1], .38, .08, 0.2, 0)
                    traj3 = spline_poly(self.IP7[2], self.FP7[2], IPN[2], .38, .08, 0, .7)
                    traj4 = spline_poly(self.IP7[3], self.FP7[3], IPN[3], .38, .08, .2, 0)
                    traj5 = spline_poly(self.IP7[4], self.FP7[4], IPN[4], .38, .08, 0, 0.7)
                    traj6 = spline_poly(self.IP7[5], self.FP7[5], IPN[5], .38, .08, .2, 0)
                    traj7 = spline_poly(self.IP7[6], self.FP7[6], IPN[6], .38, .08, 0, 0.7)

                    if self.IP7 == CP:
                        CPpass = 1
                        CP = IPN

                if pnote == 2:
                    traj1 = spline_poly(self.IP8[0], self.FP8[0], IPN[0], .46, .08, 0, .7)
                    traj2 = spline_poly(self.IP8[1], self.FP8[1], IPN[1], .46, .08, 0.2, 0)
                    traj3 = spline_poly(self.IP8[2], self.FP8[2], IPN[2], .46, .08, 0, .7)
                    traj4 = spline_poly(self.IP8[3], self.FP8[3], IPN[3], .46, .08, .2, 0)
                    traj5 = spline_poly(self.IP8[4], self.FP8[4], IPN[4], .46, .08, 0, 0.7)
                    traj6 = spline_poly(self.IP8[5], self.FP8[5], IPN[5], .46, .08, .2, 0)
                    traj7 = spline_poly(self.IP8[6], self.FP8[6], IPN[6], .46, .08, 0, 0.7)

                    if self.IP8 == CP:
                        CPpass = 1
                        CP = IPN
                ###these are hits with veloctiy 1 representing the IP!###

                # if note is 3 (midi 60) (normal strike)
                if pnote == 3:
                    traj1 = spline_poly(self.IP1[0], self.FP1[0], IPN[0], .2, .08, 0, .7)
                    traj2 = spline_poly(self.IP1[1], self.FP1[1], IPN[1], .4, .08, 0.1, 0)
                    traj3 = spline_poly(self.IP1[2], self.FP1[2], IPN[2], .2, .08, 0, .7)
                    traj4 = spline_poly(self.IP1[3], self.FP1[3], IPN[3], .32, .08, .23, 0)
                    traj5 = spline_poly(self.IP1[4], self.FP1[4], IPN[4], .2, .08, 0, 0.7)
                    traj6 = spline_poly(self.IP1[5], self.FP1[5], IPN[5], .18, .08, .45, 0)
                    traj7 = spline_poly(self.IP1[6], self.FP1[6], IPN[6], .2, .08, 0, 0.7)

                    if self.IP1 == CP:
                        CPpass = 1
                        CP = IPN

                # if note is 4 (double strike)
                elif (pnote == 4):
                    # added .1 to all stopbots
                    traj1 = spline_poly(self.IP1[0], self.FP5[0], IPN[0], .2, .08, 0, .8)
                    traj2 = spline_poly(self.IP1[1], self.FP5[1], IPN[1], .5, .08, 0, 0.02)
                    traj3 = spline_poly(self.IP1[2], self.FP5[2], IPN[2], .2, .08, 0, .8)
                    traj4 = spline_poly(self.IP1[3], self.FP5[3], IPN[3], .32, .08, .13, 0.156)
                    traj5 = spline_poly(self.IP1[4], self.FP5[4], IPN[4], .2, .08, 0, 0.8)
                    traj6 = spline_poly(self.IP1[5], self.FP5[5], IPN[5], .2, .08, .35, 0.116)
                    traj7 = spline_poly(self.IP1[6], self.FP5[6], IPN[6], .2, .08, 0, 0.8)

                    if self.IP1 == CP:
                        CPpass = 1
                        CP = IPN

                # if note is 5 (triple strike)
                elif (pnote == 5):
                    # added .2 to all stopbots
                    traj1 = spline_poly(self.IP1[0], self.FP5[0], IPN[0], .2, .08, 0, .8)
                    traj2 = spline_poly(self.IP1[1], self.FP5[1], IPN[1], .5, .08, 0, 0.08)
                    traj3 = spline_poly(self.IP1[2], self.FP5[2], IPN[2], .2, .08, 0, .8)
                    traj4 = spline_poly(self.IP1[3], self.FP5[3], IPN[3], .32, .08, .13, 0.216)
                    traj5 = spline_poly(self.IP1[4], self.FP5[4], IPN[4], .2, .08, 0, 0.8)
                    traj6 = spline_poly(self.IP1[5], self.FP5[5], IPN[5], .2, .08, .35, 0.176)
                    traj7 = spline_poly(self.IP1[6], self.FP5[6], IPN[6], .2, .08, 0, 0.8)

                    if self.IP1 == CP:
                        CPpass = 1
                        CP = IPN

                # if note is 6 (pure rim)
                elif (pnote == 6):
                    traj1 = spline_poly(self.IP2[0], self.FP2[0], IPN[0], .2, .08, 0.05, .32)
                    traj2 = spline_poly(self.IP2[1], self.FP2[1], IPN[1], .4, .08, 0.05, 0)
                    traj3 = spline_poly(self.IP2[2], self.FP2[2], IPN[2], .2, .08, 0.05, .32)
                    traj4 = spline_poly(self.IP2[3], self.FP2[3], IPN[3], .32, .08, 0.05, 0.18)
                    traj5 = spline_poly(self.IP2[4], self.FP2[4], IPN[4], .2, .08, 0.05, 0.32)
                    traj6 = spline_poly(self.IP2[5], self.FP2[5], IPN[5], .2, .08, 0.05, 0.32)
                    traj7 = spline_poly(self.IP2[6], self.FP2[6], IPN[6], .2, .08, 0.05, 0.32)

                    if (self.IP2 == CP):
                        CPpass = 1
                        CP = IPN

                ###these are hits with veloctiy 3 representing the IP!###

                # if note is 7 (pure wood)
                elif (pnote == 7):
                    traj1 = spline_poly(self.IP3[0], self.FP3[0], IPN[0], .2, .08, 0, .32)
                    traj2 = spline_poly(self.IP3[1], self.FP3[1], IPN[1], .4, .08, 0, 0)
                    traj3 = spline_poly(self.IP3[2], self.FP3[2], IPN[2], .2, .08, 0, .32)
                    traj4 = spline_poly(self.IP3[3], self.FP3[3], IPN[3], .32, .08, 0, 0.18)
                    traj5 = spline_poly(self.IP3[4], self.FP3[4], IPN[4], .2, .08, 0, 0.32)
                    traj6 = spline_poly(self.IP3[5], self.FP3[5], IPN[5], .2, .08, 0, 0.32)
                    traj7 = spline_poly(self.IP3[6], self.FP3[6], IPN[6], .2, .08, 0, 0.32)

                    if self.IP3 == CP:
                        CPpass = 1
                        CP = IPN

                ###these are hits with veloctiy 4 representing the IP!###

                # if note is 8 (rimshot rim + skin)
                elif pnote == 8:
                    traj1 = spline_poly(self.IP4[0], self.FP4[0], IPN[0], .2, .08, 0, .32)
                    traj2 = spline_poly(self.IP4[1], self.FP4[1], IPN[1], .4, .08, 0, 0)
                    traj3 = spline_poly(self.IP4[2], self.FP4[2], IPN[2], .2, .08, 0, .32)
                    traj4 = spline_poly(self.IP4[3], self.FP4[3], IPN[3], .32, .08, 0, 0.18)
                    traj5 = spline_poly(self.IP4[4], self.FP4[4], IPN[4], .2, .08, 0, 0.32)
                    traj6 = spline_poly(self.IP4[5], self.FP4[5], IPN[5], .2, .08, 0, 0.32)
                    traj7 = spline_poly(self.IP4[6], self.FP4[6], IPN[6], .2, .08, 0, 0.32)

                    if self.IP4 == CP:
                        CPpass = 1
                        CP = IPN

            # when num is 6 (bodhron)
            elif num == 6:
                # only one IP
                CPpass = 1
                # if note is 3 (midi 60) (normal strike)
                if pnote == 3:
                    traj1 = spline_poly(self.BIP1[0], self.BFP1[0], self.BIP1[0], .2, .08, 0, .7)
                    traj2 = spline_poly(self.BIP1[1], self.BFP1[1], self.BIP1[1], .4, .08, 0.1, 0)
                    traj3 = spline_poly(self.BIP1[2], self.BFP1[2], self.BIP1[2], .2, .08, 0, .7)
                    traj4 = spline_poly(self.BIP1[3], self.BFP1[3], self.BIP1[3], .32, .08, .23, 0)
                    traj5 = spline_poly(self.BIP1[4], self.BFP1[4], self.BIP1[4], .2, .08, 0, 0.7)
                    traj6 = spline_poly(self.BIP1[5], self.BFP1[5], self.BIP1[5], .18, .08, .45, 0)
                    traj7 = spline_poly(self.BIP1[6], self.BFP1[6], self.BIP1[6], .2, .08, 0, 0.7)

                # if note is 4 (double strike)
                elif (pnote == 4):
                    # added .1 to all stopbots
                    CPpass = 0
                    traj1 = spline_poly(self.BIP1[0], self.BFP1[0], self.BIP1[0], .2, .08, 0, .8)
                    traj2 = spline_poly(self.BIP1[1], self.BFP1[1], self.BIP1[1], .5, .08, 0, 0.02)
                    traj3 = spline_poly(self.BIP1[2], self.BFP1[2], self.BIP1[2], .2, .08, 0, .8)
                    traj4 = spline_poly(self.BIP1[3], self.BFP1[3], self.BIP1[3], .32, .08, .13, 0.156)
                    traj5 = spline_poly(self.BIP1[4], self.BFP1[4], self.BIP1[4], .2, .08, 0, 0.8)
                    traj6 = spline_poly(self.BIP1[5], self.BFP1[5], self.BIP1[5], .2, .08, .35, 0.116)
                    traj7 = spline_poly(self.BIP1[6], self.BFP1[6], self.BIP1[6], .2, .08, 0, 0.8)

                # if note is 5 (triple strike)
                elif pnote == 5:
                    CPpass = 0
                    # added .2 to all stopbots
                    traj1 = spline_poly(self.BIP1[0], self.BFP1[0], self.BIP1[0], .2, .08, 0, .8)
                    traj2 = spline_poly(self.BIP1[1], self.BFP1[1], self.BIP1[1], .5, .08, 0, 0.08)
                    traj3 = spline_poly(self.BIP1[2], self.BFP1[2], self.BIP1[2], .2, .08, 0, .8)
                    traj4 = spline_poly(self.BIP1[3], self.BFP1[3], self.BIP1[3], .32, .08, .13, 0.216)
                    traj5 = spline_poly(self.BIP1[4], self.BFP1[4], self.BIP1[4], .2, .08, 0, 0.8)
                    traj6 = spline_poly(self.BIP1[5], self.BFP1[5], self.BIP1[5], .2, .08, .35, 0.176)
                    traj7 = spline_poly(self.BIP1[6], self.BFP1[6], self.BIP1[6], .2, .08, 0, 0.8)

                # outside rim of bodhron
                elif (pnote == 6):
                    traj1 = spline_poly(self.BIP1[0], self.BFP2[0], self.BIP1[0], .2, .08, 0.05, .32)
                    traj2 = spline_poly(self.BIP1[1], self.BFP2[1], self.BIP1[1], .4, .08, 0.05, 0)
                    traj3 = spline_poly(self.BIP1[2], self.BFP2[2], self.BIP1[2], .2, .08, 0.05, .32)
                    traj4 = spline_poly(self.BIP1[3], self.BFP2[3], self.BIP1[3], .32, .08, 0.15, 0.08)
                    traj5 = spline_poly(self.BIP1[4], self.BFP2[4], self.BIP1[4], .2, .08, 0.05, 0.32)
                    traj6 = spline_poly(self.BIP1[5], self.BFP2[5], self.BIP1[5], .2, .08, 0.35, 0.02)
                    traj7 = spline_poly(self.BIP1[6], self.BFP2[6], self.BIP1[6], .2, .08, 0.05, 0.32)

            # send trajectories to drumbot to perform (unless CP is not met)
            if (CPpass == 1):
                self.drumbot(traj1, traj2, traj3, traj4, traj5, traj6, traj7, num)
                CPpass = 0
            else:
                print("WARNING!!! Discontinuity")

    def drumbot(self, trajz, trajp, arm):
        # j_angles = pos
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(trajz)):
            # run command
            # start_time = time.time()
            # j_angles[4] = traj[i]
            # arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
            mvpose = [492, 0, trajz[i], 180, trajp[i], 0]
            # print(mvpose[2])
            self.arms[arm].set_servo_cartesian(mvpose, speed=100, mvacc=2000)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

    def strumbot(self, numarm, traj):
        pos = self.IP[numarm]
        j_angles = pos
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(traj)):
            # run command
            start_time = time.time()
            j_angles[4] = traj[i]
            self.arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

    def prepGesture(self, numarm, traj):
        pos = self.IP[numarm]
        j_angles = pos.copy()
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(traj)):
            # run command
            j_angles[1] = pos[1] + traj[i]
            j_angles[3] = pos[3] + traj[i]
            self.arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
            # print(j_angles)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

    def strummer(self, inq, num):
        i = 0
        uptraj = self.fifth_poly(-self.strumD / 2, self.strumD / 2, self.speed)
        downtraj = self.fifth_poly(self.strumD / 2, -self.strumD / 2, self.speed)
        both = [uptraj, downtraj]
        tension = self.fifth_poly(0, -20, 0.5)
        release = self.fifth_poly(-20, 0, 0.75)
        strumMode = 0

        while True:
            newmode, play = inq.get()  # WHERE I AM GETTING A PLAY A NOT COMMAND
            if newmode != strumMode:
                i = 0
                # posetoPose(,)
            print("got!")
            if play == 1:
                direction = i % 2
                time.sleep(self.delayarray[direction, num])  # time delay before playing
                print(num)
                print(self.delayarray[0, num])
                self.strumbot(num, both[direction])
                i += 1
            elif play == 2:
                self.prepGesture(num, tension)
                time.sleep(0.25)
                self.prepGesture(num, release)

    def robomove(self, numarm, trajectory):
        track_time = time.time()
        initial_time = time.time()
        for j_angles in trajectory:
            # run command
            start_time = time.time()
            self.arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

    def poseToPose(self, poseI, poseF, t):
        traj = []
        for p in range(len(poseI)):
            traj.append(self.fifth_poly(poseI[p], poseF[p], t))
            print(p)
        return traj

    def gotoPose(self, numarm, traj):
        track_time = time.time()
        initial_time = time.time()
        for ang in range(len(traj[0])):
            angles = [traj[0][ang], traj[1][ang], traj[2][ang], traj[3][ang], traj[4][ang], traj[5][ang], traj[6][ang]]
            start_time = time.time()
            self.arms[numarm].set_servo_angle_j(angles=angles, is_radian=False)
            # print(angles)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.001)
            initial_time += 0.004
