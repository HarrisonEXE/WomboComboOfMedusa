import time
import numpy as np
import serial
from xarm import XArmAPI
from queue import Queue
from threading import Thread
from Helpers import positions
from Helpers.Utils import createRandList, delay
from Helpers.TrajectoryGeneration import fifth_poly, fifth_poly2, spline_poly
from Helpers.DataFilters import save_joint_data


class RobotHandler:
    def __init__(self, is_lab_work=True):
        self.is_lab_work = is_lab_work
        self.lightMode = True
        self.arms = []
        self.strumD = 30
        self.speed = 0.25
        # self.arduino = serial.Serial('/dev/ttyACM0', 9600) # for Linux
        # self.arduino = serial.Serial('com4', 9600)    # for PC
        self.IP = self.setIPs()
        self.randLists = self.setRandList()

        self.qList = self.setQList()
        self.strumArmThreads = self.setStrumArmThreadList()

        # self.drumQ = Queue()
        self.xArmDrumThread1 = Thread(
            target=self.drummer, args=(5,))
        self.xArmDrumThread2 = Thread(
            target=self.drummer, args=(6,))
        self.drumvel = 3
        self.drumtrajs = self.setDrummingTraj()

        self.lightQ = Queue()
        self.lightThread = Thread(
            target=self.lightController, args=(self.lightQ,))

        # TODO: replace tracking_offest for vision pos offest with a better solution
        self.tracking_offset = 0

    def setQList(self):
        q0 = Queue()
        q1 = Queue()
        q2 = Queue()
        q3 = Queue()
        q4 = Queue()
        q5 = Queue()
        q6 = Queue()
        return [q0, q1, q2, q3, q4, q5, q6]

    def setStrumArmThreadList(self):
        xArm0Thread = Thread(
            target=self.strumController, args=(self.qList[0], 0,))  # num 2
        xArm1Thread = Thread(
            target=self.strumController, args=(self.qList[1], 1,))  # num 4
        xArm2Thread = Thread(
            target=self.strumController, args=(self.qList[2], 2,))  # num 1
        xArm3Thread = Thread(
            target=self.strumController, args=(self.qList[3], 3,))  # num 3
        xArm4Thread = Thread(
            target=self.strumController, args=(self.qList[4], 4,))  # num 5
        #drumming arms
        #parameters self,queue,num,velocity
        xArm5Thread = Thread(
            target=self.drumController, args=(self.qList[5], 5,))  # num 5
        xArm6Thread = Thread(
            target=self.drumController, args=(self.qList[6], 6,))  # num 6

        return [xArm0Thread, xArm1Thread,
                xArm2Thread, xArm3Thread, xArm4Thread, xArm5Thread, xArm6Thread]

    def setIPs(self):
        IP0 = [-0.25, 87.38, -2, 126.5, -self.strumD / 2, 51.73, -45]
        IP1 = [2.62, 86.2, 0, 127.1, -self.strumD / 2, 50.13, -45]
        IP2 = [1.3, 81.68, 0.0, 120, -self.strumD / 2, 54.2, -45]
        IP3 = [-1.4, 83.95, 0, 120, -self.strumD / 2, 50.65, -45]
        IP4 = [-1.8, 81.8, 0, 120, -self.strumD / 2, 50.65, -45]
        IP5 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        FP5 = [0, 51, 0, 60, 0, -12, 0] #refer as 7th in IP
        IP6 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        FP6 = [0, 51, 0, 60, 0, -12, 0] #refer as 8th in IP
        return [IP0, IP1, IP2, IP3, IP4, IP5, IP6, FP5, FP6]

    def setupRobots(self):
        self.buildArmsList()
        self.connectToArms()
        print("Robots are connected and ready")

    def buildArmsList(self):
        if self.is_lab_work:
            arm0 = XArmAPI('192.168.1.208')
            arm1 = XArmAPI('192.168.1.226')
            arm2 = XArmAPI('192.168.1.244')
            arm3 = XArmAPI('192.168.1.203')
            arm4 = XArmAPI('192.168.1.237')
            arm5 = XArmAPI('192.168.1.236')
            arm6 = XArmAPI('192.168.1.204')
            self.arms = [arm0, arm1, arm2, arm3, arm4, arm5, arm6]

    def connectToArms(self):
        for i in range(len(self.arms)):
            self.arms[i].set_simulation_robot(on_off=False)
            self.arms[i].motion_enable(enable=True)
            self.arms[i].clean_warn()
            self.arms[i].clean_error()
            self.arms[i].set_mode(0)
            self.arms[i].set_state(0)
            self.arms[i].set_servo_angle(angle=self.IP[i], wait=False, speed=20, acceleration=0.25,
                                         is_radian=False)
        print(f"{len(self.arms)} arms connected.")

    def startThreads(self):
        # for thread in self.strumArmThreads:
        #     thread.start()
        # self.xArmDrumThread1.start()
        # time.sleep(1.5)
        # self.xArmDrumThread2.start()
        # self.lightThread.start()
        print("Robot threads started")

    def moveToStart(self, index):
        self.arms[index].set_servo_angle(angle=[0.0, 0.0, 0.0, 1.57, 0.0, 0, 0.0], wait=False, speed=0.4,
                                         acceleration=0.25,
                                         is_radian=True)

    def moveToStrumPos(self, index):
        self.arms[index].set_servo_angle(angle=self.IP[index], wait=False,
                                         speed=20, acceleration=0.25, is_radian=False)

    def turnOffLive(self):
        for i in range(len(self.arms)):
            self.arms[i].set_mode(0)
            self.arms[i].set_state(0)
            self.moveToStart(i)
        self.waitForRobots()

    def turnOnLive(self):
        for i in range(len(self.arms)):
            self.moveToStrumPos(i)

        self.waitForRobots()

        for i in range(len(self.arms)):
            self.arms[i].set_mode(1)
            self.arms[i].set_state(0)

    def waitForRobots(self):
        not_safe_to_continue = True
        while not_safe_to_continue:
            not_safe_to_continue = False
            for arm in self.arms:
                if arm.get_is_moving():
                    not_safe_to_continue = True

    def playString(self, robotNum):
        print(f"Loading note on {robotNum + 1}")
        self.loadQueue(robotNum, 'strum', 'N/A')

    def playTestString(self, robotNum):
        print(f"Loading note on robot {robotNum + 1}")

    def loadQueue(self, index, mode, data):
        self.qList[index].put([mode, data])

    def switchLightMode(self):
        self.lightMode = not self.lightMode

    # ------------------------ Controllers ------------------------ #
    def lightController(self, lightQ):
        while True:
            if not self.lightMode:  # gradient mode
                self.sendSyncVal('gradient')
                self.listSend(self.getAngles(
                    2), self.randLists[0])  # [2,3,4,5]
                self.listSend(self.getAngles(
                    0), self.randLists[1])  # [1,2,3,4,5,6]
                self.listSend(self.getAngles(
                    3), self.randLists[2])  # [1,2,3,4,5,6]
                self.listSend(self.getAngles(
                    1), self.randLists[3])  # [1,2,3,4,5,6]
                self.listSend(self.getAngles(
                    4), self.randLists[4])  # [2,3,4,5]

            else:  # flash mode
                received = lightQ.get()
                self.sendSyncVal('flash')
                self.sendSyncVal(str(received + 1))

    def strumController(self, queue, robotNum):
        i = 0

        # TODO: Move to seperate method
        upStrumTraj = fifth_poly(-self.strumD / 2, self.strumD / 2, self.speed)
        downStrumTraj = fifth_poly(
            self.strumD / 2, -self.strumD / 2, self.speed)
        strumTrajectories = [upStrumTraj, downStrumTraj]

        while True:
            # TODO: Add "mode" parameter to all queue items across the board
            #   -- Updated it for the audio portions
            # TODO: Define "mode" policies and mappings
            mode, data = queue.get()

            if mode == 'live':
                print("Tracking Command Received for Robot " + str(robotNum))
                self.trackbot(robotNum, data)

            elif mode == 'pose':
                print("Pose Command Received for Robot " + str(robotNum))
                self.posebot(robotNum, data)

            elif mode == 'strum':
                print("Strum Command Received for Robot " + str(robotNum))

                strumDirection = i % 2

                self.lightQ.put(robotNum)
                self.strumbot(robotNum, strumTrajectories[strumDirection])

                i += 1

    def setDrummingTraj(self):

        traj2_4 = spline_poly(self.IP[5][1], self.IP[7][1] + 1, self.IP[5][1], .4, .08, 0, 0, 32, .5, 0)
        traj4_4 = spline_poly(self.IP[5][3], self.IP[7][3] + 1, self.IP[5][3], .3, .08, .13, .1, 0, .5, 0)
        traj6_4 = spline_poly(self.IP[5][5], self.IP[7][5] + 1, self.IP[5][5], .2, .08, .35, .1, 32, .5, 0)

        traj2_3 = spline_poly(self.IP[5][1], self.IP[7][1] - 2, self.IP[5][1], .5, .08, 0, 0, 24, .5, 0)
        traj4_3 = spline_poly(self.IP[5][3], self.IP[7][3] - 2, self.IP[5][3], .4, .08, .13, .1, 0, .5, 0)
        traj6_3 = spline_poly(self.IP[5][5], self.IP[7][5] - 2, self.IP[5][5], .3, .08, .35, .1, 24, .5, 0)

        traj2_2 = spline_poly(self.IP[5][1], self.IP[7][1] - 4, self.IP[5][1], .6, .08, 0, 0, 16, .5, 0)
        traj4_2 = spline_poly(self.IP[5][3], self.IP[7][3] - 4, self.IP[5][3], .5, .08, .13, .1, 0, .5, 0)
        traj6_2 = spline_poly(self.IP[5][5], self.IP[7][5] - 4, self.IP[5][5], .4, .08, .35, .1, 16, .5, 0)

        traj2_1 = spline_poly(self.IP[5][1], self.IP[7][1] - 6, self.IP[5][1], .7, .08, 0, 0, 8, .5, 0)
        traj4_1 = spline_poly(self.IP[5][3], self.IP[7][3] - 6, self.IP[5][3], .6, .08, .13, .1, 0, .5, 0)
        traj6_1 = spline_poly(self.IP[5][5], self.IP[7][5] - 6, self.IP[5][5], .5, .08, .35, .1, 8, .5, 0)

        traj2_14 = spline_poly(self.IP[5][1], self.IP[7][1] + 6, self.IP[5][1], .4, .08, 0, 0, 32, .5, 0)
        traj4_14 = spline_poly(self.IP[5][3], self.IP[7][3] + 6, self.IP[5][3], .3, .08, .13, .1, 0, .5, 0)
        traj6_14 = spline_poly(self.IP[5][5], self.IP[7][5] + 6, self.IP[5][5], .2, .08, .35, .1, 32, .5, 0)

        traj2_13 = spline_poly(self.IP[5][1], self.IP[7][1] - 2, self.IP[5][1], .5, .08, 0, 0, 24, .5, 0)
        traj4_13 = spline_poly(self.IP[5][3], self.IP[7][3] - 2, self.IP[5][3], .4, .08, .13, .1, 0, .5, 0)
        traj6_13 = spline_poly(self.IP[5][5], self.IP[7][5] - 2, self.IP[5][5], .3, .08, .35, .1, 24, .5, 0)

        traj2_12 = spline_poly(self.IP[5][1], self.IP[7][1] - 4, self.IP[5][1], .6, .08, 0, 0, 16, .5, 0)
        traj4_12 = spline_poly(self.IP[5][3], self.IP[7][3] - 4, self.IP[5][3], .5, .08, .13, .1, 0, .5, 0)
        traj6_12 = spline_poly(self.IP[5][5], self.IP[7][5] - 4, self.IP[5][5], .4, .08, .35, .1, 16, .5, 0)

        traj2_11 = spline_poly(self.IP[5][1], self.IP[7][1] - 6, self.IP[5][1], .7, .08, 0, 0, 8, .5, 0)
        traj4_11 = spline_poly(self.IP[5][3], self.IP[7][3] - 6, self.IP[5][3], .6, .08, .13, .1, 0, .5, 0)
        traj6_11 = spline_poly(self.IP[5][5], self.IP[7][5] - 6, self.IP[5][5], .5, .08, .35, .1, 8, .5, 0)

        return {
            '1': [traj2_1, traj4_1, traj6_1],
            '2': [traj2_2, traj4_2, traj6_2],
            '3': [traj2_3, traj4_3, traj6_3],
            '4': [traj2_4, traj4_4, traj6_4],
            '11': [traj2_11, traj4_11, traj6_11],
            '12': [traj2_12, traj4_12, traj6_12],
            '13': [traj2_13, traj4_13, traj6_13],
            '14': [traj2_14, traj4_14, traj6_14]
        }

    def drummer(self, num):
        t = time.time()
        while True:
            if(num == 5):
                traj2 = self.drumtrajs[str(self.drumvel)][0]
                traj4 = self.drumtrajs[str(self.drumvel)][1]
                traj6 = self.drumtrajs[str(self.drumvel)][2]
            if (num == 6):
                traj2 = self.drumtrajs[str(self.drumvel+10)][0]
                traj4 = self.drumtrajs[str(self.drumvel+10)][1]
                traj6 = self.drumtrajs[str(self.drumvel+10)][2]

            print(self.drumvel)
            if time.time() - t >= 2:
                self.drumbot(traj2, traj4, traj6, num)
                print("drum vel ", self.drumvel)
                t = time.time()

    def drumController(self, queue, num):
        #formerly known as drummer funciton
        while True:
            mode, data = queue.get()
            if data == "up":
                if self.drumvel < 4:
                    self.drumvel += 1
            elif data == "down":
                if self.drumvel > 1:
                    self.drumvel -= 1

    # --------------- Controller Helpers --------------- #
    def trackbot(self, num, data):
        head = data[0]
        shoulder = data[1]

        if self.tracking_offset <= 300:
            offset0 = head['x']
            offset1 = head['y']
            offset3 = shoulder['y']
            offset4 = shoulder['x']

        # TODO: Following values are hardcoded for xArm 0 for testing purposes, need to be generalized for all robots
        j3 = np.interp(shoulder['x'] - offset4, [-0.5, 0.5], [-30, 30])
        j4 = np.interp(shoulder['y'] - offset3, [-0.5, 0.5], [70, 120])
        j5 = np.interp(head['x'] - offset0, [-0.5, 0.5], [-60, 60])
        j6 = np.interp(head['y'] - offset1, [-0.5, 0.5], [-70, 70])

        p = self.getAngles(num)
        p[2] = j3
        p[3] = j4
        p[4] = j5
        p[5] = j6

        # save_joint_data('joint_data.csv', time.time(), p)
        self.setAngles(num, p)
        self.tracking_offset += 1

    def posebot(self, num, play):
        if play == 1:  # stop
            poseI = self.getAngles(num)
            # TODO: Remove hard-coded values for robot positions
            poseF = [0, 0, 0, 90, 0, 0, 0]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)

        if play == 2:  # go
            poseI = self.getAngles(num)
            poseF = self.IP[num]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)

        if play == 3:  # swipe_left
            poseI = self.getAngles(num)
            poseF = positions.IPc[num]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)
            self.robomove(num, positions.circletraj[num])

        if play == 4:  # swipe_left
            poseI = self.getAngles(num)
            poseF = positions.IPw[num]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)
            self.robomove(num, positions.wtraj[num])

        if play == 5:  # twirl
            poseI = self.getAngles(num)
            poseF = positions.IPus[num]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)
            self.robomove(num, positions.spintraj[num])

    def strumbot(self, num, traj):
        pos = self.IP[num]
        j_angles = pos
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(traj)):
            j_angles[4] = traj[i]
            self.arms[num].set_servo_angle_j(
                angles=j_angles, is_radian=False)

            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

    def drumbot(self, traj2, traj4, traj6, arm):
        # def drumbot(traj1, traj2, traj3, traj4, traj5, traj6, traj7, arm):

        # j_angles = pos
        track_time = time.time()
        initial_time = time.time()
        for i in range(min(len(traj2), len(traj4), len(traj6))):
            # for i in range(min(len(traj1),len(traj2),len(traj3),len(traj4),len(traj5),len(traj6),len(traj7))):
            # run command
            # start_time = time.time()
            # j_angles[4] = traj[i]
            # arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)

            jointangles = [0, traj2[i], 0, traj4[i], 0, traj6[i], 0]
            # jointangles = [traj1[i], traj2[i], traj3[i], traj4[i], traj5[i], traj6[i], traj7[i]]

            print(traj6[i])
            self.arms[arm].set_servo_angle_j(angles=jointangles, is_radian=False)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

    # Picks and sends indexes, defined by anglesToSend, of a 6 item list, defined by listToSend
    def listSend(self, listToSend, anglesToSend):
        sentList = []
        j = 0
        for i in anglesToSend:
            sentList.append(
                (round(listToSend[i] * 2.5 * 256 / 360)) % 256 + (i * 20))
            self.arduino.write(str(sentList[j]).encode())
            delay()
            j += 1

    def sendSyncVal(self, value):
        self.arduino.write(value.encode())
        delay()

    # TODO: rename to something light specific
    def setRandList(self):
        randList1 = createRandList(4)
        randList2 = createRandList(6)
        randList3 = createRandList(6)
        randList4 = createRandList(6)
        randList5 = createRandList(4)
        return [randList1, randList2, randList3, randList4, randList5]

    def scare(self):
        self.arms[0].set_servo_angle(angle=[-.1, -15, 71.6, 83.8, -7.3, 0.8, 4.6], wait=False, speed=40,
                                     acceleration=0.6,
                                     is_radian=False)
        self.arms[1].set_servo_angle(angle=[0.0, -43.9, 26.5, 90.1, 18.1, 1.2, 28], wait=False, speed=25,
                                     acceleration=0.4,
                                     is_radian=False)
        self.arms[2].set_servo_angle(angle=[25.7, 34.9, 25.8, 93.9, -25.2, 5.1, 2], wait=False, speed=25,
                                     acceleration=0.4,
                                     is_radian=False)
        self.arms[3].set_servo_angle(angle=[41.4, 0.0, 0.0, 128.3, 0.0, 54.4, 3.1], wait=False, speed=25,
                                     acceleration=0.4,
                                     is_radian=False)
        # arms[4].set_servo_angle(angle=IP[4], wait=False, speed=20, acceleration=0.25,
        #                         is_radian=False)

    def getAngles(self, index):
        return self.arms[index].angles

    def setAngles(self, index, angles, is_radian=False):
        self.arms[index].set_servo_angle_j(angles=angles, is_radian=is_radian)

    # --------------- Controller Helpers - Needs Refactoring --------------- #
    # TODO: Refactor this to be more readable, and less redundant
    # TODO: Check if duplicate code can be removed
    def poseToPose(self, poseI, poseF, t):
        traj = []
        for p in range(len(poseI)):
            traj.append(fifth_poly(poseI[p], poseF[p], t))
        return traj

    def gotoPose(self, num, traj):
        track_time = time.time()
        initial_time = time.time()
        for ang in range(len(traj[0])):
            angles = [traj[0][ang], traj[1][ang], traj[2][ang],
                      traj[3][ang], traj[4][ang], traj[5][ang], traj[6][ang]]
            self.setAngles(num, angles, is_radian=False)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.001)
            initial_time += 0.004

    def robomove(self, num, trajectory):
        track_time = time.time()
        initial_time = time.time()
        for j_angles in trajectory:
            self.setAngles(num, j_angles, is_radian=False)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004
