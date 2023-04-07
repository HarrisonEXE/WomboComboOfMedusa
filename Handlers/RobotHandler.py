import time
import numpy as np
import serial
from xarm import XArmAPI
from queue import Queue
from threading import Thread
from Helpers import positions
from Helpers.Utils import createRandList, delay
from Helpers.TrajectoryGeneration import fifth_poly, spline_poly
from Helpers.DataFilters import save_joint_data


class RobotHandler:
    def __init__(self, is_lab_work=True):
        self.is_lab_work = is_lab_work
        self.lightMode = True
        self.arms = []
        self.strumD = 30
        self.speed = 0.25
        self.basesamp = 40
        self.usamp = 30
        # self.arduino = serial.Serial('/dev/ttyACM0', 9600) # for Linux
        # self.arduino = serial.Serial('com4', 9600)    # for PC
        self.IP = self.setIPs()
        self.IPus = self.setIPus()
        self.randLists = self.setRandList()

        self.qList = self.setQList()
        self.strumArmThreads = self.setStrumArmThreadList()

        self.drumQ = Queue()
        self.xArmDrumThread = Thread(
            target=self.drumController, args=(self.drumQ, 5,))

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
        return [q0, q1, q2, q3, q4]

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
        return [xArm0Thread, xArm1Thread,
                xArm2Thread, xArm3Thread, xArm4Thread]

    def setIPs(self):
        IP0 = [-0.25, 87.38, -2, 126.5, -self.strumD / 2, 51.73, -45]
        IP1 = [2.62, 86.2, 0, 127.1, -self.strumD / 2, 50.13, -45]
        IP2 = [1.3, 81.68, 0.0, 120, -self.strumD / 2, 54.2, -45]
        IP3 = [-1.4, 83.95, 0, 120, -self.strumD / 2, 50.65, -45]
        IP4 = [-1.8, 81.8, 0, 120, -self.strumD / 2, 50.65, -45]
        return [IP0, IP1, IP2, IP3, IP4]

    def setIPus(self):
        IP0us = [-0.25 - self.basesamp / 2, 87.5 -
                 self.usamp, -2, 126.5, 0, 51.7, -45]
        IP1us = [2.67 - self.basesamp / 2, 86.32 -
                 self.usamp, 0, 127.1, 0, 50.1, -45]
        IP2us = [1.3 - self.basesamp / 2, 81.8 -
                 self.usamp, 0, 120, 0, 54.2, -45]
        IP3us = [-1.4 - self.basesamp / 2, 83.95 -
                 self.usamp, 0, 120, 0, 50.75, -45]
        IP4us = [-1.8 - self.basesamp / 2, 81.88 -
                 self.usamp, 0, 120, 0, 50.75, -45]
        return [IP0us, IP1us, IP2us, IP3us, IP4us]

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
            self.arms = [arm0, arm1, arm2, arm3, arm4]

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
        for thread in self.strumArmThreads:
            thread.start()
        # self.xArmDrumThread.start()
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

    def drumController(self, queue, num):
        # drumQ.put(1)
        trajz = spline_poly(325, 35, .08, .08, 0.01)
        trajp = spline_poly(-89, -28, .08, .08, 0.01)

        trajz2 = spline_poly(325, 35, .08, .08, .1)
        trajp2 = spline_poly(-89, -28, .08, .08, .1)

        trajz3 = spline_poly(325, 35, .08, .08, .15)
        trajp3 = spline_poly(-89, -28, .08, .08, .15)

        i = 0
        while True:
            i += 1
            queue.get()

            if i % 3 == 1:
                self.drumbot(trajz, trajp, num)

            elif i % 3 == 2:
                self.drumbot(trajz2, trajp2, num)

            elif i % 3 == 0:
                self.drumbot(trajz3, trajp3, num)

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
        if play == 1:  # waving HI
            poseI = self.getAngles(num)
            # TODO: Remove hard-coded values for robot positions
            poseF = [0, 0, 0, 90, 0, 0, 0]
            newPos = self.poseToPose(poseI, poseF, 5)
            self.gotoPose(num, newPos)

        if play == 2:  # waving BYE
            poseI = self.getAngles(num)
            poseF = self.IP[num]
            newPos = self.poseToPose(poseI, poseF, 5)
            self.gotoPose(num, newPos)

        if play == 3:  # twirl
            poseI = self.getAngles(num)
            poseF = self.IPus[num]
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

    def drumbot(self, trajz, trajp, arm):
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(trajz)):
            mvpose = [492, 0, trajz[i], 180, trajp[i], 0]
            self.drums[0].set_servo_cartesian(mvpose, speed=100, mvacc=2000)

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
        self.randLists = [randList1, randList2,
                          randList3, randList4, randList5]

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
