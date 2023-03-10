import time
import numpy as np
import serial
from xarm import XArmAPI
from queue import Queue
from threading import Thread
from Helpers.Helpers import createRandList, delay
from Helpers.TrajectoryGeneration import fifth_poly, spline_poly


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

        self.drumQ = Queue()
        self.xArmDrumThread = Thread(
            target=self.drumController, args=(self.drumQ, 5,))

        self.lightQ = Queue()
        self.lightThread = Thread(
            target=self.lightController, args=(self.lightQ,))

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
        self.arms[index].set_servo_angle(angle=[0.0, 0.0, 0.0, 1.57, 0.0, 0, 0.0], wait=False, speed=0.4, acceleration=0.25,
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

    def playString(self, noteInfo):
        degree, delay = noteInfo
        degree -= 1

        time.sleep(delay)
        print(f"Loading note {degree} with a {delay} delay")
        self.loadQueue(degree, 'X')

    def playStringTemp(self, noteInfo):
        print(f"Loading note {noteInfo}")
        self.loadQueue(noteInfo, 'X')

    def playTestStringTemp(self, robotNum):
        print(f"Loading note on robot {robotNum + 1}")

    def playTestString(self, noteInfo):
        degree, delay = noteInfo
        time.sleep(delay)
        print(f"Playing note {degree} with a {delay} delay")

    def loadQueue(self, index, delay):
        self.qList[index].put(delay)

    def loadQueues(self, indexes, delays):
        for i in indexes:
            self.qList[i].put(delays[i])

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
            queue.get()
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

    def strumbot(self, numarm, traj):
        pos = self.IP[numarm]
        j_angles = pos
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(traj)):
            j_angles[4] = traj[i]
            self.arms[numarm].set_servo_angle_j(
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
            self.delay()
            j += 1

    def sendSyncVal(self, value):
        self.arduino.write(value.encode())
        self.delay()

    # TODO: rename to something light specific

    def delay():
        time.sleep(0.013)

    def setRandList(self):
        randList1 = createRandList(4)
        randList2 = createRandList(6)
        randList3 = createRandList(6)
        randList4 = createRandList(6)
        randList5 = createRandList(4)
        self.randLists = [randList1, randList2,
                          randList3, randList4, randList5]

    def scare(self):
        self.arms[0].set_servo_angle(angle=[-.1, -15, 71.6, 83.8, -7.3, 0.8, 4.6], wait=False, speed=40, acceleration=0.6,
                                     is_radian=False)
        self.arms[1].set_servo_angle(angle=[0.0, -43.9, 26.5, 90.1, 18.1, 1.2, 28], wait=False, speed=25, acceleration=0.4,
                                     is_radian=False)
        self.arms[2].set_servo_angle(angle=[25.7, 34.9, 25.8, 93.9, -25.2, 5.1, 2], wait=False, speed=25, acceleration=0.4,
                                     is_radian=False)
        self.arms[3].set_servo_angle(angle=[41.4, 0.0, 0.0, 128.3, 0.0, 54.4, 3.1], wait=False, speed=25, acceleration=0.4,
                                     is_radian=False)
        # arms[4].set_servo_angle(angle=IP[4], wait=False, speed=20, acceleration=0.25,
        #                         is_radian=False)

    def getAngles(self, index): return self.arms[index].angles
