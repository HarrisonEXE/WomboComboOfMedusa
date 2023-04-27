import math
import time
import numpy as np
import serial
from xarm import XArmAPI
from queue import Queue
from threading import Thread, Event
from Helpers import positions
from Helpers.Utils import createRandList, delay
from Helpers.TrajectoryGeneration import fifth_poly, fifth_poly2, spline_poly
from Helpers.DataFilters import save_joint_data, save_vision_data
from Helpers import VisionResponse


class RobotHandler:

    def __init__(self, is_lab_work=True):
        self.is_lab_work = is_lab_work
        self.lightMode = False
        self.arms = []
        self.strumD = 30
        self.speed = 0.25
        self.arduino = serial.Serial('/dev/ttyACM0', 9600) # for Linux
        # self.arduino = serial.Serial('com4', 9600)    # for PC
        self.IP = self.setIPs()
        self.randLists = self.setRandList()

        self.qList = self.setQList()
        self.armThreads = self.setArmThreadList()

        # self.xArmDrumThread1 = Thread(target=self.drummer, args=(5,))
        # self.xArmDrumThread2 = Thread(target=self.drummer, args=(6,))
        self.xArmDrummers = Thread(target=self.drummer2, args=([5, 6],))
        self.drumvel = 3
        self.drumtrajs = self.setDrummingTraj()
        self.playDrums = False
        self.playDrums_event = Event()

        self.lightQ = Queue()
        self.lightThread = Thread(
            target=self.lightController, args=(self.lightQ,))

        self.offset_counter = 0
        self.tracking_offsets = [0., 0., 0., 0., 0., 0.]
        self.IPts = self.setIPts()
        self.reset_discontinuity = [True, True, True, True, True]
        self.temp_time_tracker = 0
        self.custom_mappings = self.setVisionMappings()

    def setQList(self):
        q0 = Queue()
        q1 = Queue()
        q2 = Queue()
        q3 = Queue()
        q4 = Queue()
        q5 = Queue()
        q6 = Queue()
        return [q0, q1, q2, q3, q4, q5, q6]

    def setArmThreadList(self):
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
        xArm5Thread = Thread(
            target=self.drumController, args=(self.qList[5],))  # num 5
        xArm6Thread = Thread(
            target=self.drumController, args=(self.qList[6],))  # num 6

        return [xArm0Thread, xArm1Thread, xArm2Thread, xArm3Thread, xArm4Thread, xArm5Thread, xArm6Thread]

    def setIPs(self):
        IP0 = [-0.25, 87.38, -2, 126.5, -self.strumD / 2, 51.73, -45]
        IP1 = [2.62, 86.2, 0, 127.1, -self.strumD / 2, 50.13, -45]
        IP2 = [1.3, 81.68, 0.0, 120, -self.strumD / 2, 54.2, -45]
        IP3 = [-1.4, 83.95, 0, 120, -self.strumD / 2, 50.65, -45]
        IP4 = [-1.8, 81.8, 0, 120, -self.strumD / 2, 50.65, -45]
        IP5 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        FP5 = [0, 51, 0, 60, 0, -12, 0]  # refer as 7th in IP
        IP6 = [0, 23.1, 0, 51.4, 0, -60.8, 0]
        FP6 = [0, 51, 0, 60, 0, -12, 0]  # refer as 8th in IP
        return [IP0, IP1, IP2, IP3, IP4, IP5, IP6, FP5, FP6]

    def setIPts(self):
        """
        Initial Positions for tracking mode
        """
        IPt0 = [0, -51.0, -2, 95.0, 0, 10.1, -45]
        IPt1 = [-3.5, -51.6, -7.5, 93.3, -3, 12., -45]
        IPt2 = [40.0, 10, 0, 118, 0, 37.5, -45]
        IPt3 = [-1.4, 56.8, 0, 180, -15, 44.5, -45]
        IPt4 = [-30, 20, 0, 135, 0, 35.5, -45]
        return [IPt0, IPt1, IPt2, IPt3, IPt4]

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
        for thread in self.armThreads:
            thread.start()
        self.xArmDrummers.start()
        self.lightThread.start()
        print("Robot threads started")
        # self.xArmDrumThread1.start()
        # self.xArmDrumThread2.start()

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

    def flashLights(self, robotNum):
        self.switchLightMode()
        # self.sendSyncVal('flash')
        # self.sendSyncVal(str(robotNum + 1))
        # print("LETS GOOO")
        self.lightQ.put(robotNum)
        # print("send flash")
        # self.sendSyncVal('flash')
        # self.sendSyncVal(str(2))
        # self.switchLightMode()

    # ------------------------ Controllers ------------------------ #
    def lightController(self, lightQ):
        flash_count = 0
        while True:
            if not self.lightMode:  # gradient mode
                # print("BACK TO NORMAL")
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

                flash_count += 1
                # print("Flash count : ", flash_count)
                # print("FLASHED")
                # self.sendSyncVal('flash')

                if flash_count == 5:
                    flash_count = 0
                    time.sleep(1)
                    self.sendSyncVal('flash')
                    self.sendSyncVal(str(1))
                    self.sendSyncVal('flash')
                    self.sendSyncVal(str(2))
                    self.sendSyncVal('flash')
                    self.sendSyncVal(str(3))
                    self.sendSyncVal('flash')
                    self.sendSyncVal(str(4))
                    self.sendSyncVal('flash')
                    self.sendSyncVal(str(5))
                    time.sleep(1)
                    self.switchLightMode()
                    print("switch lights back")

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
                # print("Tracking Command Received for Robot " + str(robotNum))
                self.trackbot(robotNum, data)

            elif mode == 'pose':
                print("Pose Command Received for Robot " + str(robotNum))
                self.flashLights(robotNum)
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

        traj2_2 = spline_poly(self.IP[5][1], self.IP[7][1] - 4, self.IP[5][1], .6, .08, 0, 0, 16, .35, 0)
        traj4_2 = spline_poly(self.IP[5][3], self.IP[7][3] - 4, self.IP[5][3], .5, .08, .13, .1, 0, .35, 0)
        traj6_2 = spline_poly(self.IP[5][5], self.IP[7][5] - 4, self.IP[5][5], .4, .08, .35, .1, 16, .35, 0)

        traj2_1 = spline_poly(self.IP[5][1], self.IP[7][1] - 6, self.IP[5][1], .7, .08, 0, 0, 8, .25, 0)
        traj4_1 = spline_poly(self.IP[5][3], self.IP[7][3] - 6, self.IP[5][3], .6, .08, .13, .06, 0, .25, 0)
        traj6_1 = spline_poly(self.IP[5][5], self.IP[7][5] - 6, self.IP[5][5], .5, .08, .35, .1, 8, .25, 0)

        traj2_14 = spline_poly(self.IP[5][1], self.IP[7][1] + 6, self.IP[5][1], .4, .08, 0, 0, 32, .5, 0)
        traj4_14 = spline_poly(self.IP[5][3], self.IP[7][3] + 6, self.IP[5][3], .3, .08, .13, .1, 0, .5, 0)
        traj6_14 = spline_poly(self.IP[5][5], self.IP[7][5] + 6, self.IP[5][5], .2, .08, .35, .1, 32, .5, 0)

        traj2_13 = spline_poly(self.IP[5][1], self.IP[7][1] - 2, self.IP[5][1], .5, .08, 0, 0, 24, .5, 0)
        traj4_13 = spline_poly(self.IP[5][3], self.IP[7][3] - 2, self.IP[5][3], .4, .08, .13, .1, 0, .5, 0)
        traj6_13 = spline_poly(self.IP[5][5], self.IP[7][5] - 2, self.IP[5][5], .3, .08, .35, .1, 24, .5, 0)

        traj2_12 = spline_poly(self.IP[5][1], self.IP[7][1] - 4, self.IP[5][1], .6, .08, 0, 0, 16, .35, 0)
        traj4_12 = spline_poly(self.IP[5][3], self.IP[7][3] - 4, self.IP[5][3], .5, .08, .13, .1, 0, .35, 0)
        traj6_12 = spline_poly(self.IP[5][5], self.IP[7][5] - 4, self.IP[5][5], .4, .08, .35, .1, 16, .35, 0)

        traj2_11 = spline_poly(self.IP[5][1], self.IP[7][1] - 6, self.IP[5][1], .7, .08, 0, 0, 8, .25, 0)
        traj4_11 = spline_poly(self.IP[5][3], self.IP[7][3] - 6, self.IP[5][3], .6, .08, .13, .06, 0, .25, 0)
        traj6_11 = spline_poly(self.IP[5][5], self.IP[7][5] - 6, self.IP[5][5], .5, .08, .35, .1, 8, .25, 0)

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

    # def drummer(self, num):
    #     t = time.time()
    #     while True:
    #         if num == 5:
    #             traj2 = self.drumtrajs[str(self.drumvel)][0]
    #             traj4 = self.drumtrajs[str(self.drumvel)][1]
    #             traj6 = self.drumtrajs[str(self.drumvel)][2]
    #
    #         if num == 6:
    #             traj2 = self.drumtrajs[str(self.drumvel+10)][0]
    #             traj4 = self.drumtrajs[str(self.drumvel+10)][1]
    #             traj6 = self.drumtrajs[str(self.drumvel+10)][2]
    #
    #         if time.time() - t >= 3 and num == 6:
    #             self.drumbot(traj2, traj4, traj6, num)
    #             print("drum vel ", self.drumvel)
    #             t = time.time()

    def drummer2(self, arms):

        t_init = time.time()
        print("initial time set")

        t1 = time.time()
        t2 = time.time()

        #these are the delays for different hit velocities
        delayarray = [0, .042, .076, .195]

        # Create an Event object
        self.playDrums_event = Event()

        while True:

            if self.playDrums:
                t1 = time.time()
                if arms[0] == 5:
                    traj2 = self.drumtrajs[str(self.drumvel)][0]
                    traj4 = self.drumtrajs[str(self.drumvel)][1]
                    traj6 = self.drumtrajs[str(self.drumvel)][2]

                    if (t1 - t_init)%2.353 > delayarray[self.drumvel-1] and (t1 - t_init)%2.353 < (delayarray[self.drumvel-1] + 0.01):
                        #wait for any extra time, so slow hits hit at the same time as hard
                        self.drumbot(traj2, traj4, traj6, arms[0])


                # if arms[1] == 6:
                #     traj2 = self.drumtrajs[str(self.drumvel + 10)][0]
                #     traj4 = self.drumtrajs[str(self.drumvel + 10)][1]
                #     traj6 = self.drumtrajs[str(self.drumvel + 10)][2]
                #     if time.time() - t2 >= 2:
                #         #self.drumbot(traj2, traj4, traj6, arms[1])
                #         t2 = time.time()

            else:
                self.playDrums_event.clear()
                self.playDrums_event.wait()

    def drumController(self, queue):
        # formerly known as drummer function
        while True:
            mode, data = queue.get()
            if data == "up":
                if self.drumvel < 4:
                    self.drumvel += 1
                    print("drum vel increased to", self.drumvel)
            elif data == "down":
                if self.drumvel > 1:
                    self.drumvel -= 1
                    print("drum vel decreased to", self.drumvel)

    # --------------- Controller Helpers --------------- #
    def trackbot(self, num, data):
        save_joint_data(f'logs/joint_data_arm_{num}.csv', time.time(), self.getAngles(num))

        if self.reset_discontinuity[num]:
            self.reset_discontinuity[num] = False
            poseI = self.getAngles(num)
            poseF = self.IPts[num]

            newPos = self.poseToPose(poseI, poseF, 6)
            self.gotoPose(num, newPos)
            print("tracking position set")
        else:
            head = {'x': float(data[0]), 'y': float(data[1]), 'z': float(data[2])}
            shoulder = {'x': float(data[3]), 'y': float(data[4]), 'z': float(data[5])}

            if self.offset_counter < 150:
                if self.offset_counter == 0:
                    self.temp_time_tracker = time.time()
                self.tracking_offsets[0] += head['x']
                self.tracking_offsets[1] += head['y']
                self.tracking_offsets[2] += head['z']
                self.tracking_offsets[3] += shoulder['x']
                self.tracking_offsets[4] += shoulder['y']
                self.tracking_offsets[5] += shoulder['z']
                self.offset_counter += 1
            elif self.offset_counter == 150:
                self.tracking_offsets = [x / 150 for x in self.tracking_offsets]
                self.offset_counter += 1
                self.temp_time_tracker = time.time() - self.temp_time_tracker
                print("Time to get offsets: ", self.temp_time_tracker, "secs")
            else:
                offset_head = {
                    'x': head['x'] - self.tracking_offsets[0],
                    'y': head['y'] - self.tracking_offsets[1],
                    'z': head['z'] - self.tracking_offsets[2]
                }
                offset_shoulder = {
                    'x': shoulder['x'] - self.tracking_offsets[3],
                    'y': shoulder['y'] - self.tracking_offsets[4],
                    'z': shoulder['z'] - self.tracking_offsets[5]
                }

                save_vision_data(f'logs/vision_data_arm_{num}.csv', time.time(),
                                 smoothed_values=[*offset_head.values(), *offset_shoulder.values()])

                # TODO: add individual mappings for each arm and add more joints
                j1 = self.apply_vision_mapping(num, joint=0, value=offset_head['x'])
                j2 = self.apply_vision_mapping(num, joint=1, value=offset_head['y'])
                j3 = self.apply_vision_mapping(num, joint=2, value=offset_head['x'])
                j4 = self.apply_vision_mapping(num, joint=3, value=offset_head['y'])
                j5 = self.apply_vision_mapping(num, joint=4, value=offset_head['x'])
                j6 = self.apply_vision_mapping(num, joint=5, value=offset_head['x'])
                # j7 = self.apply_vision_mapping(num, joint=6, value=offset_head['x'])

                p = self.getAngles(num)
                p[0] = j1
                p[1] = j2
                p[2] = j3
                p[3] = j4
                p[4] = j5
                p[5] = j6
                # p[7] = j7

                self.setAngles(num, p)

    def preprogrambot(self, num):
        # controls playing of hope's preprogrammed dance
        # get gestures to play based on curr arm number
        gestures_to_play = VisionResponse.hope_arm_gesture[num]
        poseI = self.getAngles(num)
        newPos = self.poseToPose(poseI, [0, 0, 0, 90, 0, 0, 0], 8)
        hope_gestures = []
        for i in gestures_to_play:
            hope_gestures.append(VisionResponse.make_traj(i))

        self.gotoPose(num, newPos)
        for arm_gest in hope_gestures:
            self.k_robomove(num, arm_gest)

    def posebot(self, num, play):

        VisionResponse.is_moving = True
        if play == 1:  # stop
            VisionResponse.rtp_dispatcher.map("anything", 1)
            poseI = self.getAngles(num)
            # TODO: Remove hard-coded values for robot positions
            poseF = [0, 0, 0, 90, 0, 0, 0]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)


        if play == 2:  # go
            VisionResponse.rtp_dispatcher.map("anything", 1)
            poseI = self.getAngles(num)
            poseF = self.IP[num]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)

        if play == 3:  # swipe_left
            # TODO: SEND TO UDP FOR POSE DETECTION SOUND
            #VisionResponse.dispatcher.map("/anything", 1)
            VisionResponse.rtp_dispatcher.map("anything", 1)
            poseI = self.getAngles(num)
            poseF = positions.IPc[num]
            #newPos = self.poseToPose(poseI, poseF, 4)
            newPos = self.poseToPose(poseI, [0,0,0,90,0,0,0],8)
            curr_gesture = VisionResponse.choose_swipe_gesture()
            swipe_left_response = VisionResponse.make_traj(curr_gesture)
            # print(len(swipe_left_response))
            # # for item in swipe_left_response:
            # #     print(item[0])
            # print(swipe_left_response[0][0:20])
            self.gotoPose(num, newPos)
            self.k_robomove(num, swipe_left_response)

        if play == 4:  # swipe_left
            VisionResponse.rtp_dispatcher.map("anything", 1)
            poseI = self.getAngles(num)
            poseF = positions.IPw[num]
            #newPos = self.poseToPose(poseI, poseF, 4)
            newPos = self.poseToPose(poseI, [0,0,0,90,0,0,0], 8)
            curr_gesture = VisionResponse.choose_swipe_gesture()
            swipe_right_response = VisionResponse.make_traj(curr_gesture)
            self.gotoPose(num, newPos)
            self.k_robomove(num, swipe_right_response)


        if play == 5:  # twirl
            VisionResponse.rtp_dispatcher.map("anything", 1)
            poseI = self.getAngles(num)
            poseF = positions.IPus[num]
            newPos = self.poseToPose(poseI, poseF, 4)
            self.gotoPose(num, newPos)
            self.robomove(num, positions.spintraj[num])

        VisionResponse.is_moving = False
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

    def getIsMoving(self):
        return self.is_moving

    def drumbot(self, traj2, traj4, traj6, arm):
        a = time.time()
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
            # print(traj6[i])
            self.arms[arm].set_servo_angle_j(angles=jointangles, is_radian=False)

            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004

        #Used to track how long a drum strike takes
        #print("arm: ", arm, "traj time: ", time.time() - a)

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

    def setVisionMappings(self):
        """
        return: {arm_num : {joint_num: {'input_range': [], 'output_range':[]}}}
        """
        # TODO: 2. Add more joint mappings
        # TODO: 3. Measure max range of vision data values
        return {
            0: {
                0: {'input_range': [-0.5, 0.5], 'output_range': [22, -22]},
                2: {'input_range': [-0.5, 0.5], 'output_range': [78, -82]},
                4: {'input_range': [-0.5, 0.5], 'output_range': [60, -60]},
            },
            1: {
                0: {'input_range': [-0.5, 0.5], 'output_range': [25, -32]},
                2: {'input_range': [-0.5, 0.5], 'output_range': [45, -60]},
                4: {'input_range': [-0.5, 0.5], 'output_range': [-66, 60]},
            },
            2: {
                1: {'input_range': [-0.5, 0.5], 'output_range': [-40, 60]},
                3: {'input_range': [-0.5, 0.5], 'output_range': [50, 186]},
                5: {'input_range': [-0.5, 0.5], 'output_range': [-13, 88]},
            },
            3: {
                3: {'input_range': [-0.5, 0.5], 'output_range': [150, 210]},
                4: {'input_range': [-0.5, 0.5], 'output_range': [30, -60]},
            },
            4: {
                1: {'input_range': [-0.5, 0.5], 'output_range': [-30, 70]},
                3: {'input_range': [-0.5, 0.5], 'output_range': [60, 210]},
                5: {'input_range': [-0.5, 0.5], 'output_range': [-13, 82]},
            },
        }

    def apply_vision_mapping(self, num, joint, value):
        current_angles = self.getAngles(num)

        if num in self.custom_mappings and joint in self.custom_mappings[num]:
            mapping = self.custom_mappings[num][joint]
            return np.interp(value, mapping['input_range'], mapping['output_range'])
        else:
            return current_angles[joint]

    def switch_drum_state(self):
        self.playDrums = not self.playDrums
        self.playDrums_event.set()

    def k_robomove(self, num, trajectory):
        track_time = time.time()
        initial_time = time.time()
        for i in range(len(trajectory[0])):
            angles = [trajectory[0][i], trajectory[1][i], trajectory[2][i],
                      trajectory[3][i], trajectory[4][i], trajectory[5][i],
                      trajectory[6][i]]
            self.setAngles(num, angles, is_radian=False)
            while track_time < initial_time + 0.004:
                track_time = time.time()
                time.sleep(0.0001)
            initial_time += 0.004