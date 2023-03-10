import time
import numpy as np
import serial
# from xarm import XArmAPI
from queue import Queue
from threading import Thread
from Helpers.Helpers import createRandList, delay
from Helpers.TrajectoryGeneration import fifth_poly, spline_poly


# ------------------------ Robot Logistics ------------------------ #
def setupRobots(is_lab_work):
    buildArmsList(is_lab_work)
    connectToArms()
    print("Robots are connected and ready")


def buildArmsList(is_lab_work):
    global arms
    if is_lab_work:
        arm0 = XArmAPI('192.168.1.208')
        arm1 = XArmAPI('192.168.1.226')
        arm2 = XArmAPI('192.168.1.244')
        arm3 = XArmAPI('192.168.1.203')
        arm4 = XArmAPI('192.168.1.237')
        arms = [arm0, arm1, arm2, arm3, arm4]
    else:
        arms = []


def connectToArms():
    for i in range(len(arms)):
        arms[i].set_simulation_robot(on_off=False)
        arms[i].motion_enable(enable=True)
        arms[i].clean_warn()
        arms[i].clean_error()
        arms[i].set_mode(0)
        arms[i].set_state(0)
        arms[i].set_servo_angle(angle=IP[i], wait=False, speed=20, acceleration=0.25,
                                is_radian=False)
    print(f"{len(arms)} arms connected.")


def moveToStart(index):
    arms[index].set_servo_angle(angle=[0.0, 0.0, 0.0, 1.57, 0.0, 0, 0.0], wait=False, speed=0.4, acceleration=0.25,
                                is_radian=True)


def moveToStrumPos(index):
    arms[index].set_servo_angle(angle=IP[index], wait=False,
                                speed=20, acceleration=0.25, is_radian=False)


def turnOffLive():
    for i in range(len(arms)):
        arms[i].set_mode(0)
        arms[i].set_state(0)
        moveToStart(i)
    waitForRobots()


def turnOnLive():
    for i in range(len(arms)):
        moveToStrumPos(i)

    waitForRobots()

    for i in range(len(arms)):
        arms[i].set_mode(1)
        arms[i].set_state(0)


def waitForRobots():
    not_safe_to_continue = True
    while not_safe_to_continue:
        not_safe_to_continue = False
        for arm in arms:
            if arm.get_is_moving():
                not_safe_to_continue = True


# ------------------------ Client Facing ------------------------ #
def playString(noteInfo):
    degree, delay = noteInfo
    degree -= 1

    time.sleep(delay)
    print(f"Loading note {degree} with a {delay} delay")
    loadQueue(degree, 'X')


def playStringTemp(noteInfo):
    print(f"Loading note {noteInfo}")
    loadQueue(noteInfo, 'X')


def playTestStringTemp(robotNum):
    print(f"Loading note on robot {robotNum + 1}")


def playTestString(noteInfo):
    degree, delay = noteInfo
    time.sleep(delay)
    print(f"Playing note {degree} with a {delay} delay")


def loadQueue(index, delay):
    qList[index].put(delay)


def loadQueues(indexes, delays):
    for i in indexes:
        qList[i].put(delays[i])


def scare():
    arms[0].set_servo_angle(angle=[-.1, -15, 71.6, 83.8, -7.3, 0.8, 4.6], wait=False, speed=40, acceleration=0.6,
                            is_radian=False)
    arms[1].set_servo_angle(angle=[0.0, -43.9, 26.5, 90.1, 18.1, 1.2, 28], wait=False, speed=25, acceleration=0.4,
                            is_radian=False)
    arms[2].set_servo_angle(angle=[25.7, 34.9, 25.8, 93.9, -25.2, 5.1, 2], wait=False, speed=25, acceleration=0.4,
                            is_radian=False)
    arms[3].set_servo_angle(angle=[41.4, 0.0, 0.0, 128.3, 0.0, 54.4, 3.1], wait=False, speed=25, acceleration=0.4,
                            is_radian=False)
    # arms[4].set_servo_angle(angle=IP[4], wait=False, speed=20, acceleration=0.25,
    #                         is_radian=False)


# ------------------------ Controllers ------------------------ #
def lightController(lightQ):
    while True:
        # print(lightMode)
        if not lightMode:  # gradient mode
            sendSyncVal('gradient')
            listSend(getAngles(2), randList1)  # [2, 3, 4, 5])
            listSend(getAngles(0), randList2)  # [1, 2, 3, 4, 5, 6])
            listSend(getAngles(3), randList3)  # [1, 2, 3, 4, 5, 6])
            listSend(getAngles(1), randList4)  # [1, 2, 3, 4, 5, 6])
            listSend(getAngles(4), randList5)  # [2, 3, 4, 5])

        if lightMode:  # flash mode
            received = lightQ.get()
            sendSyncVal('flash')
            sendSyncVal(str(received + 1))


def switchLightMode():
    global lightMode
    lightMode = not lightMode
    # print("made it to light mode")


def strumController(queue, robotNum):
    i = 0

    # TODO: Move to seperate method
    upStrumTraj = fifth_poly(-strumD / 2, strumD / 2, speed)
    downStrumTraj = fifth_poly(strumD / 2, -strumD / 2, speed)
    strumTrajectories = [upStrumTraj, downStrumTraj]

    while True:
        queue.get()
        print("Strum Command Received for Robot " + str(robotNum))

        strumDirection = i % 2

        # time.sleep(delayArray[strumDirection, robotNum])
        lightQ.put(robotNum)
        strumbot(robotNum, strumTrajectories[strumDirection])

        i += 1


def drumController(queue, num):
    drumQ.put(1)
    trajz = spline_poly(325, 35, .08, .08, 0.01)
    trajp = spline_poly(-89, -28, .08, .08, 0.01)

    trajz2 = spline_poly(325, 35, .08, .08, .1)
    trajp2 = spline_poly(-89, -28, .08, .08, .1)

    trajz3 = spline_poly(325, 35, .08, .08, .15)
    trajp3 = spline_poly(-89, -28, .08, .08, .15)

    i = 0
    while True:
        i += 1
        play = queue.get()

        if i % 3 == 1:
            drumbot(trajz, trajp, num)

        elif i % 3 == 2:
            drumbot(trajz2, trajp2, num)

        elif i % 3 == 0:
            drumbot(trajz3, trajp3, num)


# --------------- Controller Helpers --------------- #
def strumbot(numarm, traj):
    pos = IP[numarm]
    j_angles = pos
    track_time = time.time()
    initial_time = time.time()
    for i in range(len(traj)):
        j_angles[4] = traj[i]
        arms[numarm].set_servo_angle_j(angles=j_angles, is_radian=False)

        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


def drumbot(trajz, trajp, arm):
    track_time = time.time()
    initial_time = time.time()
    for i in range(len(trajz)):
        mvpose = [492, 0, trajz[i], 180, trajp[i], 0]
        drums[0].set_servo_cartesian(mvpose, speed=100, mvacc=2000)

        while track_time < initial_time + 0.004:
            track_time = time.time()
            time.sleep(0.0001)
        initial_time += 0.004


# Picks and sends indexes, defined by anglesToSend, of a 6 item list, defined by listToSend
def listSend(listToSend, anglesToSend):
    sentList = []
    j = 0
    for i in anglesToSend:
        sentList.append(
            (round(listToSend[i] * 2.5 * 256 / 360)) % 256 + (i * 20))
        arduino.write(str(sentList[j]).encode())
        delay()
        j += 1


def sendSyncVal(value):
    arduino.write(value.encode())
    delay()


# ------------------------ Weird Robot Stuff ------------------------ #
global IP
global arms
global drums
global strumD
global speed
global notes
global lightMode
global lightQ
global arduino


# --------------- Light Attributes --------------- #
lightMode = False
# arduino = serial.Serial('/dev/ttyACM0', 9600)
# arduino = serial.Serial('com4', 9600)    # for PC

randList1 = createRandList(4)
randList2 = createRandList(6)
randList3 = createRandList(6)
randList4 = createRandList(6)
randList5 = createRandList(4)

# --------------- Arm Attributes --------------- #
ROBOT = "xArms"
PORT = 5003
speed = 0.25

# --------------- Initial Positinos --------------- #
strumD = 30
IP0 = [-0.25, 87.38, -2, 126.5, -strumD / 2, 51.73, -45]
IP1 = [2.62, 86.2, 0, 127.1, -strumD / 2, 50.13, -45]
IP2 = [1.3, 81.68, 0.0, 120, -strumD / 2, 54.2, -45]
IP3 = [-1.4, 83.95, 0, 120, -strumD / 2, 50.65, -45]
IP4 = [-1.8, 81.8, 0, 120, -strumD / 2, 50.65, -45]
IP = [IP0, IP1, IP2, IP3, IP4]


# --------------- Arm Addresses --------------- #
# TODO: Consider adding drum arm to arms list
# arm0 = XArmAPI('192.168.1.208')
# arm1 = XArmAPI('192.168.1.226')
# arm2 = XArmAPI('192.168.1.244')
# arm3 = XArmAPI('192.168.1.203')
# arm4 = XArmAPI('192.168.1.237')
# arms = [arm0, arm1, arm2, arm3, arm4]

# armDrum = XArmAPI('192.168.1.204')
# drums = [armDrum]

# ---- PC Debugging ---- #
# arms = []
drums = []

# --------------- Queues --------------- #
# Arm Queues
q0 = Queue()
q1 = Queue()
q2 = Queue()
q3 = Queue()
q4 = Queue()
qList = [q0, q1, q2, q3, q4]

# Drum Queue
drumQ = Queue()

# Light Queue
lightQ = Queue()


# --------------- Threads --------------- #
xArm0 = Thread(target=strumController, args=(q0, 0,))  # num 2
xArm1 = Thread(target=strumController, args=(q1, 1,))  # num 4
xArm2 = Thread(target=strumController, args=(q2, 2,))  # num 1
xArm3 = Thread(target=strumController, args=(q3, 3,))  # num 3
xArm4 = Thread(target=strumController, args=(q4, 4,))  # num 5
xArmDrum = Thread(target=drumController, args=(drumQ, 5,))
lights = Thread(target=lightController, args=(lightQ,))


def startThreads():
    xArm0.start()
    xArm1.start()
    xArm2.start()
    xArm3.start()
    xArm4.start()
    # xArmDrum.start()
    # lights.start()
    print("Robot threads started")


# --------------- Getter Methods --------------- #
def getArms(): return arms


def getAngles(a): return arms[a].angles
