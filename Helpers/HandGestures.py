from google.protobuf.json_format import MessageToDict
import math
from datetime import datetime, timedelta
import time
import numpy as np

#TODO: Removed redundant code
#TODO: Merge with VisionHandler.py
#TODO: Removed duplicate functions with VisionHandler.py

def detectOpen(hand_landmarks):
    firstFingerIsOpen = False
    secondFingerIsOpen = False
    thirdFingerIsOpen = False
    fourthFingerIsOpen = False

    kp = hand_landmarks.landmark[6].y
    if (hand_landmarks.landmark[7].y < kp and hand_landmarks.landmark[8].y < kp): firstFingerIsOpen = True

    kp = hand_landmarks.landmark[10].y
    if (hand_landmarks.landmark[11].y < kp and hand_landmarks.landmark[12].y < kp): secondFingerIsOpen = True

    kp = hand_landmarks.landmark[14].y
    if (hand_landmarks.landmark[15].y < kp and hand_landmarks.landmark[16].y < kp): thirdFingerIsOpen = True

    kp = hand_landmarks.landmark[18].y
    if (hand_landmarks.landmark[19].y < kp and hand_landmarks.landmark[20].y < kp): fourthFingerIsOpen = True

    if firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen:
        return True
    else:
        return False


def detectBack(hand_landmarks, handedness):
    if (handedness == "R"):
        if (hand_landmarks.landmark[17].x < hand_landmarks.landmark[13].x < hand_landmarks.landmark[9].x < hand_landmarks.landmark[5].x):
            return True
    else:
        if (hand_landmarks.landmark[17].x > hand_landmarks.landmark[13].x > hand_landmarks.landmark[9].x > hand_landmarks.landmark[5].x):
            return True
    return False


def detectTwirlEnd(hand_landmarks, handedness):
    back = detectBack(hand_landmarks, handedness)
    near = isNear(hand_landmarks.landmark[8], hand_landmarks.landmark[12], .06) and isNear(hand_landmarks.landmark[12], hand_landmarks.landmark[16], .06) and isNear(hand_landmarks.landmark[16], hand_landmarks.landmark[20], .08)
    thumbBack = hand_landmarks.landmark[4].z > hand_landmarks.landmark[8].z and hand_landmarks.landmark[4].z > hand_landmarks.landmark[12].z and hand_landmarks.landmark[4].z > hand_landmarks.landmark[16].z
    upright = detectUpright(hand_landmarks)

    if (back and near and thumbBack and upright):
        return True


def detectUpright(hand_landmarks):
    pointerDistance = math.sqrt((hand_landmarks.landmark[8].x - hand_landmarks.landmark[5].x)**2 + (hand_landmarks.landmark[8].y - hand_landmarks.landmark[5].y)**2)
    middleDistance = math.sqrt((hand_landmarks.landmark[12].x - hand_landmarks.landmark[9].x)**2 + (hand_landmarks.landmark[12].y - hand_landmarks.landmark[9].y)**2)
    ringDistance = math.sqrt((hand_landmarks.landmark[16].x - hand_landmarks.landmark[13].x)**2 + (hand_landmarks.landmark[16].y - hand_landmarks.landmark[13].y)**2)

    pointerStraight = hand_landmarks.landmark[8].y < abs(hand_landmarks.landmark[5].y - pointerDistance*4/5)
    middleStraight = hand_landmarks.landmark[12].y < abs(hand_landmarks.landmark[9].y - middleDistance*4/5)
    ringStraight = hand_landmarks.landmark[16].y < abs(hand_landmarks.landmark[13].y - ringDistance*4/5)

    if (pointerStraight and middleStraight and ringStraight): return True
    else: return False


def detectFront(hand_landmarks, handedness):
    if (handedness == "R"):
        if (hand_landmarks.landmark[17].x > hand_landmarks.landmark[13].x > hand_landmarks.landmark[9].x > hand_landmarks.landmark[5].x) and (hand_landmarks.landmark[18].x > hand_landmarks.landmark[14].x > hand_landmarks.landmark[10].x > hand_landmarks.landmark[6].x) and (hand_landmarks.landmark[19].x > hand_landmarks.landmark[15].x > hand_landmarks.landmark[11].x > hand_landmarks.landmark[7].x) and (hand_landmarks.landmark[20].x > hand_landmarks.landmark[16].x > hand_landmarks.landmark[12].x > hand_landmarks.landmark[8].x):
            return True
    else:
        if (hand_landmarks.landmark[17].x < hand_landmarks.landmark[13].x < hand_landmarks.landmark[9].x < hand_landmarks.landmark[5].x) and (hand_landmarks.landmark[18].x < hand_landmarks.landmark[14].x < hand_landmarks.landmark[10].x < hand_landmarks.landmark[6].x) and (hand_landmarks.landmark[19].x < hand_landmarks.landmark[15].x < hand_landmarks.landmark[11].x < hand_landmarks.landmark[7].x) and (hand_landmarks.landmark[20].x < hand_landmarks.landmark[16].x < hand_landmarks.landmark[12].x < hand_landmarks.landmark[8].x):
            return True
    return False


def detectStraight(hand_landmarks):
    length = determineLength(hand_landmarks)
    count = 4
    for i in length:
        if (abs(i - euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[count])) > .03):
            return False
        count += 4
    return True


def detectFlat(hand_landmarks, handedness):
    length = determineLength(hand_landmarks)
    straight = detectStraight(hand_landmarks)

    line = sameLine(hand_landmarks.landmark[4], hand_landmarks.landmark[8], hand_landmarks.landmark[12], hand_landmarks.landmark[16], hand_landmarks.landmark[20], "y", length[2])

    if line and straight: return True
    else: return False


def variance(data_y):
    # Number of observations
    n = len(data_y)
    # Mean of the data
    mean = sum(data_y) / n
    # Square deviations
    deviations = [(y - mean) ** 2 for y in data_y]
    # Variance
    variance = sum(deviations) / n
    return variance


def euclideanDistance(fingerOne, fingerTwo):
    return math.sqrt(math.pow((fingerOne.x-fingerTwo.x),2) + math.pow((fingerOne.y-fingerTwo.y),2) + math.pow((fingerOne.z-fingerTwo.z), 2))


def isNear(fingerOne, fingerTwo, threshold):
    return euclideanDistance(fingerOne, fingerTwo) < threshold


def determineLength(hand_landmarks):
    thumb = euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[1])
    pointer = euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[5])
    middle = euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[9])
    ring = euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[13])
    pinky = euclideanDistance(hand_landmarks.landmark[0], hand_landmarks.landmark[17])

    for i in range(1, 4):
        thumb += euclideanDistance(hand_landmarks.landmark[i], hand_landmarks.landmark[i + 1])
    for i in range(5, 8):
        pointer += euclideanDistance(hand_landmarks.landmark[i], hand_landmarks.landmark[i + 1])
    for i in range(9, 12):
        middle += euclideanDistance(hand_landmarks.landmark[i], hand_landmarks.landmark[i + 1])
    for i in range(13, 16):
        ring += euclideanDistance(hand_landmarks.landmark[i], hand_landmarks.landmark[i + 1])
    for i in range(17, 20):
        pinky += euclideanDistance(hand_landmarks.landmark[i], hand_landmarks.landmark[i + 1])

    return [thumb, pointer, middle, ring, pinky]


def sameLine(fingerOne, fingerTwo, fingerThree, fingerFour, fingerFive, axis, length):
    # if axis == "x":
    #     var = variance([fingerOne.x, fingerTwo.x, fingerThree.x, fingerFour.x, fingerFive.x])
    #     if abs(fingerThree.y - fingerOne.y) < length * .2:
    #         return True
    if axis == "y":
        var = variance([fingerOne.y, fingerTwo.y, fingerThree.y, fingerFour.y, fingerFive.y])
        if abs(fingerThree.y - fingerOne.y) < length * .2:
            return True
    # elif axis == "z":
    #     var = variance([fingerOne.z, fingerTwo.z, fingerThree.z, fingerFour.z, fingerFive.z])
    #     if abs(fingerThree.y - fingerOne.y) < length * .2:
    #         return True


def detectNear1D(fingerOne, fingerTwo, threshold):
    return abs(fingerOne - fingerTwo) < threshold


def detectBasic(hand_landmarks, handedness):
    return detectStraight(hand_landmarks) and detectUpright(hand_landmarks) and detectFront(hand_landmarks, handedness)


def detectSideways(hand_landmarks, handedness):
    poseCheck = hand_landmarks.landmark[5].y < hand_landmarks.landmark[9].y < hand_landmarks.landmark[13].y < hand_landmarks.landmark[17].y
    xAxisCheck = detectNear1D(hand_landmarks.landmark[5].x, hand_landmarks.landmark[9].x, .02) and detectNear1D(hand_landmarks.landmark[9].x, hand_landmarks.landmark[13].x, .02) and detectNear1D(hand_landmarks.landmark[13].x, hand_landmarks.landmark[17].x, .02) and detectNear1D(hand_landmarks.landmark[5].x, hand_landmarks.landmark[17].x, .02)

    return poseCheck and xAxisCheck


def detectClosed(hand_landmarks, handedness):
    firstFingerClosed = False
    secondFingerClosed = False
    thirdFingerClosed = False
    fourthFingerClosed = False

    if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y and hand_landmarks.landmark[6].y < hand_landmarks.landmark[5].y: firstFingerClosed = True
    if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y and hand_landmarks.landmark[10].y < hand_landmarks.landmark[9].y: secondFingerClosed = True
    if hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y and hand_landmarks.landmark[14].y < hand_landmarks.landmark[13].y: thirdFingerClosed = True
    if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y and hand_landmarks.landmark[18].y < hand_landmarks.landmark[17].y: fourthFingerClosed = True

    if firstFingerClosed and secondFingerClosed and thirdFingerClosed and fourthFingerClosed: return True
    else: return False
