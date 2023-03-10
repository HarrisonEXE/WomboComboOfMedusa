# ----------------------- Notes --------------------- #
# TODO: Not a huge fan of the misc helpers. Refactor  #
# --------------------------------------------------- #

import random
import time
import math


# Vision ######################################
def euclideanDistance(self, a_x, a_y, a_z, b_x, b_y, b_z):
    return math.sqrt(math.pow((a_x-b_x), 2) + math.pow((a_y-b_y), 2) + math.pow((a_z-b_z), 2))


def isNear(self, fingerOne, fingerTwo, threshold):
    return euclideanDistance(fingerOne.x, fingerOne.y, fingerOne.z, fingerTwo.x, fingerTwo.y, fingerTwo.z) < threshold


def variance(self, data_y):
    # Number of observations
    n = len(data_y)
    # Mean of the data
    mean = sum(data_y) / n
    # Square deviations
    deviations = [(y - mean) ** 2 for y in data_y]
    # Variance
    variance = sum(deviations) / n
    return variance


# Robots ##################################
def createRandList(size):
    a = []
    for i in range(size):
        a.append(random.randint(0, 6))
    return a


def delay():
    time.sleep(0.013)
