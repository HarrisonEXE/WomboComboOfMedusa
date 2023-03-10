# ----------------------- Notes --------------------- #
# TODO: Not a huge fan of the misc helpers. Refactor  #
# --------------------------------------------------- #

import random
import time


def createRandList(size):
    a = []
    for i in range(size):
        a.append(random.randint(0, 6))
    return a


def delay():
    time.sleep(0.013)
