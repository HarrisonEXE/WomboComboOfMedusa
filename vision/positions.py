import numpy as np
import math

########### sine and circle values ##############

def sintrajmaker(IP):
    amp = 40
    speed = 4
    strikes = speed / 4
    tup = np.arange(0, 2 * speed, 0.004)
    # tstrike = np.arange(0, 2 * strikes, 0.004)
    wave = []
    strike = []
    pos = IP
    j_angles = pos.copy()
    traj = []
    for t in tup:
        wave.append(math.cos((math.pi / speed) * (t)))
        strike.append(-math.cos((math.pi / strikes) * (t)))
    for i in range(len(wave)):
        pos = IP
        j_angles[1] = pos[1] + 0.5 * amp * wave[i] - 0.5 * amp
        j_angles[0] = pos[0] + 0.5 * amp2 * strike[i] + 0.5 * amp2
        j_angles[3] = pos[3] + 1.556 * (0.5 * amp * wave[i] - 0.5 * amp)
        buffer = j_angles.copy()
        # print(j_angles)
        traj.append(buffer)
    # print(traj)
    return traj

def circletrajmaker(IP):
    amp = 40
    speed = 4
    strikes = speed / 2
    tup = np.arange(0, 2 * speed, 0.004)
    # tstrike = np.arange(0, 2 * strikes, 0.004)
    wave = []
    strike = []
    pos = IP
    j_angles = pos.copy()
    traj = []
    for t in tup:
        wave.append(math.cos((math.pi / speed) * (t)))
        strike.append(-math.cos((math.pi / strikes) * (t)))
    for i in range(len(wave)):
        pos = IP
        j_angles[1] = pos[1] + 0.5 * amp * wave[i] - 0.5 * amp
        j_angles[0] = pos[0] + 0.5 * amp2 * strike[i] + 0.5 * amp2
        j_angles[3] = pos[3] + 1.556 * (0.5 * amp * wave[i] - 0.5 * amp)
        buffer = j_angles.copy()
        # print(j_angles)
        traj.append(buffer)
    # print(traj)
    return traj


def utrajmaker(IP):
    amp = uamp

    speed = 0.75
    strikes = speed * 2
    tup = np.arange(0, 2 * speed, 0.004)
    # tstrike = np.arange(0, 2 * strikes, 0.004)
    wave = []
    strike = []
    pos = IP
    j_angles = pos.copy()
    traj = []
    for t in tup:
        wave.append(-math.cos((math.pi / speed) * (t)))
        strike.append(-math.cos((math.pi / strikes) * (t)))
    for i in range(len(wave)):
        pos = IP
        j_angles[1] = pos[1] + 0.5 * amp * wave[i] + 0.5 * amp
        j_angles[0] = pos[0] + 0.5 * baseamp * strike[i] + 0.5 * baseamp
        buffer = j_angles.copy()
        # print(j_angles)
        traj.append(buffer)
    # print(traj)
    return traj

def spintrajmaker(IP):
    amp = usamp

    speed = 3.5
    tup = np.arange(0, 2 * speed, 0.004)
    strikes = speed * 2
    striket = 0.25
    spin = speed - striket/2
    tspinstart = np.arange(0, spin, 0.004)
    point = int(striket / 0.004)
    tspinmid = np.array([spin] * point)
    # print(tspinmid)
    tspinend = np.arange(spin, 2 * spin, 0.004)
    spint = np.concatenate((tspinstart, tspinmid, tspinend))
    # print(len(tup),len(spint))
    # input("fdjfshgasjsd")
    # tstrike = np.arange(0, 2 * strikes, 0.004)
    wave = []
    strike = []
    spina = []

    pos = IP
    j_angles = pos.copy()
    traj = []
    for t in tup:
        wave.append(-math.cos((math.pi / speed) * (t)))
        strike.append(-math.cos((math.pi / strikes) * (t)))
    for t in spint:
        spina.append(math.cos((math.pi / spin) * (t)))
    for i in range(len(strike)):
        pos = IP
        j_angles[1] = pos[1] + 0.5 * amp * wave[i] + 0.5 * amp
        j_angles[0] = pos[0] + 0.5 * basesamp * strike[i] + 0.5 * basesamp
        j_angles[4] = pos[4] + 359.8/2 * spina[i] - 359.8/2
        buffer = j_angles.copy()
        # print(buffer)
        traj.append(buffer)
    # print(traj)
    return traj

def wavetrajmaker(pos):
    wavetraj = []
    speed = 4
    timearray = np.arange(0, 2 * speed, 0.004)
    wave = []
    for t in timearray:
        wave.append(-math.cos((math.pi / speed) * (t)))
    j_angles = pos.copy()
    sign = -1
    amp = 30
    for w in wave:
        j_angles[0] = pos[0] - 0.5 * amp * w - 0.5 * amp
        j_angles[4] = pos[4] + 3 * amp * w + 3 * amp
        j_angles[1] = pos[1] + 0.5 * amp * w + 0.5 * amp
        j_angles[3] = pos[3] + amp * w + amp
        j_angles[5] = pos[5] + 0.5 * amp * w + 0.5 * amp
        buffer = j_angles.copy()
        # print(buffer)
        wavetraj.append(buffer)
    return wavetraj


amp2 = 7
IP0s = [-1-amp2/2, 86.0, -2, 126.5, 0, 51.7, -45]
IP1s = [2.1-amp2/2, 85.3, 0, 127.1, 0, 50.1, -45]
IP2s = [1.5-amp2/2, 80.70, 0.0, 120, 0, 54.2, -45]
IP3s = [-0.2-amp2/2, 83.0, 0, 120, 0, 50.75, -45]
IP4s = pos = [-1.6-amp2/2, 81.0, 0, 120, 0, 50.65, -45]
global IPs
IPs = [IP0s, IP1s, IP2s, IP3s, IP4s]


IP0c = [-1-amp2/2, 87.1, -2, 126.5, 0, 51.7, -45]
IP1c = [2.1-amp2/2, 86.3, 0, 127.1, 0, 50.1, -45]
IP2c = [1.5-amp2/2, 81.68, 0.0, 120, 0, 54.2, -45]
IP3c = [-0.2-amp2/2, 83.95, 0, 120, 0, 50.75, -45]
IP4c = pos = [-1.6-amp2/2, 81.87, 0, 120, 0, 50.65, -45]
global IPc
IPc = [IP0c, IP1c, IP2c, IP3c, IP4c]

global baseamp
global uamp
baseamp = 30
uamp = 10
IP0u = [-0.25-baseamp/2, 87.5 - uamp, -2, 126.5, 0, 51.7, -45]
IP1u = [2.67-baseamp/2, 86.32 - uamp, 0, 127.1, 0, 50.1, -45] # [2.67 , 86.1, 0, 127.1, -strumD / 2, 50.1, -45]
IP2u = [1.3-baseamp/2, 81.8 - uamp, 0, 120, 0, 54.2, -45]
IP3u = [-1.4-baseamp/2, 83.95 - uamp, 0, 120, 0, 50.75, -45] #[-0.2, 83.8, 0, 120, -strumD/2, 50.75, -45]
IP4u = [-1.8-baseamp/2, 81.88 - uamp, 0, 120, 0, 50.75, -45]


global basesamp
global usamp
basesamp = 40
usamp = 30
IP0us = [-0.25-basesamp/2, 87.5 - usamp, -2, 126.5, 0, 51.7, -45]
IP1us = [2.67-basesamp/2, 86.32 - usamp, 0, 127.1, 0, 50.1, -45] # [2.67 , 86.1, 0, 127.1, -strumD / 2, 50.1, -45]
IP2us = [1.3-basesamp/2, 81.8 - usamp, 0, 120, 0, 54.2, -45]
IP3us = [-1.4-basesamp/2, 83.95 - usamp, 0, 120, 0, 50.75, -45] #[-0.2, 83.8, 0, 120, -strumD/2, 50.75, -45]
IP4us = [-1.8-basesamp/2, 81.88 - usamp, 0, 120, 0, 50.75, -45]


IP0w = [162, -60, 0, 30, 0, 40, 0]
IP1w = [-162, -60, 0, 30, 0, 40, 0]
IP2w = [90, -60, 0, 40, 0, 40, 0]
IP3w = [0, -30, 0, 60, 0, 30, 0]
IP4w = [-90, -80, 0, 30, 0, 30, 0]  # [-3.9, 65, 3.5, 100.3, -strumD/2, 42.7, 101.1]
IP5w = [165, -45, 0, 45, 0, 24, 0]  # Snare
IP6w = [-165, -70, 0, 30, 0, 30, 0]  # Bodharn
global IPw
IPw = [IP0w, IP1w, IP2w, IP3w, IP4w, IP5w, IP6w]

global snakeIP
snakeIP0 = [0, -60, 0, 120, 0, 0, 0]
snakeIP1 = [0, -60, 0, 120, 0, 0, 0]
snakeIP2 = [0.0, -50, 0, 100, 0, 0, 0]
snakeIP3 = [0.0, -50, 0, 100, 0, 0, 0]
snakeIP4 = [0.0, -50, 0, 100, 0, 0, 0]  # [-1.6, 81.8, 0, 120, -strumD/2, 50.13, -45]
snakeIP5 = [120, -10, 0, 95, 0, 0, 0]
snakeIP6 = [-120, -20, 0, 95, 0, 10, 0]

snakeIP = [snakeIP0, snakeIP1, snakeIP2, snakeIP3, snakeIP4, snakeIP5, snakeIP6]

global IPu
IPu = [IP0u, IP1u, IP2u, IP3u, IP4u]
sintraj = []
for robots in IPs:
    # trajmaker(robots)
    sintraj.append(sintrajmaker(robots))

global IPus
IPus = [IP0us, IP1us, IP2us, IP3us, IP4us]
spintraj = []
for robots in IPus:
    # trajmaker(robots)
    spintraj.append(spintrajmaker(robots))
    # input("NEX BOT")

circletraj = []
for robots in IPc:
    # trajmaker(robots)
    circletraj.append(circletrajmaker(robots))

utraj = []
for robots in IPu:
    # trajmaker(robots)
    utraj.append(utrajmaker(robots))

wtraj = []
for robots in IPw:
    # trajmaker(robots)
    wtraj.append(wavetrajmaker(robots))
    # input("ready for next bot")




    # input("check traj")


############ U traj ########3