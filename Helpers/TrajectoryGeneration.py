import math
import numpy as np

# For strings
def fifth_poly(q_i, q_f, t):
    traj_t = np.arange(0, t, 0.004)
    dq_i = 0
    dq_f = 0
    ddq_i = 0
    ddq_f = 0
    a0 = q_i
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * t ** 3) * (20 * (q_f - q_i) - (8 * dq_f +
                                                 12 * dq_i) * t - (3 * ddq_f - ddq_i) * t ** 2)
    a4 = 1 / (2 * t ** 4) * (30 * (q_i - q_f) + (14 * dq_f +
                                                 16 * dq_i) * t + (3 * ddq_f - 2 * ddq_i) * t ** 2)
    a5 = 1 / (2 * t ** 5) * (12 * (q_f - q_i) -
                             (6 * dq_f + 6 * dq_i) * t - (ddq_f - ddq_i) * t ** 2)
    traj_pos = a0 + a1 * traj_t + a2 * traj_t ** 2 + a3 * \
        traj_t ** 3 + a4 * traj_t ** 4 + a5 * traj_t ** 5
    return traj_pos

# For drums
def fifth_poly2(q_i, q_f, t, ttopstop, tbotstop):
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

    traj_top = np.ones(int(ttopstop / 0.004)) * q_i  # time stopped at top of trajectory, before strike

    traj_bot = np.ones(int(tbotstop / 0.004)) * q_f  # time stopped at bottom of trajectory, after strike

    half_traj = np.concatenate((traj_top, traj_pos, traj_bot))
    full_traj = np.append(half_traj, np.flip(half_traj))

    return full_traj

def spline_poly(q_i, q_f, q_in, ta, tt, ttopstop, tbotstop, pba, pbt, pbtopstop):
    # qi is initial pos, qf is final pos (strike), qin is new initial (return pos)

    # initial accel (using first half of a 5th order poly)
    # ta is double the time till max acceleration (time doing 5th order poly)

    # pba is pullback amount, pbt is pullback time (.5 is good default)

    ########### this code calculates the trajectory for the first half (the way down) #############

    traj_ta = np.arange(0, ta, 0.004)
    dq_i = 0
    dq_f = 0
    ddq_i = 0
    ddq_f = 0
    a0 = (q_i - pba)
    a1 = dq_i
    a2 = 0.5 * ddq_i
    a3 = 1 / (2 * ta ** 3) * (
            20 * (q_f - (q_i - pba)) / 2 - (8 * dq_f + 12 * dq_i) * ta - (3 * ddq_f - ddq_i) * ta ** 2)
    a4 = 1 / (2 * ta ** 4) * (
            30 * ((q_i - pba) - q_f) / 2 + (14 * dq_f + 16 * dq_i) * ta + (3 * ddq_f - 2 * ddq_i) * ta ** 2)
    a5 = 1 / (2 * ta ** 5) * (12 * (q_f - (q_i - pba)) / 2 - (6 * dq_f + 6 * dq_i) * ta - (ddq_f - ddq_i) * ta ** 2)
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
    delta3 = abs((q_i - pba) - q_f) - delta1 - delta2
    if (delta3 < 0):
        print("accel time and turnaround time too big")

    tc = delta3 / abs(hv)

    traj_tc = np.arange(0, tc, 0.004)
    pc = fifth_pos[hp] + traj_tc * hv

    # stall time at top / bottom
    # traj_top = np.ones(int(ttopstop / 0.004)) * q_i  # time stopped at top of trajectory, before strike
    # traj_top can be used for the pullback

    traj_top = fifth_poly2(q_i, q_i - pba, pbt, pbtopstop, 0)
    thp_pb = math.floor(len(traj_top) / 2)  # halfway point of turnaround traj
    traj_top = traj_top[0:thp_pb]

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
    if (delta3 < 0):
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