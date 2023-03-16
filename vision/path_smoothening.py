import atexit
import time
import os

from pythonosc import dispatcher
from pythonosc import osc_server

import numpy as np
import csv


def buffered_smooth(buffer_x, buffer_y, buffer_z, value_x, value_y, value_z, algorithm='ema'):
    buffer_x.append(value_x)
    buffer_y.append(value_y)
    buffer_z.append(value_z)

    if len(buffer_x) >= BUFFER_SIZE:
        if algorithm == 'ma':
            # Calculate moving average
            x = np.mean(buffer_x[-BUFFER_SIZE:])
            y = np.mean(buffer_y[-BUFFER_SIZE:])
            z = np.mean(buffer_z[-BUFFER_SIZE:])
            return x, y, z
        elif algorithm == 'ema':
            # Calculate exponential moving average
            x = exponential_moving_average(buffer_x, 0.076)
            y = exponential_moving_average(buffer_y, 0.076)
            z = exponential_moving_average(buffer_z, 0.076)
            return x, y, z
    else:
        return None


def exponential_moving_average(values, alpha):
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema[-1]


def moving_average(values, window_size):
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')


# # Define Kalman filter update functions (Assuming 1D motion)
# def kalman_filter_update(x_est, P, z_meas, R, Q):
#     # Prediction
#     x_pred = x_est
#     P_pred = P + Q
#
#     # Update
#     K = P_pred / (P_pred + R)
#     x_upd = x_pred + K * (z_meas - x_pred)
#     P_upd = (1 - K) * P_pred
#
#     return x_upd, P_upd

def save_vision_data(filename, timestamp, raw_values, smoothed_values):
    row_data = [timestamp] + raw_values + smoothed_values
    columns = ['timestamp', 'raw_head_x', 'raw_head_y', 'raw_head_z', 'raw_shoulder_x', 'raw_shoulder_y',
               'raw_shoulder_z', 'smoothed_head_x', 'smoothed_head_y', 'smoothed_head_z', 'smoothed_shoulder_x',
               'smoothed_shoulder_y', 'smoothed_shoulder_z']

    # Check if the file exists
    if not os.path.isfile(filename):
        # Create a new file and write the header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

    # Append the data to the existing file
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)


def save_joint_data(filename, timestamp, joint_angles):
    if len(joint_angles) != 7:
        raise ValueError("Joint angles list must have exactly 7 elements.")

    with open(filename, 'a', newline='') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=',')
        data_writer.writerow([timestamp,
                              joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3],
                              joint_angles[4], joint_angles[5], joint_angles[6]])


def respondToHead(address, *args):
    if address == "/head":

        x1, y1, z1 = args[0], args[1], args[2]
        x2, y2, z2 = args[3], args[4], args[5]

        smoothed_head_values = buffered_smooth(head_x, head_y, head_z, x1, y1, z1, 'ema')
        smoothed_shoulder_values = buffered_smooth(shoulder_x, shoulder_y, shoulder_z, x2, y2, z2, 'ema')

        if smoothed_head_values is not None and smoothed_shoulder_values is not None:
            smoothed_head_x, smoothed_head_y, smoothed_head_z = smoothed_head_values
            smoothed_shoulder_x, smoothed_shoulder_y, smoothed_shoulder_z = smoothed_shoulder_values

            timestamp = time.time()
            save_vision_data('vision_data.csv', timestamp, [x1, y1, z1, x2, y2, z2],
                             [smoothed_head_x, smoothed_head_y, smoothed_head_z, smoothed_shoulder_x,
                              smoothed_shoulder_y, smoothed_shoulder_z])

            # map_angle_q.put([avg_head_x, avg_head_y, avg_head_y, avg_shoulder_x, avg_shoulder_y, avg_shoulder_z])


def server():
    server = osc_server.ThreadingOSCUDPServer((UDP_IP, UDP_PORT), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
    atexit.register(server.server_close())


if __name__ == "__main__":
    UDP_IP = "0.0.0.0"  # local IP
    UDP_PORT = 12346  # port to retrieve data from Max

    BUFFER_SIZE = 10  # number of values to average over

    dispatcher = dispatcher.Dispatcher()  # dispatcher to send
    dispatcher.map("/head", respondToHead)

    ######################################################
    head_x = []
    head_y = []
    head_z = []
    shoulder_x = []
    shoulder_y = []
    shoulder_z = []

    server()
