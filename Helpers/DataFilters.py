import csv
import os

BUFFER_SIZE = 5


def calculate_alpha(buffer, threshold):
    if len(buffer) < 2:
        return 0.5

    diff = abs(buffer[-1] - buffer[-2])

    if diff > threshold:
        return 0.2
    else:
        return 0.5


def buffered_smooth(buffer_x, buffer_y, buffer_z, coordinates):
    buffer_x.append(float(coordinates['x']))
    buffer_y.append(float(coordinates['y']))
    buffer_z.append(float(coordinates['z']))

    # Keep only the most recent BUFFER_SIZE elements
    # Buffer handling: To implement a sliding window approach
    buffer_x = buffer_x[-BUFFER_SIZE:]
    buffer_y = buffer_y[-BUFFER_SIZE:]
    buffer_z = buffer_z[-BUFFER_SIZE:]

    # adaptive smoothening approach, need to define threshold based on data
    # alpha_x = calculate_alpha(buffer_x, THRESHOLD)
    # alpha_y = calculate_alpha(buffer_y, THRESHOLD)
    # alpha_z = calculate_alpha(buffer_z, THRESHOLD)

    if len(buffer_x) >= BUFFER_SIZE:
        x = exponential_moving_average(buffer_x, 0.6)
        y = exponential_moving_average(buffer_y, 0.6)
        z = exponential_moving_average(buffer_z, 0.6)
        return float(x), float(y), float(z)
    else:
        return None


def exponential_moving_average(values, alpha):
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema[-1]


def save_vision_data(filename, timestamp, smoothed_values):
    row_data = [timestamp] + smoothed_values
    columns = ['timestamp', 'smoothed_head_x', 'smoothed_head_y', 'smoothed_head_z', 'smoothed_shoulder_x',
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

    columns = ["timestamp", "j1", "j2", "j3", "j4", "j5", "j6", "j7"]

    # Check if the file exists
    if not os.path.isfile(filename):
        # Create a new file and write the header
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)

    # Append the data to the existing file
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        row_data = [timestamp, joint_angles[0], joint_angles[1], joint_angles[2], joint_angles[3], joint_angles[4], joint_angles[5], joint_angles[6]]
        writer.writerow(row_data)
