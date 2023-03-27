import csv
import os

BUFFER_SIZE = 5


def buffered_smooth(buffer_x, buffer_y, buffer_z, coordinates):
    buffer_x.append(coordinates['x'])
    buffer_y.append(coordinates['y'])
    buffer_z.append(coordinates['z'])

    if len(buffer_x) >= BUFFER_SIZE:
        x = exponential_moving_average(buffer_x, 0.1)
        y = exponential_moving_average(buffer_y, 0.1)
        z = exponential_moving_average(buffer_z, 0.1)
        return x, y, z
    else:
        return None


def exponential_moving_average(values, alpha):
    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    return ema[-1]


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
