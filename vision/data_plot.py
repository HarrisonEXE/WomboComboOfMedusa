from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_data(processed_data):
    algorithms = ['raw', 'smoothed']
    coords = ['head_x', 'head_y', 'head_z', 'shoulder_x', 'shoulder_y', 'shoulder_z']

    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axs = axs.ravel()

    for idx, coord in enumerate(coords):
        for algo in algorithms:
            axs[idx].plot(processed_data['timestamp'], processed_data[f'{algo}_{coord}'], label=f'{algo}')
        axs[idx].set_title(f'{coord} over time')
        axs[idx].set_ylabel(f'{coord} value')
        axs[idx].legend()

    plt.xlabel('Timestamp')
    plt.tight_layout()
    plt.show()


def plot_joint_angles(processed_data):
    algorithms = ['raw', 'smoothed']
    joints = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']

    for joint in joints:
        plt.figure(figsize=(12, 6))
        plt.title(f'{joint} joint angles over time')

        for algo in algorithms:
            plt.plot(processed_data['timestamp'], processed_data[f'{algo}_{joint}'], label=f'{algo}')

        plt.xlabel('Timestamp')
        plt.ylabel('Joint Angle (Degrees)')
        plt.legend()
        plt.show()


def map_coordinates_to_angles(data):
    angles = pd.DataFrame()
    angles['j1'] = 0
    angles['j2'] = 0
    angles['j3'] = np.interp(data['shoulder_x'], [0.0, 1.0], [-90, 90])
    angles['j4'] = np.interp(data['shoulder_y'], [0.0, 1.0], [70, 120])
    angles['j5'] = np.interp(data['head_x'], [0.0, 1.0], [-100, 100])
    angles['j6'] = np.interp(data['head_y'], [0.0, 1.0], [-90, 90])
    angles['j7'] = 0
    return angles


def process_data(filename):
    data = pd.read_csv(filename, header=0)
    data['timestamp'] = data['timestamp'] - data['timestamp'][0]
    return data


def process_joint_data(data: pd.DataFrame):
    algorithms = ['raw', 'smoothed']

    for algo in algorithms:
        columns = [f'{algo}_head_x', f'{algo}_head_y', f'{algo}_head_z', f'{algo}_shoulder_x', f'{algo}_shoulder_y', f'{algo}_shoulder_z']
        algo_data = data[columns]
        algo_data.columns = ['head_x', 'head_y', 'head_z', 'shoulder_x', 'shoulder_y', 'shoulder_z']
        joint_angles = map_coordinates_to_angles(algo_data)
        for joint in ['j1', 'j2','j3', 'j4', 'j5', 'j6', 'j7']:
            data[f'{algo}_{joint}'] = joint_angles[joint]

    return data


if __name__ == '__main__':
    data = process_data('vision_data.csv')
    joint_data = process_joint_data(data)
    plot_joint_angles(joint_data)
