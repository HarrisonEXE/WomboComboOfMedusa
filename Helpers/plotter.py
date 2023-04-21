# import matplotlib.pyplot as plt
import pandas as pd
import argparse
import plotly.graph_objs as go
import plotly.subplots as sp

def plot_joint_angles(filename):
    data = pd.read_csv(filename, header=0)
    data['timestamp'] = data['timestamp'] - data['timestamp'][0]

    joints = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']

    fig = go.Figure()

    for joint in joints:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'{joint}'],
                                 mode='lines',
                                 name=f'{joint}'))

    fig.update_layout(title='Joint angles over time',
                      xaxis_title='Time (sec)',
                      yaxis_title='Joint Angle (deg)')

    fig.show()

def plot_vision(filename):
    data = pd.read_csv(filename, header=0)

    fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Head Coordinates", "Shoulder Coordinates"))

    coords = ['X', 'Y', 'Z']

    for idx, coord in enumerate(coords):
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'smoothed_head_{coord.lower()}'],
                                 mode='lines',
                                 name=f'Head {coord}'), row=1, col=1)

    for idx, coord in enumerate(coords):
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data[f'smoothed_shoulder_{coord.lower()}'],
                                 mode='lines',
                                 name=f'Shoulder {coord}'), row=2, col=1)

    fig.update_layout(title='Head and Coordinates',
                      xaxis_title='Timestamp')

    fig.update_yaxes(title_text='Coordinates', row=1, col=1)
    fig.update_yaxes(title_text='Coordinates', row=2, col=1)

    fig.show()

# def plot_vision(filename):
#     # Read data from the CSV file
#     data = pd.read_csv(filename)
#
#     # Adjust the timestamps
#     data['timestamp'] = data['timestamp'] - data['timestamp'][0]
#
#     # Plot the data
#     plt.figure(figsize=(10, 6))
#
#     plt.plot(data['timestamp'], data['smoothed_head_x'], label='Head X')
#     plt.plot(data['timestamp'], data['smoothed_head_y'], label='Head Y')
#     plt.plot(data['timestamp'], data['smoothed_head_z'], label='Head Z')
#     plt.plot(data['timestamp'], data['smoothed_shoulder_x'], label='Shoulder X')
#     plt.plot(data['timestamp'], data['smoothed_shoulder_y'], label='Shoulder Y')
#     plt.plot(data['timestamp'], data['smoothed_shoulder_z'], label='Shoulder Z')
#
#     plt.xlabel('Time (sec)')
#     plt.ylabel('Coordinates')
#     plt.title('Head and Coordinates')
#
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# def plot_joint_angles(filename):
#     data = pd.read_csv(filename, header=0)
#     data['timestamp'] = data['timestamp'] - data['timestamp'][0]
#
#     joints = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7']
#
#     for joint in joints:
#         plt.figure(figsize=(12, 6))
#         plt.title(f'{joint} joint angles over time')
#
#         plt.plot(data['timestamp'], data[f'{joint}'], label=f'{joint}')
#
#         plt.xlabel('Time (sec)')
#         plt.ylabel('Joint Angle (deg)')
#         plt.legend()
#         plt.show()
def main():
    parser = argparse.ArgumentParser(description='Plot vision or joint data.')
    parser.add_argument('--vision', action='store_true', help='Plot vision data')
    parser.add_argument('--joint', action='store_true', help='Plot joint data')
    parser.add_argument('-v', '--vision_file', type=str, default='vision_ema.csv', help='Vision data file')
    parser.add_argument('-j', '--joint_file', type=str, default='joint_data.csv', help='Joint data file')

    args = parser.parse_args()

    if args.vision:
        print(f"Plotting vision data from {args.vision_file}...")
        plot_vision(args.vision_file)

    if args.joint:
        print(f"Plotting joint angles from {args.joint_file}...")
        plot_joint_angles(args.joint_file)

if __name__ == "__main__":
    main()

