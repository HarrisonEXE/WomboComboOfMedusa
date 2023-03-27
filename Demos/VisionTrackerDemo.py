from Demos.IRobotDemo import IRobotDemo
from Handlers.RobotHandler import RobotHandler
from Helpers.DataFilters import buffered_smooth, save_joint_data, save_vision_data

import atexit
import numpy as np
import queue
import time
from threading import Thread
from pythonosc import dispatcher, osc_server

UDP_IP = "0.0.0.0"  # local IP
UDP_PORT = 12346  # port to retrieve data from Max


class VisionTrackerDemo(IRobotDemo):
    def __init__(self, robotHandler=RobotHandler, is_lab_work=True) -> None:
        self.name = "Live Vision Tracker"
        super().__init__(robotHandler, is_lab_work)
        self.tracking_offset = 0
        self.arms = self.robotHandler.arms
        self.strumArmThreads = self.robotHandler.strumArmThreads

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/gesture", self._listener)
        self.dispatcher.map("/head", self._listener)

        self.head_x = []
        self.head_y = []
        self.head_z = []
        self.shoulder_x = []
        self.shoulder_y = []
        self.shoulder_z = []

        # TODO: Update this to use the new queue system
        # TODO: Update this to use use previous threads
        # TODO: Add threads and queue for each arm
        self.map_angle_q = queue.Queue()
        self.xArm1_Play = Thread(target=self.play_arm, args=(0, self.map_angle_q))

    def start(self):
        self.announceStart()
        self.readyRobots()
        self.run_udp_server()

    def _listener(self, address, *args):
        if address == "/gesture":
            if args[0] == "wave_hello":
                poseI = self.arms[0].angles
                poseF = [0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0]
                newPos = self.robotHandler.poseToPose(poseI, poseF, 5)
                self.robotHandler.gotoPose(0, newPos)
                self.xArm1_Play.start()
            elif args[0] == "wave_bye":
                self.strumArmThreads[0].put(2)
            elif args[0] == "twirl":
                self.strumArmThreads[0].put(3)
        elif address == "/head":
            head = {'x': args[0], 'y': args[1], 'z': args[2]}
            shoulder = {'x': args[3], 'y': args[4], 'z': args[5]}

            smoothed_head = buffered_smooth(self.head_x, self.head_y, self.head_z, head)
            smoothed_shoulder = buffered_smooth(self.shoulder_x, self.shoulder_y, self.shoulder_z, shoulder)

            if smoothed_head is not None and smoothed_shoulder is not None:
                timestamp = time.time()
                save_vision_data('vision_data.csv', timestamp, [head, shoulder], [smoothed_head, smoothed_shoulder])

                self.map_angle_q.put([smoothed_head, smoothed_shoulder])

    def play_arm(self, num, que):
        while True:
            data = que.get()
            smoothed_head = data[0]
            smoothed_shoulder = data[1]

            if self.tracking_offset <= 300:
                offset0 = smoothed_head['x']
                offset1 = smoothed_head['y']
                offset3 = smoothed_shoulder['y']
                offset4 = smoothed_shoulder['x']

            j3 = np.interp(smoothed_shoulder['x'] - offset4, [-0.5, 0.5], [-30, 30])
            j4 = np.interp(smoothed_shoulder['y'] - offset3, [-0.5, 0.5], [70, 120])
            j5 = np.interp(smoothed_head['x'] - offset0, [-0.5, 0.5], [-60, 60])
            j6 = np.interp(smoothed_head['y'] - offset1, [-0.5, 0.5], [-70, 70])

            p = self.arms[num].angles
            p[2] = j3
            p[3] = j4
            p[4] = j5
            p[5] = j6

            save_joint_data('joint_data.csv', time.time(), p)
            self.arms[num].set_servo_angle_j(angles=p, is_radian=False)
            self.tracking_offset += 1

    def run_udp_server(self):
        server = osc_server.ThreadingOSCUDPServer((UDP_IP, UDP_PORT), self.dispatcher)
        print("Serving on {}".format(server.server_address))
        server.serve_forever()
        atexit.register(server.server_close())
