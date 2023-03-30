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
    def __init__(self, robotHandler, is_lab_work=True) -> None:
        self.name = "Live Vision Tracker"
        super().__init__(robotHandler, is_lab_work)

        self.arms = self.robotHandler.arms
        self.qList = self.robotHandler.qList

        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/gesture", self._listener)
        self.dispatcher.map("/head", self._listener)

        self.head_x = []
        self.head_y = []
        self.head_z = []
        self.shoulder_x = []
        self.shoulder_y = []
        self.shoulder_z = []

    def start(self):
        self.announceStart()
        self.readyRobots()
        self.run_udp_server()

    #TODO: Add a method to stop the server.
    #TODO: Add different poses/tracking for different arms.
    def _listener(self, address, *args):
        if address == "/gesture":
            mode = "pose"
            if args[0] == "wave_hello":
                self.qList[0].put([mode, 1])
            elif args[0] == "wave_bye":
                self.qList[0].put([mode, 2])
            elif args[0] == "twirl":
                self.qList[0].put([mode, 3])
            elif args[0] == "swipe_right":
                #TODO: Add swipe right gesture 0
                self.qList[0].put([mode, 0])
            elif args[0] == "swipe_left":
                #TODO: Add swipe left gesture 1
                self.qList[0].put([mode, 1])

        elif address == "/live":
            head = {'x': args[0], 'y': args[1], 'z': args[2]}
            shoulder = {'x': args[3], 'y': args[4], 'z': args[5]}

            smoothed_head = buffered_smooth(self.head_x, self.head_y, self.head_z, head)
            smoothed_shoulder = buffered_smooth(self.shoulder_x, self.shoulder_y, self.shoulder_z, shoulder)
            mode = "live"

            if smoothed_head is not None and smoothed_shoulder is not None:
                # timestamp = time.time()
                # save_vision_data('vision_data.csv', timestamp, [head, shoulder], [smoothed_head, smoothed_shoulder])
                data = [smoothed_head, smoothed_shoulder]
                self.qList[0].put([mode, data])

    def run_udp_server(self):
        server = osc_server.ThreadingOSCUDPServer((UDP_IP, UDP_PORT), self.dispatcher)
        print("Serving on {}".format(server.server_address))
        server.serve_forever()
        atexit.register(server.server_close())
