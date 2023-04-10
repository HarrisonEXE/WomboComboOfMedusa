from Demos.Interfaces.IRobotDemo import IRobotDemo
from Handlers.VisionHandler import VisionHandler
from Helpers.DataFilters import buffered_smooth, save_joint_data, save_vision_data

import time
from threading import Thread
from queue import Queue


class VisionTrackerDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True, robots_already_awake=False):
        self.name = "Live Vision Tracker"
        super().__init__(robotHandler, is_lab_work, robots_already_awake)

        self.arms = self.robotHandler.arms
        self.qList = self.robotHandler.qList

        self.communication_queue = Queue()
        # self.vision = PosenetHandler(device=0, communication_queue=self.communication_queue)
        self.vision = VisionHandler(device=6, communication_queue=self.communication_queue)
        self.vision_thread = Thread(target=self.vision.start)
        self.listener_thread = Thread(target=self._listener)

        self.head_x = []
        self.head_y = []
        self.head_z = []
        self.shoulder_x = []
        self.shoulder_y = []
        self.shoulder_z = []

    def start(self):
        self.announceStart()
        self.readyRobots()
        self.listener_thread.start()
        self.vision_thread.start()

    # TODO: Add a method to stop the server.
    # TODO: Add different poses/tracking for different arms.

    def _listener(self):
        print("Listening queue started")
        while True:
            address, gesture = self.communication_queue.get()

            if address == "/gesture":
                mode = "pose"
                print(gesture)
                if gesture == "stop":
                    self.qList[0].put([mode, 1])
                    self.qList[1].put([mode, 1])
                    self.qList[2].put([mode, 1])
                    self.qList[3].put([mode, 1])
                    self.qList[4].put([mode, 1])
                elif gesture == "go":
                    self.qList[0].put([mode, 2])
                    self.qList[1].put([mode, 2])
                    self.qList[2].put([mode, 2])
                    self.qList[3].put([mode, 2])
                    self.qList[4].put([mode, 2])
                elif gesture == "swipe_left":
                    self.qList[0].put([mode, 3])
                    self.qList[1].put([mode, 3])
                    self.qList[2].put([mode, 3])
                    self.qList[3].put([mode, 3])
                    self.qList[4].put([mode, 3])
                elif gesture == "swipe_right":
                    self.qList[0].put([mode, 4])
                    self.qList[1].put([mode, 4])
                    self.qList[2].put([mode, 4])
                    self.qList[3].put([mode, 4])
                    self.qList[4].put([mode, 4])
                elif gesture == "twirl":
                    self.qList[0].put([mode, 5])
                    self.qList[1].put([mode, 5])
                    self.qList[2].put([mode, 5])
                    self.qList[3].put([mode, 5])
                    self.qList[4].put([mode, 5])

            # TODO: replace args[0], args[1] .... from communication queue
            elif address == "/head":
                raise NotImplementedError
                head = {'x': args[0], 'y': args[1], 'z': args[2]}
                shoulder = {'x': args[3], 'y': args[4], 'z': args[5]}

                smoothed_head = buffered_smooth(self.head_x, self.head_y, self.head_z, head)
                smoothed_shoulder = buffered_smooth(self.shoulder_x, self.shoulder_y, self.shoulder_z, shoulder)
                mode = "live"

                if smoothed_head is not None and smoothed_shoulder is not None:
                    # timestamp = time.time()
                    # save_vision_data('vision_data.csv', timestamp, [head, shoulder], [smoothed_head, smoothed_shoulder])
                    data = [smoothed_head, smoothed_shoulder]
                    # self.qList[0].put([mode, data])