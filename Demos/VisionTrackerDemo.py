from Demos.Interfaces.IRobotDemo import IRobotDemo
from Handlers.VisionHandler import VisionHandler

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
        # self.vision = VisionHandler(device=0, communication_queue=self.communication_queue)
        self.vision = VisionHandler(device=4, communication_queue=self.communication_queue)
        self.vision_thread = Thread(target=self.vision.start)
        self.listener_thread = Thread(target=self._listener)

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
            # print(self.communication_queue)
            address, data = self.communication_queue.get()

            if address == "/gesture":
                mode = "pose"
                print(data)
                if data == "stop":
                    self.qList[0].put([mode, 1])
                    self.qList[1].put([mode, 1])
                    self.qList[2].put([mode, 1])
                    self.qList[3].put([mode, 1])
                    self.qList[4].put([mode, 1])
                elif data == "go":
                    self.qList[0].put([mode, 2])
                    self.qList[1].put([mode, 2])
                    self.qList[2].put([mode, 2])
                    self.qList[3].put([mode, 2])
                    self.qList[4].put([mode, 2])
                # elif data == "swipe_left":
                #     self.qList[0].put([mode, 3])
                #     self.qList[1].put([mode, 3])
                #     self.qList[2].put([mode, 3])
                #     self.qList[3].put([mode, 3])
                #     self.qList[4].put([mode, 3])
                # elif data == "swipe_right":
                #     self.qList[0].put([mode, 4])
                #     self.qList[1].put([mode, 4])
                #     self.qList[2].put([mode, 4])
                #     self.qList[3].put([mode, 4])
                #     self.qList[4].put([mode, 4])
                elif data == "twirl":
                    self.qList[0].put([mode, 5])
                    self.qList[1].put([mode, 5])
                    self.qList[2].put([mode, 5])
                    self.qList[3].put([mode, 5])
                    self.qList[4].put([mode, 5])
                elif data == "up":
                    self.qList[5].put([mode, "up"])
                elif data == "down":
                    self.qList[5].put([mode, "down"])
            elif address == "/live":
                mode = "live"
                self.qList[0].put([mode, data])
                self.qList[1].put([mode, data])
                self.qList[2].put([mode, data])
                self.qList[3].put([mode, data])
                self.qList[4].put([mode, data])
            elif address == "/updateState":
                if data == "drums":
                    self.robotHandler.switch_drum_state()
