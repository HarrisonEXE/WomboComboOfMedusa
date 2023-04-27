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
        self.running = True
        self.readyRobots()
        self.listener_thread.start()
        self.vision_thread.start()

    def kill(self):
        super().kill()
        self.vision.kill()
        self.listener_thread.join()
        self.vision_thread.join()

    def _listener(self):
        print("Listening queue started")

        gesture_mapping = {"stop": 1, "go": 2, "swipe_left": 3, "swipe_right": 4, "twirl": 5}
        while self.running:
            address, data = self.communication_queue.get()

            if address == "/gesture":
                mode = "pose"
                if data in gesture_mapping:
                    gesture_value = gesture_mapping[data]
                    for q in self.qList[:5]:
                        q.put([mode, gesture_value])
                elif data == "up" or data == "down":
                    self.qList[5].put([mode, data])
            elif address == "/live":
                mode = "live"
                for q in self.qList[:5]:
                    q.put([mode, data])
            elif address == "/updateState":
                if data == "drums":
                    self.robotHandler.switch_drum_state()

        print("Listening queue closed")
        return
