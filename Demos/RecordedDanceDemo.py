from Demos.Interfaces.IRobotDemo import IRobotDemo
from Handlers import RobotHandler
from Helpers.VisionResponse import is_moving
from threading import Thread
from queue import Queue
import time

class RecordedDanceDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True, robots_already_awake=False):
        self.name = "Hope's Dance Demo"
        print("rippp")
        super().__init__(robotHandler, is_lab_work, robots_already_awake)

        self.arms = self.robotHandler.arms
        self.qList = self.robotHandler.qList

    def start(self):
        self.announceStart()
        self.running = True
        self.readyRobots()
        # dancethread =
        self.run()
    #
    # def kill(self):
    #     self.running = False

    def run(self):
        mode = 'dance'
        data = ' '
        self.robotHandler.qList[0].put([mode, data])
        self.robotHandler.qList[1].put([mode, data])
        self.robotHandler.qList[2].put([mode, data])
        self.robotHandler.qList[3].put([mode, data])
        self.robotHandler.qList[4].put([mode, data])


        print("Queued dances")
        if not is_moving:
            return
        # while True:
        #     print("hi")
        #     time.sleep(1)

