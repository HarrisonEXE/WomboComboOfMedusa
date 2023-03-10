from Demos.IDemo import IDemo
from Handlers.RobotHandler import RobotHandler


class IRobotDemo(IDemo):
    def __init__(self, robotHandler, is_lab_work):
        self.is_lab_work = is_lab_work
        self.robotHandler = RobotHandler(is_lab_work=self.is_lab_work)

    def readyRobotsWithoutLive(self):
        self.robotHandler.setupRobots()
        self.robotHandler.startThreads()

    def readyRobots(self):
        self.readyRobotsWithoutLive()
        self.robotHandler.turnOnLive()
