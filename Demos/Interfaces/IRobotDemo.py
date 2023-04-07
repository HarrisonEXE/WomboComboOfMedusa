from Demos.Interfaces.IDemo import IDemo
from Handlers.RobotHandler import RobotHandler


class IRobotDemo(IDemo):
    def __init__(self, robotHandler, is_lab_work, robots_already_awake=False):
        super().__init__()
        self.is_lab_work = is_lab_work
        self.robots_already_awake = robots_already_awake
        self.robotHandler = robotHandler

    def readyRobotsWithoutLive(self):
        if not self.robots_already_awake:
            self.robotHandler.setupRobots()
            self.robotHandler.startThreads()

    def readyRobots(self):
        self.readyRobotsWithoutLive()
        self.robotHandler.turnOnLive()
