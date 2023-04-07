from Demos.Interfaces.IDemo import IDemo
from Demos.Interfaces.IRobotDemo import IRobotDemo
from Handlers.InputHandler import getManualInput
from Handlers.RobotHandler import RobotHandler


class KeyboardDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True) -> None:
        self.name = "Keyboard Demo with Matching Rhythm and Notes"
        super().__init__(robotHandler, is_lab_work)

    def start(self):
        self.announceStart()
        self.readyRobots()

        phrase = getManualInput()
        print(f"Received the following phrase: \n{phrase}")

        for i in range(len(phrase)):
            self.robotHandler.playString(phrase[i])
