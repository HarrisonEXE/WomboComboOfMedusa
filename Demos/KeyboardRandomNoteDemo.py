from Demos.Interfaces.IDemo import IDemo
from Demos.Interfaces.IRobotDemo import IRobotDemo
from Handlers.InputHandler import getManualInput
from numpy.random import randint

from Handlers.RobotHandler import RobotHandler


class KeyboardRandomNoteDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True):
        super().__init__(robotHandler, is_lab_work)
        self.name = "Keyboard Demo with Randomized Notes and Matching Rhythm"

    def start(self):
        self.announceStart()

        self.readyRobots()

        phrase = getManualInput()
        print(f"Recieved the following phrase: \n{phrase}")

        phrase.notes = self.getRandomizedNotes()
        print(f"Changed to the following phrase: \n{phrase}")

        for i in range(len(phrase)):
            self.robotHandler.playString(phrase[i])

    def getRandomizedNotes(self):
        return randint(1, 5, 5)
