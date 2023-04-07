import time
from Demos.Interfaces.IDemo import IDemo
from Handlers.InputHandler import getManualInput


class KeyboardRobotlessDemo(IDemo):
    def __init__(self) -> None:
        self.name = "Keyboard Robotless Demo with Matching Rhythm and Notes"

    def start(self):
        self.announceStart()

        phrase = getManualInput()
        print(f"Recieved the following phrase: \n{phrase}")

        for i in range(len(phrase)):
            degree, delay = phrase[i]
            time.sleep(delay)
            print(f"Playing note {degree} with a {delay} delay")
