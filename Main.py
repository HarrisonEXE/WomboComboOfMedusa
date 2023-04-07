from Demos.KeyboardDemo import KeyboardDemo
from Demos.KeyboardRandomNoteDemo import KeyboardRandomNoteDemo
from Demos.KeyboardRobotlessDemo import KeyboardRobotlessDemo
from Demos.MicDemo import MicDemo
from Demos.RandomizedMicDemo import RandomizedMicDemo
from Demos.VoiceRecognizer import VoiceDemo
from Demos.VisionTrackerDemo import VisionTrackerDemo
from Handlers.RobotHandler import RobotHandler
# from test import mainmethod
from Handlers.VisionHandler import VisionHandler

from multiprocessing import Process

from Performance import Performance


class MedusaDemo:
    def __init__(self, robotHandler):
        self.keyboard_robotless_demo = KeyboardRobotlessDemo()
        self.keyboard_demo = KeyboardDemo(robotHandler)
        self.keyboard_random_note_demo = KeyboardRandomNoteDemo(robotHandler)
        self.mic_demo = MicDemo(robotHandler, is_lab_work=False)
        self.randomized_mic_demo = RandomizedMicDemo(robotHandler)
        self.voice_demo = VoiceDemo(robotHandler)
        self.vision_tracker_demo = VisionTrackerDemo(robotHandler)

        self.current_demo = self.vision_tracker_demo

    def run(self):
        self.current_demo.start()


# def f():
#     mainmethod()
#
#
# def g():
#     robotHandler = RobotHandler()
#     is_lab_work = False
#     demo = MedusaDemo(robotHandler)
#     demo.run()


if __name__ == '__main__':
    # visionProcess = Process(target=f)
    # audioProcess = Process(target=g)

    # visionProcess.start()
    # audioProcess.start()

    # visionHandler = VisionHandler()
    # visionHandler.start()
    # MedusaDemo(RobotHandler()).run()
    robotHandler = RobotHandler(is_lab_work=True)
    performance = Performance(robotHandler, is_lab_work=True)
    performance.runSequence()


