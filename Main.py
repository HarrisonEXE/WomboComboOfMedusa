import argparse
from Demos.KeyboardDemo import KeyboardDemo
from Demos.KeyboardRandomNoteDemo import KeyboardRandomNoteDemo
from Demos.KeyboardRobotlessDemo import KeyboardRobotlessDemo
from Demos.MicDemo import MicDemo
from Demos.RandomizedMicDemo import RandomizedMicDemo
from Demos.VoiceRecognizer import VoiceDemo
from Demos.VisionTrackerDemo import VisionTrackerDemo
from Demos.RecordedDanceDemo import RecordedDanceDemo
from Handlers.RobotHandler import RobotHandler
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
        self.recorded_dance_demo = RecordedDanceDemo(robotHandler)

        self.current_demo = self.recorded_dance_demo

    def run(self):
        self.current_demo.start()


def run_demo():
    robotHandler = RobotHandler()
    demo = MedusaDemo(robotHandler)
    try:
        demo.run()
    except KeyboardInterrupt:
        demo.current_demo.kill()


def run_performance():
    robotHandler = RobotHandler(is_lab_work=True)
    performance = Performance(robotHandler, is_lab_work=True)
    try:
        performance.runSequence()
    except KeyboardInterrupt:
        performance.stopSequence()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Medusa Demo")
    parser.add_argument("--mode", choices=["demo", "performance"], default="performance", help="Select the mode to "
                                                                                               "run (demo or "
                                                                                               "performance)")

    args = parser.parse_args()

    try:
        if args.mode == "demo":
            run_demo()
        elif args.mode == "performance":
            run_performance()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Killing program...")
    # finally:
    #     print("Exiting...")
    #     exit(0)
