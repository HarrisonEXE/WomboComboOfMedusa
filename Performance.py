from threading import Thread
import time
from Demos.Haltable.HaltableMicDemo import HaltableMicDemo
from Demos.Haltable.HaltableVoiceDemo import HaltableVoiceDemo
from Demos.VisionTrackerDemo import VisionTrackerDemo
from Handlers.RobotHandler import RobotHandler


class Performance:
    def __init__(self, robotHandler, is_lab_work=True):
        self.robotHandler = robotHandler

        self.voiceDemo = HaltableVoiceDemo(robotHandler, is_lab_work)
        self.micDemo = HaltableMicDemo(
            robotHandler, is_lab_work, robots_already_awake=True)
        self.visionTrackerDemo = VisionTrackerDemo(robotHandler, is_lab_work)

        self.mainThread = Thread()

    def runSequence(self):
        # 0:00 ------------------- 1 -
        # Voice Command (Audio)
        # ----------------------------
        mainThread = Thread(target=self.voiceDemo.start)
        mainThread.start()
        mainThread.join()

        # TODO: Setup Xylophone mic (seperate from voice commands)

        # 0:01 - 0:30 ------------ 2 -
        # Xylophone Mimicking (Audio)
        # ----------------------------
        self.micDemo.start()
        mainThread = Thread(target=self.micDemo.runDemoThread)
        mainThread.start()
        # TODO: Kill based on r2pmidi
        time.sleep(30)
        self.micDemo.kill()
        mainThread.join()

        # 0:31 - END ------------- 2 -
        # Xylophone Mimicking (Audio)
        # ----------------------------
        mainThread = Thread(target=self.visionTrackerDemo.start)
        mainThread.start()


robotHandler = RobotHandler(is_lab_work=False)
performance = Performance(robotHandler, is_lab_work=False)
performance.runSequence()
