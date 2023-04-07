from threading import Thread
from Demos.IMicDemo import IMicDemo


# Purely for documentation purposes
def override(f):
    return f


class HaltableMicDemo(IMicDemo):
    def __init__(self, robotHandler, is_lab_work=True, sr=48000, frame_size=2400, activation_threshold=0.02, n_wait=16, robots_already_awake=False):
        super().__init__(robotHandler, is_lab_work, sr, frame_size, activation_threshold,
                         n_wait, robots_already_awake)
        self.name = "Mic Demo with matching notes and rhythm"

    @override
    def start(self):
        self.connectAudioDevice()
        self.readyRobots()
        self.announceStart()
        if self.process_thread.is_alive():
            self.process_thread.join()
