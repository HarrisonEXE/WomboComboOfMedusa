import time
import threading


class PerformanceHandler:
    def __init__(self, robotHandler, is_lab_work=True):
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.robotHandler = robotHandler
        self.is_lab_work = is_lab_work

    def perform(self, phrase):
        prev_note_start = 0
        multiplier = 1.5  # Idk why it's 2. It just works.
        now = time.time()

        for i in range(len(phrase)):
            note = phrase.notes[i]
            corrected_pitch = self.correct_pitch(note.pitch)

            dly = max(0, ((note.start - prev_note_start) *
                      multiplier) - (time.time() - now))
            self.event.wait(dly)

            now = time.time()

            self.lock.acquire()  # TODO: Check to see if actually neccesary
            if self.is_lab_work:
                self.robotHandler.playString(corrected_pitch)
            else:
                self.robotHandler.playTestString(corrected_pitch)
            self.lock.release()

            prev_note_start = note.start

        print("")

    def correct_pitch(self, note):
        # Note Info:
        # 9 - A
        # 0 - C
        # 2 - D
        # 4 - E
        # 7 - G
        scale = [4, 0, 9, 2, 7]
        return scale.index(min(scale, key=lambda x: abs(x - (note % 12))))
