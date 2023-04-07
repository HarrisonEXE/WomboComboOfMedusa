import numpy as np
from typing import Tuple, Dict

import pyaudio
import time

from threading import Thread, Lock, Event
from Classes.Phrase import Phrase
from Classes.audioDevice import AudioDevice
from Demos.Interfaces.IDemo import IDemo
from Demos.Interfaces.IRobotDemo import IRobotDemo
from Handlers.PerformanceHandler import PerformanceHandler
from Handlers.RobotHandler import RobotHandler
from Helpers.audioToMidi import AudioMidiConverter


class IMicDemo(IRobotDemo):
    def __init__(self, robotHandler, is_lab_work=True, sr=48000, frame_size=2400, activation_threshold=0.02, n_wait=16, robots_already_awake=False):
        super().__init__(robotHandler, is_lab_work, robots_already_awake)
        self.name = "Mic Demo Interface"

        self.is_lab_work = is_lab_work
        self.active = False
        self.activation_threshold = activation_threshold
        self.n_wait = n_wait
        self.wait_count = 0
        self.playing = False
        self.phrase = []
        self.midi_notes = []
        self.midi_onsets = []

        # Create RobotDemo Interface and move this
        self.performance_handler = PerformanceHandler(robotHandler=self.robotHandler,
                                                      is_lab_work=self.is_lab_work)

        self.process_thread = Thread()
        self.event = Event()
        self.lock = Lock()

        # TODO: Remove Raga Map
        self.audio2midi = AudioMidiConverter(
            raga_map=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], sr=sr, frame_size=frame_size)

    def connectAudioDevice(self):
        try:
            audioDevice = AudioDevice(self.listener)
            print("Device connected!")
            audioDevice.start()
            print("Device stream started.")

        except AssertionError:
            print("Device not found. You probably did something wrong.")

    def start(self):
        self.connectAudioDevice()
        self.readyRobots()
        self.announceStart()
        if self.process_thread.is_alive():
            self.process_thread.join()
        self.runDemoThread()

    def runDemoThread(self):
        self.lock.acquire()
        self.active = True
        self.running = True
        self.lock.release()
        self.process_thread = Thread(target=self._process)
        self.process_thread.start()
        self.event.clear()

    def reset_var(self):
        self.wait_count = 0
        self.playing = False
        self.phrase = []
        self.last_time = time.time()

    def waitForInput(self):
        while True and self.running:
            time.sleep(0.1)
            self.lock.acquire()
            if not self.active:
                self.lock.release()
                return

            if not (self.playing or len(self.phrase) == 0):
                self.lock.release()
                break
            self.lock.release()
        print(f"\nYour noises have been noted for the record.")

    def inputToPhrase(self):
        self.lock.acquire()
        phrase = np.hstack(self.phrase)
        self.phrase = []
        self.lock.release()

        if len(phrase) > 0:
            notes, onsets = self.audio2midi.convert(phrase, return_onsets=True)
            print("notes:", notes)
            # print("onsets:", onsets)
            phrase = Phrase(notes, onsets)
            return phrase
        else:
            return None

    def alterPhrase(self, phrase):
        return phrase

    def _process(self):
        try:
            if not self.running:
                print("Demo has been killed.")
                return

            self.waitForInput()
            phrase = self.inputToPhrase()
            if phrase:
                phrase = self.alterPhrase(phrase)
                self.performance_handler.perform(phrase)
            self._process()
        except ValueError:
            print("Mic Demo was forcefully stopped.")


    def listener(self, in_data: bytes, frame_count: int, time_info: Dict[str, float], status: int) -> Tuple[
            bytes, int]:
        if not self.active:
            self.reset_var()
            return in_data, pyaudio.paContinue

        y = np.frombuffer(in_data, dtype=np.int16)
        y = y[::2]

        y = self.int16_to_float(y)
        activation = np.abs(y).mean()
        if activation > self.activation_threshold:

            ''' Makes for a cool visual - Harrison '''
            print(activation)

            self.playing = True
            self.wait_count = 0
            self.lock.acquire()
            self.phrase.append(y)
            self.lock.release()
        else:
            if self.wait_count > self.n_wait:
                self.playing = False
                self.wait_count = 0
            else:
                self.lock.acquire()
                if self.playing:
                    self.phrase.append(y)
                self.lock.release()
                self.wait_count += 1

        return in_data, pyaudio.paContinue

    @staticmethod
    def int16_to_float(x):
        return x / (1 << 15)
