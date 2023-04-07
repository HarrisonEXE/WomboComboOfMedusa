import numpy as np
import pyaudio
from Demos.Interfaces.IMicDemo import IMicDemo

from Helpers.NoteRandomizer import beat_randomizer, basic_randomizer


# Purely for documentation purposes
def override(f):
    return f


class RandomizedMicDemo(IMicDemo):
    def __init__(self, robotHandler, is_lab_work=True, sr=48000, frame_size=2400, activation_threshold=0.02, n_wait=16):
        super().__init__(robotHandler, is_lab_work, sr, frame_size, activation_threshold,
                         n_wait)
        self.name = "Mic Demo with Randomized Beat"

    @override
    def alterPhrase(self, phrase):
        print("Randomizing beat...")
        random_phrase = beat_randomizer(phrase)
        return random_phrase
