import numpy as np


class Phrase:
    def __init__(self, notes=None, onsets=None, tempo=None, name=None):
        self.name = name
        self.notes = notes if notes is not None else []
        self.onsets = onsets if onsets is not None else []
        self.tempo = tempo

    def get(self):
        return self.notes, self.onsets

    def get_raw_notes(self):
        notes = []
        for note in self.notes:
            notes.append(note.pitch)
        return np.array(notes)

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, item):
        if len(self.notes) > item:
            return self.notes[item], self.onsets[item]
        return None, None

    def __setitem__(self, key, value: tuple):
        if len(self.notes) > key:
            self.notes[key] = value[0]
            self.onsets[key] = value[1]

    def append(self, note, onset):
        self.notes.append(note)
        self.onsets.append(onset)

    def __str__(self):
        ret = ""
        for i in range(len(self.notes)):
            ret = ret + f"{self.notes[i]}, Onset: {self.onsets[i]}\n"
        return ret
