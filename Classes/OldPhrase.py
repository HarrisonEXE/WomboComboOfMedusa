import numpy as np


class Phrase:
    def __init__(self, notes=None, onsets=None, tempo=None, name=None):
        self.name = name
        self.notes = notes if notes is not None else []
        self.onsets = onsets if onsets is not None else []
        self.aggregated_onsents = self.getAggregatedOnsets() if onsets is not None else []
        self.tempo = tempo

    def __str__(self):
        ret = ""
        for i in range(len(self.notes)):
            ret = ret + f"Degree: {self.notes[i]}, Onset: {self.onsets[i]}\n"
        return ret

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

    def get(self):
        return self.notes, self.onsets

    def getRawNotes(self):
        notes = []
        for note in self.notes:
            notes.append(note.pitch)
        return np.array(notes)

    def getAggregatedOnsets(onsets):
        aggregate = 0
        aggregated_onsets = []
        for onset in onsets:
            aggregate += onset
            aggregated_onsets.append(aggregate)
        return aggregated_onsets

    def append(self, note, onset):
        self.notes.append(note)
        self.onsets.append(onset)
        if not self.aggregated_onsents:
            self.aggregated_onsents.append(onset)
        else:
            self.aggregated_onsents.append(onset + self.aggregated_onsents[-1])
