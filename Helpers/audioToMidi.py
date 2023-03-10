"""
Adapted By: Harrison Melton
Original Author: Raghavasimhan Sankaranarayanan
Date: 25/02/2023
"""

import sys
import librosa
import madmom
import numpy as np
from pretty_midi import Note
np.set_printoptions(threshold=sys.maxsize)


class AudioMidiConverter:
    def __init__(self, raga_map=None, root='D3', sr=16000, note_min='D2', note_max='A5', frame_size=2048,
                 hop_length=441, threshold: float = 0.35, pre_max: int = 3, post_max: int = 3):
        self.fmin = librosa.note_to_hz(note_min)
        self.fmax = librosa.note_to_hz(note_max)
        self.hop_length = hop_length
        self.frame_size = frame_size
        self.raga_map = np.array(raga_map) if raga_map else None
        self.sr = sr
        self.root = librosa.note_to_midi(root)
        self.threshold = threshold
        self.pre_max = pre_max
        self.post_max = post_max
        self.empty_arr = np.array([])

        # Schlüter, Jan, and Sebastian Böck. "Improved musical onset detection with convolutional neural networks." 2014 ieee international conference on acoustics, speech and signal processing (icassp). IEEE, 2014.
        self.onset_processor = madmom.features.CNNOnsetProcessor()

        # viterbi decoding
        self.note_range = (50, 86)
        self.key = 62
        self.trans_mat = self.build_trans_matrix(
            self.raga_map, key=self.key, p_adj=0.3, p_repeat=0.15, note_range=self.note_range)
        self.note_idx = self.get_note_indices(
            self.raga_map, key=self.key, note_range=self.note_range)
        note_prob = np.zeros(self.note_range[1] - self.note_range[0])
        note_prob[self.note_idx - self.note_range[0]] = 1 / len(self.note_idx)
        self.start_prob = note_prob

    def convert(self, y, return_onsets=False, velocity=100):
        f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=self.fmin * 0.9, fmax=self.fmax * 1.1, sr=self.sr,
                                                    frame_length=self.frame_size, hop_length=self.hop_length)
        if len(f0) == 0:
            print("No f0")
            if return_onsets:
                return self.empty_arr, self.empty_arr
            return self.empty_arr

        pitch = librosa.hz_to_midi(f0)
        pitch[np.isnan(pitch)] = 0
        # There is at-least one onset at [0]
        onsets = self.get_onsets(
            y, threshold=self.threshold, pre_max=self.pre_max, post_max=self.post_max)
        # print(onsets)
        notes = np.zeros(len(onsets), dtype=int)
        for i in range(len(onsets) - 1):
            notes[i] = np.round(np.nanmedian(pitch[onsets[i]: onsets[i + 1]]))
        notes[-1] = np.round(np.nanmedian(pitch[onsets[-1]:]))

        onsets = onsets[notes > 0] * self.hop_length / self.sr
        notes = notes[notes > 0]

        temp = []
        for i in range(len(notes)):
            temp.append(
                Note(velocity, notes[i], start=onsets[i], end=onsets[i] + 0.1))

        if return_onsets:
            return temp, onsets

        return temp

    def get_onsets(self, y, threshold: float, pre_max: int, post_max: int):
        act = self.onset_processor(y)
        onsets = madmom.features.onsets.peak_picking(activations=act, threshold=threshold, pre_max=pre_max,
                                                     post_max=post_max)
        return np.unique(np.hstack([0, onsets]))

    @staticmethod
    def fix_outliers(arr, m=2):
        arr_mean = np.mean(arr)
        arr_std = m * np.std(arr)
        for _i in range(len(arr)):
            n = arr[_i]
            if np.abs(n - arr_mean) > arr_std:
                arr[_i] = AudioMidiConverter.shift_octave(arr[_i], arr_mean)
        return arr

    @staticmethod
    def shift_octave(val, ref):
        x = (ref - val) // 6
        res = x % 2
        x = x // 2
        return val + ((x + res) * 12)

    @staticmethod
    def get_tempo(y):
        return librosa.beat.tempo(y=y)[0]

    @staticmethod
    def build_trans_matrix(raga_map, key, p_adj, p_repeat, note_range):
        n_states = note_range[1] - note_range[0]
        if 2*p_adj + p_repeat > 1:
            raise Exception("Probability should be <= 1")
        mat = np.eye(n_states)
        note_idx = AudioMidiConverter.get_note_indices(
            raga_map, key, note_range)
        # print(note_idx)
        s = note_range[0]
        mat[note_idx[0] - s, note_idx[0] - s] = p_repeat
        mat[note_idx[-1] - s, note_idx[-1] - s] = p_repeat
        mat[note_idx[0] - s, note_idx[1] - s] = p_adj*2
        mat[note_idx[-1] - s, note_idx[-2] - s] = p_adj*2
        for i in range(1, len(note_idx)-1):
            mat[note_idx[i] - s, note_idx[i] - s] = p_repeat
            mat[note_idx[i] - s, note_idx[i - 1] - s] = p_adj
            mat[note_idx[i] - s, note_idx[i + 1] - s] = p_adj

        for r in mat:
            remaining = 1 - r.sum()
            num_non_zeros = len(np.where(r > 0)[0])
            p = remaining / (len(note_idx) - num_non_zeros)
            for i in range(len(r)):
                if not r[i] > 0 and i + s in note_idx:
                    r[i] = p

        return mat

    @staticmethod
    def get_note_indices(raga_map, key, note_range):
        original_note_idx = np.where(raga_map == 1)[0] + key
        note_idx = original_note_idx.copy()
        i = 1
        while note_idx[0] > note_range[0]:
            note_idx = np.concatenate([original_note_idx - (12 * i), note_idx])
            # print(note_idx)
            i += 1

        note_idx = note_idx[note_idx >= note_range[0]]

        i = 1
        while note_idx[-1] < note_range[1]:
            note_idx = np.concatenate([note_idx, original_note_idx + (12 * i)])
            i += 1

        return note_idx[note_idx < note_range[1]]

    @staticmethod
    def get_prior_probabilities(notes, key, raga_map, prob, note_range):
        n_states = note_range[1] - note_range[0]
        note_idx = AudioMidiConverter.get_note_indices(
            raga_map, key, note_range)
        n_steps = len(notes)
        prior = np.zeros((n_states, n_steps))
        s = note_range[0]
        for i in range(n_steps):
            for j in range(n_states):
                if notes[i] - s == j:
                    prior[j, i] = prob
                elif j + s in note_idx:
                    prior[j, i] = (1 - prob) / (len(note_idx) - 1)

        return prior

    def get_most_likely_sequence(self, notes, key, prob=0.4):
        prior = self.get_prior_probabilities(notes, key=key, raga_map=self.raga_map, prob=prob,
                                             note_range=self.note_range)
        path = librosa.sequence.viterbi(
            prob=prior, transition=self.trans_mat, p_init=self.start_prob)
        return path + self.note_range[0]
