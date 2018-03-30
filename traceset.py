# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np

class Trace(object):
    def __init__(self, signal, plaintext, ciphertext, key):
        self.signal = signal
        self.plaintext = plaintext
        self.ciphertext = ciphertext
        self.key = key

class TraceSet(object):
    def __init__(self, name=None, traces=None, plaintexts=None, ciphertexts=None, keys=None):
        self.name = name
        self.traces = self._zip_traces(traces, plaintexts, ciphertexts, keys)
        self.num_traces = traces.shape[0]
        self.windowed = False
        self.window = None

    def _zip_traces(self, traces, plaintexts, ciphertexts, keys):
        if traces is None:
            raise Exception("Tried to create TraceSet without traces")

        zipped_traces = [Trace(None, None, None, None) for i in range(0, traces.shape[0])]

        # Signals
        for i in range(0, traces.shape[0]):
            zipped_traces[i].signal = traces[i]

        # Plaintexts
        if not plaintexts is None:
            assert(traces.shape[0] == plaintexts.shape[0])
            for i in range(0, plaintexts.shape[0]):
                zipped_traces[i].plaintext = plaintexts[i]

        # Ciphertexts
        if not ciphertexts is None:
            assert(traces.shape[0] == ciphertexts.shape[0])
            for i in range(0, ciphertexts.shape[0]):
                zipped_traces[i].ciphertext = ciphertexts[i]

        # Keys
        if not keys is None:
            assert(traces.shape[0] == keys.shape[0])
            for i in range(0, keys.shape[0]):
                zipped_traces[i].key = keys[i]

        return np.array(zipped_traces)

    def set_traces(self, traces):
        if not type(traces) is np.ndarray:
            traces = np.array(traces)
        self.traces = traces
        self.num_traces = traces.shape[0]

    def save(self, path, fmt='cw'):
        if fmt == 'cw':
            np.save(self.name + "_p_traces.npy", np.array([t.signal for t in self.traces]))
        elif fmt == 'sigmf':
            raise NotImplementedError
