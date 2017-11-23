# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np

class Trace(object):
    def __init__(self, signal, plaintext, ciphertext=None):
        self.signal = signal
        self.plaintext = plaintext
        self.ciphertext = ciphertext

class TraceSet(object):
    def __init__(self, name=None, traces=None, plaintexts=None, ciphertexts=None, key=None):
        self.name = name
        assert(traces.shape[0] == plaintexts.shape[0])
        self.traces = np.array([Trace(*x) for x in zip(traces, plaintexts)])
        self.key = key
        self.num_traces = traces.shape[0]

    def set_traces(self, traces):
        self.traces = traces
        self.num_traces = traces.shape[0]

    def save(self, path, fmt='cw'):
        if fmt == 'cw':
            np.save(self.name + "_p_traces.npy", np.array([t.signal for t in self.traces]))
        elif fmt == 'sigmf':
            raise NotImplementedError
