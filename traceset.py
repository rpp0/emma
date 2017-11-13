# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np

class TraceSet(object):
    def __init__(self, name=None, traces=None, plaintexts=None, ciphertexts=None, key=None):
        self.name = name
        self.traces = traces
        self.plaintexts = plaintexts
        self.ciphertexts = ciphertexts
        self.key = key
        self.num_traces = traces.shape[0]
        self.num_samples = traces.shape[1]

    def set_traces(self, traces):
        self.traces = traces
        self.num_traces = traces.shape[0]
        self.num_samples = traces.shape[1]

    def assert_validity(self):
        assert(self.traces.shape[0] == self.plaintexts.shape[0])
