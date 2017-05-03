# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

class TraceSet(object):
    def __init__(self, name=None, traces=None, plaintexts=None, ciphertexts=None, key=None):
        self.name = name
        self.traces = traces
        self.plaintexts = plaintexts
        self.ciphertexts = ciphertexts
        self.key = key
