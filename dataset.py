# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import configparser
import emio
from os import listdir
from os.path import isfile, join

class Dataset():
    def __init__(self, id, format, reference_index=0):
        self.id = id
        self.format = format
        self.reference_index = reference_index

        self._setup()

    def _setup(self):
        '''
        Get a list of relative trace set paths for the dataset identifier and retrieve
        a reference signal for the entire dataset.

        Example trace set paths:
        em-arduino/trace1.npy
        em-arduino/trace2.npy
        ...
        em-arduino/tracen.npy

        Where trace1.npy is loaded as the reference signal.

        At a later time, the relative paths need to be resolved to absolute paths
        on the workers.
        '''

        # Get local absolute path for worker in order to get listing
        settings = configparser.RawConfigParser()
        settings.read('settings.conf')
        prefix = settings.get("Datasets", "datasets_path")
        path = join(prefix, self.id)

        # Assign trace set paths
        if self.format == "cw":  # .npy
            self.trace_set_paths = sorted([join(self.id, f) for f in listdir(path) if isfile(join(path, f)) and '_traces.npy' in f])
        elif self.format == "sigmf":  # .meta
            self.trace_set_paths = None
            raise NotImplementedError
        elif self.format == "gnuradio":  # .cfile
            self.trace_set_paths = None
            raise NotImplementedError
        else:
            raise Exception("Unknown input format '%s'" % inform)

        # Assign reference signal
        self.reference_signal = emio.get_trace_set(join(prefix, self.trace_set_paths[0]), self.format, ignore_malformed=False).traces[self.reference_index].signal