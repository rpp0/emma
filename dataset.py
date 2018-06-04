# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import configparser
import emio
from os import listdir
from os.path import isfile, join

class Dataset():
    def __init__(self, id, format, reference_index=0, conf=None):
        self.id = id
        self.format = format
        self.reference_index = reference_index
        self.traces_per_set = 0

        self._setup(conf)

    def _setup(self, conf):
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
        self.prefix = prefix

        # Assign trace set paths
        if self.format == "cw":  # .npy
            path = join(prefix, self.id)
            self.trace_set_paths = sorted([join(self.id, f) for f in listdir(path) if isfile(join(path, f)) and '_traces.npy' in f])
        elif self.format == "sigmf":  # .meta
            self.trace_set_paths = None
            raise NotImplementedError
        elif self.format == "gnuradio":  # .cfile
            self.trace_set_paths = None
            raise NotImplementedError
        elif self.format == "ascad":  # ASCAD .h5
            # Hack to force split between validation and training set in ASCAD
            validation_set = join(prefix, 'ASCAD/ASCAD_data/ASCAD_databases/%s.h5-val' % self.id)
            training_set = join(prefix, 'ASCAD/ASCAD_data/ASCAD_databases/%s.h5-train' % self.id)

            # Make sure we never use training set when attacking or classifying
            if not conf is None and ('attack' in conf.actions or 'classify' in conf.actions):
                self.trace_set_paths = [validation_set]
            else:
                self.trace_set_paths = [validation_set, training_set]
        else:
            raise Exception("Unknown input format '%s'" % self.format)

        assert(len(self.trace_set_paths) > 0)

        # Assign reference signal
        reference_trace_set = emio.get_trace_set(join(self.prefix, self.trace_set_paths[0]), self.format, ignore_malformed=False)

        self.traces_per_set = len(reference_trace_set.traces)
        self.reference_signal = reference_trace_set.traces[self.reference_index].signal

# Statistics precomputed with get_dataset_statistics.py
def get_dataset_normalization_mean_std(name):
    if name == 'em-corr-arduino' or name == 'em-cpa-arduino':
        mean = 0.014595353784991782
        std = 0.006548281541447703
    elif name == 'ASCAD':
        mean = -11.587280595238095
        std = 25.75363459386104
    elif name == 'ASCAD_desync50':
        mean = -11.195121833333333
        std = 25.89963055607876
    elif name == 'ASCAD_desync100':
        mean = -11.093145738095238
        std = 26.11483790582092
    else:
        return 0.0, 1.0

    return mean, std
