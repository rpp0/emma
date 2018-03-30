# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import configparser
from os import listdir
from os.path import isfile, join

class Dataset():
    def __init__(self, id, format):
        self.id = id
        self.format = format
        self.trace_set_paths = self._get_trace_set_paths()

    def _get_trace_set_paths(self):
        '''
        Get a list of relative trace set paths for the dataset identifier
        '''
        settings = configparser.RawConfigParser()
        settings.read('settings.conf')
        self.prefix = settings.get("Datasets", "datasets_path")
        path = join(self.prefix, self.id)

        if self.format == "cw":  # .npy
            return sorted([join(self.id, f) for f in listdir(path) if isfile(join(path, f)) and '_traces.npy' in f])
        elif self.format == "sigmf":  # .meta
            raise NotImplementedError
        elif self.format == "gnuradio":  # .cfile
            raise NotImplementedError
        else:
            raise Exception("Unknown input format '%s'" % inform)
