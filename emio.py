# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np
import configparser
from sigmf import SigMFFile
from os import listdir
from os.path import isfile, join
from traceset import TraceSet

def get_trace_paths(input_path, inform):
    '''
    Get a list of trace set paths given a specified input format.
    '''
    if inform == "cw":  # .npy
        return sorted([join(input_path,f) for f in listdir(input_path) if isfile(join(input_path, f)) and '_traces.npy' in f])
    elif inform == "sigmf":  # .meta
        raise NotImplementedError
    elif inform == "gnuradio":  # .cfile
        raise NotImplementedError
    else:
        print("Unknown input format '%s'" % inform)
        exit(1)

def get_trace_set(trace_set_path, inform):  # TODO wrap in some kind of SigMF class?
    '''
    Load traces in trace_set_path into a TraceSet object depending on the format.
    '''

    trace_set = TraceSet()

    if inform == "cw":
        trace_name = trace_set_path.rpartition('_traces')[0]
        plaintext_set_path = trace_name + '_textin.npy'

        trace_set.name = trace_name
        trace_set.traces = np.load(trace_set_path)  # TODO make more robust towards non-existing paths
        trace_set.plaintexts = np.load(plaintext_set_path)
    elif inform == "sigmf":  # .meta
        raise NotImplementedError
    elif inform == "gnuradio":  # .cfile
        raise NotImplementedError
    else:
        print("Unknown input format '%s'" % inform)
        exit(1)

    return trace_set

def update_cw_config(path, trace_set, update_dict):
    '''
    Update ChipWhisperer config file in order to reflect changes made to
    the traces by EMMA.
    '''
    cp = configparser.RawConfigParser()
    cp.optionxform = str  # Preserve case sensitivity

    # Read file
    config_file_path = join(path, 'config_' + trace_set.name + '_.cfg')
    cp.read(config_file_path)

    for key in update_dict:
        cp.get("Trace Config", key)
        cp.set("Trace Config", key, update_dict[key])

    with open(config_file_path, 'w') as config_file_path_fp:
        cp.write(config_file_path_fp)
