#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from os import listdir
from os.path import isfile, join
from dsp import *
from ops import *
from debug import DEBUG
from time import sleep
from emma_worker import app
from sigmf import SigMFFile
from celery import group
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import configparser

def filter_trace_set(trace_set):
    filtered_trace_set = []

    for trace in trace_set:
        filtered_trace = butter_filter(trace)
        filtered_trace_set.append(filtered_trace)

    return np.array(filtered_trace_set)

def update_cw_config(path, trace_set, update_dict):
    cp = configparser.RawConfigParser()
    cp.optionxform = str  # Preserve case sensitivity

    # Read file
    config_file_path = join(path, 'config_' + trace_set.rpartition('_')[0] + '_.cfg')
    cp.read(config_file_path)

    for key in update_dict:
        cp.get("Trace Config", key)
        cp.set("Trace Config", key, update_dict[key])

    with open(config_file_path, 'w') as config_file_path_fp:
        cp.write(config_file_path_fp)

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA)')  # ['align','attack','filter','save']
    parser.add_argument('actions', type=str, choices=ops.keys(), help='Action to perform', nargs='+')
    parser.add_argument('trace_set_path', type=str, help='Input directory where the trace sets are located')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw','sigmf','plot'], default='sigmf', help='Output format to use when saving')
    parser.add_argument('--outpath', '-O', dest='outpath', type=str, default='./export/', help='Output path to use when saving')
    args, unknown = parser.parse_known_args()

    try:
        # Get a list of filenames depending on the format TODO
        input_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces_unaligned/"
        output_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces/"
        output_path_gnuradio = "./export/"

        # List files TODO join here with input_path so it doesn't need to be provided to workers
        trace_set_paths = sorted([join(input_path,f) for f in listdir(input_path) if isfile(join(input_path, f)) and '_traces.npy' in f])

        # Worker-specific configuration
        conf = argparse.Namespace(
            reference_trace=None,  # TODO: Get one from the trace_set_paths (default to trace 0)
            window=Window(begin=1600, end=14000),
        )

        # Distribute files among workers TODO
        result = group([work.s(trace_set_paths, args, conf)])()
        print(result.get())
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.backend.cleanup()
