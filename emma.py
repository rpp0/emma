#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

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
import emutils
import emio

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
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA)')  # ['align','attack','filter','save']
    parser.add_argument('actions', type=str, choices=ops.keys(), help='Action to perform', nargs='+')
    parser.add_argument('inpath', type=str, help='Input path where the trace sets are located')
    parser.add_argument('--inform', dest='inform', type=str, choices=['cw','sigmf','gnuradio'], default='cw', help='Input format to use when loading')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw','sigmf','plot'], default='sigmf', help='Output format to use when saving')
    parser.add_argument('--outpath', '-O', dest='outpath', type=str, default='./export/', help='Output path to use when saving')
    parser.add_argument('--num-cores', dest='num_cores', type=int, default=4, help='Number of CPU cores')
    args, unknown = parser.parse_known_args()
    print(emutils.BANNER)

    try:
        output_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces/"
        output_path_gnuradio = "./export/"

        # Get a list of filenames depending on the format
        trace_set_paths = emio.get_trace_paths(args.inpath, args.inform)

        # Worker-specific configuration
        window = Window(begin=1600, end=14000)
        conf = argparse.Namespace(
            reference_trace=emio.get_trace_set(trace_set_paths[0], args.inform)[0][window.begin:window.end],
            window=window,
            **args.__dict__
        )

        jobs = []
        for part in emutils.partition(trace_set_paths, int(len(trace_set_paths) / args.num_cores)):
            jobs.append(work.s(part, conf))

        result = group(jobs)()
        print(result.get())
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.backend.cleanup()
