#!/usr/bin/python2
# ----------------------------------------------------
# Tool to align ChipWhisperer and GNU Radio signals
# for EM analysis
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from os import listdir
from os.path import isfile, join
from dsp import *
from debug import DEBUG
import collections
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import configparser

Bounds = collections.namedtuple('Bounds', ['begin', 'end'])

def align_traces(traces, reference):
    aligned_traces = []

    for trace in traces:
        aligned_trace = align(trace, reference_trace)
        if not aligned_trace is None:
            aligned_traces.append(aligned_trace)

    return np.array(aligned_traces)

def filter_traces(traces):
    filtered_traces = []

    for trace in traces:
        filtered_trace = butter_filter(trace)
        filtered_traces.append(filtered_trace)

    return np.array(filtered_traces)

def update_config(path, trace_set, update_dict):
    cp = configparser.RawConfigParser()
    cp.optionxform = str

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
    parser = argparse.ArgumentParser(description='Tool to align ChipWhisperer and GNU Radio signals for EM analysis')
    parser.add_argument('format', type=str, choices=['cw','gnuradio','plot'], help='Output format')
    args, unknown = parser.parse_known_args()

    input_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces_unaligned/"
    output_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces/"
    output_path_gnuradio = "/home/pieter/deleteme/"
    window = Bounds(begin=1600, end=14000)

    # List files
    trace_sets = sorted([f for f in listdir(input_path) if isfile(join(input_path, f)) and '_traces.npy' in f])
    reference_trace = None
    for trace_set in trace_sets:
        # Print progress
        print("\rAligning %s...             " % trace_set),
        sys.stdout.flush()

        # Load traces and a reference trace
        traces = np.load(join(input_path, trace_set))
        if reference_trace is None:
            reference_trace = traces[0][window.begin:window.end]

        # Align traces
        traces = align_traces(traces, reference_trace)
        #traces = filter_traces(traces)  # Lennard

        if args.format == 'plot':
            for a in traces:
                plt.plot(range(0, len(a)), butter_filter(a))
            plt.show()
        elif args.format == 'cw':
            # Save back to output file
            np.save(join(output_path, trace_set), traces)

            # Update the corresponding config file
            update_config(output_path, trace_set, {"numPoints": len(reference_trace)})
        elif args.format == 'gnuradio':
            count = 1
            for trace in traces:
                trace.tofile(join(output_path_gnuradio, "%s-%d.rf32_le" % (trace_set.rpartition('_')[0], count)))
                count += 1
        else:
            print("Unknown format: %s" % args.format)
            exit(1)
