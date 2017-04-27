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
import collections
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import configparser

Window = collections.namedtuple('Window', ['begin', 'end'])

def filter_traces(traces):
    filtered_traces = []

    for trace in traces:
        filtered_trace = butter_filter(trace)
        filtered_traces.append(filtered_trace)

    return np.array(filtered_traces)

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
    parser.add_argument('traces', type=str, help='Input directory where the traces are located')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw','sigmf','plot'], default='sigmf', help='Output format to use when saving')
    parser.add_argument('--outpath', '-O', dest='outpath', type=str, default='./export/', help='Output path to use when saving')
    args, unknown = parser.parse_known_args()

    try:
        # Get a list of filenames depending on the format TODO
        input_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces_unaligned/"
        output_path = "/home/pieter/chipwhisperer/projects/tmp/default_data/traces/"
        output_path_gnuradio = "./export/"
        window = Window(begin=1600, end=14000)

        # List files
        trace_sets = sorted([f for f in listdir(input_path) if isfile(join(input_path, f)) and '_traces.npy' in f])
        reference_trace = None

        # Perform actions on the sample sets
        for trace_set in trace_sets:
            # Load traces and a reference trace depending on the format TODO
            traces = np.load(join(input_path, trace_set))
            if reference_trace is None:
                reference_trace = traces[0][window.begin:window.end]

            for action in args.actions: # TODO
                # Print progress
                print("\rAligning %s...             " % trace_set),
                sys.stdout.flush()

                # Align traces
                traces = ops['align'](traces, reference_trace)
                #align_traces.delay(traces, reference_trace)
                #traces = filter_traces(traces)

                # Write to output
                if args.outform == 'plot':
                    for a in traces:
                        plt.plot(range(0, len(a)), butter_filter(a))
                    plt.show()
                elif args.outform == 'cw':
                    # Save back to output file
                    np.save(join(output_path, trace_set), traces)

                    # Update the corresponding config file
                    update_cw_config(output_path, trace_set, {"numPoints": len(reference_trace)})
                elif args.outform == 'sigmf':
                    count = 1
                    for trace in traces:
                        trace.tofile(join(output_path_gnuradio, "%s-%d.rf32_le" % (trace_set.rpartition('_')[0], count)))
                        count += 1
                else:
                    print("Unknown format: %s" % args.format)
                    exit(1)
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.backend.cleanup()
