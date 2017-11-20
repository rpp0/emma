#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from ops import *
from debug import DEBUG
from time import sleep
from emma_worker import app, backend
from celery import group, chord, chain
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import configparser
import emutils
import emio
import subprocess
import time

def partition_work(trace_set_paths, conf):
    num_partitions = min(conf.max_subtasks, len(trace_set_paths))
    result = []
    for part in emutils.partition(trace_set_paths, num_partitions):
        result.append(work.si(part, conf))
    return chord(result, body=merge.s())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA)')
    parser.add_argument('actions', type=str, choices=ops.keys(), help='Action to perform', nargs='+')
    parser.add_argument('inpath', type=str, help='Input path where the trace sets are located')
    parser.add_argument('--inform', dest='inform', type=str, choices=['cw','sigmf','gnuradio'], default='cw', help='Input format to use when loading')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw','sigmf','gnuradio'], default='sigmf', help='Output format to use when saving')
    parser.add_argument('--outpath', '-O', dest='outpath', type=str, default='./export/', help='Output path to use when saving')
    parser.add_argument('--max-subtasks', type=int, default=2, help='Maximum number of subtasks')
    parser.add_argument('--num-subkeys', type=int, default=16, help='Number of subkeys to break')
    parser.add_argument('--kill-workers', default=False, action='store_true', help='Kill workers after finishing the tasks.')
    args, unknown = parser.parse_known_args()
    print(emutils.BANNER)

    try:
        # Clear any previous results. Sadly, there is no cleaner way atm.
        subprocess.check_output(["redis-cli", "flushall"])

        # Get a list of filenames depending on the format
        trace_set_paths = emio.get_trace_paths(args.inpath, args.inform)

        # Worker-specific configuration
        window = Window(begin=1600, end=14000)
        conf = argparse.Namespace(
            reference_trace=emio.get_trace_set(trace_set_paths[0], args.inform, ignore_malformed=False).traces[0][window.begin:window.end],
            window=window,
            #attack_window = Window(begin=1080, end=1082),
            #attack_window = Window(begin=980, end=1700),
            #attack_window = Window(begin=980, end=1008),
            attack_window = Window(begin=1280, end=1308),
            **args.__dict__
        )

        task = partition_work(trace_set_paths, conf)
        async_result = task()
        count = 0
        while not async_result.ready():
            print("\rElapsed: %d" % count, end='')
            count += 1
            time.sleep(1)
        print("")
        result = async_result.result.correlations
        print("Num entries: %d" % result[0][0][0]._n)

        # Print results
        max_correlations = np.zeros([16, 256])
        for subkey_idx in range(0, conf.num_subkeys):
            for subkey_guess in range(0, 256):
                max_correlations[subkey_idx, subkey_guess] = np.max(np.abs(result[subkey_idx,subkey_guess,:]))
        emutils.pretty_print_correlations(max_correlations, limit_rows=20)

        # Print key
        most_likely_bytes = np.argmax(max_correlations, axis=1)
        print(emutils.numpy_to_hex(most_likely_bytes))
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.control.purge()
    app.backend.cleanup()
    if args.kill_workers:
        subprocess.check_output(['pkill', '-9', '-f', 'celery'])
