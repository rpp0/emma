#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from ops import *
from debug import DEBUG
from time import sleep
from emma_worker import app, backend
from celery import group, chord
from asyncio import Semaphore
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import configparser
import emutils
import emio
import subprocess

mutex = Semaphore()
windowsize = Window(begin=1080, end=1308).size
result = Correlation.init([256, windowsize]) # 256 * window size

def result_callback(task_id, value):
    global result
    mutex.acquire()
    if not value is None:
        subkey_idx = 0
        for subkey_guess in range(0, 256):
            for p in range(0, windowsize):
                result[subkey_guess, p].merge(value[subkey_guess, p])
    print("Job %s done!" % task_id)
    mutex.release()
    app.AsyncResult(task_id).forget()

def build_task_graph(paths, conf):
    if len(paths) == 0:
        return None
    elif len(paths) == 1:
        print(paths[0])
        return work.s(paths[0], conf)
    else:
        mid = int(len(paths) / 2)
        left = build_task_graph(paths[0:mid], conf)
        right = build_task_graph(paths[mid:], conf)
        return chord([left, right], body=merge.s())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA)')
    parser.add_argument('actions', type=str, choices=ops.keys(), help='Action to perform', nargs='+')
    parser.add_argument('inpath', type=str, help='Input path where the trace sets are located')
    parser.add_argument('--inform', dest='inform', type=str, choices=['cw','sigmf','gnuradio'], default='cw', help='Input format to use when loading')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw','sigmf','gnuradio'], default='sigmf', help='Output format to use when saving')
    parser.add_argument('--outpath', '-O', dest='outpath', type=str, default='./export/', help='Output path to use when saving')
    parser.add_argument('--num-cores', dest='num_cores', type=int, default=4, help='Number of CPU cores')  # TODO remove, useless
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
            **args.__dict__
        )

        #jobs = []
        #for path in trace_set_paths:  # Create job for each path
        #    jobs.append(work.s(path, conf))

        # Execute jobs
        #group_task = group(jobs)()

        damnboi = build_task_graph(trace_set_paths, conf)
        result = damnboi().get_leaf().data['correlations']

        # Print results
        max_correlations = np.zeros([16, 256])
        for subkey_guess in range(0, 256):
            max_correlations[0, subkey_guess] = np.max(np.abs(result[subkey_guess,:]))
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
    subprocess.check_output(['pkill', '-9', '-f', 'celery'])
