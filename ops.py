# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np
import sys
import matplotlib.pyplot as plt
import emio
import redis_lock
import pickle
from emma_worker import app, broker
from dsp import *
from correlation import Correlation
from functools import wraps
from os.path import join, basename
from emutils import Window
from celery.utils.log import get_task_logger
from lut import hw, sbox
from celery import Task
from emresult import EMResult

logger = get_task_logger(__name__)  # Logger
ops = {}  # Op registry

class EMMATask(Task):
    test = 'a'

def op(name):
    '''
    Defines the @op decorator
    '''
    def decorator(func):
        ops[name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@op('align')
def align_trace_set(trace_set, result, conf):
    '''
    Align a set of traces based on a single reference trace using cross-correlation.
    If a trace is empty, it is discarded.
    '''
    aligned_trace_set = []
    reference = conf.reference_trace

    for trace in trace_set.traces:
        aligned_trace = align(trace, reference)
        if not aligned_trace is None:
            aligned_trace_set.append(aligned_trace)

    trace_set.set_traces(np.array(aligned_trace_set))

@op('filter')
def filter_trace_set(trace_set, result, conf=None):
    '''
    Apply a Butterworth filter to the traces.
    '''
    filtered_trace_set = []

    for trace in trace_set.traces:
        filtered_trace = butter_filter(trace)
        filtered_trace_set.append(filtered_trace)

    trace_set.set_traces(np.array(filtered_trace_set))

@op('save')
def save_trace_set(trace_set, result, conf):
    '''
    Save the trace set to a file using the output format specified in the conf object.
    '''
    if conf.outform == 'cw':
        # Save back to output file
        np.save(join(conf.outpath, trace_set.name + '_traces.npy'), trace_set.traces)

        # Update the corresponding config file
        emio.update_cw_config(conf.outpath, trace_set, {"numPoints": len(conf.reference_trace)})
    elif conf.outform == 'sigmf':  # TODO make SigMF compliant
        count = 1
        for trace in trace_set.traces:
            trace.tofile(join(output_path_gnuradio, "%s-%d.rf32_le" % (trace_set.name, count)))
            count += 1
    else:
        print("Unknown format: %s" % conf.outform)
        exit(1)

@op('plot')
def plot_trace_set(trace_set, result, conf=None):
    '''
    Plot each trace in a trace set using Matplotlib
    '''
    for trace in trace_set.traces:
        plt.plot(range(0, len(trace)), trace)
    plt.show()

@op('attack')
def attack_trace_set(trace_set, result, conf=None):
    '''
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    '''
    logger.info("Attacking trace set %s..." % trace_set.name)
    trace_set.assert_validity()

    for subkey_idx in range(0, conf.num_subkeys):
        hypotheses = np.empty([256, trace_set.num_traces])

        # 1. Build hypotheses for all 256 possibilities of the key and all traces
        for subkey_guess in range(0, 256):
            for i in range(0, trace_set.num_traces):
                hypotheses[subkey_guess, i] = hw[sbox[trace_set.plaintexts[i][subkey_idx] ^ subkey_guess]]  # Model of the power consumption

        # 2. Given point j of trace i, calculate the correlation between all hypotheses
        for j in range(0, conf.attack_window.size):
            # Get measurements (columns) from all traces
            measurements = np.empty(trace_set.num_traces)
            for i in range(0, trace_set.num_traces):
                measurements[i] = trace_set.traces[i][conf.attack_window.begin+j]

            # Correlate measurements with 256 hypotheses
            for subkey_guess in range(0, 256):
                # Update correlation
                result.correlations[subkey_idx,subkey_guess,j].update(hypotheses[subkey_guess,:], measurements)

@app.task(bind=True)
def merge(self, to_merge):
    if type(to_merge) is EMResult:
        to_merge = [to_merge]

    if len(to_merge) >= 1:
        # Get size of correlations
        shape = to_merge[0].correlations.shape

        # Init result
        result = EMResult(task_id=self.request.id)
        result.correlations = Correlation.init(shape)

        # Start merging
        for m in to_merge:
            for subkey_idx in range(0, shape[0]):
                for subkey_guess in range(0, shape[1]):
                    for point in range(0, shape[2]):
                        result.correlations[subkey_idx,subkey_guess, point].merge(m.correlations[subkey_idx,subkey_guess, point])

            # Done with this task, so delete it
            logger.warning("Deleting %s" % m.task_id)
            app.AsyncResult(m.task_id).forget()

        return result
    else:
        return None

@app.task(bind=True)
def work(self, trace_set_paths, conf):
    '''
    Actions to be performed by workers on the trace set given in trace_set_path.
    '''

    if type(trace_set_paths) is list:
        # TODO build this from within the ops themselves by passing as ref!
        result = EMResult(task_id=self.request.id)
        result.correlations = Correlation.init([16, 256, conf.attack_window.size])

        for trace_set_path in trace_set_paths:
            logger.info("Node performing %s on trace set '%s'" % (str(conf.actions), trace_set_path))

            # Get trace name from path
            trace_set_name = basename(trace_set_path)

            # Load trace
            trace_set = emio.get_trace_set(trace_set_path, conf.inform, ignore_malformed=False)
            if trace_set is None:  # TODO FIX ME!
                logger.warning("Skipping trace set %s" % trace_set_path)
                continue

            # Perform actions
            for action in conf.actions:
                if action in ops:
                    ops[action](trace_set, result, conf=conf)
                else:
                    logger.warning("Ignoring unknown action '%s'." % action)

        return result
    else:
        logger.error("Must provide a list of trace set paths to worker!")
        return None
