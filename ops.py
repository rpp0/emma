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
def align_trace_set(trace_set, conf):
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
def filter_trace_set(trace_set, conf=None):
    '''
    Apply a Butterworth filter to the traces.
    '''
    filtered_trace_set = []

    for trace in trace_set.traces:
        filtered_trace = butter_filter(trace)
        filtered_trace_set.append(filtered_trace)

    trace_set.set_traces(np.array(filtered_trace_set))

@op('save')
def save_trace_set(trace_set, conf):
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
def plot_trace_set(trace_set, conf=None):
    '''
    Plot each trace in a trace set using Matplotlib
    '''
    for trace in trace_set.traces:
        plt.plot(range(0, len(trace)), trace)
    plt.show()

@op('attack')
def attack_trace_set(trace_set, conf=None):
    '''
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    '''
    logger.info("Attacking trace set %s..." % trace_set.name)
    trace_set.assert_validity()  # TODO temporary solution (plaintext vs traces problem in CW)
    #window = Window(begin=1080, end=1082)  # test
    #window = Window(begin=980, end=1700)
    window = Window(begin=980, end=1008)
    #window = Window(begin=1080, end=1308)
    subkey_idx = 0
    hypotheses = np.empty([256, trace_set.num_traces])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i] = hw[sbox[trace_set.plaintexts[i][subkey_idx] ^ subkey_guess]]  # Model of the power consumption

    # 2. Given point j of trace i, calculate the correlation between all hypotheses
    trace_set.correlations = Correlation.init([256,window.size])
    for j in range(0, window.size):
        # Get measurements (columns) from all traces
        measurements = np.empty(trace_set.num_traces)
        for i in range(0, trace_set.num_traces):
            measurements[i] = trace_set.traces[i][window.begin+j]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Update correlation
            trace_set.correlations[subkey_guess,j].update(hypotheses[subkey_guess,:], measurements)

@app.task
def dummy(results):
    return results

@app.task(bind=True, queue='priority.high')
def merge(self, results):  # results length should always be 2 TODO check
    left_result = results[0]
    right_result = results[1]
    if left_result is None and right_result is None:
        return None
    elif left_result is None:
        return right_result
    elif right_result is None:
        return left_result
    else:
        left = results[0].data['correlations']
        right = results[1].data['correlations']
        if left is None and right is None:
            return None
        elif left is None:
            return right_result
        elif right is None:
            return left_result
        else:
            # Build result
            subkey_idx = 0
            shape = left.shape
            for subkey_guess in range(0, shape[0]):
                for point in range(0, shape[1]):
                    left[subkey_guess, point].merge(right[subkey_guess, point])

            result = EMResult(data={'correlations': left}, task_id=self.request.id)

            # Clean up
            left_task = results[0].task_id
            right_task = results[1].task_id
            logger.warning("Deleting %s" % left_task)
            app.AsyncResult(left_task).forget()
            logger.warning("Deleting %s" % right_task)
            app.AsyncResult(right_task).forget()

            return result

@app.task(bind=True)
def work(self, trace_set_path, conf):
    '''
    Actions to be performed by workers on the trace set given in trace_set_path.
    '''
    logger.info("Node performing %s on trace set '%s'" % (str(conf.actions), trace_set_path))

    # Get trace name from path
    trace_set_name = basename(trace_set_path)

    # Load trace
    trace_set = emio.get_trace_set(trace_set_path, conf.inform)
    if trace_set is None:
        logger.warning("Skipping trace set %s" % trace_set_path)
        return None

    # Perform actions
    for action in conf.actions:
        if action in ops:
            ops[action](trace_set, conf=conf)
        else:
            logger.warning("Ignoring unknown action '%s'." % action)

    # TODO build this from within the ops themselves by passing as ref!
    result = EMResult(task_id=self.request.id)
    result.data['correlations'] = trace_set.correlations
    return result
