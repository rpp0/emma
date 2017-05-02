import numpy as np
import sys
import matplotlib.pyplot as plt
import emio
from emma_worker import app
from dsp import *
from functools import wraps
from os.path import join, basename
from namedtuples import Window
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)  # Logger
ops = {}  # Op registry

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

    for trace in trace_set:
        aligned_trace = align(trace, reference)
        if not aligned_trace is None:
            aligned_trace_set.append(aligned_trace)

    return np.array(aligned_trace_set)

@op('filter')
def filter_trace_set(trace_set, conf=None):
    '''
    Apply a Butterworth filter to the traces.
    '''
    filtered_trace_set = []

    for trace in trace_set:
        filtered_trace = butter_filter(trace)
        filtered_trace_set.append(filtered_trace)

    return np.array(filtered_trace_set)

@op('save')
def save_trace_set(trace_set, conf):
    if conf.outform == 'cw':
        # Save back to output file
        np.save(join(output_path, trace_set_name), trace_set)

        # Update the corresponding config file
        emio.update_cw_config(output_path, trace_set, {"numPoints": len(conf.reference_trace)})
    elif conf.outform == 'sigmf':
        count = 1
        for trace in trace_set:
            trace.tofile(join(output_path_gnuradio, "%s-%d.rf32_le" % (trace_set.rpartition('_')[0], count)))
            count += 1
    else:
        print("Unknown format: %s" % conf.outform)
        exit(1)

    return trace_set

@op('plot')
def plot_trace_set(trace_set, conf=None):
    for trace in trace_set:
        plt.plot(range(0, len(trace)), trace)
    plt.show()

    return trace_set

@app.task
def work(trace_set_paths, conf):
    logger.info("Node performing %s on trace set of length %d" % (str(conf.actions), len(trace_set_paths)))

    # Perform actions on the sample sets
    for trace_set_path in trace_set_paths:
        # Get trace name from path
        trace_set_name = basename(trace_set_path)

        # Load trace
        trace_set = emio.get_trace_set(trace_set_path, conf.inform)

        # Perform actions
        for action in conf.actions:
            if action in ops:
                trace_set = ops[action](trace_set, conf=conf)
                #trace_set = filter_trace_set(trace_set)
            else:
                logger.warning("Ignoring unknown action '%s'." % action)
    return "Finished"
