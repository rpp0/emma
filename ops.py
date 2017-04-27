import numpy as np
from emma_worker import app
from dsp import *
from functools import wraps

ops = {}

def op(name):
    def decorator(func):
        ops[name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@op('align')
@app.task
def align_traces(traces, reference):
    aligned_traces = []

    for trace in traces:
        aligned_trace = align(trace, reference)
        if not aligned_trace is None:
            aligned_traces.append(aligned_trace)

    return np.array(aligned_traces)
