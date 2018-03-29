# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from debug import DEBUG
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def normalize(trace):
    '''
    Z-score normalize trace
    '''
    mean = np.mean(trace)
    std = np.std(trace)
    if std == 0:
        raise ValueError
    return (trace - mean) / std

def butter_filter(trace, order=1, cutoff=0.01, filter_type='low'):
    '''
    Apply butter filter to trace
    '''
    b, a = signal.butter(order, cutoff, filter_type)
    trace_filtered = signal.filtfilt(b, a, trace)
    return trace_filtered

def align(trace, reference):
    '''
    Use a butterworth filter to filter the original and reference trace, and
    determine their offset using cross-correlation. This offset is then used to
    align the original signals.
    '''
    # Preprocess
    try:
        processed_trace = normalize(trace)
        processed_reference = normalize(reference)
    except ValueError: # Something is wrong with the signal
        return None

    # Correlated processed traces to determine lag
    result = signal.correlate(processed_trace, processed_reference, mode='valid')
    lag = np.argmax(result)

    # Align the original trace based on this calculation
    aligned_trace = trace[lag:]

    # Vertical align as well TODO add as new separate op?
    #bias = np.mean(aligned_trace)
    #aligned_trace -= bias

    if DEBUG:
        plt.plot(range(0, len(processed_reference)), processed_reference, label="Reference")
        plt.plot(range(0, len(aligned_trace)), aligned_trace, label="Trace")
        plt.show()

    return aligned_trace
