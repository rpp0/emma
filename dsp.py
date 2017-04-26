from debug import DEBUG
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def normalize(trace):
    mean = np.mean(trace)
    std = np.std(trace)
    if std == 0:
        raise ValueError
    return (trace - mean) / std

def butter_filter(trace, order=4, cutoff=0.004, filter_type='low'):
    b, a = signal.butter(order, cutoff, filter_type)
    trace_filtered = signal.filtfilt(b, a, trace)
    return trace_filtered

def align(trace, reference):
    # Preprocess
    try:
        processed_trace = normalize(butter_filter(trace))
        processed_reference = normalize(butter_filter(reference))
    except ValueError: # Something is wrong with the signal
        return None

    # Correlated processed traces to determine lag
    result = signal.correlate(processed_trace, processed_reference, mode='valid') / len(processed_reference)
    lag = np.argmax(result)

    # Align the original trace based on this calculation
    aligned_trace = trace[lag:lag+len(processed_reference)]

    if DEBUG:
        plt.plot(range(0, len(processed_reference)), processed_reference)
        plt.plot(range(0, len(aligned_trace)), normalize(butter_filter(aligned_trace)))
        plt.gca().set_ylim([-2.0,2.0])
        plt.show()

    return aligned_trace
