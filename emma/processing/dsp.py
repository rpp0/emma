import numpy as np
import matplotlib.pyplot as plt
from emma.utils.debug import DEBUG
from scipy import signal


def normalize(trace):
    """
    Z-score normalize trace
    """
    mean = np.mean(trace)
    std = np.std(trace)
    if std == 0:
        raise ValueError
    return (trace - mean) / std


def normalize_p2p(trace):
    return (trace - trace.min(0)) / trace.ptp(0)


def ifreq(signal):
    instantaneous_phase = np.unwrap(np.angle(signal))
    instantaneous_frequency = np.diff(instantaneous_phase)
    return instantaneous_frequency


def butter_filter(trace, order=1, cutoff=0.01, filter_type='low', fs=None):
    """
    Apply butter filter to trace
    """
    b, a = signal.butter(order, cutoff, btype=filter_type, fs=fs)
    trace_filtered = signal.filtfilt(b, a, trace)
    return trace_filtered


def align(trace, reference, cutoff=0.01, order=1, prefilter=False):
    """
    Determine their offset using cross-correlation. This offset is then used to
    align the original signals.
    """
    # Preprocess
    try:
        trace = np.array(trace)
        reference = np.array(reference)
        if prefilter:
            processed_trace = butter_filter(trace, order=order, cutoff=cutoff)
            processed_reference = butter_filter(reference, order=order, cutoff=cutoff)
            processed_trace = normalize_p2p(processed_trace)  # normalize() seems to work pretty well too
            processed_reference = normalize_p2p(processed_reference)
        else:
            processed_trace = normalize_p2p(trace)  # normalize() seems to work pretty well too
            processed_reference = normalize_p2p(reference)
    except ValueError:  # Something is wrong with the signal
        return None

    # Correlated processed traces to determine lag
    result = signal.correlate(processed_trace, processed_reference, mode='valid')
    lag = np.argmax(result)

    # Align the original trace based on this calculation
    aligned_trace = trace[lag:]

    # Vertical align as well TODO add as new separate op?
    #bias = np.mean(aligned_trace)
    #aligned_trace -= bias
    #DEBUG = True

    if DEBUG:
        plt.plot(range(0, len(processed_reference)), processed_reference, label="Normalized reference")
        plt.plot(range(0, len(processed_trace)), processed_trace, label="Normalized trace")
        plt.plot(range(0, len(result)), result, label="Correlation")
        #plt.plot(range(0, len(aligned_trace)), aligned_trace, label="Aligned trace")
        plt.legend()
        plt.show()

    return aligned_trace
