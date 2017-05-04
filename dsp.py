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

def butter_filter(trace, order=4, cutoff=0.004, filter_type='low'):
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

class Correlation(float):
    def __init__(self, init_value):
        super().__init__()
        self._corr = init_value
        self._sumx = 0.0
        self._sumy = 0.0
        self._sumxy = 0.0
        self._sumxsq = 0.0
        self._sumysq = 0.0
        self._n = 0

    def __float__(self):
        self.request()  # TODO: temporary
        return self._corr

    def __repr__(self):
        self.request()  # TODO: temporary
        return str(self._corr)

    # TODO is there a way to access a float's internal value?
    def __lt__(self, other):
        return self._corr < other._corr

    def __le__(self, other):
        return self._corr <= other._corr

    def __eq__(self, other):
        return self._corr == other._corr

    def __ne__(self, other):
        return self._corr != other._corr

    def __gt__(self, other):
        return self._corr > other._corr

    def __ge__(self, other):
        return self._corr >= other._corr

    def _update_corr(self):
        if self._n != 0:
            num = self._sumxy - (self._sumx * self._sumy / self._n)
            denom1 = self._sumxsq - (self._sumx**2 / self._n)
            denom2 = self._sumysq - (self._sumy**2 / self._n)
            self._corr = num / np.sqrt(denom1 * denom2)

    def update(self, x, y):
        self._sumx += np.sum(x)
        self._sumy += np.sum(y)
        self._sumxy += np.sum(np.multiply(x, y))
        self._sumxsq += np.sum(np.square(x))
        self._sumysq += np.sum(np.square(y))
        self._n += len(x)

    def merge(self, correlation):
        self._sumx += correlation._sumx
        self._sumy += correlation._sumy
        self._sumxy += correlation._sumxy
        self._sumxsq += correlation._sumxsq
        self._sumysq += correlation._sumysq
        self._n += correlation._n

    def request(self):  # Performance TODO describe why
        self._update_corr()

    @classmethod
    def init(cls, shape):
        result = np.empty(shape, dtype=cls)
        result = np.reshape(result, -1)
        for i in range(0, len(result)):
            result[i] = Correlation(0.0)
        result = np.reshape(result, shape)
        return result
