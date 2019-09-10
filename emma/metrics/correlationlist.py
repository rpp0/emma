import numpy as np
import pickle


class CorrelationList:
    def __init__(self, shape):
        self._sumx = np.zeros(shape, dtype=float)
        self._sumy = np.zeros(shape, dtype=float)
        self._sumxy = np.zeros(shape, dtype=float)
        self._sumxsq = np.zeros(shape, dtype=float)
        self._sumysq = np.zeros(shape, dtype=float)
        self._n = np.zeros(shape, dtype=float)  # TODO: Since this should be the same for every point, we can maybe use a single point for it

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, tuple) or isinstance(key, list):
            num = self._sumxy[key] - (self._sumx[key] * self._sumy[key] / self._n[key])
            denom1 = self._sumxsq[key] - (self._sumx[key]**2 / self._n[key])
            denom2 = self._sumysq[key] - (self._sumy[key]**2 / self._n[key])
            corr = num / np.maximum(np.sqrt(denom1 * denom2), 1e-15)
            return corr
        if isinstance(key, slice):
            raise NotImplementedError
        else:
            raise TypeError

    def update(self, key, x, y):
        self._sumx[key] += np.sum(x)
        self._sumy[key] += np.sum(y)
        self._sumxy[key] += np.dot(x, y)
        self._sumxsq[key] += np.sum(np.square(x))
        self._sumysq[key] += np.sum(np.square(y))
        self._n[key] += len(x)

    def merge(self, correlation_array):
        if isinstance(correlation_array, CorrelationList):
            self._sumx += correlation_array._sumx
            self._sumy += correlation_array._sumy
            self._sumxy += correlation_array._sumxy
            self._sumxsq += correlation_array._sumxsq
            self._sumysq += correlation_array._sumysq
            self._n += correlation_array._n
        else:
            raise TypeError

    def save(self):
        pickle.dump(self, open("/tmp/correlations.p", "wb"))
