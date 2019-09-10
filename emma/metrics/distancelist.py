import numpy as np
import pickle


class DistanceList:
    def __init__(self, shape):
        self._dxy = np.zeros(shape, dtype=float)
        self._n = np.zeros(shape, dtype=float)  # TODO: Since this should be the same for every point, we can maybe use a single point for it

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, tuple) or isinstance(key, list):
            return self._dxy[key]
        if isinstance(key, slice):
            raise NotImplementedError
        else:
            raise TypeError

    def update(self, key, true, pred):
        self._dxy[key] += np.sum(np.abs(true - pred))
        self._n[key] += len(pred)

    def merge(self, other):
        if isinstance(other, DistanceList):
            self._dxy += other._dxy
            self._n += other._n
        else:
            raise TypeError

    def save(self):
        pickle.dump(self, open("/tmp/distancelist.p", "wb"))
