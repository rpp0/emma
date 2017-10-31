import numpy as np

class Correlation():
    def __init__(self, init_value=0.0):
        super().__init__()
        self._corr = init_value
        self._sumx = 0.0
        self._sumy = 0.0
        self._sumxy = 0.0
        self._sumxsq = 0.0
        self._sumysq = 0.0
        self._n = 0

    def __float__(self):
        return float(self._corr)

    def __repr__(self):
        return str(self._corr)

    def __str__(self):
        return str(self._corr)

    def __sub__(self, other):
        return self._corr - float(other)

    def __rsub__(self, other):
        return self._corr - float(other)

    def __lt__(self, other):
        return self._corr < float(other)

    def __le__(self, other):
        return self._corr <= float(other)

    def __eq__(self, other):
        return self._corr == float(other)

    def __ne__(self, other):
        return self._corr != float(other)

    def __gt__(self, other):
        return self._corr > float(other)

    def __ge__(self, other):
        return self._corr >= float(other)

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
        self._update_corr()

    def merge(self, correlation):
        self._sumx += correlation._sumx
        self._sumy += correlation._sumy
        self._sumxy += correlation._sumxy
        self._sumxsq += correlation._sumxsq
        self._sumysq += correlation._sumysq
        self._n += correlation._n
        self._update_corr()

    @classmethod
    def init(cls, shape):
        result = np.empty(shape, dtype=cls)
        result = np.reshape(result, -1)
        for i in range(0, len(result)):
            result[i] = Correlation(0.0)
        result = np.reshape(result, shape)
        return result
