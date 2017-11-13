class EMResult():
    def __init__(self, data=None, task_id=None):
        self._task_id = task_id
        self._data = {'correlations': None}
        if not data is None:
            self._data.update(data)

    @property
    def task_id(self):
        return self._task_id

    def _get_correlations(self):
        return self._data['correlations']

    def _set_correlations(self, correlations):
        self._data['correlations'] = correlations

    correlations = property(_get_correlations, _set_correlations)
