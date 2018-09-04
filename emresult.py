class EMResult():
    def __init__(self, data=None, task_id=None):
        self._task_id = task_id
        self._data = {'distances': None, 'correlations': None, 'trace_sets': [], 'state': None, 'predictions': [], 'labels': []}
        if not data is None:
            self._data.update(data)

    @property
    def task_id(self):
        return self._task_id

    def _get_correlations(self):
        return self._data['correlations']

    def _set_correlations(self, correlations):
        self._data['correlations'] = correlations

    def _get_distances(self):
        return self._data['distances']

    def _set_distances(self, distances):
        self._data['distances'] = distances

    def _get_trace_sets(self):
        return self._data['trace_sets']

    def _set_trace_sets(self, trace_sets):
        self._data['trace_sets'] = trace_sets

    correlations = property(_get_correlations, _set_correlations)
    distances = property(_get_distances, _set_distances)
    trace_sets = property(_get_trace_sets, _set_trace_sets)
