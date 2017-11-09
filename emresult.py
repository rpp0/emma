from collections import defaultdict

class EMResult():
    def __init__(self, data=None, task_id=None):
        self._task_id = task_id
        self._data = defaultdict(self.default)
        if not data is None:
            self._data.update(data)

    def default(self):
        return None

    @property
    def task_id(self):
        return self._task_id

    @property
    def data(self):
        return self._data
