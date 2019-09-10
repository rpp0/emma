import argparse
from collections import namedtuple


class EMResult(argparse.Namespace):
    def __init__(self, **kwargs):
        kwargs.update({
            'trace_sets': [],
            'predictions': [],
            'labels': [],
            'logprobs': [],
        })
        super().__init__(**kwargs)

    def __getattr__(self, item):  # If the item doesn't exist, return None
        return None

    def __getstate__(self):  # Prevent pickle from using __getattr__ and calling None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


SalvisResult = namedtuple('SalvisResult', ['examples_batch', 'gradients'])
