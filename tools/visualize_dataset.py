#!/usr/bin/python

import argparse
import emma.io.io as emio
import numpy as np
import matplotlib.pyplot as plt
from emma.utils.utils import numpy_to_hex
from os.path import join


def plot_signal(signal):
    plt.plot(signal)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA): Dataset visualization tool', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', type=str, help='Identifier of dataset to use', nargs='+')
    parser.add_argument('--mean', default=False, action='store_true', help='Plot mean of signals')
    args, unknown = parser.parse_known_args()

    for dataset_name in args.dataset:
        dataset = emio.get_dataset(dataset_name, remote=False)
        print("Dataset: %s\nFormat: %s" % (dataset.id, dataset.format))

        # Calculate mean
        for trace_set_path in dataset.trace_set_paths:
            trace_set = emio.get_trace_set(join(dataset.root, trace_set_path), dataset.format, ignore_malformed=False, remote=False)

            if args.mean:
                signals = []
                for trace in trace_set.traces:
                    signals.append(trace.signal)
                signals = np.array(signals)
                plot_signal(np.mean(signals, axis=0))
            else:
                for trace in trace_set.traces:
                    print("Key: %s" % numpy_to_hex(trace.key))
                    print("Plaintext: %s" % numpy_to_hex(trace.plaintext))
                    print("---------------------------------------")
                    plot_signal(trace.signal)
