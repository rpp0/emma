#!/usr/bin/python3

"""
Script to get mean and standard deviation of a given dataset.
"""

import argparse
from emma.io.io import get_trace_set, get_dataset
import numpy as np
from prettytable import PrettyTable
from os.path import join

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA): Dataset statistics tool', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', type=str, help='Identifier of dataset to use', nargs='+')
    args, unknown = parser.parse_known_args()

    table = PrettyTable(['Dataset', 'Number of samples', 'Mean of samples', 'Standard deviation of samples'])
    for dataset_name in args.dataset:
        dataset = get_dataset(dataset_name, remote=False)
        print("Dataset: %s\nFormat: %s" % (dataset.id, dataset.format))
        mean_sum = 0.0
        std_sum = 0.0
        n = 0

        # Calculate mean
        for trace_set_path in dataset.trace_set_paths:
            trace_set = get_trace_set(join(dataset.root, trace_set_path), dataset.format, ignore_malformed=False, remote=False)
            for trace in trace_set.traces:
                mean_sum += np.sum(trace.signal)
                n += len(trace.signal)

        mean = mean_sum / n

        # Calculate std dev
        for trace_set_path in dataset.trace_set_paths:
            trace_set = get_trace_set(join(dataset.root, trace_set_path), dataset.format, ignore_malformed=False, remote=False)
            for trace in trace_set.traces:
                std_sum += np.sum(np.square(trace.signal - mean))

        std = np.sqrt(std_sum / (n - 1))
        table.add_row([dataset_name, n, mean, std])
    print(table)
