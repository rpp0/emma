#!/usr/bin/python3
# get_dataset_statistics.py

import argparse
import emio
import numpy as np
from prettytable import PrettyTable
from os.path import join

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA): Dataset statistics tool', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', type=str, help='Identifier of dataset to use', nargs='+')
    args, unknown = parser.parse_known_args()

    table = PrettyTable(['Dataset', 'Num items', 'Mean', 'Std'])
    for dataset_name in args.dataset:
        dataset = emio.get_dataset(dataset_name)
        print("Dataset: %s\nFormat: %s" % (dataset.id, dataset.format))
        mean_sum = 0.0
        std_sum = 0.0
        n = 0

        # Calculate mean
        for trace_set_path in dataset.trace_set_paths:
            trace_set = emio.get_trace_set(join(dataset.prefix, trace_set_path), dataset.format, ignore_malformed=False, remote=False)
            for trace in trace_set.traces:
                mean_sum += np.sum(trace.signal)
                n += len(trace.signal)

        mean = mean_sum / n

        # Calculate std dev
        for trace_set_path in dataset.trace_set_paths:
            trace_set = emio.get_trace_set(join(dataset.prefix, trace_set_path), dataset.format, ignore_malformed=False, remote=False)
            for trace in trace_set.traces:
                std_sum += np.sum(np.square(trace.signal - mean))

        std = np.sqrt(std_sum / (n - 1))
        table.add_row([dataset_name, n, mean, std])
    print(table)
