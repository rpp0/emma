#!/usr/bin/env python

"""
Script to perform PCA on a given dataset.
"""

import os.path
import numpy as np
import argparse
import pickle
from emma.io.io import get_dataset, get_trace_set
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="Name of dataset to apply PCA to. Must be in datasets.conf.")
parser.add_argument("--save-dataset", default=False, action='store_true', help="Write PCA-transformed dataset to folder.")
parser.add_argument("--limit", type=int, default=0, help="Limit number of trace sets (0=infinite)")
args = parser.parse_args()

dataset_name = args.dataset_name
dataset_name_pca = dataset_name + "-pca"

# Gather signals
dataset = get_dataset(dataset_name, remote=False)
all_signals = []
for count, trace_set_path in enumerate(dataset.trace_set_paths):
    print("\rGathering signal %d           " % count)
    trace_set_path = os.path.join(dataset.root, trace_set_path)
    trace_set = get_trace_set(trace_set_path, "cw", remote=False)
    for trace in trace_set.traces:
        all_signals.append(trace.signal)
    if count+1 == args.limit:
        break
all_signals = np.array(all_signals)

# Do PCA
print("Performing PCA")
pca = PCA()
pca.fit(all_signals)

# Save PCA model
with open("pca-components-%s.p" % dataset_name, 'wb') as f:
    pickle.dump(pca, f)
    print("Dumped PCA model to pca-components-%s.p" % dataset_name)

if args.save_dataset:
    # Transform dataset
    result = pca.transform(all_signals)

    # Make dest dir
    save_path = os.path.join(dataset.root, dataset_name_pca)
    os.makedirs(save_path, exist_ok=True)

    # Transform and store traces
    for trace_set_path in dataset.trace_set_paths:
        trace_set_path = os.path.join(dataset.root, trace_set_path)
        trace_set = get_trace_set(trace_set_path, "cw", remote=False)
        for trace in trace_set.traces:
            trace.signal = pca.transform([trace.signal])[0]
        trace_set.save(save_path, dry=False)
