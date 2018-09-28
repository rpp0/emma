#!/usr/bin/env python

import os.path
import numpy as np
import argparse
from emio import get_dataset, get_trace_set
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", type=str, help="Name of dataset to apply PCA to. Must be in datasets.conf.")
args = parser.parse_args()

dataset_name = args.dataset_name
dataset_name_pca = dataset_name + "-pca"

# Gather signals
dataset = get_dataset(dataset_name, remote=False)
all_signals = []
for trace_set_path in dataset.trace_set_paths:
    trace_set_path = os.path.join(dataset.root, trace_set_path)
    trace_set = get_trace_set(trace_set_path, "cw", remote=False)
    for trace in trace_set.traces:
        all_signals.append(trace.signal)
all_signals = np.array(all_signals)

# Do PCA
pca = PCA(n_components=16)
pca.fit(all_signals)
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
