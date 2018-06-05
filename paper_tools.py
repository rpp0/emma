#!/usr/bin/python

import argparse
import os
import pickle
import ai
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def download_files(remote_file_paths, destination):
    dest_path = os.path.abspath(destination)
    for source_path in remote_file_paths:
        print("Downloading %s..." % source_path)
        command = ["/usr/bin/scp", source_path, dest_path]
        scp_process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
        stdout, stderr = scp_process.communicate()
        scp_process.wait()

def get_hash(file_path, is_remote=False):
    if is_remote:
        host, _, path = file_path.rpartition(':')
        command = ["/usr/bin/ssh", host, "/usr/bin/md5sum " + path]
    else:
        command = ["/usr/bin/md5sum", file_path]

    stat_process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
    stdout, stderr = stat_process.communicate()
    stat_process.wait()
    hash = stdout[0:32]

    return hash

def get_remote_model(model_id, suffix, remote):
    # Check if model already exists
    remote_model_path = os.path.join(remote, "models", model_id + "-" + suffix + ".h5")
    remote_model_history_path = os.path.join(remote, "models", model_id + "-history.p")
    local_model_path =  os.path.abspath("./models/%s-%s.h5" % (model_id, suffix))
    local_model_history_path =  os.path.abspath("./models/%s-history.p" % model_id)

    if os.path.exists(local_model_path):
        # Is there a newer model?
        local_model_hash = get_hash(local_model_path, is_remote=False)
        remote_model_hash = get_hash(remote_model_path, is_remote=True)
        if local_model_hash != remote_model_hash:
            download_files([remote_model_path], "./models/")

        # Is there a newer history?
        local_model_history_hash = get_hash(local_model_history_path, is_remote=False)
        remote_model_history_hash = get_hash(remote_model_history_path, is_remote=True)
        if local_model_history_hash != remote_model_history_hash:
            download_files([remote_model_history_path], "./models/")
    else:
        # Download model
        download_files([remote_model_path,remote_model_history_path], "./models/")

def generate_history_graphs(model_id, suffix, history):
    for key, values in history.items():
        print("Generating %s graph" % key)
        fig = plt.figure()
        plt.plot(np.arange(len(values)), values)
        fig.savefig("./paper_data/%s-%s-%s.pdf" % (model_id, suffix, key), bbox_inches='tight')

def generate_model_graphs(model):
    print(model.get_weights())
    print(model.summary())

def generate_stats(model_id, suffix="last", remote=None):
    if os.path.exists("./models"):
        if not remote is None:
            get_remote_model(model_id, suffix, remote)

        # Make directory for resulting data and graphs
        os.makedirs("./paper_data/", exist_ok=True)

        # History graphs
        history = pickle.load(open(os.path.join("./models", model_id + "-history.p"), "rb"))
        generate_history_graphs(model_id, suffix, history)

        # Model graphs
        model = ai.AI(name=model_id, suffix=suffix)
        model.load()
        generate_model_graphs(model.model)
    else:
        print("No models/ directory found, exiting.")
        exit(1)

if __name__ == "__main__":
    """
    Tools for creating the paper. Possible commands:
        * run <suite_name>: Run a suite of experiments
        * stats <model_id>: Generate statistics and graphs for model_id
    """
    parser = argparse.ArgumentParser(description='Tools for CEMA correlation optimization paper.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', type=str, help='Command to execute')
    parser.add_argument('parameters', type=str, help='Parameters for the command', nargs='+')
    parser.add_argument('--remote', type=str, default=None, help='Remote location to fetch model from.')
    args, unknown = parser.parse_known_args()

    if args.command == 'stats':
        generate_stats(args.parameters[0], remote=args.remote)
