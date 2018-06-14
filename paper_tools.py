#!/usr/bin/python

import argparse
import os
import pickle
import ai
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def download_files(remote_file_paths, destination):
    dest_path = os.path.abspath(destination)
    print("Creating directory %s" % dest_path)
    os.makedirs(dest_path, exist_ok=True)
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

def normalize(values):
    return (values - np.min(values)) / np.ptp(values)

def is_remote(path):
    return ':' in path

class FigureGenerator():
    def __init__(self, input_path, model_id, model_suffix="last"):
        if not 'models' in input_path:
            raise Exception

        self.is_remote = is_remote(input_path)
        self.model_id = model_id
        self.model_suffix = model_suffix

        if self.is_remote:
            self.remote_path = input_path
            self.input_path = os.path.join("./models/", self.remote_path.rpartition('models')[2][1:])
        else:
            self.input_path = input_path

        self.input_path = os.path.abspath(self.input_path)
        self.output_path = os.path.abspath(os.path.join("./paper_data", self.input_path.rpartition('models')[2][1:]))

    def get_remote_model(self):
        remote_model_path = os.path.join(self.remote_path, self.model_id + "-" + self.model_suffix + ".h5")
        remote_model_history_path = os.path.join(self.remote_path, self.model_id + "-history.p")
        remote_model_ranks_path = os.path.join(self.remote_path, self.model_id + "-t-ranks.p")
        local_model_path =  os.path.join(self.input_path, "%s-%s.h5" % (self.model_id, self.model_suffix))
        local_model_history_path =  os.path.join(self.input_path, "%s-history.p" % self.model_id)
        local_model_ranks_path =  os.path.join(self.input_path, "%s-t-ranks.p" % self.model_id)

        # Check if model already exists
        if os.path.exists(local_model_path):
            # Is there a newer model?
            local_model_hash = get_hash(local_model_path, is_remote=False)
            remote_model_hash = get_hash(remote_model_path, is_remote=True)
            if local_model_hash != remote_model_hash:
                download_files([remote_model_path], self.input_path)

            # Is there a newer history?
            local_model_history_hash = get_hash(local_model_history_path, is_remote=False)
            remote_model_history_hash = get_hash(remote_model_history_path, is_remote=True)
            if local_model_history_hash != remote_model_history_hash:
                download_files([remote_model_history_path], self.input_path)

            # Is there a newer ranks file?
            local_model_ranks_hash = get_hash(local_model_ranks_path, is_remote=False)
            remote_model_ranks_hash = get_hash(remote_model_ranks_path, is_remote=True)
            if local_model_ranks_hash != remote_model_ranks_hash:
                download_files([remote_model_ranks_path], self.input_path)
        else:
            # Download model
            download_files([remote_model_path,remote_model_history_path,remote_model_ranks_path], self.input_path)

    def generate_stats(self):
        if self.is_remote:
            self.get_remote_model()

        # Make directory for resulting data and graphs
        os.makedirs(self.output_path, exist_ok=True)

        # History graphs
        try:
            history = pickle.load(open(os.path.join(self.input_path, self.model_id + "-history.p"), "rb"))
            self.generate_history_graphs(history)
        except FileNotFoundError:
            print("File not found; skipping history graphs")

        # Rank graphs
        try:
            ranks_confidences = pickle.load(open(os.path.join(self.input_path, self.model_id + "-t-ranks.p"),   "rb"))
            self.generate_ranks_graphs(ranks_confidences)
        except FileNotFoundError:
            print("File not found; skipping rank graphs")

        # Model graphs
        try:
            model = ai.AI(name=self.model_id, suffix=self.model_suffix, path=self.input_path)
            model.load()
            self.generate_model_graphs(model.model)
        except OSError:
            print("File not found; skipping model graphs")

    def generate_history_graphs(self, history):
        for key, values in history.items():
            print("Generating %s graph" % key)
            fig = plt.figure()
            plt.plot(np.arange(len(values)), values)
            fig.savefig(os.path.join(self.output_path, "%s-%s-%s.pdf" % (self.model_id, self.model_suffix, key)), bbox_inches='tight')

    def generate_ranks_graphs(self, ranks_confidences):
        ranks = ranks_confidences['ranks']
        confidences = ranks_confidences['confidences']
        step = ranks_confidences['rank_trace_step']
        num_validation_traces = ranks_confidences['num_validation_traces']
        conf = ranks_confidences['conf']
        t = ranks_confidences['folds']

        x = range(0, num_validation_traces + step, step)
        ranks_y = np.array([256] + list(np.mean(ranks, axis=0)), dtype=np.float32)
        confidences_y = np.array([0] + list(np.mean(confidences, axis=0)), dtype=np.float32)
        fig, ax1 = plt.subplots()
        rank_series, = ax1.plot(x, ranks_y, color='tab:blue', label="mean rank")
        ax1.set_xlabel('validation set size')
        ax1.set_ylabel('mean rank')
        ax1.set_ylim([0,256])
        ax2 = ax1.twinx()
        ax2.set_ylabel('confidence')
        confidence_series, = ax2.plot(x, confidences_y, color='tab:orange', label="confidence")

        legend = plt.legend(handles=[rank_series, confidence_series], loc=9)
        plt.gca().add_artist(legend)
        plt.title("%d-fold cross-validation of dataset '%s'" % (t, conf.dataset_id))

        fig.savefig(os.path.join(self.output_path, "%s-%s-tfold.pdf" % (self.model_id, self.model_suffix)), bbox_inches='tight')

    def generate_model_graphs(self, model):
        print(model.get_weights())
        print(model.summary())

class ModelFinder():
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.is_remote = is_remote(models_dir)
        self.keywords = ('-t-ranks', '-last', '-history')

    def find_models(self):
        if self.is_remote:
            host, _, path = self.models_dir.rpartition(':')
            python_command = "python -c \"import os; import json; print(json.dumps(list(os.walk('%s'))))\"" % path

            command = ["/usr/bin/ssh", host, python_command]
            walk_process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=False)
            stdout, stderr = walk_process.communicate()
            walk_process.wait()

            subwalks = json.loads(stdout.decode("utf-8"))
        else:
            walk_generator = os.walk(self.models_dir)
            model_directories = next(walk_generator)[1]
            subwalks = list(walk_generator)

        model_locations = []
        for subwalk in subwalks:
            subdirectory = subwalk[0]
            if self.is_remote:
                subdirectory = host + ':' + subdirectory
            files = subwalk[2]
            model_names = set()
            for file in files:
                for keyword in self.keywords:
                    if keyword in file:
                        model_names.add(file.rpartition(keyword)[0])
            if len(model_names) == 1:
                model_locations.append((subdirectory, model_names.pop()))

        return model_locations

if __name__ == "__main__":
    """
    Tools for creating the paper. Possible commands:
        * run <suite_name>: Run a suite of experiments
        * stats <model_id>: Generate statistics and graphs for model_id
    """
    parser = argparse.ArgumentParser(description='Tools for CEMA correlation optimization paper.', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', type=str, help='Command to execute')
    parser.add_argument('parameters', type=str, help='Parameters for the command', nargs='+')
    args, unknown = parser.parse_known_args()

    if args.command == 'stats':
        if len(args.parameters) >= 2:
            f = FigureGenerator(args.parameters[0], args.parameters[1])
            f.generate_stats()
        else:
            print("Not enough parameters")
    elif args.command == 'autostats':
        if len(args.parameters) >= 1:
            mf = ModelFinder(args.parameters[0])
            for model_location in mf.find_models():
                f = FigureGenerator(model_location[0], model_location[1])
                f.generate_stats()
        else:
            print("Not enough parameters")
    else:
        print("Unknown command %s" % args.command)
