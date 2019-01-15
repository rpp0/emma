#!/usr/bin/python

import argparse
import os
import pickle
import ai
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

from argparse import Namespace
from leakagemodels import LeakageModelType
from aiinputs import AIInputType
from action import Action


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

    stat_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    stdout, stderr = stat_process.communicate()
    stat_process.wait()
    hash = stdout[0:32]

    if b'No such file' in stderr:
        print("Skipping %s (no such file)" % file_path)

    return hash

def normalize(values):
    return (values - np.min(values)) / np.ptp(values)

def is_remote(path):
    return ':' in path

class RankConfidencePlot():
    def __init__(self, x_label='number of traces', y1_label='mean rank', y2_label='mean confidence', includey2=True):
        self.fig, self.ax1 = plt.subplots()
        self.ax1.set_xlabel(x_label)
        self.ax1.set_ylabel(y1_label)
        self.ax1.set_ylim([0,256])
        if includey2:
            self.ax2 = self.ax1.twinx()
            self.ax2.set_ylabel(y2_label)
        #self.ax1.spines['top'].set_visible(False)
        #self.ax2.spines['top'].set_visible(False)
        self.handles = []

    def add_series(self, x, ranks_y, confidences_y, rank_color='tab:blue', confidence_color='tab:blue', rank_style='-', confidence_style=':', rank_label="mean rank", confidence_label="mean confidence"):
        rank_series, = self.ax1.plot(x, ranks_y, color=rank_color, linestyle=rank_style, label=rank_label)
        confidence_series, = self.ax2.plot(x, confidences_y, color=confidence_color, linestyle=confidence_style, label=confidence_label, alpha=0.5)
        self.handles.extend([rank_series, confidence_series])

    def add_rank_series(self, x, ranks_y, rank_color='tab:blue', rank_style='-', rank_label="mean rank"):
        rank_series, = self.ax1.plot(x, ranks_y, color=rank_color, linestyle=rank_style, label=rank_label)
        self.handles.append(rank_series)

    def set_title(self, title):
        plt.title(title)

    def save(self, path):
        legend = plt.legend(handles=self.handles, loc=9, fontsize=8)
        plt.gca().add_artist(legend)
        self.fig.savefig(path, bbox_inches='tight')

def get_series_from_tfold_blob(tfold_blob):
    ranks = tfold_blob['ranks']
    confidences = tfold_blob['confidences']
    step = tfold_blob['rank_trace_step']
    num_validation_traces = tfold_blob['num_validation_traces']
    print("Number of validation traces: %d" % num_validation_traces)

    x = range(0, num_validation_traces + step, step)
    ranks_y = np.array([256] + list(np.mean(ranks, axis=0)), dtype=np.float32)
    confidences_y = np.array([0] + list(np.mean(confidences, axis=0)), dtype=np.float32)

    return x[0:len(ranks_y)], ranks_y, confidences_y


def insert_attribute_if_absent(instance, attr_name, default):
    dummy = getattr(instance, attr_name, None)
    if dummy is None:
        setattr(instance, attr_name, default)


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

    def get_remote_model(self, input_path, remote_path, model_id, model_suffix):
        remote_model_path = os.path.join(remote_path, model_id + "-" + model_suffix + ".h5")
        remote_model_history_path = os.path.join(remote_path, model_id + "-history.p")
        remote_model_ranks_path = os.path.join(remote_path, model_id + "-t-ranks.p")
        remote_model_testrank_path = os.path.join(remote_path, model_id + "-bestrank-testrank.p")
        local_model_path =  os.path.join(input_path, "%s-%s.h5" % (model_id, model_suffix))
        local_model_history_path =  os.path.join(input_path, "%s-history.p" % model_id)
        local_model_ranks_path =  os.path.join(input_path, "%s-t-ranks.p" % model_id)
        local_model_testrank_path =  os.path.join(input_path, "%s-bestrank-testrank.p" % model_id)

        # Check if model already exists
        if os.path.exists(input_path):
            # Is there a newer model?
            local_model_hash = get_hash(local_model_path, is_remote=False)
            remote_model_hash = get_hash(remote_model_path, is_remote=True)
            if local_model_hash != remote_model_hash:
                download_files([remote_model_path], input_path)

            # Is there a newer history?
            local_model_history_hash = get_hash(local_model_history_path, is_remote=False)
            remote_model_history_hash = get_hash(remote_model_history_path, is_remote=True)
            if local_model_history_hash != remote_model_history_hash:
                download_files([remote_model_history_path], input_path)

            # Is there a newer ranks file?
            local_model_ranks_hash = get_hash(local_model_ranks_path, is_remote=False)
            remote_model_ranks_hash = get_hash(remote_model_ranks_path, is_remote=True)
            if local_model_ranks_hash != remote_model_ranks_hash:
                download_files([remote_model_ranks_path], input_path)

            # Newer testrank file?
            local_model_testrank_hash = get_hash(local_model_testrank_path, is_remote=False)
            remote_model_testrank_hash = get_hash(remote_model_testrank_path, is_remote=True)
            if local_model_testrank_hash != remote_model_testrank_hash:
                download_files([remote_model_testrank_path], input_path)
        else:
            # Download model
            download_files([remote_model_path,remote_model_history_path,remote_model_ranks_path,remote_model_testrank_path], input_path)

    def generate_stats(self):
        tfold_blob = None

        if self.is_remote:
            self.get_remote_model(self.input_path, self.remote_path, self.model_id, self.model_suffix)

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
            tfold_blob = pickle.load(open(os.path.join(self.input_path, self.model_id + "-t-ranks.p"),   "rb"))
            self.generate_ranks_graphs(tfold_blob)
        except FileNotFoundError:
            print("File not found; skipping rank graphs")

        # Testrank graphs
        try:
            tfold_blob = pickle.load(open(os.path.join(self.input_path, self.model_id + "-bestrank-testrank.p"),   "rb"))
            self.generate_testrank_graphs(tfold_blob)
        except FileNotFoundError:
            print("File not found; skipping testrank graphs")

        # Model graphs
        try:
            if tfold_blob is not None and "conf" in tfold_blob:
                # TODO hack because some old blobs don't have use_bias
                insert_attribute_if_absent(tfold_blob["conf"], "use_bias", True)
                insert_attribute_if_absent(tfold_blob["conf"], "batch_norm", True)
                insert_attribute_if_absent(tfold_blob["conf"], "cnn", False)
                insert_attribute_if_absent(tfold_blob["conf"], "metric_freq", 10)
                insert_attribute_if_absent(tfold_blob["conf"], "regularizer", None)
                insert_attribute_if_absent(tfold_blob["conf"], "reglambda", 0.001)
                insert_attribute_if_absent(tfold_blob["conf"], "key_low", 2)
                insert_attribute_if_absent(tfold_blob["conf"], "key_high", 3)
                insert_attribute_if_absent(tfold_blob["conf"], "loss_type", "correlation")
                insert_attribute_if_absent(tfold_blob["conf"], "leakage_model", LeakageModelType.HAMMING_WEIGHT_SBOX)
                insert_attribute_if_absent(tfold_blob["conf"], "input_type", AIInputType.SIGNAL)
                insert_attribute_if_absent(tfold_blob["conf"], "n_hidden_nodes", 256)
                insert_attribute_if_absent(tfold_blob["conf"], "n_hidden_layers", 1)
                insert_attribute_if_absent(tfold_blob["conf"], "lr", 0.0001)
                insert_attribute_if_absent(tfold_blob["conf"], "activation", "leakyrelu")

                actions = []
                for action in tfold_blob["conf"].actions:
                    if isinstance(action, str):
                        actions.append(Action(action))
                    else:
                        actions.append(action)
                tfold_blob["conf"].actions = actions

                model = ai.AI(model_type=self.model_id, conf=tfold_blob["conf"])
                model.load()
                self.generate_model_graphs(model.model)
            else:
                print("No tfold blob containing conf. Skipping model graphs.")
        except OSError:
            print("File not found; skipping model graphs")

    def generate_history_graphs(self, history):
        for key, values in history.items():
            print("Generating %s graph" % key)
            fig = plt.figure()
            plt.plot(np.arange(len(values)), values)
            fig.savefig(os.path.join(self.output_path, "%s-%s-%s.pdf" % (self.model_id, self.model_suffix, key)), bbox_inches='tight')

    def generate_ranks_graphs(self, tfold_blob):
        conf = tfold_blob['conf']
        t = tfold_blob['folds']

        x, ranks_y, confidences_y = get_series_from_tfold_blob(tfold_blob)

        plot = RankConfidencePlot()
        #plot.set_title("%d-fold cross-validation of dataset '%s'" % (t, conf.dataset_id))
        plot.add_series(x, ranks_y, confidences_y, rank_label="rank", confidence_label="confidence")
        plot.save(os.path.join(self.output_path, "%s-%s-tfold.pdf" % (self.model_id, self.model_suffix)))

    def generate_testrank_graphs(self, tfold_blob):
        conf = tfold_blob['conf']
        t = tfold_blob['folds']

        tfold_blob['ranks'] = np.expand_dims(tfold_blob['ranks'], axis=0)
        tfold_blob['confidences'] = np.expand_dims(tfold_blob['confidences'], axis=0)
        x, ranks_y, confidences_y = get_series_from_tfold_blob(tfold_blob)

        plot = RankConfidencePlot(y1_label="rank", y2_label="confidence")
        #plot.set_title("Rank test of dataset '%s'" % conf.dataset_id)
        plot.add_series(x, ranks_y, confidences_y, rank_label="rank", confidence_label="confidence")
        plot.save(os.path.join(self.output_path, "%s-%s-testrank.pdf" % (self.model_id, self.model_suffix)))

    def generate_model_graphs(self, model):
        print(model.get_weights())
        print(model.summary())

def ascad_sort_name(name):
    if 'desync100' in name:
        return 2
    elif 'desync50' in name:
        return 1
    else:
        return 0

class CombinedFigureGenerator(FigureGenerator):
    def __init__(self, input_tuples, name="unknown", model_suffix="last"):
        self.input_tuples = input_tuples
        self.model_suffix = model_suffix
        self.output_path = os.path.abspath(os.path.join("./paper_data", "combined-" + name))

    def dump_text(self, name, x, ranks_y, confidences_y):
        '''
        Dump data to text. Useful for getting raw data for the paper.
        '''

        with open(os.path.join(self.output_path, "data-%s.txt" % name), "w") as f:
            min_rank = np.amin(ranks_y)
            min_rank_x = x[np.argmin(ranks_y)]
            max_rank = np.amax(ranks_y)
            max_rank_x = x[np.argmax(ranks_y)]
            min_conf = np.amin(confidences_y)
            min_conf_x = x[np.argmin(confidences_y)]
            max_conf = np.amax(confidences_y)
            max_conf_x = x[np.argmax(confidences_y)]
            f.write("Ranks:\n")
            f.write(str(list(zip(x, ranks_y))))
            f.write("\n\n")
            f.write("Confidences:\n")
            f.write(str(list(zip(x, confidences_y))))
            f.write("\n\n")
            f.write("Min rank: (%d, %d)\n" % (min_rank_x, min_rank))
            f.write("Max rank: (%d, %d)\n" % (max_rank_x, max_rank))
            f.write("Last rank: (%d, %d)\n" % (x[-1], ranks_y[-1]))
            f.write("Min confidence: (%d, %f)\n" % (min_conf_x, min_conf))
            f.write("Max confidence: (%d, %f)\n" % (max_conf_x, max_conf))
            f.write("Last confidence: (%d, %f)\n" % (x[-1], confidences_y[-1]))

        print("Dumped %s data to text" % name)


    def generate_stats(self, title="", dump_text=True):
        plot = RankConfidencePlot(includey2=True)  # TODO need to manually change this includey2... Fix

        linestyles = ['-', '--', ':', '-.']
        #colors = ['xkcd:aqua', 'xkcd:azure', 'xkcd:green']
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        if len(self.input_tuples) < 1:
            print("Nothing to do")
            return

        for input_tuple in sorted(self.input_tuples, key=lambda f: ascad_sort_name(f[0])):
            input_path, model_id = input_tuple

            if is_remote(input_path):
                remote_path = input_path
                input_path = os.path.join("./models/", remote_path.rpartition('models')[2][1:])
                self.get_remote_model(input_path, remote_path, model_id, self.model_suffix)

            tfold_blob = pickle.load(open(os.path.join(input_path, model_id + "-t-ranks.p"),   "rb"))
            x, ranks_y, confidences_y = get_series_from_tfold_blob(tfold_blob)
            dataset_name = input_path[input_path.find('ASCAD'):]
            rank_label = "mean rank (%s)" % dataset_name
            confidence_label = "mean confidence (%s)" % dataset_name
            linestyle = linestyles.pop(0)
            color = colors.pop(0)
            if 'aiascad' in model_id:  # Only plot ranks for ASCAD
                plot.add_rank_series(x, ranks_y, rank_label=rank_label, rank_color=color)
            else:
                plot.add_series(x, ranks_y, confidences_y, rank_label=rank_label, confidence_label=confidence_label, rank_color=color, confidence_color=color)
            if dump_text:
                self.dump_text(dataset_name + '-' + model_id, x, ranks_y, confidences_y)

        os.makedirs(self.output_path, exist_ok=True)
        plot.save(os.path.join(self.output_path, "combined-%s-tfold.pdf" % model_id))
        print("Combined:")
        print(self.input_tuples)

    def generate_stats_testrank(self, title="", dump_text=True):
        plot = RankConfidencePlot()

        linestyles = ['-', '--', ':', '-.']
        #colors = ['xkcd:aqua', 'xkcd:azure', 'xkcd:green']
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        if len(self.input_tuples) < 1:
            print("Nothing to do")
            return

        for input_tuple in sorted(self.input_tuples, key=lambda f: ascad_sort_name(f[0])):
            input_path, model_id = input_tuple

            if is_remote(input_path):
                remote_path = input_path
                input_path = os.path.join("./models/", remote_path.rpartition('models')[2][1:])
                self.get_remote_model(input_path, remote_path, model_id, self.model_suffix)

            tfold_blob = pickle.load(open(os.path.join(input_path, model_id + "-bestrank-testrank.p"),   "rb"))
            tfold_blob['ranks'] = np.expand_dims(tfold_blob['ranks'], axis=0)
            tfold_blob['confidences'] = np.expand_dims(tfold_blob['confidences'], axis=0)
            x, ranks_y, confidences_y = get_series_from_tfold_blob(tfold_blob)
            dataset_name = input_path[input_path.find('ASCAD'):]
            rank_label = "rank (%s)" % dataset_name
            confidence_label = "confidence (%s)" % dataset_name
            linestyle = linestyles.pop(0)
            color = colors.pop(0)
            plot.add_series(x, ranks_y, confidences_y, rank_label=rank_label, confidence_label=confidence_label, rank_color=color, confidence_color=color)

            if dump_text:
                self.dump_text(dataset_name + '-' + model_id, x, ranks_y, confidences_y)

        os.makedirs(self.output_path, exist_ok=True)
        plot.save(os.path.join(self.output_path, "combined-%s-testrank.pdf" % model_id))
        print("Combined:")
        print(self.input_tuples)


class ModelFinder():
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.is_remote = is_remote(models_dir)
        self.keywords = ('-t-ranks', '-last', '-history')

    def find_models(self, dir_filter=None, model_filter=None):
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
            if not dir_filter is None and not dir_filter in subdirectory:
                continue
            if self.is_remote:
                subdirectory = host + ':' + subdirectory
            files = subwalk[2]
            model_names = set()
            for file in files:
                for keyword in self.keywords:
                    if keyword in file:
                        model_names.add(file.rpartition(keyword)[0])
            for model_name in model_names:
                if not model_filter is None and not model_filter in model_name:
                    continue
                model_locations.append((subdirectory, model_name))

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
            print("Not enough parameters. Expected <model_subdirectory> <model_id>")
    elif args.command == 'autostats':
        if len(args.parameters) >= 1:
            mf = ModelFinder(args.parameters[0])
            for model_location in mf.find_models():
                f = FigureGenerator(model_location[0], model_location[1])
                f.generate_stats()
        else:
            print("Not enough parameters. Expected <models_root_directory>")
    elif args.command == 'combinedtfold':
        if len(args.parameters) >= 3:
            mf = ModelFinder(args.parameters[0])
            model_locations = mf.find_models(dir_filter=args.parameters[1], model_filter=args.parameters[2])
            f = CombinedFigureGenerator(model_locations, name=args.parameters[1])
            try: # Deadline approaching, fix later
                f.generate_stats()
            except FileNotFoundError:
                pass
            try:  # Deadline approaching, fix later
                f.generate_stats_testrank()
            except FileNotFoundError:
                pass
        else:
            print("Not enough parameters. Expected <models_root_directory> <dir_filter> <model_filter>")
    else:
        print("Unknown command %s" % args.command)
