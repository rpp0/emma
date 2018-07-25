#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from ops import ops
from activities import *
from debug import DEBUG
from emma_worker import app, backend
import matplotlib.pyplot as plt
import argparse
import subprocess

def args_epilog():
    result = "Actions can take the following parameters between square brackets ('[]'):\n"
    for op in ops.keys():
        result += "{:>20s} ".format(op)
        if op in ops_optargs:
            result += "["
            for optarg in ops_optargs[op]:
                result += "{:s}, ".format(optarg)
            result = result.strip().rstrip(',')
            result += "]"
        result += "\n"
    return result

def clear_redis():
    '''
    Clear any previous results from Redis. Sadly, there is no cleaner way atm.
    '''
    try:
        subprocess.check_output(["redis-cli", "flushall"])
        logger.info("Redis cleared")
    except FileNotFoundError:
        logger.warning("Could not clear local Redis database")

class EMMAHost():
    def __init__(self, args):
        self.dataset, self.dataset_ref, self.dataset_val = self._get_datasets(args)
        self.conf = self._generate_conf(args)

    def _get_datasets(self, args):
        # Load dataset from worker node
        dataset = emio.remote_get_dataset(dataset=args.dataset, conf=args)

        # Load reference set if applicable. Otherwise just use reference from dataset
        if not args.refset is None:
            dataset_ref = emio.remote_get_dataset(dataset=args.refset, conf=args)
        else:
            dataset_ref = dataset

        # Load validation set if applicable
        if not args.valset is None:
            dataset_val = emio.remote_get_dataset(dataset=args.valset, conf=args)
        else:
            dataset_val = None

        return dataset, dataset_ref, dataset_val

    def _generate_conf(self, args):
        if self.dataset is None or self.dataset_ref is None:
            raise Exception("Tried to generate configuration without loading datasets.")

        conf = argparse.Namespace(
            format=self.dataset.format,
            reference_signal=self.dataset_ref.reference_signal,
            traces_per_set=self.dataset.traces_per_set,
            datasets_path=self.dataset.prefix,
            dataset_id=self.dataset.id,
            subkey=0,
            **args.__dict__
        )

        return conf

    def _determine_activity(self):
        num_activities = 0
        activity = None

        for activity_name in activities.keys():
            for action in self.conf.actions:
                if activity_name == action:
                    activity = activities[activity_name]
                    num_activities += 1

        if num_activities > 1:
            raise Exception("Only one activity can be executed at a time. Choose from: %s" % str(activities.keys()))

        if activity is None:
            activity = activities['default']

        return activity

    def run(self):
        activity = self._determine_activity()
        activity(self)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Electromagnetic Mining Array (EMMA)', epilog=args_epilog(), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('actions', type=str, help='Action to perform. Choose from %s' % str(ops.keys()), nargs='+')
    parser.add_argument('dataset', type=str, help='Identifier of dataset to use')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw','sigmf','gnuradio'], default='sigmf', help='Output format to use when saving')
    parser.add_argument('--outpath', '-O', dest='outpath', type=str, default='./export/', help='Output path to use when saving')
    parser.add_argument('--max-subtasks', type=int, default=32, help='Maximum number of subtasks')
    parser.add_argument('--skip-subkeys', type=int, default=0, help='Number of subkeys to skip')
    parser.add_argument('--num-subkeys', type=int, default=16, help='Number of subkeys to break')
    parser.add_argument('--kill-workers', default=False, action='store_true', help='Kill workers after finishing the tasks.')
    parser.add_argument('--butter-order', type=int, default=1, help='Order of Butterworth filter')
    parser.add_argument('--butter-cutoff', type=float, default=0.01, help='Cutoff of Butterworth filter')
    parser.add_argument('--windowing-method', type=str, default='rectangular', help='Windowing method')
    parser.add_argument('--hamming', default=False, action='store_true', help='Use Hamming weight instead of true byte values.')
    parser.add_argument('--augment-roll', default=False, action='store_true', help='Roll signal during data augmentation.')
    parser.add_argument('--augment-noise', default=False, action='store_true', help='Add noise to the signal during data augmentation.')
    parser.add_argument('--update', default=False, action='store_true', help='Update existing AI model instead of replacing.')
    parser.add_argument('--online', default=False, action='store_true', help='Fetch samples from remote EMcap instance online (without storing to disk).')
    parser.add_argument('--refset', type=str, default=None, help='Dataset to take reference signal from for alignment (default = same as dataset argument)')
    parser.add_argument('--valset', type=str, default=None, help='Dataset to take validation set traces from (default = same as dataset argument)')
    parser.add_argument('--model-suffix', type=str, default=None, help='Suffix for model name.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--max-cache', type=int, default=None, help='Max trace sets in cache. Default: all.')
    parser.add_argument('--num-valsets', type=int, default=128, help='Number of validation trace sets to use')
    parser.add_argument('--normalize', default=False, action='store_true', help='Normalize input data before feeding to NN')
    parser.add_argument('--tfold', default=False, action='store_true', help='Train using t-fold cross-validation')
    parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in model.')
    parser.add_argument('--activation', type=str, default='leakyrelu', help='Activation function of model.')
    parser.add_argument('--cnn', default=False, action='store_true', help='Use ASCAD CNN for AICorrNet')
    parser.add_argument('--testrank', default=False, action='store_true', help='Load model and test rank for varying test set sizes.')
    parser.add_argument('--regularizer', type=str, default=None, help='Reg')
    parser.add_argument('--reglambda', type=float, default=0.001, help='Reg')
    parser.add_argument('--nomodel', default=False, action='store_true', help='Do not assume power consumption model.')
    parser.add_argument('--ptinput', default=False, action='store_true', help='Also use plaintext as inputs.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size.')
    parser.add_argument('--metric-freq', type=int, default=10, help='Frequency of calculating metrics (e.g. rank) of model.')
    args, unknown = parser.parse_known_args()
    print(emutils.BANNER)

    try:
        clear_redis()

        emma = EMMAHost(args)
        emma.run()
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.control.purge()
    app.backend.cleanup()
    if args.kill_workers:
        subprocess.check_output(['pkill', '-9', '-f', 'celery'])
