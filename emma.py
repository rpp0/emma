#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from ops import *
from debug import DEBUG
from time import sleep
from emma_worker import app, backend
from celery import group, chord, chain
from celery.result import AsyncResult, GroupResult
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import configparser
import emutils
import emio
import subprocess
import time

def parallel_actions_merge_corr(trace_set_paths, conf):
    num_partitions = min(conf.max_subtasks, len(trace_set_paths))
    result = []
    for part in emutils.partition(trace_set_paths, num_partitions):
        result.append(work.si(part, conf))
    return chord(result, body=merge.s(conf))()

def parallel_actions(trace_set_paths, conf):
    num_partitions = min(conf.max_subtasks, len(trace_set_paths))
    result = []
    for part in emutils.partition(trace_set_paths, num_partitions):
        result.append(work.si(part, conf))
    return group(result)()

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

def wait_until_completion(async_result, message="Task"):
    count = 0
    while not async_result.ready():
        print("\r%s: elapsed: %ds" % (message, count), end='')
        count += 1
        time.sleep(1)
    print("")

    if isinstance(async_result, AsyncResult):
        return async_result.result
    elif isinstance(async_result, GroupResult):
        return async_result.results
    else:
        raise TypeError

def perform_cpa_attack(dataset, conf):
    logger.info("Attacking traces: %s" % str(dataset.trace_set_paths))
    max_correlations = np.zeros([conf.skip_subkeys + conf.num_subkeys, 256])

    for subkey in range(conf.skip_subkeys, min(conf.skip_subkeys + conf.num_subkeys, 16)):
        conf.subkey = subkey

        # Execute task
        async_result = parallel_actions_merge_corr(dataset.trace_set_paths, conf)
        em_result = wait_until_completion(async_result, message="Attacking subkey %d" % conf.subkey)

        # Parse results
        if not em_result is None:
            corr_result = em_result.correlations
            print("Num entries: %d" % corr_result._n[0][0])

            # Get maximum correlations over all points
            for subkey_guess in range(0, 256):
                max_correlations[conf.subkey, subkey_guess] = np.max(np.abs(corr_result[subkey_guess,:]))

            print("{:02x}".format(np.argmax(max_correlations[conf.subkey])))

    # Print results to stdout
    emutils.pretty_print_correlations(max_correlations, limit_rows=20)
    most_likely_bytes = np.argmax(max_correlations, axis=1)
    print(emutils.numpy_to_hex(most_likely_bytes))

def perform_ml_attack(dataset, dataset_val, conf):
    """
    Train ML algorithm. Use only one core because Tensorflow is not thread-safe.
    """
    if dataset_val is None:  # No validation dataset provided, so split training data
        split_size = 1  # Number of trace sets to use for validation
        validation_split = dataset.trace_set_paths[0:split_size]
        training_split = dataset.trace_set_paths[split_size:]
    else:
        validation_split = dataset_val.trace_set_paths[0:conf.num_valsets]
        training_split = dataset.trace_set_paths

    logger.info("Training set: %s" % str(training_split))
    logger.info("Validation set: %s" % str(validation_split))
    async_result = aitrain.si(training_split, validation_split, conf).delay()
    wait_until_completion(async_result, message="Training neural network")

def perform_base_test(dataset, conf):
    async_result = basetest.si(dataset.trace_set_paths, conf).delay()
    wait_until_completion(async_result, message="Performing base test")

def perform_actions(dataset, conf):
    async_result = parallel_actions(dataset.trace_set_paths, conf)
    wait_until_completion(async_result, message="Performing actions")

def perform_classification_attack(dataset, conf):
    async_result = parallel_actions(dataset.trace_set_paths, conf)
    celery_results = wait_until_completion(async_result, message="Classifying")

    if conf.hamming:
        predict_count = np.zeros(9, dtype=int)
        label_count = np.zeros(9, dtype=int)
    else:
        predict_count = np.zeros(256, dtype=int)
        label_count = np.zeros(256, dtype=int)
    accuracy = 0
    num_samples = 0

    # Get results from all workers and store in prediction dictionary
    for celery_result in celery_results:
        em_result = celery_result.get()
        for i in range(0, len(em_result._data['labels'])):
            label = em_result._data['labels'][i]
            prediction = em_result._data['predictions'][i]
            if label == prediction:
                accuracy += 1
            predict_count[prediction] += 1
            label_count[label] += 1
            num_samples += 1
    accuracy /= float(num_samples)

    print("Labels")
    print(label_count)
    print("Predictions")
    print(predict_count)
    print("Best prediction: %d" % np.argmax(predict_count))
    print("Accuracy: %.4f" % accuracy)

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
    args, unknown = parser.parse_known_args()
    print(emutils.BANNER)

    try:
        clear_redis()

        # Get a list of filenames from a dataset
        dataset = emio.remote_get_dataset(dataset=args.dataset, conf=args)
        if not args.refset is None:
            dataset_ref = emio.remote_get_dataset(dataset=args.refset, conf=args)
        else:
            dataset_ref = dataset
        if not args.valset is None:
            dataset_val = emio.remote_get_dataset(dataset=args.valset, conf=args)
        else:
            dataset_val = None

        # Worker-specific configuration. Add properties of the loaded dataset
        conf = argparse.Namespace(
            format=dataset.format,
            reference_signal=dataset_ref.reference_signal,
            traces_per_set=dataset.traces_per_set,
            datasets_path=dataset.prefix,
            dataset_id=dataset.id,
            subkey=0,
            **args.__dict__
        )

        if 'attack' in conf.actions:  # Group of tasks and merge correlation results
            perform_cpa_attack(dataset, conf)
        elif 'basetest' in conf.actions:
            perform_base_test(dataset, conf)
        elif True in [a.find('train') > -1 for a in conf.actions]:
            perform_ml_attack(dataset, dataset_val, conf)
        elif 'classify' in conf.actions:
            perform_classification_attack(dataset, conf)
        else:  # Regular group of tasks
            perform_actions(dataset, conf)
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.control.purge()
    app.backend.cleanup()
    if args.kill_workers:
        subprocess.check_output(['pkill', '-9', '-f', 'celery'])
