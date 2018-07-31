# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import time
import emutils

from ops import *
from celery import group, chord
from celery.result import AsyncResult, GroupResult
from celery.utils.log import get_task_logger
from functools import wraps

logger = get_task_logger(__name__)  # Logger
activities = {}


def activity(name):
    """
    Defines the @activity decorator
    """
    def decorator(func):
        activities[name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


def wait_until_completion(async_result, message="Task"):
    """
    Wait for a Celery async_result to complete and measure the time taken.
    :param async_result:
    :param message:
    :return:
    """
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


def parallel_actions(trace_set_paths, conf, merge_results=False):
    """
    Divide the trace set paths into `conf.max_subtasks` partitions that are distributed to available workers. The
    actions are performed in parallel on these partitions. Optionally, after processing is completed by all workers,
    the results can be merged.
    :param trace_set_paths:
    :param conf:
    :param merge_results:
    :return:
    """
    num_partitions = min(conf.max_subtasks, len(trace_set_paths))
    result = []
    for part in emutils.partition(trace_set_paths, num_partitions):
        result.append(work.si(part, conf))
    if merge_results:  # Merge correlation subresult from all workers into one final result
        return chord(result, body=merge.s(conf))()
    else:
        return group(result)()


@activity('attack')
def perform_cpa_attack(emma):
    """
    Activity that performs a regular Correlation Power Analysis attack on the dataset.
    :param emma:
    :return:
    """
    logger.info("Attacking traces: %s" % str(emma.dataset.trace_set_paths))
    max_correlations = np.zeros([emma.conf.key_high, 256])

    for subkey in range(emma.conf.key_low, min(emma.conf.key_high, 16)):
        emma.conf.subkey = subkey

        # Execute task
        async_result = parallel_actions(emma.dataset.trace_set_paths, emma.conf, merge_results=True)
        em_result = wait_until_completion(async_result, message="Attacking subkey %d" % emma.conf.subkey)

        # Parse results
        if not em_result is None:
            corr_result = em_result.correlations
            print("Num entries: %d" % corr_result._n[0][0])

            # Get maximum correlations over all points
            for subkey_guess in range(0, 256):
                max_correlations[emma.conf.subkey, subkey_guess] = np.max(np.abs(corr_result[subkey_guess,:]))

            print("{:02x}".format(np.argmax(max_correlations[emma.conf.subkey])))

    # Print results to stdout
    emutils.pretty_print_correlations(max_correlations, limit_rows=20)
    most_likely_bytes = np.argmax(max_correlations, axis=1)
    print(emutils.numpy_to_hex(most_likely_bytes))


@activity('corrtrain')
@activity('ascadtrain')
@activity('shacputrain')
@activity('shacctrain')
def perform_ml_attack(emma):
    """
    Trains a machine learning algorithm on the training samples from a dataset.
    """
    if emma.dataset is None:
        raise Exception("No dataset provided")

    if emma.dataset_val is None:  # No validation dataset provided, so split training data in two parts
        if emma.dataset.format == "ascad":  # ASCAD uses different formatting
            validation_split = [x for x in emma.dataset.trace_set_paths if x.endswith('-val')]
            training_split = [x for x in emma.dataset.trace_set_paths if x.endswith('-train')]
        else:
            validation_split = emma.dataset.trace_set_paths[0:emma.conf.num_valsets]
            training_split = emma.dataset.trace_set_paths[emma.conf.num_valsets:]
    else:
        validation_split = emma.dataset_val.trace_set_paths[0:emma.conf.num_valsets]
        training_split = emma.dataset.trace_set_paths

    logger.info("Training set: %s" % str(training_split))
    logger.info("Validation set: %s" % str(validation_split))

    if emma.conf.remote:
        async_result = aitrain.si(training_split, validation_split, emma.conf).delay()
        wait_until_completion(async_result, message="Waiting for worker to train neural network")
    else:
        aitrain(training_split, validation_split, emma.conf)


@activity('basetest')
def perform_base_test(emma):
    async_result = basetest.si(emma.dataset.trace_set_paths, emma.conf).delay()
    wait_until_completion(async_result, message="Performing base test")


@activity('default')
def perform_actions(emma, message="Performing actions"):
    """
    Default activity: split trace_set_paths in partitions and let each node execute the actions on its assigned partition.
    :param emma:
    :param message:
    :return:
    """
    async_result = parallel_actions(emma.dataset.trace_set_paths, emma.conf)
    return wait_until_completion(async_result, message=message)


@activity('classify')
def perform_classification_attack(emma):
    async_result = parallel_actions(emma.dataset.trace_set_paths, emma.conf)
    celery_results = wait_until_completion(async_result, message="Classifying")

    if emma.conf.hamming:
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
