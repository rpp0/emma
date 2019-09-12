import time
from emma.utils import utils, visualizations
from emma.processing import ops
import numpy as np
from emma.ai import saliency

from celery import group, chord
from celery.result import AsyncResult, GroupResult
from celery.utils.log import get_task_logger
from emma.utils.registry import activity
from emma.utils.utils import EMMAException, conf_has_op
from emma.attacks.leakagemodels import LeakageModel

logger = get_task_logger(__name__)  # Logger


def submit_task(task, *args, remote=True, message="Working", **kwargs):
    if remote:
        async_result = task.si(*args, **kwargs).delay()
        em_result = wait_until_completion(async_result, message=message + " (remote)")
    else:
        logger.info(message)
        em_result = task(*args, **kwargs)

    return em_result


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


def parallel_work(trace_set_paths, conf, merge_results=False):
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
    for part in utils.partition(trace_set_paths, num_partitions):
        result.append(ops.work.si(part, conf))
    if merge_results:  # Merge subresults from all workers into one final result
        return chord(result, body=ops.merge.s(conf))()
    else:
        return group(result)()


@activity('spattack')
@activity('attack')
def __perform_cpa_attack(emma):
    """
    Attack that predicts the best subkey guess using the maximum correlation with the key value.
    :param emma:
    :return:
    """

    def update_correlations(max_correlations, em_result, subkey_index):
        corr_result = em_result.correlations
        print("Num correlation entries: %d" % corr_result._n[0][0])

        # Get maximum correlations over all points
        for subkey_guess in range(0, 256):
            max_correlations[subkey_index, subkey_guess] = np.max(np.abs(corr_result[subkey_guess, :]))

        print("{:02x}".format(np.argmax(max_correlations[subkey_index])))

    def print_results(max_correlations):
        utils.pretty_print_subkey_scores(max_correlations, limit_rows=20)
        most_likely_bytes = np.argmax(max_correlations, axis=1)
        print(utils.numpy_to_hex(most_likely_bytes))

    __attack_subkeys(emma, update_correlations, print_results)


# TODO: Duplicate code, fix me
@activity('pattack')
def __perform_prob_cpa_attack(emma):
    """
    Attack that predicts the best subkey guess using the maximum probability of the key value.
    :param emma:
    :return:
    """

    def update_probabilities(max_probs, em_result, subkey_index):
        prob_result = em_result.probabilities

        # Get maximum correlations over all points
        for subkey_guess in range(0, 256):
            max_probs[subkey_index, subkey_guess] = np.max(prob_result[subkey_guess, :])

        print("{:02x}".format(np.argmax(max_probs[subkey_index])))

    def print_results(max_probs):
        utils.pretty_print_subkey_scores(max_probs, limit_rows=20)
        most_likely_bytes = np.argmax(max_probs, axis=1)
        print(utils.numpy_to_hex(most_likely_bytes))

    __attack_subkeys(emma, update_probabilities, print_results)


def __attack_subkeys(emma, subkey_score_cb, final_score_cb):
    score = np.zeros([emma.conf.key_high, 256])

    # Determine dataset to attack
    if emma.dataset_val is not None:
        raise EMMAConfException("Validation set should only be given when training a model.")

    logger.info("Attacking traces: %s" % str(emma.dataset.trace_set_paths))

    # Attack each subkey separately
    for subkey in range(emma.conf.key_low, emma.conf.key_high):
        emma.conf.subkey = subkey  # Set in conf, so the workers know which subkey to attack

        # Execute task
        async_result = parallel_work(emma.dataset.trace_set_paths, emma.conf, merge_results=True)
        em_result = wait_until_completion(async_result, message="Attacking subkey %d" % emma.conf.subkey)

        # Parse results
        if em_result is not None:
            subkey_score_cb(score, em_result, subkey)

    # Print results to stdout
    final_score_cb(score)


@activity('dattack')
def __perform_dis_attack(emma):
    """
    Attack that predicts the best subkey guess using the minimum absolute distance to the key value.
    :param emma:
    :return:
    """

    # Define callbacks
    def update_distances(min_distances, em_result, subkey_index):
        # Get score for this subkey determined by the workers
        dis_result = em_result.distances
        print("Num distance entries: %d" % dis_result._n[0][0])

        # Get minimum distances over all points in the trace or encoding
        for subkey_guess in range(0, 256):
            min_distances[subkey_index, subkey_guess] = np.min(dis_result[subkey_guess, :])

        # Print best subkey guess for this subkey index
        print("{:02x}".format(np.argmin(min_distances[subkey_index])))

    def print_results(min_distances):
        utils.pretty_print_subkey_scores(min_distances, limit_rows=20, descending=False)
        most_likely_bytes = np.argmin(min_distances, axis=1)
        print(utils.numpy_to_hex(most_likely_bytes))

    __attack_subkeys(emma, update_distances, print_results)


@activity('corrtrain')
@activity('ascadtrain')
@activity('shacputrain')
@activity('shacctrain')
@activity('autoenctrain')
def __perform_ml_attack(emma):
    """
    Trains a machine learning algorithm on the training samples from a dataset.
    """
    if emma.dataset is None:
        raise EMMAException("No dataset provided")

    if emma.dataset_val is None:  # No validation dataset provided, so split training data in two parts
        validation_split = emma.dataset.trace_set_paths[0:emma.conf.num_valsets]
        training_split = emma.dataset.trace_set_paths[emma.conf.num_valsets:]
    else:
        validation_split = emma.dataset_val.trace_set_paths[0:emma.conf.num_valsets]  # TODO: allow setting this per dataset?
        training_split = emma.dataset.trace_set_paths

    logger.info("Training set: %s" % str(training_split))
    logger.info("Validation set: %s" % str(validation_split))

    submit_task(ops.aitrain,
                training_split,
                validation_split,
                emma.conf,
                remote=emma.conf.remote,
                message="Training neural network")


@activity('plot')
def __perform_plot(emma, *params):
    trace_sets_to_get = max(int(emma.conf.plot_num_traces / emma.dataset.traces_per_set), 1)
    em_result = submit_task(ops.work,  # Op
                            emma.dataset.trace_set_paths[0:trace_sets_to_get], emma.conf, keep_trace_sets=True, keep_scores=False,  # Op parameters
                            remote=emma.conf.remote,
                            message="Performing actions")

    visualizations.plot_trace_sets(
        em_result.reference_signal,
        em_result.trace_sets,
        params=params,
        no_reference_plot=emma.conf.plot_no_reference,
        num_traces=emma.conf.plot_num_traces,
        title=emma.conf.plot_title,
        xlabel=emma.conf.plot_xlabel,
        ylabel=emma.conf.plot_ylabel,
        colorbar_label=emma.conf.plot_colorbar_label,
        time_domain=(not (conf_has_op(emma.conf, 'spec') or conf_has_op(emma.conf, 'fft'))) or emma.conf.plot_force_timedomain,
        sample_rate=1.0)


@activity('specgram')
def __perform_specgram(emma, *params):
    em_result = submit_task(ops.work,
                            emma.dataset.trace_set_paths[0:1], emma.conf, keep_trace_sets=True, keep_scores=False,  # Op parameters
                            remote=emma.conf.remote,
                            message="Performing actions")

    for trace_set in em_result.trace_sets:
        visualizations.plot_spectogram(trace_set,
                                       emma.conf.specgram_samprate,
                                       params=params,
                                       num_traces=emma.conf.plot_num_traces)


@activity('basetest')
def __perform_base_test(emma):
    async_result = ops.basetest.si(emma.dataset.trace_set_paths, emma.conf).delay()
    wait_until_completion(async_result, message="Performing base test")


@activity('default')
def __perform_actions(emma, message="Performing actions"):
    """
    Default activity: split trace_set_paths in partitions and let each node execute the actions on its assigned partition.
    :param emma:
    :param message:
    :return:
    """
    if emma.conf.remote:
        async_result = parallel_work(emma.dataset.trace_set_paths, emma.conf)
        return wait_until_completion(async_result, message=message)
    else:
        ops.work(emma.dataset.trace_set_paths, emma.conf)


@activity('keyplot')
def __perform_keyplot(emma, message="Grouping keys..."):
    for subkey in range(emma.conf.key_low, emma.conf.key_high):
        emma.conf.subkey = subkey  # Set in conf, so the workers know which subkey to attack

        if emma.conf.remote:
            async_result = parallel_work(emma.dataset.trace_set_paths, emma.conf)
            em_result = wait_until_completion(async_result, message=message)
        else:
            em_result = ops.work(emma.dataset.trace_set_paths, emma.conf)
            em_result = ops.merge(em_result, emma.conf)

        import pickle
        with open("/tmp/means.p", "wb") as f:
            pickle.dump(em_result.means, f)
        visualizations.plot_keyplot(em_result.means,
                                    time_domain=(not (conf_has_op(emma.conf, 'spec') or conf_has_op(emma.conf, 'fft'))) or emma.conf.plot_force_timedomain,
                                    sample_rate=1.0,
                                    show=True)


@activity('classify')
def __perform_classification_attack(emma):
    for subkey in range(emma.conf.key_low, emma.conf.key_high):
        emma.conf.subkey = subkey  # Set in conf, so the workers know which subkey to attack

        async_result = parallel_work(emma.dataset_val.trace_set_paths, emma.conf, merge_results=False)
        celery_results = wait_until_completion(async_result, message="Classifying")

        lm_outputs = LeakageModel(emma.conf).onehot_outputs  # Determine leakage model number of outputs (classes)
        predict_count = np.zeros(lm_outputs, dtype=int)
        label_count = np.zeros(lm_outputs, dtype=int)
        logprobs = np.zeros(lm_outputs, dtype=float)
        accuracy = 0
        num_samples = 0

        # Get results from all workers and store in prediction dictionary
        for celery_result in celery_results:
            em_result = celery_result.get()
            assert(len(em_result.labels) == len(em_result.predictions))

            for i in range(0, len(em_result.labels)):
                label = em_result.labels[i]
                prediction = em_result.predictions[i]
                logprob = em_result.logprobs[i]

                if label == prediction:
                    accuracy += 1
                predict_count[prediction] += 1
                label_count[label] += 1
                num_samples += 1

                logprobs += np.array(logprob)

        accuracy /= float(num_samples)

        print("Labels")
        print(label_count)
        print("Predictions")
        print(predict_count)
        print("Best argmax prediction: %02x (hex)" % np.argmax(predict_count))
        print("Argmax accuracy: %.4f" % accuracy)

        if np.sum(label_count) == np.max(label_count):
            print("Best logprob prediction: %02x" % np.argmax(logprobs))
            print("True key               : %02x" % np.argmax(label_count))
        else:
            print("WARNING: logprob prediction not available because there is more than 1 true key label.")


@activity('salvis')
def __visualize_model(emma, model_type, vis_type='2doverlay', *args, **kwargs):
    vis_type = vis_type.lower()
    if emma.dataset_val is not None:
        trace_sets = emma.dataset_val.trace_set_paths[0:emma.conf.num_valsets]
    else:
        trace_sets = emma.dataset.trace_set_paths

    salvis_result = submit_task(ops.salvis,
                                trace_sets,
                                model_type,
                                vis_type,
                                emma.conf,
                                remote=emma.conf.remote,
                                message="Getting trace set gradients %s" % str(trace_sets))

    logger.info("Getting saliency of %d traces" % salvis_result.examples_batch.shape[0])

    if vis_type == '1d':
        saliency.plot_saliency_1d(emma.conf, salvis_result)
    elif vis_type == '2d':
        saliency.plot_saliency_2d(emma.conf, salvis_result)
    elif vis_type == '2doverlay':
        saliency.plot_saliency_2d_overlay(emma.conf, salvis_result)
    elif vis_type == 'kerasvis':
        saliency.plot_saliency_kerasvis(emma.conf, salvis_result)
    elif vis_type == '2doverlayold':
        saliency.plot_saliency_2d_overlayold(emma.conf, salvis_result)
    else:
        logger.error("Unknown visualization type: %s" % vis_type)


@activity('optimize_capture')
def __optimize_capture(emma):
    submit_task(ops.optimize_capture,
                emma.dataset.trace_set_paths, emma.conf,
                remote=emma.conf.remote,
                message="Fitting PCA")
