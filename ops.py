# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import os
import matplotlib
if not 'DISPLAY' in os.environ:  # Do not attempt to show plot windows when headless
    matplotlib.use('Agg')
import numpy as np
import sys
import matplotlib.pyplot as plt
import emio
import pickle
import configparser
import aiiterators
import ai
import traceset
import rank
import saliency
import registry
from emma_worker import app, broker
from dsp import *
from correlationlist import CorrelationList
from distancelist import DistanceList
from os.path import join, basename
from emutils import Window, conf_to_id, get_action_op_params
from celery.utils.log import get_task_logger
from lut import hw, sbox
from emresult import EMResult, SalvisResult
from registry import op

logger = get_task_logger(__name__)  # Logger


@op('align', optargs=['ref_window_begin', 'ref_window_end'])
def align_trace_set(trace_set, result, conf, params=None):
    """
    Align a set of traces based on a single reference trace using cross-correlation.
    If a trace is empty, it is discarded.
    """
    logger.info("align %s" % (str(params) if not params is None else ""))
    if params is None:  # If no parameters provided, assume percent% max offset
        percent = 0.30
        length = len(conf.reference_signal)
        end = int(length - length*percent)
        begin = int(0 + length*percent)
        window = Window(begin=begin, end=end)
    else:
        window = Window(begin=int(params[0]), end=int(params[1]))

    logger.info("Aligning %d traces" % len(trace_set.traces))
    aligned_trace_set = []
    reference = conf.reference_signal[window.begin:window.end]

    discarded = 0
    for trace in trace_set.traces:
        aligned_trace = align(trace.signal, reference)
        if not aligned_trace is None:
            trace.signal = aligned_trace
            aligned_trace_set.append(trace)
        else:
            discarded += 1

    if discarded > 0:
        logger.warning("Discarded %d unable to align traces." % discarded)

    trace_set.set_traces(np.array(aligned_trace_set))


@op('filterkey', optargs=['key'])
def filterkey_trace_set(trace_set, result, conf, params=None):
    """
    Filter traces by key prefix
    """
    logger.info("filterkey %s" % (str(params) if not params is None else ""))
    if params is None:
        logger.warning("No argument specified for filterkey. Skipping op.")
        return

    key_param = params[0]

    filtered_trace_set = []
    discarded = 0
    for trace in trace_set.traces:
        hex_key = ''.join(["%02x" % x for x in list(trace.key.astype(int))])
        if hex_key.startswith(key_param):
            filtered_trace_set.append(trace)
        else:
            discarded += 1

    if discarded > 0:
        logger.info("Discarded %d keys not matching %s." % (discarded, key_param))

    trace_set.set_traces(np.array(filtered_trace_set))


@op('spec')
def spectogram_trace_set(trace_set, result, conf, params=None):
    """
    Calculate the spectogram of the trace set.
    """
    logger.info("spec %s" % (str(params) if not params is None else ""))
    if not trace_set.windowed:
        logger.warning("Taking the FFT of non-windowed traces will result in variable FFT sizes.")

    for trace in trace_set.traces:
        trace.signal = np.square(np.abs(np.fft.fft(trace.signal)))
        #if True: # If real signal
        #    trace.signal = trace.signal[0:int(len(trace.signal) / 2)]


@op('abs')
def magnitude_trace_set(trace_set, result, conf, params=None):
    """
    Calculate the magnitude of the signals in trace_set
    """
    logger.info("abs %s" % (str(params) if not params is None else ""))

    for trace in trace_set.traces:
        trace.signal = np.abs(trace.signal)
    conf.reference_signal = np.abs(conf.reference_signal)


@op('norm')
def normalize_trace_set(trace_set, result, conf, params=None):
    """
    Normalize the signals (amplitudes) in a trace set.
    """
    logger.info("norm %s" % (str(params) if not params is None else ""))

    for trace in trace_set.traces:
        trace.signal = trace.signal - np.mean(trace.signal)


@op('fft')
def fft_trace_set(trace_set, result, conf, params=None):
    logger.info("fft %s" % (str(params) if not params is None else ""))
    if not trace_set.windowed:
        logger.warning("Taking the FFT of non-windowed traces will result in variable FFT sizes.")

    for trace in trace_set.traces:
        trace.signal = np.fft.fft(trace.signal)


@op('rwindow', optargs=['window_begin', 'window_end', 'offset'])
def random_window_trace_set(trace_set, result, conf, params=None):
    """
    Like window, but with a random begin offset. Used to artificially increase training set.
    """
    # logger.info("rwindow %s" % (str(params) if not params is None else ""))
    if params is None:
        logger.error("3 params must be provided to rwindow (begin, end, offset)")
        exit(1)
    else:
        if len(params) > 2:
            begin = int(params[0])
            end = int(params[1])
            offset = int(params[2])
        else:
            logger.error("3 params must be provided to rwindow (begin, end, offset)")
            exit(1)

    length = end - begin
    rand_offset = np.random.randint(low=-offset, high=offset+1)
    new_begin = max(begin + rand_offset, 0)
    new_end = new_begin + length
    window_trace_set(trace_set, result, conf, params=[str(new_begin), str(new_end), 'rectangular'])


@op('window', optargs=['window_begin', 'window_end', 'method'])
def window_trace_set(trace_set, result, conf, params=None):
    """
    Perform windowing on a specific trace set. See https://en.wikipedia.org/wiki/Window_function#Spectral_analysis
    for a good overview of the effects of the different windowing methods on the PSD of the signal.

    The trace is windowed according to conf.window.size, or according to the size of the reference trace if the
    window is not configured.

    Interesting excerpt: 'What cannot be seen from the graphs is that the rectangular window has the best noise bandwidth, which makes it a good candidate for detecting low-level sinusoids in an otherwise white noise environment. Interpolation techniques, such as zero-padding and frequency-shifting, are available to mitigate its potential scalloping loss.'

    Params: (window start, window end)
    """
    logger.info("window %s" % (str(params) if not params is None else ""))
    windowing_method = conf.windowing_method  # Default windowing
    if params is None:  # If no parameters provided, window according to reference signal
        window = Window(begin=0, end=len(conf.reference_signal))
    else:
        window = Window(begin=int(params[0]), end=int(params[1]))
        if len(params) > 2:  # Override windowing
            windowing_method = params[2]

    for trace in trace_set.traces:
        length_diff = len(trace.signal[window.begin:]) - window.size
        # Pad or cut
        if length_diff < 0:
            trace.signal = np.lib.pad(trace.signal[window.begin:], (0, abs(length_diff)), 'constant', constant_values=(0.0))
        else:
            trace.signal = trace.signal[window.begin:window.end]
        assert(len(trace.signal) == window.size)

        # Apply window
        if windowing_method == 'rectangular':
            continue # Already cut rectangularly
        elif windowing_method == 'kaiser':
            trace.signal = trace.signal * np.kaiser(window.size, 14)
        elif windowing_method == 'blackman':
            trace.signal = trace.signal * np.blackman(window.size)
        else:
            logger.warning("Requested unknown windowing method '%d'. Skipping." % windowing_method)
            return
    trace_set.windowed = True
    trace_set.window = window


@op('filter')
def filter_trace_set(trace_set, result, conf, params=None):
    """
    Apply a Butterworth filter to the traces.
    """
    logger.info("filter %s" % (str(params) if not params is None else ""))
    for trace in trace_set.traces:
        trace.signal = butter_filter(trace.signal, order=conf.butter_order, cutoff=conf.butter_cutoff)

    conf.reference_signal = butter_filter(conf.reference_signal, order=conf.butter_order, cutoff=conf.butter_cutoff)


@op('rmoutliers')
def rmoutliers_trace_set(trace_set, result, conf, params=None):
    """
    Remove outliers in terms of amplitude.
    """
    logger.info("rmoutliers %s" % (str(params) if not params is None else ""))
    reference_mean = np.mean(conf.reference_signal)
    threshold = 0.001

    new_traces = []
    for trace in trace_set.traces:
        trace_mean = np.mean(trace.signal)
        diff = reference_mean - trace_mean
        if np.abs(diff) <= threshold:
            new_traces.append(trace)

    trace_set.set_traces(new_traces)


@op('roll')
def roll_trace_set(trace_set, result, conf, params=None):
    logger.info("roll %s" % (str(params) if not params is None else ""))
    if params is None:  # If no parameters provided, window according to reference signal
        roll_window = Window(begin=0, end=len(conf.reference_signal))
    else:
        roll_window = Window(begin=int(params[0]), end=int(params[1]))

    for trace in trace_set.traces:
        trace.signal = np.roll(trace.signal, np.random.randint(roll_window.begin, roll_window.end))


@op('save')
def save_trace_set(trace_set, result, conf, params=None):
    """
    Save the trace set to a file using the output format specified in the conf object.
    """
    logger.info("save %s" % (str(params) if not params is None else ""))
    if conf.outform == 'cw':
        # Save back to output file
        np.save(join(conf.outpath, trace_set.name + '_traces.npy'), trace_set.traces)

        # Update the corresponding config file
        emio.update_cw_config(conf.outpath, trace_set, {"numPoints": len(conf.reference_signal)})
    elif conf.outform == 'sigmf':  # TODO make SigMF compliant
        count = 1
        for trace in trace_set.traces:
            trace.tofile(join(conf.outpath, "%s-%d.rf32_le" % (trace_set.name, count)))
            count += 1
    else:
        print("Unknown format: %s" % conf.outform)
        exit(1)


@op('attack')
def attack_trace_set(trace_set, result, conf=None, params=None):
    """
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    """
    logger.info("attack %s" % (str(params) if not params is None else ""))

    # Use mask?
    usemask = False
    if not params is None:
        if len(params) == 1:
            if params[0] == 'usemask':
                usemask = True

    # Init if first time
    if result.correlations is None:
        result.correlations = CorrelationList([256, trace_set.window.size])

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return

    if trace_set.num_traces <= 0:
        logger.warning("Skipping empty trace set.")
        return

    hypotheses = np.empty([256, trace_set.num_traces])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            if usemask:
                mask = trace_set.traces[i].mask[conf.subkey] if not trace_set.traces[i].mask is None else 0
            else:
                mask = 0
            if conf.nomodel:
                hypotheses[subkey_guess, i] = sbox[trace_set.traces[i].plaintext[conf.subkey] ^ subkey_guess]
            else:
                hypotheses[subkey_guess, i] = hw[sbox[trace_set.traces[i].plaintext[conf.subkey] ^ subkey_guess] ^ mask]  # Model of the power consumption

    # 2. Given point j of trace i, calculate the correlation between all hypotheses
    for j in range(0, trace_set.window.size):
        # Get measurements (columns) from all traces
        measurements = np.empty(trace_set.num_traces)
        for i in range(0, trace_set.num_traces):
            measurements[i] = trace_set.traces[i].signal[j]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Update correlation
            result.correlations.update((subkey_guess,j), hypotheses[subkey_guess,:], measurements)

# TODO: Duplicate code, fix me
@op('dattack')
def dattack_trace_set(trace_set, result, conf=None, params=None):
    """
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    """
    logger.info("dattack %s" % (str(params) if not params is None else ""))

    # Init if first time
    if result.distances is None:
        result.distances = DistanceList([256, trace_set.window.size])

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return

    if trace_set.num_traces <= 0:
        logger.warning("Skipping empty trace set.")
        return

    hypotheses = np.empty([256, trace_set.num_traces])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            if conf.nomodel:
                hypotheses[subkey_guess, i] = sbox[trace_set.traces[i].plaintext[conf.subkey] ^ subkey_guess]
            elif conf.nomodelpt:
                hypotheses[subkey_guess, i] = subkey_guess / 255.0
            else:
                hypotheses[subkey_guess, i] = hw[sbox[trace_set.traces[i].plaintext[conf.subkey] ^ subkey_guess]]  # Model of the power consumption

    # 2. Given point j of trace i, calculate the distance between all hypotheses
    for j in range(0, trace_set.window.size):
        # Get measurements (columns) from all traces
        measurements = np.empty(trace_set.num_traces)
        for i in range(0, trace_set.num_traces):
            measurements[i] = trace_set.traces[i].signal[j]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Update distamces
            result.distances.update((subkey_guess, j), hypotheses[subkey_guess, :], measurements)


@op('memattack')
def memattack_trace_set(trace_set, result, conf=None, params=None):
    logger.info("memattack %s" % (str(params) if not params is None else ""))
    if result.correlations is None:
        result.correlations = CorrelationList([16, 256, trace_set.window.size])

    for byte_idx in range(0, conf.key_high - conf.key_low):
        for j in range(0, trace_set.window.size):
            # Get measurements (columns) from all traces
            measurements = np.empty(trace_set.num_traces)
            for i in range(0, trace_set.num_traces):
                measurements[i] = trace_set.traces[i].signal[j]

            # Correlate measurements with 256 hypotheses
            for byte_guess in range(0, 256):
                # Update correlation
                hypotheses = [hw[byte_guess]] * trace_set.num_traces
                result.correlations.update((byte_idx,byte_guess,j), hypotheses, measurements)


@op('memtrain')
def memtrain_trace_set(trace_set, result, conf=None, params=None):
    if trace_set.windowed:
        if result.ai is None:
            logger.debug("Initializing Keras")
            result.ai = ai.AIMemCopyDirect(input_dim=len(trace_set.traces[0].signal), hamming=conf.hamming)

        signals = np.array([trace.signal for trace in trace_set.traces])
        values = np.array([hw[trace.plaintext[0]] for trace in trace_set.traces])
        logger.warning("Training %d signals" % len(signals))
        result.ai.train_set(signals, values)
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")


@op('weight', optargs=['weight_filename'])
def weight_trace_set(trace_set, result, conf=None, params=None):
    """
    Multiply trace signal element-wise with weights stored in a file.
    """
    logger.info("weight %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if params is None:
            filename = "weights.p"
        else:
            filename = str(params[0])

        weights = pickle.load(open(filename, "rb"))
        if len(weights) == trace_set.window.size:
            for trace in trace_set.traces:
                trace.signal = np.multiply(trace.signal, weights)
        else:
            logger.error("Weight length is not equal to signal length.")
    else:
        logger.error("The trace set must be windowed before applying weights.")


@op('sum')
def sum_trace_set(trace_set, result, conf=None, params=None):
    logger.info("sum %s" % (str(params) if not params is None else ""))
    for trace in trace_set.traces:
        trace.signal = np.array([np.sum(trace.signal)])

    trace_set.windowed = True
    trace_set.window = Window(begin=0, end=1)


@op('corrtest')
def corrtest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("corrtest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result.ai is None:
            result.ai = ai.AI(conf, "aicorrnet")
            result.ai.load()

        # Fetch signals from traces
        if conf.ptinput:
            x = np.array([np.concatenate((trace.signal, trace.plaintext)) for trace in trace_set.traces], dtype=float)
        elif conf.kinput:
            x = np.array([np.concatenate((trace.signal, trace.key)) for trace in trace_set.traces], dtype=float)
        else:
            x = np.array([trace.signal for trace in trace_set.traces])

        # Get encodings of signals
        encodings = result.ai.predict(x)

        # Replace original signal with encoding
        assert(encodings.shape[0] == len(trace_set.traces))
        for i in range(0, len(trace_set.traces)):
            trace_set.traces[i].signal = encodings[i]

        # Adjust window size
        trace_set.window = Window(begin=0, end=encodings.shape[1])
        trace_set.windowed = True
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")


@op('shacputest')
def shacputest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("shacputest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result.ai is None:
            result.ai = ai.AI(conf, "aishacpu")
            result.ai.load()

        for trace in trace_set.traces:
            if conf.hamming:
                result.labels.append(hw[trace.plaintext[0] ^ 0x36])
            else:
                result.labels.append(trace.plaintext[0] ^ 0x36)
            result.predictions.append(np.argmax(result.ai.predict(np.array([trace.signal], dtype=float))))


@op('shacctest')
def shacctest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("shacctest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result.ai is None:
            result.ai = ai.AI(conf, "aishacc")
            result.ai.load()

        for trace in trace_set.traces:
            if conf.hamming:
                result.labels.append(hw[trace.plaintext[0] ^ 0x36])
            else:
                result.labels.append(trace.plaintext[0] ^ 0x36)

            cc_out = result.ai.predict(np.array([trace.signal], dtype=float))
            predicted_classes = np.argmax(cc_out, axis=1)
            result.predictions.append(predicted_classes[0])


@app.task(bind=True)
def merge(self, to_merge, conf):
    if type(to_merge) is EMResult:
        to_merge = [to_merge]

    # Is it useful to merge?
    if len(to_merge) >= 1:
        result = EMResult(task_id=self.request.id)

        # If we are attacking, merge the correlations
        if 'attack' in conf.actions or 'memattack' in conf.actions:
            # Get size of correlations
            shape = to_merge[0].correlations._n.shape  # TODO fixme init hetzelfde als in attack

            # Init result
            result.correlations = CorrelationList(shape)

            # Start merging
            for m in to_merge:
                result.correlations.merge(m.correlations)
        elif 'dattack' in conf.actions:  # TODO just check for presence of to_merge.distances instead of doing this
            shape = to_merge[0].distances._n.shape
            result.distances = DistanceList(shape)
            # Start merging
            for m in to_merge:
                result.distances.merge(m.distances)

        # Clean up tasks
        for m in to_merge:
            logger.warning("Deleting %s" % m.task_id)
            app.AsyncResult(m.task_id).forget()

        return result
    else:
        return None


@app.task
def remote_get_dataset(dataset, conf=None):
    return emio.get_dataset(dataset, conf=conf, remote=False)


@app.task
def remote_get_trace_set(trace_set_path, format, ignore_malformed):
    return emio.get_trace_set(trace_set_path, format, ignore_malformed, remote=False)


def process_trace_set(result, trace_set, conf, request_id=None, keep_trace_sets=False):
    # Keep copy of reference signal
    original_reference_signal = conf.reference_signal.copy()

    # Perform actions
    for action in conf.actions:
        op, params = get_action_op_params(action)
        if op in registry.operations:
            registry.operations[op](trace_set, result, conf=conf, params=params)
        else:
            if op not in registry.activities:
                logger.warning("Ignoring unknown op '%s'." % op)

    # Store result
    if keep_trace_sets:
        result.trace_sets.append(trace_set)
        result.reference_signal = conf.reference_signal

    # Restore reference signal for next trace set
    # This is required because changes to the reference need to happen in lockstep (crucial for alignment for example).
    conf.reference_signal = original_reference_signal


def process_trace_set_paths(result, trace_set_paths, conf, request_id=None, keep_trace_sets=False):
    num_todo = len(trace_set_paths)
    num_done = 0
    for trace_set_path in trace_set_paths:
        # Get trace name from path
        trace_set_name = basename(trace_set_path)
        logger.info("Processing '%s' (%d/%d)" % (trace_set_name, num_done, num_todo))

        # Load trace
        trace_set = emio.get_trace_set(trace_set_path, conf.format, ignore_malformed=False, remote=False)
        if trace_set is None:
            logger.warning("Failed to load trace set %s (got None). Skipping..." % trace_set_path)
            continue

        # Process trace set
        process_trace_set(result, trace_set, conf, request_id, keep_trace_sets)

        num_done += 1


def resolve_paths(trace_set_paths):
    """
    Determine the path on disk based on the location of the database specified in the
    worker's settings file.
    """
    settings = configparser.RawConfigParser()
    settings.read('settings.conf')
    prefix = settings.get("Datasets", "datasets_path")

    for i in range(0, len(trace_set_paths)):
            # Add prefix to path
            trace_set_paths[i] = join(prefix, trace_set_paths[i])


@app.task(bind=True)
def work(self, trace_set_paths, conf, keep_trace_sets=False, keep_scores=True, keep_ai=False):
    """
    Actions to be performed by workers on the trace set given in trace_set_path.
    """
    resolve_paths(trace_set_paths)  # Get absolute paths

    if type(trace_set_paths) is list:
        result = EMResult(task_id=self.request.id)  # Keep state and results

        # Process trace set paths and fill in results of analysis
        process_trace_set_paths(result, trace_set_paths, conf, request_id=self.request.id, keep_trace_sets=keep_trace_sets)

        if not keep_trace_sets:  # Do not return processed traces
            result.trace_sets = None
            result.reference_signal = None
        if not keep_scores:  # Do not return attack scores
            result.correlations = None
            result.distances = None
        if not keep_ai:
            result.ai = None  # Do not return AI object

        return result
    else:
        logger.error("Must provide a list of trace set paths to worker!")
        return None


def action_to_model_type(action):
    if action == 'corrtrain':
        return 'aicorrnet'
    elif action == 'shacputrain':
        return 'aishacpu'
    elif action == 'shacctrain':
        return 'aishacc'
    elif action == 'ascadtrain':
        return 'aiascad'
    else:
        return None


def get_conf_model_type(conf):
    for action in conf.actions:
        model_type = action_to_model_type(action)
        if not model_type is None:
            return model_type
    return None


@app.task(bind=True)
def basetest(self, trace_set_paths, conf, rank_trace_step=1000, t=10):
    resolve_paths(trace_set_paths)  # Get absolute paths

    if type(trace_set_paths) is list:
        result = EMResult(task_id=self.request.id)  # Keep state and results

        # Process trace set paths
        process_trace_set_paths(result, trace_set_paths, conf, request_id=self.request.id, keep_trace_sets=True)

        all_traces_list = []
        for trace_set in result.trace_sets:
            all_traces_list.extend(trace_set.traces)
        del result

        all_traces = traceset.TraceSet(name="all_traces")
        all_traces.set_traces(all_traces_list)

        num_validation_traces = 60000

        # Perform t-fold base test
        ranks = np.zeros(shape=(10, int(num_validation_traces / rank_trace_step))) + 256
        confidences = np.zeros(shape=(10, int(num_validation_traces / rank_trace_step)))
        for i in range(0, t):
            print("Fold %d" % i)
            # Randomize trace_sets
            random_indices = np.arange(len(all_traces.traces))
            np.random.shuffle(random_indices)
            validation_traces = np.take(all_traces.traces, random_indices, axis=0)[0:num_validation_traces]

            # Now, evaluate the rank for increasing number of traces from the validation set (steps of 10)
            for j in range(0, int(num_validation_traces / rank_trace_step)):
                subset = traceset.TraceSet(name="all_traces")
                subset.set_traces(validation_traces[0:(j+1)*rank_trace_step])
                subset.window = Window(begin=0, end=len(subset.traces[0].signal))
                subset.windowed = True
                r, c = rank.calculate_traceset_rank(subset, 2, subset.traces[0].key[2])
                ranks[i][j] = r
                confidences[i][j] = c
                print("Rank is %d with confidence %f (%d traces)" % (r, c, (j+1)*rank_trace_step))

        print(ranks)
        print(confidences)
        data_to_save = {
            'ranks': ranks,
            'confidences': confidences,
            'rank_trace_step': rank_trace_step,
            'folds': t,
            'num_validation_traces': num_validation_traces,
            'conf': conf,
        }
        directory = "./models/%s" % conf_to_id(conf)
        os.makedirs(directory, exist_ok=True)
        pickle.dump(data_to_save, open("%s/basetest-t-ranks.p" % directory, "wb"))
    else:
        logger.error("Must provide a list of trace set paths to worker!")
        return None


@app.task(bind=True)
def aitrain(self, training_trace_set_paths, validation_trace_set_paths, conf):
    resolve_paths(training_trace_set_paths)  # Get absolute paths for training set
    resolve_paths(validation_trace_set_paths)  # Get absolute paths for validation set

    # Hardcoded stuff
    subtype = 'custom'

    # Determine type of model to train
    model_type = get_conf_model_type(conf)  # TODO: Refactor 'name' to 'model_type' everywhere and let user specify modeltype in [] params of "train" activity

    # Select training iterator (gathers data, performs augmentation and preprocessing)
    training_iterator, validation_iterator = aiiterators.get_iterators_for_model(model_type, training_trace_set_paths, validation_trace_set_paths, conf, hamming=conf.hamming, subtype=subtype, request_id=self.request.id)

    print("Getting shape of data...")
    x, _ = training_iterator.next()
    input_shape = x.shape[1:]  # Shape excluding batch
    print("Shape of data to train: %s" % str(input_shape))

    # Select model
    model = None
    if conf.update or conf.testrank:  # Load existing model to update or test
        model = ai.AI(conf, model_type)
        model.load()
    else:  # Create new model
        if model_type == 'aicorrnet':
            model = ai.AICorrNet(conf, input_dim=input_shape[0])
        elif model_type == 'aishacpu':
            model = ai.AISHACPU(conf, input_shape=input_shape, subtype=subtype)
        elif model_type == 'aishacc':
            model = ai.AISHACC(conf, input_shape=input_shape)
        elif model_type == 'aiascad':
            model = ai.AIASCAD(conf, input_shape=input_shape)

    if conf.tfold:  # Train t times and generate tfold rank summary
        model.train_t_fold(training_iterator, batch_size=conf.batch_size, epochs=conf.epochs, num_train_traces=45000, t=10, rank_trace_step=10, conf=conf)
    elif conf.testrank:  # TODO this should not be in aitrain; refactor
        model.test_fold(validation_iterator, rank_trace_step=10, conf=conf, max_traces=5000)
    else:  # Train once
        model.train_generator(training_iterator, validation_iterator, epochs=conf.epochs, workers=1)


@app.task(bind=True)
def salvis(self, trace_set_paths, model_type, vis_type, conf):
    """
    Visualize the salience of an AI.
    :param self:
    :param trace_set_paths: List of trace set paths to be used as possible examples for the saliency visualization.
    :param model_type: Type of model to load for this configuration.
    :param conf: Configuration of the model (required preprocessing actions, architecture, etc.).
    :return:
    """
    logger.info("Loading model")
    model = ai.AI(conf, model_type)
    model.load()

    logger.info("Resolving traces")
    resolve_paths(trace_set_paths)
    examples_iterator, _ = aiiterators.get_iterators_for_model(model_type, trace_set_paths, [], conf, hamming=conf.hamming, subtype=None, request_id=self.request.id)

    logger.info("Retrieving batch of examples")
    examples_batch = np.array([x.signal for x in examples_iterator.get_all_as_trace_set(limit=2).traces])
    examples_batch = examples_batch[0:conf.saliency_num_traces, :]
    if len(examples_batch.shape) != 2:
        raise ValueError("Expected 2D examples batch for saliency visualization.")

    if conf.saliency_remove_bias:
        examples_batch = examples_batch[:, 1:]
    kerasvis = True if vis_type == 'kerasvis' else False

    return SalvisResult(examples_batch=examples_batch, gradients=saliency.get_gradients(conf, model, examples_batch, kerasvis=kerasvis))