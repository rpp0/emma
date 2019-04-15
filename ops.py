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
from emutils import Window, conf_to_id, conf_has_op, EMMAException
from celery.utils.log import get_task_logger
from lut import hw, sbox
from emresult import EMResult, SalvisResult
from registry import op
from leakagemodels import LeakageModel
from aiinputs import AIInput
from sklearn.decomposition import PCA
from collections import defaultdict

logger = get_task_logger(__name__)  # Logger


@op('align', optargs=['ref_window_begin', 'ref_window_end', 'prefilter'])
def align_trace_set(trace_set, result, conf, params=None):
    """
    Align a set of traces based on a single reference trace using cross-correlation.
    If a trace is empty, it is discarded.
    """
    logger.info("align %s" % (str(params) if not params is None else ""))
    prefilter = False
    if params is None:  # If no parameters provided, assume percent% max offset
        percent = 0.30
        length = len(conf.reference_signal)
        end = int(length - length*percent)
        begin = int(0 + length*percent)
        window = Window(begin=begin, end=end)
    else:
        window = Window(begin=int(params[0]), end=int(params[1]))
        if len(params) > 2:
            prefilter = bool(params[2])

    logger.info("Aligning %d traces" % len(trace_set.traces))
    aligned_trace_set = []
    reference = conf.reference_signal[window.begin:window.end]

    discarded = 0
    for trace in trace_set.traces:
        aligned_trace = align(trace.signal, reference, cutoff=conf.butter_cutoff, order=conf.butter_order, prefilter=prefilter)
        if not aligned_trace is None:
            trace.signal = aligned_trace
            aligned_trace_set.append(trace)
        else:
            discarded += 1

    if discarded > 0:
        logger.warning("Discarded %d unable to align traces." % discarded)

    trace_set.set_traces(np.array(aligned_trace_set))


@op('trigger_align', optargs=['threshold', 'count'])
def trigger_align_trace_set(trace_set, result, conf, params=None):
    logger.info("trigger_align %s" % (str(params) if not params is None else ""))

    for trace in trace_set.traces:
        s = trace.signal

        cnt = 0
        under_cnt = 0
        state = 0
        begin = 0
        cut = 0
        for sample in s:
            if state == 0:
                if sample < float(params[0]):
                    state = 1
                    begin = cnt
            if state == 1:
                if sample < 0.45:
                    under_cnt += 1
                else:
                    state = 0
                if under_cnt > int(params[1]):
                    cut = begin
                    break
            cnt += 1
        trace.signal = trace.signal[cut:]


@op('invert')
def invert_trace_set(trace_set, result, conf, params=None):
    logger.info("invert %s" % (str(params) if not params is None else ""))

    for trace in trace_set.traces:
        trace.signal = -trace.signal

    if conf.reference_signal is not None:
        conf.reference_signal = -conf.reference_signal


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


@op('ifreq')
def ifreq_trace_set(trace_set, result, conf, params=None):
    logger.info("ifreq %s" % (str(params) if not params is None else ""))

    for trace in trace_set.traces:
        trace.signal = ifreq(trace.signal)

    if conf.reference_signal is not None:
        conf.reference_signal = np.diff(np.unwrap(np.angle(conf.reference_signal)))


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

    if conf.reference_signal is not None:
        conf.reference_signal = np.square(np.abs(np.fft.fft(conf.reference_signal)))


def tspectogram_trace_set(trace_set, result, conf, params=None):
    logger.info("tspec %s" % (str(params) if not params is None else ""))
    if not trace_set.windowed:
        raise EMMAException("Trace set should be windowed")

    # Check params
    if params is not None:
        if len(params) == 1:
            nfft = int(params[0])
        elif len(params) == 2:
            nfft = int(params[0])
            noverlap = int(nfft * int(params[1]) / 100.0)

    print("TODO")


def detect_peaks(signal, spread, num_peaks=8):
    max_end = 0
    spread += int(spread / 2)
    signal_copy = np.copy(signal)

    for i in range(num_peaks):
        peak = np.argmax(signal_copy)
        start = max(0, peak-spread)
        end = min(len(signal_copy), peak+spread)
        if end > max_end:
            max_end = end
        for j in range(start, end):
            signal_copy[j] = 0

    # return signal_copy
    return max_end


@op('sync')
def sync_trace_set(trace_set, result, conf, params=None):
    logger.info("sync %s" % (str(params) if params is not None else ""))
    nops_per_sample = 600
    nops_per_pulse = 5000000
    samples_per_pulse = nops_per_pulse / nops_per_sample
    pulse_edge_size = int(samples_per_pulse / 2)
    corr_signal = np.array([1.0] * pulse_edge_size) + ([0.0] * pulse_edge_size)

    for trace in trace_set.traces:
        correlation = np.correlate(ifreq(trace.signal), corr_signal)
        max_end = detect_peaks(correlation, pulse_edge_size)
        trace.signal = trace.signal[max_end:]
        #trace.signal = correlation


@op('abs')
def magnitude_trace_set(trace_set, result, conf, params=None):
    """
    Calculate the magnitude of the signals in trace_set
    """
    logger.info("abs %s" % (str(params) if not params is None else ""))

    for trace in trace_set.traces:
        trace.signal = np.abs(trace.signal)

    if conf.reference_signal is not None:
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

    if conf.reference_signal is not None:
        conf.reference_signal = np.fft.fft(conf.reference_signal)


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


@op('filter', optargs=['btype', 'cutoff', 'order'])
def filter_trace_set(trace_set, result, conf, params=None):
    """
    Apply a Butterworth filter to the traces.
    """
    logger.info("filter %s" % (str(params) if params is not None else ""))

    butter_type = conf.butter_type
    butter_fs = conf.butter_fs
    butter_order = conf.butter_order
    butter_cutoff = conf.butter_cutoff

    if params is not None:
        if len(params) >= 1:
            butter_type = str(params[0])
        if len(params) >= 2:
            butter_cutoff = float(params[1])
        if len(params) >= 3:
            butter_order = int(params[2])

    for trace in trace_set.traces:
        trace.signal = butter_filter(trace.signal,
                                     order=butter_order,
                                     cutoff=butter_cutoff,
                                     filter_type=butter_type,
                                     fs=butter_fs)

    if conf.reference_signal is not None:
        conf.reference_signal = butter_filter(conf.reference_signal,
                                              order=butter_order,
                                              cutoff=butter_cutoff,
                                              filter_type=butter_type,
                                              fs=butter_fs)


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

    output_path = os.path.join(conf.datasets_path, conf.dataset_id + "-pre")

    logger.info("Saving trace set %s to: %s" % (trace_set.name, output_path))
    os.makedirs(output_path, exist_ok=True)
    trace_set.save(output_path, fmt=conf.outform)

    if conf.outform == 'cw':
        # Update the corresponding config file
        # Deprecated
        # emio.update_cw_config(conf.outpath, trace_set, {"numPoints": len(conf.reference_signal)})
        pass


@op('attack')
def attack_trace_set(trace_set, result, conf=None, params=None):
    """
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    """
    logger.info("attack %s" % (str(params) if not params is None else ""))

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return

    if trace_set.num_traces <= 0:
        logger.warning("Skipping empty trace set.")
        return

    # Init if first time
    if result.correlations is None:
        result.correlations = CorrelationList([256, trace_set.window.size])

    hypotheses = np.empty([256, trace_set.num_traces])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    leakage_model = LeakageModel(conf)
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i] = leakage_model.get_trace_leakages(trace=trace_set.traces[i], subkey_start_index=conf.subkey, key_hypothesis=subkey_guess)

    # 2. Given point j of trace i, calculate the correlation between all hypotheses
    for j in range(0, trace_set.window.size):
        # Get measurements (columns) from all traces
        measurements = np.empty(trace_set.num_traces)
        for i in range(0, trace_set.num_traces):
            measurements[i] = trace_set.traces[i].signal[j]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Update correlation
            result.correlations.update((subkey_guess, j), hypotheses[subkey_guess, :], measurements)


@op('groupkeys')
def groupkeys_trace_set(trace_set, result, conf=None, params=None):
    """
    Group traces by leakage value and return the mean trace for this leakage.
    :param trace_set: 
    :param result: 
    :param conf: 
    :param params: 
    :return: 
    """
    logger.info("groupkeys %s" % (str(params) if not params is None else ""))

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping groupkeys.")
        return

    if result.means is None:
        result.means = defaultdict(lambda: [])

    leakage_model = LeakageModel(conf)
    tmp = defaultdict(lambda: [])
    for trace in trace_set.traces:
        leakage = leakage_model.get_trace_leakages(trace, conf.subkey)
        if isinstance(leakage, list):
            for leakage_index in range(len(leakage)):
                key = "(%d,%02x)" % (leakage_index, leakage[leakage_index])
                tmp[key].append(trace.signal)
        else:
            tmp["%02x" % leakage].append(trace.signal)

    for key, traces in tmp.items():
        all_traces = np.array(traces)
        print("Mean of %d traces for leakage %s (subkey %d)" % (all_traces.shape[0], key, conf.subkey))
        result.means[key].append(np.mean(all_traces, axis=0))


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
    leakage_model = LeakageModel(conf)
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i] = leakage_model.get_trace_leakages(trace=trace_set.traces[i], subkey_start_index=conf.subkey, key_hypothesis=subkey_guess)

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


@op('spattack')
def spattack_trace_set(trace_set, result, conf=None, params=None):
    logger.info("spattack %s" % (str(params) if not params is None else ""))

    num_keys = conf.key_high - conf.key_low
    num_outputs_per_key = LeakageModel.get_num_outputs(conf) // num_keys

    # Init if first time
    if result.correlations is None:
        result.correlations = CorrelationList([256, 1])  # We only have 1 output point (correlation)

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return

    if trace_set.num_traces <= 0:
        logger.warning("Skipping empty trace set.")
        return

    hypotheses = np.empty([256, trace_set.num_traces, num_outputs_per_key])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    leakage_model = LeakageModel(conf)
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i, :] = leakage_model.get_trace_leakages(trace=trace_set.traces[i], subkey_start_index=conf.subkey, key_hypothesis=subkey_guess)

    # 2. Given point j of trace i, calculate the correlation between all hypotheses
    for i in range(0, trace_set.num_traces):
        k = conf.subkey - conf.key_low

        # Get measurements (columns) from all traces for this subkey
        measurements = trace_set.traces[i].signal[num_outputs_per_key*k:num_outputs_per_key*(k+1)]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Update correlation
            result.correlations.update((subkey_guess, 0), hypotheses[subkey_guess, i, :], measurements)


# TODO: Duplicate code, fix me
# TODO: Write unit test
@op('pattack')
def pattack_trace_set(trace_set, result, conf=None, params=None):
    logger.info("pattack %s" % (str(params) if not params is None else ""))

    num_keys = conf.key_high - conf.key_low
    num_outputs_per_key = LeakageModel.get_num_outputs(conf) // num_keys

    # Init if first time
    if result.probabilities is None:
        result.probabilities = np.zeros([256, 1])  # We have 256 probabilities for values for 1 subkey

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return

    if trace_set.num_traces <= 0:
        logger.warning("Skipping empty trace set.")
        return

    hypotheses = np.empty([256, trace_set.num_traces, num_outputs_per_key])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    leakage_model = LeakageModel(conf)
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i, :] = leakage_model.get_trace_leakages(trace=trace_set.traces[i], subkey_start_index=conf.subkey, key_hypothesis=subkey_guess)

    # 2. Given point j of trace i, calculate the correlation between all hypotheses
    for i in range(0, trace_set.num_traces):
        k = conf.subkey - conf.key_low

        # Get measurements (columns) from all traces for this subkey
        measurements = trace_set.traces[i].signal[num_outputs_per_key*k:num_outputs_per_key*(k+1)]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Get sbox[p ^ guess]
            hypo = np.argmax(hypotheses[subkey_guess, i])

            # Get probability of this hypothesis
            proba = measurements[hypo]

            # Update probabilities
            result.probabilities[subkey_guess, 0] += np.log(proba + 0.000001)


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


@op('select', optargs=['select_filename'])
def select_trace_set(trace_set, result, conf=None, params=None):
    """
    Select elements from traces based on provided indices.
    """
    logger.info("select %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if params is None:
            filename = "selection.p"
        else:
            filename = str(params[0])

        selection = pickle.load(open(filename, "rb"))
        if len(selection) == trace_set.window.size:
            for trace in trace_set.traces:
                trace.signal = trace.signal[selection]
        else:
            logger.error("Select length is not equal to signal length.")
    else:
        logger.error("The trace set must be windowed before selection can be applied.")


@op('sum')
def sum_trace_set(trace_set, result, conf=None, params=None):
    logger.info("sum %s" % (str(params) if not params is None else ""))
    for trace in trace_set.traces:
        trace.signal = np.array([np.sum(trace.signal)])

    trace_set.windowed = True
    trace_set.window = Window(begin=0, end=1)


@op('pca')
def pca_trace_set(trace_set, result, conf=None, params=None):
    logger.info("pca %s" % (str(params) if not params is None else ""))

    if result.pca is None:
        if params is None:
            params = ['manifest.emcap']

        with open(params[0], 'rb') as f:  # TODO fix path to make this more general (param?)
            manifest = pickle.load(f)
            result.pca = manifest['pca']

    for trace in trace_set.traces:
        trace.signal = result.pca.transform([trace.signal])[0]
        assert(len(trace.signal) == result.pca.n_components)

    trace_set.windowed = True
    trace_set.window = Window(begin=0, end=result.pca.n_components)


@op('corrtest', id_override="")
def corrtest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("corrtest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        # Get params
        if params is None:
            model_type = "aicorrnet"  # TODO model_type can be inferred from conf. Therefore change AI to only require conf.
        else:
            model_type = str(params[0])

        if result.ai is None:
            result.ai = ai.AI(conf, model_type)
            result.ai.load()

        # Fetch inputs from trace_set
        x = AIInput(conf).get_trace_set_inputs(trace_set)

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
        logger.error("The trace set must be windowed before testing can take place because a fixed-size input tensor is required by Tensorflow.")


@op('classify')
def classify_trace_set(trace_set, result, conf=None, params=None):
    logger.info("classify %s" % (str(params) if not params is None else ""))

    if trace_set.windowed:
        leakage_model = LeakageModel(conf)

        for trace in trace_set.traces:
            true_value = np.argmax(leakage_model.get_trace_leakages(trace, conf.subkey))  # Get argmax of one-hot true label
            predicted_value = np.argmax(trace.signal)  # Get argmax of prediction from corrtest (previous step)
            result.labels.append(true_value)
            result.predictions.append(predicted_value)
            logprobs = ai.softmax_np(np.array(trace.signal))
            result.logprobs.append(list(logprobs))
    else:
        logger.error("The trace set must be windowed before classification can take place because a fixed-size input tensor is required by Tensorflow.")


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
        # TODO this can be cleaned up
        if conf_has_op(conf, 'attack') or conf_has_op(conf, 'memattack') or conf_has_op(conf, 'spattack'):
            # Get size of correlations
            shape = to_merge[0].correlations._n.shape  # TODO fixme init hetzelfde als in attack

            # Init result
            result.correlations = CorrelationList(shape)

            # Start merging
            for m in to_merge:
                result.correlations.merge(m.correlations)
        elif conf_has_op(conf, 'dattack'):  # TODO just check for presence of to_merge.distances instead of doing this
            shape = to_merge[0].distances._n.shape
            result.distances = DistanceList(shape)

            for m in to_merge:
                result.distances.merge(m.distances)
        elif conf_has_op(conf, 'pattack'):
            shape = to_merge[0].probabilities.shape
            result.probabilities = np.zeros(shape)

            for m in to_merge:
                result.probabilities += m.probabilities
        elif conf_has_op(conf, 'keyplot'):
            result.means = {}

            tmp = defaultdict(lambda: [])
            for m in to_merge:
                for key, mean_traces in m.means.items():
                    tmp[key].extend(mean_traces)

            for key, mean_traces in tmp.items():
                all_traces = np.array(mean_traces)
                print("Merging %d traces for subkey value %s" % (all_traces.shape[0], key))
                result.means[key] = np.mean(all_traces, axis=0)

        # Clean up tasks
        if conf.remote:
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
        if action.op in registry.operations:
            registry.operations[action.op](trace_set, result, conf=conf, params=action.params)
        else:
            if action.op not in registry.activities:
                logger.warning("Ignoring unknown op '%s'." % action.op)

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
        # trace_set_name = basename(trace_set_path)
        logger.info("Processing '%s' (%d/%d)" % (trace_set_path, num_done, num_todo))

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
    if action.op == 'corrtrain':
        return 'aicorrnet'
    elif action.op == 'shacputrain':
        return 'aishacpu'
    elif action.op == 'shacctrain':
        return 'aishacc'
    elif action.op == 'ascadtrain':
        return 'aiascad'
    elif action.op == 'autoenctrain':
        return 'autoenc'
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
                r, c = rank.calculate_traceset_rank(subset, 2, subset.traces[0].key[2], conf)
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
        elif model_type == 'autoenc':
            model = ai.AutoEncoder(conf, input_dim=input_shape[0])
        else:
            raise EMMAException("Unknown model type %s" % model_type)
    logger.info(model.info())

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
    trace_set = examples_iterator.get_all_as_trace_set(limit=int(conf.saliency_num_traces/256))
    examples_batch = AIInput(conf).get_trace_set_inputs(trace_set)
    examples_batch = examples_batch[0:conf.saliency_num_traces, :]
    if len(examples_batch.shape) != 2:
        raise ValueError("Expected 2D examples batch for saliency visualization.")

    if conf.saliency_remove_bias:
        examples_batch = examples_batch[:, 1:]
    kerasvis = True if vis_type == 'kerasvis' else False

    return SalvisResult(examples_batch=examples_batch, gradients=saliency.get_gradients(conf, model, examples_batch, kerasvis=kerasvis))


@app.task(bind=True)
def optimize_capture(self, trace_set_paths, conf):
    """
    Apply PCA in order to obtain transformation that lowers the dimensionality of the input data.

    :param self:
    :param trace_set_paths: List of trace set paths to be used in the PCA fit
    :param conf: EMMA configuration blob
    :return:
    """

    logger.info("Resolving traces")
    resolve_paths(trace_set_paths)

    logger.info("Performing actions")
    result = EMResult()
    process_trace_set_paths(result, trace_set_paths, conf, request_id=None, keep_trace_sets=True)

    logger.info("Extracting signals")
    signals_to_fit = []
    for trace_set in result.trace_sets:
        if not trace_set.windowed:
            logger.warning("Skipping trace_set because not windowed")
            continue

        signals_to_fit.extend([trace.signal for trace in trace_set.traces])
    del result
    signals_to_fit = np.array(signals_to_fit)
    print(signals_to_fit.shape)

    logger.info("Performing PCA")
    pca = PCA(n_components=256)
    pca.fit(signals_to_fit)
    print(pca.explained_variance_ratio_)

    import visualizations
    dummy = traceset.TraceSet(name="test")
    traces = [traceset.Trace(signal=x, key=None, plaintext=None, ciphertext=None, mask=None) for x in pca.components_]
    dummy.set_traces(traces)
    visualizations.plot_trace_sets(np.array([0]), [dummy])
    print(pca.singular_values_)

    logger.info("Writing manifest")
    emio.write_emcap_manifest(conf, pca)
