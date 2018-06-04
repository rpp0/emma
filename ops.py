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
from emma_worker import app, broker
from dsp import *
from correlationlist import CorrelationList
from functools import wraps
from os.path import join, basename
from emutils import Window
from celery.utils.log import get_task_logger
from lut import hw, sbox
from celery import Task
from emresult import EMResult
from ai import AIMemCopyDirect, AICorrNet, AISHACPU, AI, AISHACC, AIASCAD
from matplotlib.backends.backend_pdf import PdfPages

logger = get_task_logger(__name__)  # Logger
ops = {}  # Op registry
ops_optargs = {}

class EMMATask(Task):
    test = 'a'

def op(name, optargs=None):
    '''
    Defines the @op decorator
    '''
    def decorator(func):
        ops[name] = func
        if not optargs is None:
            ops_optargs[name] = optargs
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@op('align', optargs=['ref_window_begin', 'ref_window_end'])
def align_trace_set(trace_set, result, conf, params=None):
    '''
    Align a set of traces based on a single reference trace using cross-correlation.
    If a trace is empty, it is discarded.
    '''
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
    '''
    Filter traces by key prefix
    '''
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
    '''
    Calculate the spectogram of the trace set.
    '''
    logger.info("spec %s" % (str(params) if not params is None else ""))
    if not trace_set.windowed:
        logger.warning("Taking the FFT of non-windowed traces will result in variable FFT sizes.")

    for trace in trace_set.traces:
        trace.signal = np.square(np.abs(np.fft.fft(trace.signal)))
        #if True: # If real signal
        #    trace.signal = trace.signal[0:int(len(trace.signal) / 2)]

@op('norm')
def normalize_trace_set(trace_set, result, conf, params=None):
    '''
    normalize the signals (amplitudes) in a trace set.
    '''
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

@op('window', optargs=['window_begin', 'window_end'])
def window_trace_set(trace_set, result, conf, params=None):
    '''
    Perform windowing on a specific trace set. See https://en.wikipedia.org/wiki/Window_function#Spectral_analysis
    for a good overview of the effects of the different windowing methods on the PSD of the signal.

    The trace is windowed according to conf.window.size, or according to the size of the reference trace if the
    window is not configured.

    Interesting excerpt: 'What cannot be seen from the graphs is that the rectangular window has the best noise bandwidth, which makes it a good candidate for detecting low-level sinusoids in an otherwise white noise environment. Interpolation techniques, such as zero-padding and frequency-shifting, are available to mitigate its potential scalloping loss.'

    Params: (window start, window end)
    '''
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
    '''
    Apply a Butterworth filter to the traces.
    '''
    logger.info("filter %s" % (str(params) if not params is None else ""))
    for trace in trace_set.traces:
        trace.signal = butter_filter(trace.signal, order=conf.butter_order, cutoff=conf.butter_cutoff)

    conf.reference_signal = butter_filter(conf.reference_signal, order=conf.butter_order, cutoff=conf.butter_cutoff)

@op('rmoutliers')
def rmoutliers_trace_set(trace_set, result, conf, params=None):
    '''
    Remove outliers in terms of amplitude.
    '''
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

@op('save')
def save_trace_set(trace_set, result, conf, params=None):
    '''
    Save the trace set to a file using the output format specified in the conf object.
    '''
    logger.info("save %s" % (str(params) if not params is None else ""))
    if conf.outform == 'cw':
        # Save back to output file
        np.save(join(conf.outpath, trace_set.name + '_traces.npy'), trace_set.traces)

        # Update the corresponding config file
        emio.update_cw_config(conf.outpath, trace_set, {"numPoints": len(conf.reference_signal)})
    elif conf.outform == 'sigmf':  # TODO make SigMF compliant
        count = 1
        for trace in trace_set.traces:
            trace.tofile(join(output_path_gnuradio, "%s-%d.rf32_le" % (trace_set.name, count)))
            count += 1
    else:
        print("Unknown format: %s" % conf.outform)
        exit(1)

@op('plot', optargs=['save'])
def plot_trace_set(trace_set, result, conf=None, params=None):
    '''
    Plot each trace in a trace set using Matplotlib
    '''
    logger.info("plot %s" % (str(params) if not params is None else ""))
    for trace in trace_set.traces:
        plt.plot(range(0, len(trace.signal)), trace.signal)
    plt.plot(range(0, len(conf.reference_signal)), conf.reference_signal, linewidth=2, linestyle='dashed')

    plt.title(trace_set.name)

    if (not params is None) and 'save' in params:
        pp = PdfPages('/tmp/%s.pdf' % trace_set.name)
        pp.savefig()
        pp.close()
        plt.clf()
    else:
        plt.show()

@op('attack')
def attack_trace_set(trace_set, result, conf=None, params=None):
    '''
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    '''
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

@op('memattack')
def memattack_trace_set(trace_set, result, conf=None, params=None):
    logger.info("memattack %s" % (str(params) if not params is None else ""))
    if result.correlations is None:
        result.correlations = CorrelationList([16, 256, trace_set.window.size])

    for byte_idx in range(0, conf.num_subkeys):
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
            result.ai = AIMemCopyDirect(input_dim=len(trace_set.traces[0].signal), hamming=conf.hamming)

        signals = np.array([trace.signal for trace in trace_set.traces])
        values = np.array([hw[trace.plaintext[0]] for trace in trace_set.traces])
        logger.warning("Training %d signals" % len(signals))
        result.ai.train_set(signals, values)
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")

@op('weight', optargs=['weight_filename'])
def weight_trace_set(trace_set, result, conf=None, params=None):
    '''
    Multiply trace signal element-wise with weights stored in a file.
    '''
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
        if result._data['state'] is None:
            logger.warning("Loading model aicorrnet-%s" % conf.model_suffix)
            result._data['state'] = AI("aicorrnet", suffix=conf.model_suffix)
            result._data['state'].load()

        # Fetch signals from traces
        x = np.array([trace.signal for trace in trace_set.traces])
        #import keras.backend as K
        #get_output = K.function([result._data['state'].model.layers[0].input, K.learning_phase()], [result._data['state'].model.layers[-1].output])
        #encodings = get_output([x, 1])[0]
        #encoding_dimensions = result._data['state'].model.layers[-1].output_shape[1]

        # Get encodings of signals
        encodings = result._data['state'].predict(x)

        # Replace original signal with encoding
        assert(encodings.shape[0] == len(trace_set.traces))
        for i in range(0, len(trace_set.traces)):
            trace_set.traces[i].signal = encodings[i]
            #trace_set.traces[i].signal = result._data['state'].predict(np.array([trace_set.traces[i].signal]))  # Without copy, but somewhat slower

        # Adjust window size
        trace_set.window = Window(begin=0, end=encodings.shape[1])
        trace_set.windowed = True
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")

@op('shacputest')
def shacputest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("shacputest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result._data['state'] is None:
            logger.debug("Loading Keras")
            result._data['state'] = AI("aishacpu" + ("-hw" if conf.hamming else ""), suffix=conf.model_suffix)
            result._data['state'].load()

        for trace in trace_set.traces:
            if conf.hamming:
                result._data['labels'].append(hw[trace.plaintext[0] ^ 0x36])
            else:
                result._data['labels'].append(trace.plaintext[0] ^ 0x36)
            result._data['predictions'].append(np.argmax(result._data['state'].predict(np.array([trace.signal], dtype=float))))

@op('shacctest')
def shacctest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("shacctest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result._data['state'] is None:
            logger.debug("Loading Keras")
            result._data['state'] = AI("aishacc" + ("-hw" if conf.hamming else ""), suffix=conf.model_suffix)
            result._data['state'].load()

        for trace in trace_set.traces:
            if conf.hamming:
                result._data['labels'].append(hw[trace.plaintext[0] ^ 0x36])
            else:
                result._data['labels'].append(trace.plaintext[0] ^ 0x36)

            cc_out = result._data['state'].predict(np.array([trace.signal], dtype=float))
            predicted_classes = np.argmax(cc_out, axis=1)
            result._data['predictions'].append(predicted_classes[0])

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

        # Clean up tasks
        for m in to_merge:
            logger.warning("Deleting %s" % m.task_id)
            app.AsyncResult(m.task_id).forget()

        return result
    else:
        return None

@app.task
def remote_get_dataset(dataset, conf=None):
    return emio.get_dataset(dataset, conf=conf)

@app.task
def remote_get_trace_set(trace_set_path, format, ignore_malformed):
    return emio.get_trace_set(trace_set_path, format, ignore_malformed)

def process_trace_set(result, trace_set, conf, request_id=None, keep_trace_sets=False):
    # Perform actions
    for action in conf.actions:
        params = None
        if '[' in action:
            op, _, params = action.rpartition('[')
            params = params.rstrip(']').split(',')
        else:
            op = action
        if op in ops:
            ops[op](trace_set, result, conf=conf, params=params)
        else:
            if not ('train' in action):
                logger.warning("Ignoring unknown op '%s'." % op)

    # Store result
    if keep_trace_sets:
        result.trace_sets.append(trace_set)

def process_trace_set_paths(result, trace_set_paths, conf, request_id=None, keep_trace_sets=False):
    num_todo = len(trace_set_paths)
    num_done = 0
    for trace_set_path in trace_set_paths:
        # Get trace name from path
        trace_set_name = basename(trace_set_path)
        logger.info("Processing '%s' (%d/%d)" % (trace_set_name, num_done, num_todo))

        # Load trace
        trace_set = emio.get_trace_set(trace_set_path, conf.format, ignore_malformed=False)
        if trace_set is None:
            logger.warning("Failed to load trace set %s (got None). Skipping..." % trace_set_path)
            continue

        # Process trace
        process_trace_set(result, trace_set, conf, request_id, keep_trace_sets)

        num_done += 1

def resolve_paths(trace_set_paths):
    '''
    Determine the path on disk based on the location of the database specified in the
    worker's settings file.
    '''
    settings = configparser.RawConfigParser()
    settings.read('settings.conf')
    prefix = settings.get("Datasets", "datasets_path")

    for i in range(0, len(trace_set_paths)):
            # Add prefix to path
            trace_set_paths[i] = join(prefix, trace_set_paths[i])

@app.task(bind=True)
def work(self, trace_set_paths, conf, keep_trace_sets=False, keep_correlations=True):
    '''
    Actions to be performed by workers on the trace set given in trace_set_path.
    '''
    resolve_paths(trace_set_paths)  # Get absolute paths

    if type(trace_set_paths) is list:
        result = EMResult(task_id=self.request.id)  # Keep state and results

        # Process trace set paths and fill in results of analysis
        process_trace_set_paths(result, trace_set_paths, conf, request_id=self.request.id, keep_trace_sets=keep_trace_sets)

        if not keep_trace_sets:  # Do not return processed traces
            result.trace_sets = None
        if not keep_correlations:  # Do not return correlations
            result.correlations = None
        result._data['state'] = None  # Never return state

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
def aitrain(self, training_trace_set_paths, validation_trace_set_paths, conf):
    resolve_paths(training_trace_set_paths)  # Get absolute paths for training set
    resolve_paths(validation_trace_set_paths)  # Get absolute paths for validation set

    # Hardcoded stuff
    subtype = 'custom'

    # Determine type of model to train
    model_type = get_conf_model_type(conf)

    # Select training iterator (gathers data, performs augmentation and preprocessing)
    training_iterator, validation_iterator = aiiterators.get_iterators_for_model(model_type, training_trace_set_paths, validation_trace_set_paths, conf, hamming=conf.hamming, subtype=subtype, request_id=self.request.id)

    x, _ = training_iterator.next()
    input_shape = x.shape[1:]  # Shape excluding batch
    print("Shape of data to train: %s" % str(input_shape))

    # Select model
    model = None
    if conf.update:  # Load existing model to update
        logger.warning("Loading model %s%s" % (model_type, '-' + conf.model_suffix if not conf.model_suffix is None else ''))
        model = AI(model_type, suffix=conf.model_suffix)
        model.load()
    else:  # Create new model
        if model_type == 'aicorrnet':
            model = AICorrNet(input_dim=input_shape[0], suffix=conf.model_suffix)
        elif model_type == 'aishacpu':
            model = AISHACPU(input_shape=input_shape, hamming=conf.hamming, subtype=subtype, suffix=conf.model_suffix)
        elif model_type == 'aishacc':
            model = AISHACC(input_shape=input_shape, hamming=conf.hamming, suffix=conf.model_suffix)
        elif model_type == 'aiascad':
            model = AIASCAD(input_shape=input_shape, suffix=conf.model_suffix)

    logger.debug("Training...")
    model.train_generator(training_iterator, validation_iterator, epochs=conf.epochs, workers=1)
