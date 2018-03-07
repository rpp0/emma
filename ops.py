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
from ai import AIMemCopyDirect, AICorrNet, AISHACPU, AI, AISHACC

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
    if params is None:  # If no parameters provided, window according to reference signal
        window = Window(begin=0, end=len(conf.reference_signal))
    else:
        window = Window(begin=int(params[0]), end=int(params[1]))

    for trace in trace_set.traces:
        length_diff = len(trace.signal[window.begin:]) - window.size
        # Pad or cut
        if length_diff < 0:
            trace.signal = np.lib.pad(trace.signal[window.begin:], (0, abs(length_diff)), 'constant', constant_values=(0.0))
        else:
            trace.signal = trace.signal[window.begin:window.end]
        assert(len(trace.signal) == window.size)

        # Apply window
        if conf.windowing_method == 'rectangular':
            continue # Already cut rectangularly
        elif conf.windowing_method == 'kaiser':
            trace.signal = trace.signal * np.kaiser(window.size, 14)
        elif conf.windowing_method == 'blackman':
            trace.signal = trace.signal * np.blackman(window.size)
        else:
            logger.warning("Requested unknown windowing method '%d'. Skipping." % conf.windowing_method)
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

@op('plot')
def plot_trace_set(trace_set, result, conf=None, params=None):
    '''
    Plot each trace in a trace set using Matplotlib
    '''
    logger.info("plot %s" % (str(params) if not params is None else ""))
    for trace in trace_set.traces:
        plt.plot(range(0, len(trace.signal)), trace.signal)

    plt.title(trace_set.name)
    plt.show()

@op('attack')
def attack_trace_set(trace_set, result, conf=None, params=None):
    '''
    Perform CPA attack on a trace set. Assumes the traces in trace_set are real time domain signals.
    '''
    logger.info("attack %s" % (str(params) if not params is None else ""))

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return
    # Init if first time
    if result.correlations is None:
        result.correlations = CorrelationList([256, trace_set.window.size])

    hypotheses = np.empty([256, trace_set.num_traces])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i] = hw[sbox[trace_set.traces[i].plaintext[conf.subkey] ^ subkey_guess]]  # Model of the power consumption

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
            logger.debug("Loading Keras")
            result._data['state'] = AI("aicorrnet")
            result._data['state'].load()

        for trace in trace_set.traces:
            trace.signal = result._data['state'].predict(np.array([trace.signal], dtype=float))

        trace_set.window = Window(begin=0, end=len(trace_set.traces[0].signal))
        trace_set.windowed = True
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")

@op('shacputest')
def shacputest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("shacputest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result._data['state'] is None:
            logger.debug("Loading Keras")
            result._data['state'] = AI("aishacpu" + ("-hw" if conf.hamming else ""))
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
            result._data['state'] = AI("aishacc" + ("-hw" if conf.hamming else ""))
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
def remote_get_trace_paths(input_path, inform):
    return emio.get_trace_paths(input_path, inform)

@app.task
def remote_get_trace_set(trace_set_path, inform, ignore_malformed):
    return emio.get_trace_set(trace_set_path, inform, ignore_malformed)

class AISignalIteratorBase():
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None):
        self.trace_set_paths = trace_set_paths
        self.conf = conf
        self.batch_size = batch_size
        self.cache = {}
        self.index = 0
        self.values_batch = []
        self.signals_batch = []
        self.request_id = request_id
        self.max_cache = 1000
        self.augment_roll = not self.conf.no_augment_roll

    def __iter__(self):
        return self

    def _preprocess_trace_set(self, trace_set):
        # X
        signals = np.array([trace.signal for trace in trace_set.traces], dtype=float)

        # Y
        values = np.array([trace.plaintext for trace in trace_set.traces], dtype=float)

        return signals, values

    def fetch_features(self, trace_set_path):
        '''
        Fethes the features (raw trace and y-values) for a single trace path.
        '''
        # Memoize
        if trace_set_path in self.cache:
            return self.cache[trace_set_path]

        # Apply actions from work()
        result = process_trace_sets([trace_set_path], self.conf, keep_trace_sets=True, request_id=self.request_id)

        if len(result.trace_sets) > 0:
            signals, values = self._preprocess_trace_set(result.trace_sets[0])  # Since we iterate per path, there will be only 1 result in trace_sets

            # Cache
            if len(self.cache.keys()) < self.max_cache:
                self.cache[trace_set_path] = (signals, values)

            return signals, values
        else:
            return None

    def _augment_roll(self, signals, roll_limit=None):  # TODO unit test!
        roll_limit = roll_limit if not roll_limit is None else len(signals[0,:])
        roll_limit_start = -roll_limit if not roll_limit is None else 0
        logger.debug("Data augmentation: rolling signals")
        num_signals, signal_len = signals.shape
        for i in range(0, num_signals):
            signals[i,:] = np.roll(signals[i,:], np.random.randint(roll_limit_start, roll_limit))
        return signals

    def next(self):
        # Bound checking
        if self.index < 0 or self.index >= len(self.trace_set_paths):
            return None

        while True:
            # Do we have enough samples in buffer already?
            if len(self.signals_batch) > self.batch_size:
                # Get exactly batch_size training examples
                signals_return_batch = np.array(self.signals_batch[0:self.batch_size])
                values_return_batch = np.array(self.values_batch[0:self.batch_size])

                # Keep the remainder for next iteration
                self.signals_batch = self.signals_batch[self.batch_size:]
                self.values_batch = self.values_batch[self.batch_size:]

                # Return
                return signals_return_batch,values_return_batch

            # Determine next trace set path
            trace_set_path = self.trace_set_paths[self.index]
            self.index += 1
            if self.index >= len(self.trace_set_paths):
                self.index = 0

            # Fetch features from selected path
            result = self.fetch_features(trace_set_path)
            if result is None:
                continue
            signals, values = result

            # Augment if enabled
            if self.augment_roll:
                signals = self._augment_roll(signals, roll_limit=1024)

            # Concatenate arrays until batch obtained
            self.signals_batch.extend(signals)
            self.values_batch.extend(values)

    def __next__(self):
        return self.next()

class AICorrSignalIterator(AISignalIteratorBase):
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None):
        super(AICorrSignalIterator, self).__init__(trace_set_paths, conf, batch_size, request_id)

    def _preprocess_trace_set(self, trace_set):
        '''
        Preprocessing specifically for AICorrNet
        '''

        # Get training data
        signals = np.array([trace.signal for trace in trace_set.traces], dtype=float)

        # Get model labels (key bytes to correlate)
        key = [0x0E, 0xEB, 0xA7, 0x43, 0x00, 0x9D, 0x67, 0xD2, 0xE5, 0x63, 0xCF, 0x4C, 0x5C, 0xB0, 0x77, 0xCB]
        values = np.zeros((len(trace_set.traces), len(key)), dtype=float)
        for i in range(len(trace_set.traces)):
            for j in range(len(key)):
                values[i, j] = hw[sbox[trace_set.traces[i].plaintext[j] ^ key[j]]]

        # Normalize key labels: required for correct correlation calculation! Note that x is normalized using batch normalization. In Keras, this function also remembers the mean and variance from the training set batches. Therefore, there's no need to normalize before calling model.predict
        values = values - np.mean(values, axis=0)

        return signals, values

class AISHACPUSignalIterator(AISignalIteratorBase):
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None, hamming=True, subtype='vgg16'):
        super(AISHACPUSignalIterator, self).__init__(trace_set_paths, conf, batch_size, request_id)
        self.hamming = hamming
        self.subtype = subtype

    def _adapt_input_vgg(self, traces):
        batch = []
        for trace in traces:
            side_len = int(np.sqrt(len(trace.signal) / 3.0))
            max_len = side_len * side_len * 3
            image = np.array(trace.signal[0:max_len], dtype=float).reshape(side_len, side_len, 3)
            batch.append(image)
        return np.array(batch)

    def _preprocess_trace_set(self, trace_set):
        '''
        Preprocessing specifically for AISHACPU
        '''

        # Get training data
        if self.subtype == 'vgg16':
            signals = self._adapt_input_vgg(trace_set.traces)
        else:
            signals = np.array([trace.signal for trace in trace_set.traces], dtype=float)

        # Get one-hot labels (bytes XORed with 0x36)
        if self.hamming:
            values = np.zeros((len(trace_set.traces), 9), dtype=float)
        else:
            values = np.zeros((len(trace_set.traces), 256), dtype=float)
        index_to_find = 0  # Byte index of SHA-1 key
        for i in range(len(trace_set.traces)):
            trace = trace_set.traces[i]
            key_byte = trace.plaintext[index_to_find]
            if self.hamming:
                values[i, hw[key_byte ^ 0x36]] = 1.0
            else:
                values[i, key_byte ^ 0x36] = 1.0

        return signals, values

def process_trace_sets(trace_set_paths, conf, request_id=None, keep_trace_sets=False):
    result = EMResult(task_id=request_id)
    num_todo = len(trace_set_paths)
    num_done = 0
    for trace_set_path in trace_set_paths:
        # Get trace name from path
        trace_set_name = basename(trace_set_path)
        logger.info("Processing '%s' (%d/%d)" % (trace_set_name, num_done, num_todo))

        # Load trace
        trace_set = emio.get_trace_set(trace_set_path, conf.inform, ignore_malformed=False)
        if trace_set is None:
            logger.warning("Failed to load trace set %s (got None). Skipping..." % trace_set_path)
            continue

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
                logger.warning("Ignoring unknown op '%s'." % op)

        # Store result
        if keep_trace_sets:
            result.trace_sets.append(trace_set)
        num_done += 1
    return result

@app.task(bind=True)
def work(self, trace_set_paths, conf, keep_trace_sets=False, keep_correlations=True):
    '''
    Actions to be performed by workers on the trace set given in trace_set_path.
    '''

    if type(trace_set_paths) is list:
        result = process_trace_sets(trace_set_paths, conf, request_id=self.request.id, keep_trace_sets=keep_trace_sets)

        if not keep_trace_sets:  # Do not return processed traces
            result.trace_sets = None
        if not keep_correlations:  # Do not return correlations
            result.correlations = None
        result._data['state'] = None  # Never return state

        return result
    else:
        logger.error("Must provide a list of trace set paths to worker!")
        return None

def get_iterators_for_model(model_type, trace_set_paths, conf, batch_size=512, hamming=False, subtype='custom', request_id=None):
    num_validation_trace_sets = 1
    validation_trace_set_paths = trace_set_paths[0:num_validation_trace_sets]
    training_trace_set_paths = trace_set_paths[num_validation_trace_sets:]

    training_iterator = None
    validation_iterator = None
    if model_type == 'corrtrain':
        training_iterator = AICorrSignalIterator(training_trace_set_paths, conf, request_id=self.request.id)
        validation_iterator = AICorrSignalIterator(validation_trace_set_paths, conf, request_id=self.request.id)
    elif model_type == 'shacputrain':
        training_iterator = AISHACPUSignalIterator(training_trace_set_paths, conf, batch_size=512, request_id=request_id, hamming=hamming, subtype=subtype)
        validation_iterator = AISHACPUSignalIterator(training_trace_set_paths, conf, batch_size=512, request_id=request_id, hamming=hamming, subtype=subtype)
    elif model_type == 'shacctrain':
        training_iterator = AISHACPUSignalIterator(training_trace_set_paths, conf, batch_size=512, request_id=request_id, hamming=hamming, subtype='custom')
        validation_iterator = AISHACPUSignalIterator(training_trace_set_paths, conf, batch_size=512, request_id=request_id, hamming=hamming, subtype='custom')
    else:
        logger.error("Unknown training procedure specified.")
        exit(1)

    return training_iterator, validation_iterator

def get_conf_model_type(conf):
    for action in conf.actions:
        if action in ['corrtrain', 'shacputrain', 'shacctrain']:
            return action
    return None

@app.task(bind=True)
def aitrain(self, trace_set_paths, conf):
    logger.debug("Determining post-processed training sample size")

    # Hardcoded stuff
    subtype = 'custom'

    # Determine type of model to train
    model_type = get_conf_model_type(conf)

    # Select training iterator (gathers data, performs augmentation and preprocessing)
    training_iterator, validation_iterator = get_iterators_for_model(model_type, trace_set_paths, conf, hamming=conf.hamming, subtype=subtype, request_id=self.request.id)

    x, _ = training_iterator.next()
    input_shape = x.shape[1:]  # Shape excluding batch
    print("Shape of data to train: %s" % str(input_shape))

    # Select model
    model = None
    if model_type == 'corrtrain':
        model = AICorrNet(input_dim=input_shape[0])
    elif model_type == 'shacputrain':
        model = AISHACPU(input_shape=input_shape, hamming=conf.hamming, subtype=subtype)
    elif model_type == 'shacctrain':
        model = AISHACC(input_shape=input_shape, hamming=conf.hamming)

    logger.debug("Training...")
    model.train_generator(training_iterator, validation_iterator, epochs=900, workers=1)
