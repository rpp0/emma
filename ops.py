# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

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
from ai import AIMemCopyDirect, AICorrNet

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
            result.ai = AIMemCopyDirect(input_dim=len(trace_set.traces[0].signal), hamming=True)

        signals = np.array([trace.signal for trace in trace_set.traces])
        values = np.array([hw[trace.plaintext[0]] for trace in trace_set.traces])
        logger.warning("Training %d signals" % len(signals))
        result.ai.train(signals, values)
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

@op('corrtrain')
def corrtrain_trace_set(trace_set, result, conf=None, params=None):
    logger.info("corrtrain %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        key = [0x0E, 0xEB, 0xA7, 0x43, 0x00, 0x9D, 0x67, 0xD2, 0xE5, 0x63, 0xCF, 0x4C, 0x5C, 0xB0, 0x77, 0xCB]
        if result.ai is None:
            logger.debug("Initializing Keras")
            result.ai = AICorrNet(input_dim=len(trace_set.traces[0].signal))
            result._data['hackx'] = []
            for i in range(0, 16):
                result._data['hacky' + str(i)] = []

        """
        signals = np.array([trace.signal for trace in trace_set.traces], dtype=float)
        values = np.array([hw[sbox[trace.plaintext[0] ^ 0x0e]] for trace in trace_set.traces], dtype=float)
        logger.warning("Training %d signals" % len(signals))
        result.ai.train(signals, values)
        """

        if len(result._data['hackx']) > 19000:
            signals = np.array(result._data['hackx'], dtype=float)
            values = np.zeros([len(result._data['hacky0']), 16])
            for i in range(0, 16):
                values[:,i] = np.array(result._data['hacky' + str(i)], dtype=float)
            print(values.shape)
            print(values)
            logger.warning("Training %d signals" % len(signals))
            result.ai.train(signals, values)

            result._data['hackx'] = []
            for i in range(0, 16):
                result._data['hacky'+ str(i)] = []
        else:
            result._data['hackx'].extend([trace.signal for trace in trace_set.traces])
            # TODO New idea: pass model hw values for ALL keys (as an extra dimension) to AICorrNet. Then make the loss function so that the correlation loss function minimizes the loss of ALL models
            for i in range(0, 16):
                result._data['hacky' + str(i)].extend([hw[sbox[trace.plaintext[i] ^ key[i]]] for trace in trace_set.traces])
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")

@op('corrtest')
def corrtest_trace_set(trace_set, result, conf=None, params=None):
    logger.info("corrtest %s" % (str(params) if not params is None else ""))
    if trace_set.windowed:
        if result.ai is None:
            logger.debug("Loading Keras")
            result.ai = AICorrNet(input_dim=len(trace_set.traces[0].signal))
            result.ai.load()

        for trace in trace_set.traces:
            trace.signal = result.ai.predict(np.array([trace.signal], dtype=float))

        trace_set.window = Window(begin=0, end=len(trace_set.traces[0].signal))
        trace_set.windowed = True
    else:
        logger.error("The trace set must be windowed before training can take place because a fixed-size input tensor is required by Tensorflow.")

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

@app.task(bind=True)
def work(self, trace_set_paths, conf):
    '''
    Actions to be performed by workers on the trace set given in trace_set_path.
    '''

    if type(trace_set_paths) is list:
        result = EMResult(task_id=self.request.id) # TODO init this from within Task subclass
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

            # Count progress
            num_done += 1

        result.ai = None  # AI cannot be pickled for further processing; store separately
        return result
    else:
        logger.error("Must provide a list of trace set paths to worker!")
        return None
