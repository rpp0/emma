# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import keras
import numpy as np
import keras.backend as K
import tensorflow as tf
import ops

from lut import sbox
from traceset import TraceSet
from argparse import Namespace
from emutils import Window, conf_get_action
from emresult import EMResult
from leakagemodels import LeakageModelType
from aiinputs import AIInput


class RankCallbackBase(keras.callbacks.Callback):
    """
    Calculate the rank after passing a trace set through the model. A trace_set must be supplied
    since to calculate the rank we need metadata about the traces as well (plaintext and key).
    """
    def __init__(self, conf, log_dir, save_best=True, save_path='/tmp/model.h5'):
        self.trace_set = None
        self.writer = tf.summary.FileWriter(log_dir)
        self.save_best = save_best
        self.best_rank = 256
        self.best_confidence = 0

        self.metric_freq = conf.metric_freq
        self.cnn = conf.cnn
        self.key_low = conf.key_low
        self.key_high = conf.key_high
        self.conf = conf

        if not save_path is None:
            self.save_path = "%s-bestrank.h5" % save_path.rpartition('.')[0]

    def set_trace_set(self, trace_set):
        self.trace_set = trace_set

    def _write_rank(self, epoch, rank, confidence, tag='unknown'):
        """
        DEPRECATED. Is now added to log and added to Tensorboard automatically
        """
        # Add to TensorBoard
        summary_rank = tf.Summary()
        summary_conf = tf.Summary()

        summary_rank_value = summary_rank.value.add()
        summary_rank_value.simple_value = rank
        summary_rank_value.tag = 'rank ' + tag

        summary_conf_value = summary_conf.value.add()
        summary_conf_value.simple_value = confidence
        summary_conf_value.tag = 'confidence ' + tag

        self.writer.add_summary(summary_rank, epoch)
        self.writer.add_summary(summary_conf, epoch)
        self.writer.flush()

    def _save_best_rank_model(self, rank, confidence):
        """
        Saves the model to disk if the rank is lower and confidence is higher than the previous best.
        :param rank:
        :param confidence:
        :return:
        """
        # Save
        if self.save_best and rank <= self.best_rank:
            self.best_rank = rank
            if confidence >= self.best_confidence:
                self.best_confidence = confidence

                self.model.save(self.save_path)
            return True
        return False


class ProbRankCallback(RankCallbackBase):
    """
    RankCallback that assumes the model outputs a probability for each key byte [?, 256].
    """

    def __init__(self, conf, log_dir, save_best=True, save_path='/tmp/model.h5'):
        super(ProbRankCallback, self).__init__(conf, log_dir, save_best, save_path)

    def on_epoch_end(self, epoch, logs=None):
        if not self.trace_set is None:
            x = np.expand_dims(np.array([trace.signal for trace in self.trace_set.traces]), axis=-1)
            predictions = self.model.predict(x) # Output: [?, 256]
            key_scores = np.zeros(256)

            #
            for i in range(0, len(self.trace_set.traces)):
                trace = self.trace_set.traces[i]
                plaintext_byte = trace.plaintext[2]  # ASCAD chosen plaintext byte
                key_true = trace.key[2] # TODO show for all keys
                for key_guess in range(0, 256):
                    key_prob = predictions[i][sbox[plaintext_byte ^ key_guess]]
                    key_scores[key_guess] += np.log(key_prob + K.epsilon())

            # TODO UNTESTED
            ranks = calculate_ranks(key_scores)
            rank, confidence = get_rank_and_confidence(ranks, key_scores, key_true)
            #self._write_rank(epoch, rank, confidence, '%d' % (i-1))
            self._save_best_rank_model(rank, confidence)
        else:
            print("Warning: no trace_set supplied to RankCallback")


class CorrRankCallback(RankCallbackBase):
    """
    RankCallback that assumes the model an encoding that is highly correlated with the true key bytes.
    """

    def __init__(self, conf, log_dir, save_best=True, save_path='/tmp/model.h5'):
        super(CorrRankCallback, self).__init__(conf, log_dir, save_best, save_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if epoch % self.metric_freq != 0 or epoch == 0:
            return
        if self.trace_set is not None:
            # Fetch inputs from trace_set
            x = AIInput(self.conf).get_trace_set_inputs(self.trace_set)

            if self.cnn:
                x = np.expand_dims(x, axis=-1)

            encodings = self.model.predict(x)  # Output: [?, 16]

            # Store encodings as fake traceset
            keys = np.array([trace.key for trace in self.trace_set.traces])
            plaintexts = np.array([trace.plaintext for trace in self.trace_set.traces])
            fake_ts = TraceSet(traces=encodings, plaintexts=plaintexts, keys=keys, name="fake_ts")
            fake_ts.window = Window(begin=0, end=encodings.shape[1])
            fake_ts.windowed = True

            for i in range(self.key_low, self.key_high):
                if len(set(keys[:, i])) > 1:
                    print("Warning: nonidentical key bytes detected. Skipping rank calculation")
                    print("Subkey %d:" % i)
                    print(keys[:, i])
                    break
                rank, confidence = calculate_traceset_rank(fake_ts, i, keys[0][i], self.conf)  # TODO: It is assumed here that all true keys of the test set are the same
                self._save_best_rank_model(rank, confidence)
                logs['rank %d' % i] = rank
                logs['confidence %d' % i] = confidence
            #self._save_best_rank_model(np.mean(ranks))
        else:
            print("Warning: no trace_set supplied to RankCallback")


def calculate_traceset_rank(trace_set, key_index, true_key, orig_conf):
    conf = Namespace(subkey=key_index, leakage_model=orig_conf.leakage_model, key_low=orig_conf.key_low, key_high=orig_conf.key_high)
    result = EMResult(task_id=None)

    if orig_conf.loss_type == 'categorical_crossentropy':
        ops.pattack_trace_set(trace_set, result, conf, params=None)
        scores = result.probabilities

        key_scores = np.zeros(256)
        for key_guess in range(0, 256):
            key_scores[key_guess] = np.max(scores[key_guess, :])
    else:
        if conf.leakage_model == LeakageModelType.AES_MULTI or conf.leakage_model == LeakageModelType.AES_MULTI_TEST or conf.leakage_model == LeakageModelType.HAMMING_WEIGHT_SBOX_OH or conf.leakage_model == LeakageModelType.KEY_BITS or conf.leakage_model == LeakageModelType.KEY_OH or conf.leakage_model == LeakageModelType.AES_BITS_EX or conf.leakage_model == LeakageModelType.HMAC_BITS or conf.leakage_model == LeakageModelType.HMAC or conf.leakage_model == LeakageModelType.HMAC_HW:
            ops.spattack_trace_set(trace_set, result, conf, params=None)
        else:
            ops.attack_trace_set(trace_set, result, conf, params=None)
        scores = result.correlations
        print("Num entries: %d" % scores._n[0][0])

        # Get maximum correlations over all points and interpret as score
        key_scores = np.zeros(256)
        for key_guess in range(0, 256):
            key_scores[key_guess] = np.max(np.abs(scores[key_guess, :]))

    ranks = calculate_ranks(key_scores)
    rank, confidence = get_rank_and_confidence(ranks, key_scores, true_key)

    return rank, confidence


def calculate_ranks(key_scores):
    assert(key_scores.shape == (256,))
    key_ranks = np.zeros(256, dtype=int)

    sorted_score_indices = np.argsort(key_scores)[::-1]
    for i in range(0, 256):
        key_ranks[i] = sorted_score_indices[i]

    return key_ranks


def get_rank_and_confidence(key_ranks, key_scores, true_key):
    print_rank_top_x(key_ranks, x=5, scores=key_scores)
    print("True key is %02x" % true_key)
    rank = np.int32(list(key_ranks).index(true_key))
    print("==> Rank is %d" % rank)

    confidence = np.float32(key_scores[key_ranks[0]] - key_scores[key_ranks[1]])

    return rank, confidence


def print_rank_top_x(key_ranks, x=5, scores=None):
    print("-----------------------------")
    for i in range(0, x):
        key = key_ranks[i]
        if not scores is None:
            print("Rank %d: %02x (score: %f)" % (i, key, scores[key]))
        else:
            print("Rank %d: %02x" % (i, key))
    print("-----------------------------")
