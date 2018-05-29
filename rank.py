# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import keras
import numpy as np
import keras.backend as K
import pickle
import tensorflow as tf
import ops

from lut import sbox
from traceset import TraceSet
from argparse import Namespace
from emutils import Window
from emresult import EMResult

class RankCallbackBase(keras.callbacks.Callback):
    """
    Calculate the rank after passing a trace set through the model. A trace_set must be supplied
    since to calculate the rank we need metadata about the traces as well (plaintext and key).
    """
    def __init__(self, log_dir, save_best=True, save_path='/tmp/model.h5', freq=100):
        self.trace_set = None
        self.writer = tf.summary.FileWriter(log_dir)
        self.save_best = save_best
        self.best_rank = 256
        self.freq = freq

        if not save_path is None:
            self.save_path = "%s-bestrank.h5" % save_path.rpartition('.')[0]

    def set_trace_set(self, trace_set):
        self.trace_set = trace_set

    def _write_rank(self, epoch, rank, tag='rank'):
        # Add to TensorBoard
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = rank
        summary_value.tag = tag
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def _save_best_rank_model(self, rank):
        # Save
        if self.save_best and rank <= self.best_rank:
            self.best_rank = rank
            self.model.save(self.save_path)

class ProbRankCallback(RankCallbackBase):
    """
    RankCallback that assumes the model outputs a probability for each key byte [?, 256].
    """

    def __init__(self, log_dir, save_best=True, save_path='/tmp/model.h5'):
        super(ProbRankCallback, self).__init__(log_dir, save_best, save_path)

    def on_epoch_begin(self, epoch, logs=None):
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
                    key_scores[key_guess] += -np.log(key_prob + K.epsilon())  # Lower = better # TODO reverse argsort instead of doing this

            rank = calculate_rank(key_scores, key_true)
            self._write_rank(epoch, rank, 'rank')
            self._save_best_rank_model(rank)
        else:
            print("Warning: no trace_set supplied to RankCallback")

class CorrRankCallback(RankCallbackBase):
    """
    RankCallback that assumes the model an encoding that is highly correlated with the true key bytes.
    """

    def __init__(self, log_dir, save_best=True, save_path='/tmp/model.h5', freq=100):
        super(CorrRankCallback, self).__init__(log_dir, save_best, save_path, freq)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.freq != 0 or epoch == 0:
            return
        if not self.trace_set is None:
            x = np.array([trace.signal for trace in self.trace_set.traces])
            encodings = self.model.predict(x) # Output: [?, 16]

            # Store encodings as fake traceset
            keys = np.array([trace.key for trace in self.trace_set.traces])
            plaintexts = np.array([trace.plaintext for trace in self.trace_set.traces])
            fake_ts = TraceSet(traces=encodings, plaintexts=plaintexts, keys=keys)
            fake_ts.window = Window(begin=0, end=encodings.shape[1])
            fake_ts.windowed = True

            for i in range(2, 3):  # TODO show for all keys
                conf = Namespace(subkey=i)
                result = EMResult(task_id=None)
                ops.attack_trace_set(fake_ts, result, conf, params=None)

                corr_result = result.correlations
                print("Num entries: %d" % corr_result._n[0][0])

                # Get maximum correlations over all points and interpret as score
                key_scores = np.zeros(256)
                for key_guess in range(0, 256):
                    key_scores[key_guess] = -np.max(np.abs(corr_result[key_guess,:]))  # TODO reverse argsort instead of doing this negation

                rank = calculate_rank(key_scores, keys[0][i])  # TODO: Is is assumed here that all keys of the set are the same here
                self._write_rank(epoch, rank, 'rank %d' % i)
                self._save_best_rank_model(rank)
            #self._save_best_rank_model(np.mean(ranks))
        else:
            print("Warning: no trace_set supplied to RankCallback")

def calculate_rank(key_scores, true_key):
    assert(key_scores.shape == (256,))
    key_ranks = np.zeros(256, dtype=int)

    sorted_score_indices = np.argsort(key_scores)
    for i in range(0, 256):
        key_ranks[sorted_score_indices[i]] = i

    print_rank_top_x(key_ranks, x=5, scores=key_scores)
    #best_key = np.argmin(key_ranks)
    print("True key is %02x" % true_key)
    rank = key_ranks[true_key]
    print("==> Rank is %d" % rank)

    return rank

def print_rank_top_x(key_ranks, x=5, scores=None):
    key_indices = range(0, 256)
    zipped = list(zip(key_ranks, key_indices))
    top = sorted(zipped, key=lambda x: x[0])
    print("-----------------------------")
    for i in range(0, x):
        key = top[i][1]
        if not scores is None:
            print("Rank %d: %02x (score: %f)" % (i, key, scores[key]))
        else:
            print("Rank %d: %02x" % (i, key))
    print("-----------------------------")
