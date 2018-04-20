# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import keras
import numpy as np
import keras.backend as K
import pickle
import tensorflow as tf

from lut import sbox

class RankCallback(keras.callbacks.Callback):
    """
    Calculate the rank after passing a trace set through the model. A trace_set must be supplied
    since to calculate the rank we need metadata about the traces as well (plaintext and key).
    """

    def __init__(self, log_dir):
        #super(RankCallback, self).__init__()
        self.trace_set = None
        self.writer = tf.summary.FileWriter(log_dir)

    def set_trace_set(self, trace_set):
        self.trace_set = trace_set

    def on_epoch_begin(self, epoch, logs=None):
        if not self.trace_set is None:
            x = np.expand_dims(np.array([trace.signal for trace in self.trace_set.traces]), axis=-1)
            predictions = self.model.predict(x) # Output: [?, 256]
            key_scores = np.zeros(256)

            #
            for i in range(0, len(self.trace_set.traces)):
                trace = self.trace_set.traces[i]
                plaintext_byte = trace.plaintext[2]  # ASCAD chosen plaintext byte
                key_true = trace.key[2]
                for key_guess in range(0, 256):
                    key_prob = predictions[i][sbox[plaintext_byte ^ key_guess]]
                    key_scores[key_guess] += -np.log(key_prob + K.epsilon())  # Lower = better

            sorted_score_indices = np.argsort(key_scores)
            key_ranks = np.zeros(256, dtype=int)
            for i in range(0, 256):
                key_ranks[sorted_score_indices[i]] = i
            best_key = np.argmin(key_ranks)
            print("True key is %02x" % key_true)
            print("Best key is %02x" % best_key)
            print("Rank is %d" % key_ranks[key_true])

            # Add to TensorBoard
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = key_ranks[key_true]
            summary_value.tag = 'rank'
            self.writer.add_summary(summary, epoch)
            self.writer.flush()
        else:
            print("Warning: no trace_set supplied to RankCallback")

def rank(y_true, y_pred):
    pickle.dump(y_true, open('/tmp/test.p', 'wb'))
    print(np.argmax(y_true, axis=1))
    true_key_scores = -np.sum(np.log(y_true + K.epsilon()), axis=0)
    true_key_scores_rank = np.argsort(true_key_scores)
    true_key = np.argmin(true_key_scores_rank)
    print("True key is %02x" % true_key)
    print("True key score is %f" % true_key_scores[true_key])
