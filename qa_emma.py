#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# Test suites
# ----------------------------------------------------

from correlationlist import CorrelationList
import ops
import unittest
import numpy as np
import emutils
import tensorflow as tf
import keras.backend as K
from ai import AICORRNET_KEY_LOW, AICORRNET_KEY_HIGH
from traceset import TraceSet
from aiiterators import AICorrSignalIterator
from argparse import Namespace
from rank import CorrRankCallback

class TestCorrelationList(unittest.TestCase):
    def test_update(self):
        test_array = np.array([
            [1, 2],
            [3, 5],
            [4, 5],
            [4, 8]
        ])
        x = test_array[:,0]
        y = test_array[:,1]

        clist1 = CorrelationList(1)
        clist1.update(0, x, y)
        clist2 = CorrelationList([1, 1])
        clist2.update((0,0), x, y)

        # Checks
        self.assertAlmostEqual(clist1[0], np.corrcoef(x, y)[1,0], places=13)
        self.assertAlmostEqual(clist2[0,0], np.corrcoef(x, y)[1,0], places=13)

    def test_merge(self):
        test_array_1 = np.array([
            [1, 2],
            [3, 5],
            [4, 5],
            [4, 8]
        ])

        test_array_2 = np.array([
            [4, 3],
            [5, 4],
            [6, 1],
            [8, 8]
        ])

        test_array_check = np.array([
            [1, 2],
            [3, 5],
            [4, 5],
            [4, 8],
            [4, 3],
            [5, 4],
            [6, 1],
            [8, 8]
        ])

        x1 = test_array_1[:,0]
        y1 = test_array_1[:,1]
        x2 = test_array_2[:,0]
        y2 = test_array_2[:,1]
        x_check = test_array_check[:,0]
        y_check = test_array_check[:,1]

        c1 = CorrelationList(1)
        c1.update(0, x1, y1)

        c2 = CorrelationList(1)
        c2.update(0, x2, y2)

        c3 = CorrelationList(1)
        c3.merge(c1)
        c3.merge(c2)

        self.assertAlmostEqual(c3[0], np.corrcoef(x_check, y_check)[1,0], places=13)

class TestUtils(unittest.TestCase):
    def test_pretty_print_correlations(self):
        pass
        # TODO implement me
        test = CorrelationList([16,256])
        #emutils.pretty_print_correlations(test)

class TestAI(unittest.TestCase):
    def test_corrtrain(self):
        '''
        Artificial example to test AICorrNet and trace processing
        '''

        # ------------------------------
        # Generate data
        # ------------------------------
        traces = [ # Contains abs(trace). Shape = [trace, point]
            [1, 1, 1, -15],
            [-4, 1, 2, -12],
            [10, 1, 3, 8],
            [8, 1, 1, -14],
            [9, 1, -3, 8],
        ]

        plaintexts = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        keys = [
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        # Convert to numpy
        traces = np.array(traces)
        plaintexts = np.array(plaintexts)
        keys = np.array(keys)

        trace_set = TraceSet(name='test', traces=traces, plaintexts=plaintexts, keys=keys)

        # ------------------------------
        # Preprocess data
        # ------------------------------
        it_dummy = AICorrSignalIterator([], Namespace(max_cache=0, augment_roll=False, augment_noise=False, normalize=False, traces_per_set=4), batch_size=10000, request_id=None, stream_server=None)
        x, y = it_dummy._preprocess_trace_set(trace_set)

        # ------------------------------
        # Train and obtain encodings
        # ------------------------------
        model = ops.AICorrNet(4, name="test")
        rank = CorrRankCallback('/tmp/deleteme/', save_best=False, save_path=None, freq=100)
        rank.set_trace_set(trace_set)

        if model.using_regularization:
            print("Warning: cant do correlation loss test because regularizer will influence loss function")
            return

        # Find optimal weights
        print("The x (EM samples) and y (HW[sbox[p xor k]]) are:")
        print(x)
        print(y)
        print("When feeding x through the model without training, the encodings become:")
        print(model.predict(x))
        print("Training now")
        model.train_set(x, y, save=False, epochs=101, extra_callbacks=[rank])
        print("Done training")

        # Get the encodings of the input data using the same approach used in ops.py corrtest (iterate over rows)
        result = []
        for i in range(0, x.shape[0]):
            result.append(model.predict(np.array([x[i,:]], dtype=float))[0])  # Result contains sum of points such that corr with y[key_index] is maximal for all key indices. Shape = [trace, 16]
        result = np.array(result)
        print("When feeding x through the model after training, the encodings for key bytes %d to %d become:\n %s" % (AICORRNET_KEY_LOW, AICORRNET_KEY_HIGH, str(result)))

        # ------------------------------
        # Check loss function
        # ------------------------------
        # Evaluate the model to get the loss for the encodings
        predicted_loss = model.model.evaluate(x, y, verbose=0)

        # Manually calculate the loss using numpy to verify that we are learning a correct correlation
        calculated_loss = 0
        for i in range(AICORRNET_KEY_LOW, AICORRNET_KEY_HIGH):
            print("Subkey %d HWs   : %s" %(i, str(y[:,i])))
            print("Subkey %d encodings: %s" %(i, str(result[:,i-AICORRNET_KEY_LOW])))
            y_key = y[:,i].reshape([-1, 1])
            y_pred = result[:,i-AICORRNET_KEY_LOW].reshape([-1, 1])

            # Normalize labels
            y_key_norm = y_key - np.mean(y_key, axis=0)
            y_pred_norm = y_pred - np.mean(y_pred, axis=0)

            # Calculate correlation (vector approach)
            #denom = np.sqrt(np.dot(y_pred_norm.T, y_pred_norm)) * np.sqrt(np.dot(y_key_norm.T, y_key_norm))
            #denom = np.maximum(denom, 1e-15)
            #corr_key_i = (np.dot(y_key_norm.T, y_pred_norm) / denom)[0,0]
            #print("corr_vec: %s" % corr_key_i)

            # Calculate correlation (numpy approach)
            corr_key_i = np.corrcoef(y_pred[:,0], y_key[:,0], rowvar=False)[1,0]
            print("corr_num: %s" % corr_key_i)

            calculated_loss += 1.0 - corr_key_i

        print("These values should be close:")
        print("Predicted loss: %s" % str(predicted_loss))
        print("Calculated loss: %s" % str(calculated_loss))
        self.assertAlmostEqual(predicted_loss, calculated_loss, places=5)

if __name__ == '__main__':
    unittest.main()
