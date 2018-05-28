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
from ai import AICORRNET_KEY_LOW, AICORRNET_KEY_HIGH

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
        model = ops.AICorrNet(4, name="test")
        x = [ # Contains abs(trace). Shape = [trace, point]
            [1, 1, 1, -15],
            [2, 1, -4, -12],
            [3, 1, 10, 8],
        ]

        y = [  # Contains hw[sbox[plaintext[trace] ^ key[key_index]]]. Shape = [trace, key_index]
            [6, 16, 5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            [7, -17, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [8, 7, 23, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        ]

        x = np.array(x)
        y = np.array(y)

        # Find optimal weights
        model.train_set(x, y, save=False, epochs=150)
        result = []

        # Simulate same approach used in ops.py corrtest (iterate over rows)
        for i in range(0, 3):
            result.append(model.predict(np.array([x[i,:]], dtype=float))[0])  # Result contains sum of points such that corr with y[key_index] is maximal for all key indices. Shape = [trace, 16]
        result = np.array(result)

        print("Learned frequency sums: " + str(result))
        print("Mean:")
        print(np.mean(result, axis=0))

        calculated_loss = 0
        for i in range(AICORRNET_KEY_LOW, AICORRNET_KEY_HIGH):
            print("Subkey %d values   : %s" %(i, str(y[:,i])))
            print("Subkey %d encodings: %s" %(i, str(result[:,i-AICORRNET_KEY_LOW])))
            y_key = y[:,i].reshape([-1, 1])
            y_pred = result[:,i-AICORRNET_KEY_LOW].reshape([-1, 1])

            # Normalize labels
            y_key_norm = y_key - np.mean(y_key, axis=0)
            y_pred_norm = y_pred - np.mean(y_pred, axis=0)

            # Calculate correlation (vector approach)
            denom = np.sqrt(np.dot(y_pred_norm.T, y_pred_norm)) * np.sqrt(np.dot(y_key_norm.T, y_key_norm))
            denom = np.maximum(denom, 1e-15)
            corr_key_i = np.square(np.dot(y_key_norm.T, y_pred_norm) / denom)[0,0]
            print("corr_vec: %s" % corr_key_i)

            # Calculate correlation (numpy approach)
            #corr_key_i = np.square(np.corrcoef(y_pred[:,0], y_key[:,0], rowvar=False)[1,0])
            #print("corr_num: %s" % corr_key_i)

            calculated_loss += 1.0 - corr_key_i

        print("These values should be close:")
        print("Last loss: %s" % str(model.last_loss))
        print("Calculated loss: %s" % str(calculated_loss))
        self.assertAlmostEqual(model.last_loss, calculated_loss, places=5)

if __name__ == '__main__':
    unittest.main()
