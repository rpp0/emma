#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# Test suites
# ----------------------------------------------------

from correlation import Correlation
import unittest
import numpy as np
import emutils

class TestCorrelation(unittest.TestCase):
    def test_update(self):
        test_array = np.array([
            [1, 2],
            [3, 5],
            [4, 5],
            [4, 8]
        ])
        x = test_array[:,0]
        y = test_array[:,1]

        c = Correlation()
        c.update(x,y)
        # Note for np.corrcoef: rows = variables, columns = observations. See https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.corrcoef.html
        self.assertAlmostEqual(c, np.corrcoef(x, y)[1,0], places=13)

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

        c1 = Correlation()
        c1.update(x1, y1)

        c2 = Correlation()
        c2.update(x2, y2)

        c3 = Correlation()
        c3.merge(c1)
        c3.merge(c2)

        self.assertAlmostEqual(c3, np.corrcoef(x_check, y_check)[1,0], places=13)

class TestUtils(unittest.TestCase):
    def test_pretty_print_correlations(self):
        test = Correlation.init([16,256])
        test[0,40].update([1, 2], [1, 2])
        test[0,20].update([4, 8, 9, 10], [6, 7, 1, 10])
        test[1,255].update([2, 5, 6, 10], [4, 30, 29, 10])
        test[2,254].update([7, 2, 6, 7], [1, 2, 5, 7])
        #emutils.pretty_print_correlations(test)

if __name__ == '__main__':
    unittest.main()
