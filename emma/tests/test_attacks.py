import unittest
from emma.processing import ops
from emma.attacks.leakagemodels import *
from argparse import Namespace
from emma.io.traceset import TraceSet
from emma.io.emresult import EMResult


class TestAttacks(unittest.TestCase):
    def setUp(self):
        # Dummy EM measurement data. The third sample correlates with the key under AES SBOX HW power model
        traces = [
            [1,  1,  1, -15],
            [-4, 1,  2, -12],
            [10, 1,  3,   8],
            [8,  1,  1, -14],
            [9,  1, -3,   8],
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

        # Convert to numpy arrays
        self.traces = np.array(traces)
        self.plaintexts = np.array(plaintexts)
        self.keys = np.array(keys)

        # Store in trace set
        self.trace_set = TraceSet(name='test', traces=self.traces, plaintexts=self.plaintexts, keys=self.keys)

        # Configuration
        self.conf = Namespace(
            leakage_model=LeakageModelType.HAMMING_WEIGHT_SBOX,
            key_low=2,
            key_high=3,
            subkey=2,
            windowing_method='rectangular',
        )

        self.leakage_model = LeakageModel(self.conf)
        self.result = EMResult()

        # Window trace set
        ops.window_trace_set(self.trace_set, self.result, self.conf, params=[0, 4])

    def test_attack_trace_set(self):
        ops.attack_trace_set(self.trace_set, self.result, self.conf, params=None)

        # Make sure the calculated correlations match numpy's correlation
        for i in range(256):
            leakages = self.leakage_model.get_trace_set_leakages(self.trace_set, key_hypothesis=i).flatten()
            for j in range(4):
                cal_corr = self.result.correlations[i, j]
                np_corr = np.corrcoef(leakages, self.traces[:, j])[1, 0]
                if np.isnan(np_corr):
                    np_corr = 0.0
                self.assertAlmostEqual(cal_corr, np_corr, places=13)

        # Make sure the subkey is correctly found
        max_correlations = np.zeros(256)
        for subkey_guess in range(0, 256):
            max_correlations[subkey_guess] = np.max(np.abs(self.result.correlations[subkey_guess, :]))
        self.assertEqual(np.argmax(max_correlations), self.keys[0, self.conf.subkey])

    def test_attack_trace_set_distance(self):
        num_repetitions = 5
        for i in range(num_repetitions):  # Test multiple invocations (simulates multiple trace sets given). This has the effect of multiplying the distance by the number of invocations.
            ops.attack_trace_set_distance(self.trace_set, self.result, self.conf, params=None)

        # Make sure the calculated distances match numpy's correlation
        for i in range(256):
            leakages = self.leakage_model.get_trace_set_leakages(self.trace_set, key_hypothesis=i).flatten()
            for j in range(4):
                cal_dist = self.result.distances[i, j]
                np_dist = np.sum(np.abs(leakages - self.traces[:, j])) * num_repetitions
                self.assertAlmostEqual(cal_dist, np_dist, places=13)

