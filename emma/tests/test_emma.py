#!/usr/bin/python3

import unittest
from emma.processing import ops
from emma.ai import models, rankcallbacks
import pickle

from emma.metrics.correlationlist import CorrelationList
from emma.metrics.distancelist import DistanceList
from emma.io.traceset import TraceSet, Trace
from argparse import Namespace
from emma.ai.iterators import AICorrSignalIterator, AutoEncoderSignalIterator
from emma.attacks.leakagemodels import *
from emma.ai.inputs import *
from emma.processing.action import Action
from keras.utils import to_categorical
from emma.attacks.lut import *
from emma.tests import UnitTestSettings


class TestCorrelationList(unittest.TestCase):
    def test_update(self):
        test_array = np.array([
            [1, 2],
            [3, 5],
            [4, 5],
            [4, 8]
        ])
        x = test_array[:, 0]
        y = test_array[:, 1]

        clist1 = CorrelationList(1)
        clist1.update(0, x, y)
        clist2 = CorrelationList([1, 1])
        clist2.update((0, 0), x, y)

        # Checks
        self.assertAlmostEqual(clist1[0], np.corrcoef(x, y)[1, 0], places=13)
        self.assertAlmostEqual(clist2[0, 0], np.corrcoef(x, y)[1, 0], places=13)

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

    def test_max(self):
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

        test_array_3 = np.array([
            [-1, 1],
            [-2, 2],
            [-3, 3],
            [-4, 4]
        ])

        x1 = test_array_1[:, 0]
        y1 = test_array_1[:, 1]
        x2 = test_array_2[:, 0]
        y2 = test_array_2[:, 1]
        x3 = test_array_3[:, 0]
        y3 = test_array_3[:, 1]

        clist = CorrelationList([1, 3])
        clist.update((0, 0), x1, y1)
        clist.update((0, 1), x2, y2)
        clist.update((0, 2), x3, y3)

        max_corr_over_points = np.max(np.abs(clist[0, :]))
        self.assertEqual(max_corr_over_points, 1.0)


class TestDistanceList(unittest.TestCase):
    def test_update(self):
        test_array = np.array([
            [1, 2],
            [3, 5],
            [4, 5],
            [4, 8]
        ])
        x = test_array[:, 0]
        y = test_array[:, 1]

        clist1 = DistanceList(1)
        clist1.update(0, x, y)
        clist2 = DistanceList([1, 1])
        clist2.update((0, 0), x, y)

        # Checks
        self.assertAlmostEqual(clist1[0], np.sum(np.abs(x - y)), places=13)
        self.assertAlmostEqual(clist2[0, 0], np.sum(np.abs(x - y)), places=13)

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

        x1 = test_array_1[:, 0]
        y1 = test_array_1[:, 1]
        x2 = test_array_2[:, 0]
        y2 = test_array_2[:, 1]
        x_check = test_array_check[:, 0]
        y_check = test_array_check[:, 1]

        c1 = DistanceList(1)
        c1.update(0, x1, y1)

        c2 = DistanceList(1)
        c2.update(0, x2, y2)

        c3 = DistanceList(1)
        c3.merge(c1)
        c3.merge(c2)

        self.assertAlmostEqual(c3[0], np.sum(np.abs(x_check - y_check)), places=13)


class TestAI(unittest.TestCase):
    @unittest.skipIf(UnitTestSettings.TEST_FAST, "fast testing enabled")
    def test_corrtrain_correlation(self):
        """
        Artificial example to test AICorrNet and trace processing
        """

        # ------------------------------
        # Generate data
        # ------------------------------
        traces = [  # Contains abs(trace). Shape = [trace, point]
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
        conf = Namespace(
            max_cache=0,
            augment_roll=False,
            augment_noise=False,
            normalize=False,
            traces_per_set=4,
            online=False,
            dataset_id='qa',
            cnn=False,
            leakage_model=LeakageModelType.HAMMING_WEIGHT_SBOX,
            input_type=AIInputType.SIGNAL,
            augment_shuffle=True,
            n_hidden_layers=1,
            n_hidden_nodes=256,
            activation='leakyrelu',
            metric_freq=100,
            regularizer=None,
            reglambda=0.001,
            model_suffix=None,
            use_bias=True,
            batch_norm=True,
            hamming=False,
            key_low=2,
            key_high=3,
            loss_type='correlation',
            lr=0.0001,
            epochs=1000,
            batch_size=512,
            norank=False,
        )
        it_dummy = AICorrSignalIterator([], conf, batch_size=10000, request_id=None, stream_server=None)
        x, y = it_dummy._preprocess_trace_set(trace_set)

        # ------------------------------
        # Train and obtain encodings
        # ------------------------------
        model = models.AICorrNet(conf, input_dim=4, name="test")
        print(model.info())
        rank_cb = rankcallbacks.CorrRankCallback(conf, '/tmp/deleteme/', save_best=False, save_path=None)
        rank_cb.set_trace_set(trace_set)

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
        model.train_set(x, y, save=False, epochs=conf.epochs, extra_callbacks=[rank_cb])
        print("Done training")

        # Get the encodings of the input data using the same approach used in ops.py corrtest (iterate over rows)
        result = []
        for i in range(0, x.shape[0]):
            result.append(model.predict(np.array([x[i,:]], dtype=float))[0])  # Result contains sum of points such that corr with y[key_index] is maximal for all key indices. Shape = [trace, 16]
        result = np.array(result)
        print("When feeding x through the model after training, the encodings for key bytes %d to %d become:\n %s" % (conf.key_low, conf.key_high, str(result)))

        # ------------------------------
        # Check loss function
        # ------------------------------
        # Evaluate the model to get the loss for the encodings
        predicted_loss = model.model.evaluate(x, y, verbose=0)

        # Manually calculate the loss using numpy to verify that we are learning a correct correlation
        calculated_loss = 0
        for i in range(0, conf.key_high - conf.key_low):
            print("Subkey %d HWs   : %s" % (i + conf.key_low, str(y[:, i])))
            print("Subkey %d encodings: %s" % (i + conf.key_low, str(result[:, i])))
            y_key = y[:, i].reshape([-1, 1])
            y_pred = result[:, i].reshape([-1, 1])

            # Calculate correlation (vector approach)
            # y_key_norm = y_key - np.mean(y_key, axis=0)
            # y_pred_norm = y_pred - np.mean(y_pred, axis=0)
            # denom = np.sqrt(np.dot(y_pred_norm.T, y_pred_norm)) * np.sqrt(np.dot(y_key_norm.T, y_key_norm))
            # denom = np.maximum(denom, 1e-15)
            # corr_key_i = (np.dot(y_key_norm.T, y_pred_norm) / denom)[0,0]
            # print("corr_vec: %s" % corr_key_i)

            # Calculate correlation (numpy approach)
            corr_key_i = np.corrcoef(y_pred[:, 0], y_key[:, 0], rowvar=False)[1,0]
            print("corr_num: %s" % corr_key_i)

            calculated_loss += 1.0 - corr_key_i

        print("These values should be close:")
        print("Predicted loss: %s" % str(predicted_loss))
        print("Calculated loss: %s" % str(calculated_loss))
        self.assertAlmostEqual(predicted_loss, calculated_loss, places=5)

    @unittest.skipIf(UnitTestSettings.TEST_FAST, "fast testing enabled")
    def test_corrtrain_correlation_multi(self):
        from emma.attacks.leakagemodels import LeakageModel
        """
        Artificial example to test AICorrNet and trace processing with multiple leakage values and multiple subkeys.
        """

        # ------------------------------
        # Generate data
        # ------------------------------
        traces = [  # Contains abs(trace). Shape = [trace, point]
            [1, 1, 1, -15],
            [-4, 2, 2, -12],
            [10, 3, 3, 8],
            [8, 1, 1, -14],
            [9, 0, -3, 8],
        ]

        plaintexts = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        keys = [
            [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        # Convert to numpy
        traces = np.array(traces)
        plaintexts = np.array(plaintexts)
        keys = np.array(keys)

        trace_set = TraceSet(name='test', traces=traces, plaintexts=plaintexts, keys=keys)

        # ------------------------------
        # Preprocess data
        # ------------------------------
        conf = Namespace(
            max_cache=0,
            augment_roll=False,
            augment_noise=False,
            normalize=False,
            traces_per_set=4,
            online=False,
            dataset_id='qa',
            cnn=False,
            leakage_model=LeakageModelType.AES_MULTI,
            input_type=AIInputType.SIGNAL,
            augment_shuffle=True,
            n_hidden_layers=1,
            n_hidden_nodes=256,
            activation='leakyrelu',
            metric_freq=100,
            regularizer=None,
            reglambda=0.001,
            model_suffix=None,
            use_bias=True,
            batch_norm=True,
            hamming=False,
            key_low=1,
            key_high=3,
            loss_type='correlation',
            lr=0.001,
            epochs=5000,
            batch_size=512,
            norank=False,
        )
        it_dummy = AICorrSignalIterator([], conf, batch_size=10000, request_id=None, stream_server=None)
        x, y = it_dummy._preprocess_trace_set(trace_set)

        # ------------------------------
        # Train and obtain encodings
        # ------------------------------
        model = models.AICorrNet(conf, input_dim=4, name="test")
        print(model.info())
        rank_cb = rankcallbacks.CorrRankCallback(conf, '/tmp/deleteme/', save_best=False, save_path=None)
        rank_cb.set_trace_set(trace_set)

        if model.using_regularization:
            print("Warning: cant do correlation loss test because regularizer will influence loss function")
            return

        # Find optimal weights
        print("The x (EM samples) and y (leakage model values) are:")
        print(x)
        print(y)
        print("When feeding x through the model without training, the encodings become:")
        print(model.predict(x))
        print("Training now")
        model.train_set(x, y, save=False, epochs=conf.epochs, extra_callbacks=[rank_cb])
        print("Done training")

        # Get the encodings of the input data using the same approach used in ops.py corrtest (iterate over rows)
        result = []
        for i in range(0, x.shape[0]):
            result.append(model.predict(np.array([x[i,:]], dtype=float))[0])  # Result contains sum of points such that corr with y[key_index] is maximal for all key indices. Shape = [trace, 16]
        result = np.array(result)
        print("When feeding x through the model after training, the encodings for key bytes %d to %d become:\n %s" % (conf.key_low, conf.key_high, str(result)))

        # ------------------------------
        # Check loss function
        # ------------------------------
        # Evaluate the model to get the loss for the encodings
        predicted_loss = model.model.evaluate(x, y, verbose=0)

        # Manually calculate the loss using numpy to verify that we are learning a correct correlation
        calculated_loss = 0
        num_keys = (conf.key_high - conf.key_low)
        num_outputs = LeakageModel.get_num_outputs(conf) // num_keys
        for i in range(0, num_keys):
            subkey_hws = y[:, i*num_outputs:(i+1)*num_outputs]
            subkey_encodings = result[:, i*num_outputs:(i+1)*num_outputs]
            print("Subkey %d HWs   : %s" % (i + conf.key_low, str(subkey_hws)))
            print("Subkey %d encodings: %s" % (i + conf.key_low, str(subkey_encodings)))
            y_key = subkey_hws.reshape([-1, 1])
            y_pred = subkey_encodings.reshape([-1, 1])
            print("Flattened subkey %d HWs   : %s" % (i + conf.key_low, str(y_key)))
            print("Flattened subkey %d encodings: %s" % (i + conf.key_low, str(y_pred)))

            # Calculate correlation (numpy approach)
            corr_key_i = np.corrcoef(y_pred[:, 0], y_key[:, 0], rowvar=False)[1,0]
            print("corr_num: %s" % corr_key_i)

            calculated_loss += 1.0 - corr_key_i

        print("These values should be close:")
        print("Predicted loss: %s" % str(predicted_loss))
        print("Calculated loss: %s" % str(calculated_loss))
        self.assertAlmostEqual(predicted_loss, calculated_loss, places=2)

    @unittest.skipIf(UnitTestSettings.TEST_FAST, "fast testing enabled")
    def test_autoenctrain(self):
        """
        Artificial example to test AutoEncoder
        """

        # ------------------------------
        # Generate data
        # ------------------------------
        traces = [  # Contains abs(trace). Shape = [trace, point]
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
        conf = Namespace(
            max_cache=0,
            augment_roll=False,
            augment_noise=False,
            normalize=False,
            traces_per_set=4,
            online=False,
            dataset_id='qa',
            cnn=False,
            leakage_model=LeakageModelType.HAMMING_WEIGHT_SBOX,
            input_type=AIInputType.SIGNAL,
            augment_shuffle=True,
            n_hidden_layers=1,
            n_hidden_nodes=256,
            activation='leakyrelu',
            metric_freq=100,
            regularizer=None,
            reglambda=0.001,
            model_suffix=None,
            use_bias=True,
            batch_norm=True,
            hamming=False,
            key_low=2,
            key_high=3,
            loss_type='correlation',
            lr=0.0001,
            epochs=2000,
            batch_size=512,
            norank=False,
        )
        it_dummy = AutoEncoderSignalIterator([], conf, batch_size=10000, request_id=None, stream_server=None)
        x, y = it_dummy._preprocess_trace_set(trace_set)

        # ------------------------------
        # Train and obtain encodings
        # ------------------------------
        model = models.AutoEncoder(conf, input_dim=4, name="test")
        print(model.info())

        # Find optimal weights
        print("X, Y")
        print(x)
        print(y)
        print("When feeding x through the model without training, the encodings become:")
        print(model.predict(x))
        print("Training now")
        model.train_set(x, y, epochs=conf.epochs)
        print("Done training")

        # Get the encodings of the input data using the same approach used in ops.py corrtest (iterate over rows)
        result = []
        for i in range(0, x.shape[0]):
            result.append(model.predict(np.array([x[i, :]], dtype=float))[0])  # Result contains sum of points such that corr with y[key_index] is maximal for all key indices. Shape = [trace, 16]
        result = np.array(result)

        for i in range(result.shape[0]):
            rounded_result = np.round(result[i])
            print("Original x    : %s" % x[i])
            print("Rounded result: %s" % rounded_result)
            self.assertListEqual(list(rounded_result), list(x[i]))

    def test_softmax(self):
        test = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        a = models.softmax(test)
        b = models.softmax_np(test)
        self.assertEqual(len(a), len(b))
        for i in range(0, len(a)):
            self.assertAlmostEqual(a[i], b[i], places=6)


class TestRank(unittest.TestCase):
    def test_calculate_ranks(self):
        dummy_scores = np.array(list(range(1, 257)))  # 1, 2, 3, ..., 256 (rank scores)
        expected_outcome = list(range(255, -1, -1))   # 255, 254, 253, ..., 0 (resulting ranks)

        outcome = list(rankcallbacks.calculate_ranks(dummy_scores))
        self.assertListEqual(outcome, expected_outcome)

    def test_get_rank_and_confidence(self):
        dummy_scores = np.array(list(range(1, 257)))
        ranks = rankcallbacks.calculate_ranks(dummy_scores)

        rank_value, confidence = rankcallbacks.get_rank_and_confidence(ranks, dummy_scores, 255)
        self.assertEqual(confidence, 1)
        self.assertEqual(rank_value, 0)
        rank_value, _ = rankcallbacks.get_rank_and_confidence(ranks, dummy_scores, 254)
        self.assertEqual(rank_value, 1)
        rank_value, _ = rankcallbacks.get_rank_and_confidence(ranks, dummy_scores, 154)
        self.assertEqual(rank_value, 101)


class TestOps(unittest.TestCase):
    def test_align_trace_set(self):
        traces = np.array([[0, 1, 0, 8, 10, 8, 0, 1, 0], [8, 8, 11, 8], [8, 10, 8, 0]])
        expected = np.array([[8, 10, 8, 0, 1, 0], [8, 11, 8], [8, 10, 8, 0]])
        reference_signal = np.array([8, 10, 8])
        conf = Namespace(reference_signal=reference_signal, butter_cutoff=0.1, butter_order=1)

        ts = TraceSet(traces=traces, name='test')
        ops.align_trace_set(ts, None, conf, params=[0, len(reference_signal)])
        for i in range(0, len(ts.traces)):
            self.assertListEqual(list(ts.traces[i].signal), expected[i])

    def test_select_trace_set(self):
        test_path = "/tmp/selection.p"
        traces = np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
        expected = np.array([[3, 4], [3, 4]])
        conf = Namespace(windowing_method='rectangular')
        with open(test_path, "wb") as f:
            pickle.dump(np.array([False, False, True, True, False, False]), f)

        ts = TraceSet(traces=traces, name='test')
        ops.window_trace_set(ts, None, conf, params=[0, 6])
        ops.select_trace_set(ts, None, None, params=[test_path])
        for i in range(0, len(ts.traces)):
            self.assertListEqual(list(ts.traces[i].signal), list(expected[i]))

    def test_filterkey_trace_set(self):
        traces = np.array([[0], [1], [2]])
        keys = np.array([[0], [1], [2]])

        ts = TraceSet(traces=traces, keys=keys)
        conf = Namespace()
        ops.filterkey_trace_set(ts, None, conf, params=['01'])

        self.assertEqual(len(ts.traces), 1)
        self.assertListEqual(list(ts.traces[0].signal), list(traces[1]))

    def test_spectogram_trace_set(self):
        traces = np.array([[0, 1, 2]])

        ts = TraceSet(traces=traces)
        conf = Namespace(reference_signal=None)
        ops.spectogram_trace_set(ts, None, conf, None)

        self.assertListEqual([round(x, 8) for x in list(ts.traces[0].signal)], [9., 3., 3.])

    def test_normalize_trace_set(self):
        traces = np.array([[10, 16, 19],])
        expected = np.array([[-5, 1, 4],])

        ts = TraceSet(traces=traces)
        ops.normalize_trace_set(ts, None, None, None)
        for i in range(0, len(traces)):
            self.assertListEqual(list(ts.traces[i].signal), list(expected[i]))

    def test_fft_trace_set(self):
        traces = np.array([[0, 1, 2]])

        ts = TraceSet(traces=traces)
        conf = Namespace(reference_signal=None)
        ops.fft_trace_set(ts, None, conf, None)

        self.assertListEqual([round(x, 8) for x in list(ts.traces[0].signal)], [3.+0.j, -1.5+0.8660254j, -1.5-0.8660254j])

    def test_window_trace_set(self):
        traces = np.array([[1], [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4]])
        params = [1, 5]
        expected = np.array([[0, 0, 0, 0], [2, 3, 4, 5], [2, 3, 4, 0]])

        ts = TraceSet(traces=traces)
        conf = Namespace(windowing_method="rectangular")
        ops.window_trace_set(ts, None, conf, params=params)

        for i in range(0, len(traces)):
            self.assertListEqual(list(ts.traces[i].signal), list(expected[i]))


class TestUtils(unittest.TestCase):
    def test_int_to_one_hot(self):
        from emma.utils.utils import int_to_one_hot
        self.assertListEqual(list(int_to_one_hot(0, 256)), [1] + [0]*255)
        self.assertListEqual(list(int_to_one_hot(0, 3)), [1, 0, 0])
        self.assertListEqual(list(int_to_one_hot(1, 3)), [0, 1, 0])
        self.assertListEqual(list(int_to_one_hot(2, 3)), [0, 0, 1])


class TestIterator(unittest.TestCase):
    def test_iterator_wrapping(self):
        conf = Namespace(
            input_type=AIInputType.SIGNAL,
            leakage_model=LeakageModelType.SBOX_OH,
            max_cache=None,
            augment_roll=False,
            augment_noise=False,
            augment_shuffle=False,
            normalize=False,
            traces_per_set=32,
            online=False,
            dataset_id='test',
            format='cw',
            reference_signal=np.array([0]*128),
            actions=[],
            cnn=False,
            key_low=2,
            key_high=3,
            norank=False,
        )

        iterator = AICorrSignalIterator(
            ["./datasets/unit-test/test_traces.npy", "./datasets/unit-test/test2_traces.npy"],
            conf,
            batch_size=48
        )

        inputs, labels = next(iterator)
        for i in range(0, 48):
            self.assertListEqual(list(inputs[i]), [i] * 128)
            self.assertListEqual(list(labels[i]), list(to_categorical(sbox[1 ^ 0], num_classes=256)))
        self.assertEqual(inputs.shape, (48, 128))
        self.assertEqual(labels.shape, (48, 256))

        inputs, labels = next(iterator)
        for i in range(0, 48):
            self.assertListEqual(list(inputs[i]), [(i+48) % 64] * 128)
        self.assertEqual(inputs.shape, (48, 128))
        self.assertEqual(labels.shape, (48, 256))

    @unittest.skipIf(UnitTestSettings.TEST_FAST, "fast testing enabled")
    def test_ascad_iterator(self):
        """
        Check whether the AICorrSignalIterator returns the same output as load_ascad
        :return:
        """
        from ascad.ASCAD_train_models import load_ascad

        conf = Namespace(
            input_type=AIInputType.SIGNAL,
            leakage_model=LeakageModelType.SBOX_OH,
            max_cache=None,
            augment_roll=False,
            augment_noise=False,
            augment_shuffle=False,
            normalize=False,
            traces_per_set=50000,
            online=False,
            dataset_id='test',
            format='ascad',
            reference_signal=np.array([0]*700),
            actions=[Action('window[0,700]')],
            cnn=False,
            key_low=2,
            key_high=3,
            windowing_method='rectangular',
            norank=False,
        )

        ascad_root = "./datasets/ASCAD/ASCAD_data/ASCAD_databases/ASCAD.h5"
        ascad_paths = [
            "%s#Profiling_traces[0:256]" % ascad_root,
            "%s#Profiling_traces[256:512]" % ascad_root
        ]

        iterator = AICorrSignalIterator(
            ascad_paths,
            conf,
            batch_size=256
        )

        x = np.zeros((512, 700))
        y = np.zeros((512, 256))
        inputs, labels = next(iterator)
        x[0:256] = inputs
        y[0:256] = labels
        inputs, labels = next(iterator)
        x[256:512] = inputs
        y[256:512] = labels

        (X_profiling, Y_profiling), (X_attack, Y_attack), (meta_profiling, meta_attack) = load_ascad(ascad_root, True)
        x_ascad = X_profiling
        y_ascad = to_categorical(Y_profiling, num_classes=256)

        for i in range(0, 512):
            self.assertListEqual(list(x[i]), list(x_ascad[i]))

        for i in range(0, 512):
            self.assertListEqual(list(y[i]), list(y_ascad[i]))


class TestLUT(unittest.TestCase):
    def test_hw(self):
        self.assertEqual(hw[0xff], 8)
        self.assertEqual(hw[0x00], 0)
        self.assertEqual(hw[0x02], 1)
        self.assertEqual(hw[0x22], 2)
        self.assertEqual(hw[0x12], 2)
        self.assertEqual(hw[0x10], 1)
        self.assertEqual(hw[0xf0], 4)
        self.assertEqual(hw[0x0f], 4)

    def test_hw16(self):
        self.assertEqual(hw16[0xff], 8)
        self.assertEqual(hw16[0x00], 0)
        self.assertEqual(hw16[0xffff], 16)
        self.assertEqual(hw16[0xff00], 8)
        self.assertEqual(hw16[0x00ff], 8)
        self.assertEqual(hw16[0xf000], 4)
        self.assertEqual(hw16[0x0f00], 4)

    def test_hw32(self):
        self.assertEqual(hw32(0xff), 8)
        self.assertEqual(hw32(0x00), 0)
        self.assertEqual(hw32(0xffff), 16)
        self.assertEqual(hw32(0xff00), 8)
        self.assertEqual(hw32(0xffff00), 16)
        self.assertEqual(hw32(0xffffff00), 24)
        self.assertEqual(hw32(0xffffffff), 32)
        self.assertEqual(hw32(0xff0fffff), 28)


class TestLeakageModels(unittest.TestCase):
    def test_hmac_hw(self):
        fake_pt  = np.array([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], dtype=np.uint8)
        fake_key = np.array([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80], dtype=np.uint8)
        trace = Trace(signal=[], plaintext=fake_pt, ciphertext=[], key=fake_key, mask=[])

        lm = LeakageModel(Namespace(leakage_model="hmac_hw", key_low=0, key_high=1))
        self.assertEqual(lm.subkey_size, 4)  # HMAC uses subkeys of size 4
        leakage_subkey_0 = lm.get_trace_leakages(trace, 0)  # hw32(0x40302010)
        leakage_subkey_1 = lm.get_trace_leakages(trace, 1)  # hw32(0x80706050)
        self.assertEqual(leakage_subkey_0[0], hw32(0x40302010))
        self.assertEqual(leakage_subkey_1[0], hw32(0x80706050))
        self.assertEqual(leakage_subkey_0[1], hw32(0x40302010 ^ 0x36363636))
        self.assertEqual(leakage_subkey_1[1], hw32(0x80706050 ^ 0x36363636))
        self.assertEqual(leakage_subkey_0[2], hw32(0x40302010 ^ 0x5c5c5c5c))
        self.assertEqual(leakage_subkey_1[2], hw32(0x80706050 ^ 0x5c5c5c5c))


if __name__ == '__main__':
    unittest.main()
