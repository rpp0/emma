import numpy as np
np.random.seed(1)  # Make results reproducible
import keras.backend as K
import keras
import pickle
import time
import os
import io
import matplotlib.pyplot as plt
import tensorflow as tf
import traceset
import emutils
import rank
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Conv1D, Reshape, MaxPool1D, Flatten, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import TensorBoard, History
from matplotlib.ticker import FuncFormatter
from keras.applications.vgg16 import VGG16
from keras import regularizers
from keras.engine.topology import Layer
from keras.losses import categorical_crossentropy
from rank import ProbRankCallback, CorrRankCallback

K.set_epsilon(1e-15)
# Some temporary globals for testing purposes
AICORRNET_KEY_LOW = 2  # 0
AICORRNET_KEY_HIGH = 3  # 16


class AI():
    """
    Base class for the models.
    """
    def __init__(self, name="unknown", suffix=None, path=None):
        self.id = str(int(time.time()))
        suffix = "" if suffix is None else '-' + suffix
        self.name = name + suffix
        if path is None:
            self.models_dir = os.path.join(os.getcwd(), 'models')
        else:
            self.models_dir = os.path.abspath(path)
        if not os.path.isdir(self.models_dir):  # TODO only do this when saving. (don't forget callbacks)
            os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, "%s.h5" % self.name)
        self.base_path = self.model_path.rpartition('.')[0]
        self.model = None

        # Callbacks during training
        self.callbacks = {
            'lastloss': LastLoss(),
            'tensorboard': TensorBoard(log_dir='/tmp/keras/' + self.name + '-' + self.id),
            'save': SaveLowestValLoss(self.model_path),
            'rank': ProbRankCallback('/tmp/keras/' + self.name + '-' + self.id + '/rank/', save_best=True, save_path=self.model_path),
        }

    def train_generator(self, training_iterator, validation_iterator, epochs=2000, workers=1, save=True):
        validation_batch = validation_iterator.next()  # Get one mini-batch from validation set to quickly test validation error

        # If we have a RankCallback set, pass our supplied validation set to it
        if 'rank' in self.callbacks:
            all_validation_trace_set = validation_iterator.get_all_as_trace_set(limit=20000)
            self.callbacks['rank'].set_trace_set(all_validation_trace_set)

        steps_per_epoch = int(training_iterator.num_total_examples / training_iterator.batch_size)

        # Train model
        self.model.fit_generator(training_iterator,
                                epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_data=validation_batch,
                                workers=workers, callbacks=list(self.callbacks.values()) + [SavingHistory(self.base_path)], verbose=1)

        # Get loss from callback
        self.last_loss = self.callbacks['lastloss'].value

        self._post_train(save)

    def train_t_fold(self, training_iterator, batch_size=10000, epochs=100, num_train_traces=45000, t=10, rank_trace_step=1000, conf=None):
        """
        t-fold cross-validation according to paper by Prouff et al.
        """

        # Get all traces in memory to speed up the process
        # First, process all ops and apply them to the traces set
        all_traces = training_iterator.get_all_as_trace_set()

        # Use the preprocessing function of the iterator to convert to Keras features
        inputs, labels = training_iterator._preprocess_trace_set(all_traces)

        num_validation_traces = training_iterator.num_total_examples - num_train_traces
        model_initial_state = self.model.get_weights()

        ranks = np.zeros(shape=(10, int(num_validation_traces / rank_trace_step))) + 256
        confidences = np.zeros(shape=(10, int(num_validation_traces / rank_trace_step)))
        for i in range(0, t):
            print("Fold %d" % i)
            # Reset model to untrained state
            self.model.set_weights(model_initial_state)

            # Randomize inputs and labels in the same order
            assert(len(inputs) == len(labels) == len(all_traces.traces))
            random_indices = np.arange(len(inputs))
            np.random.shuffle(random_indices)
            shuffled_inputs = np.take(inputs, random_indices, axis=0)  # Take random input examples
            shuffled_labels = np.take(labels, random_indices, axis=0)  # Take random label examples
            shuffled_traces = np.take(all_traces.traces, random_indices, axis=0)
            assert(labels[random_indices[0]][0] == shuffled_labels[0][0])
            assert(shuffled_traces[0].signal[0] == shuffled_inputs[0][0])

            shuffled_inputs_train = shuffled_inputs[0:num_train_traces]
            shuffled_inputs_val = shuffled_inputs[num_train_traces:]
            shuffled_labels_train = shuffled_labels[0:num_train_traces]
            shuffled_labels_val = shuffled_labels[num_train_traces:]

            # Train the model
            self.model.fit(shuffled_inputs_train, shuffled_labels_train, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=None, validation_data=(shuffled_inputs_val, shuffled_labels_val))

            # Now, evaluate the rank for increasing number of traces from the validation set (steps of 10)
            validation_traces = shuffled_traces[num_train_traces:]
            for j in range(0, int(num_validation_traces / rank_trace_step)):
                validation_traces_subset = validation_traces[0:(j+1)*rank_trace_step]
                x = np.array([trace.signal for trace in validation_traces_subset])
                if(conf.cnn):
                    x = np.expand_dims(x, axis=-1)
                encodings = self.model.predict(x) # Output: [?, 16]
                keys = np.array([trace.key for trace in validation_traces_subset])
                plaintexts = np.array([trace.plaintext for trace in validation_traces_subset])
                fake_ts = traceset.TraceSet(traces=encodings, plaintexts=plaintexts, keys=keys, name="fake_ts")
                fake_ts.window = emutils.Window(begin=0, end=encodings.shape[1])
                fake_ts.windowed = True
                r, c = rank.calculate_traceset_rank(fake_ts, 2, keys[0][2], conf.nomodel)
                ranks[i][j] = r
                confidences[i][j] = c
                print("Rank is %d with confidence %f (%d traces)" % (r, c, (j+1)*rank_trace_step))

        print(ranks)
        print(confidences)
        data_to_save = {
            'ranks': ranks,
            'confidences': confidences,
            'rank_trace_step': rank_trace_step,
            'folds': t,
            'num_train_traces': num_train_traces,
            'batch_size': batch_size,
            'epochs': epochs,
            'num_validation_traces': num_validation_traces,
            'conf': conf,
        }
        pickle.dump(data_to_save, open("%s-t-ranks.p" % self.base_path, "wb"))

    def test_fold(self, validation_iterator, rank_trace_step=1000, conf=None, max_traces=5000):
        """
        Test a single fold on the validation set to generate similar plot as train_t_fold, but without retraining the model. Could probably be used as a subcomponent of train_t_fold, but running out of time therefore TODO refactor.
        """

        # Get all traces in memory to speed up the process
        all_traces = validation_iterator.get_all_as_trace_set()
        validation_traces = all_traces.traces[0:max_traces]
        num_validation_traces = len(validation_traces)

        ranks = np.zeros(shape=int(num_validation_traces / rank_trace_step)) + 256
        confidences = np.zeros(shape=int(num_validation_traces / rank_trace_step))

        for j in range(0, int(num_validation_traces / rank_trace_step)):
            validation_traces_subset = validation_traces[0:(j+1)*rank_trace_step]
            x = np.array([trace.signal for trace in validation_traces_subset])
            if(conf.cnn):
                x = np.expand_dims(x, axis=-1)
            encodings = self.model.predict(x) # Output: [?, 16]
            keys = np.array([trace.key for trace in validation_traces_subset])
            plaintexts = np.array([trace.plaintext for trace in validation_traces_subset])
            fake_ts = traceset.TraceSet(traces=encodings, plaintexts=plaintexts, keys=keys, name="fake_ts")
            fake_ts.window = emutils.Window(begin=0, end=encodings.shape[1])
            fake_ts.windowed = True
            r, c = rank.calculate_traceset_rank(fake_ts, 2, keys[0][2], conf.nomodel)
            ranks[j] = r
            confidences[j] = c
            print("Rank is %d with confidence %f (%d traces)" % (r, c, (j+1)*rank_trace_step))

        print(ranks)
        print(confidences)
        data_to_save = {
            'ranks': ranks,
            'confidences': confidences,
            'rank_trace_step': rank_trace_step,
            'folds': 1,
            'num_train_traces': 0,
            'batch_size': None,
            'epochs': 0,
            'num_validation_traces': num_validation_traces,
            'conf': conf,
        }
        pickle.dump(data_to_save, open("%s-testrank.p" % self.base_path, "wb"))

    def _post_train(self, save=True):
        """
        Do some post-train actions like printing the model weights and saving the model.
        """
        if save:
            self.model.save("%s-last.h5" % self.base_path)

    def predict(self, x):
        return self.model.predict(x, batch_size=10000, verbose=0)

    def load(self):
        self.model = load_model(self.model_path, custom_objects={'correlation_loss': correlation_loss, 'cc_loss': cc_loss, 'CCLayer': CCLayer, 'cc_catcross_loss': cc_catcross_loss})


class AIMemCopyDirect():
    """
    Extremely simple NN that attempts to find a relation between the power consumption (input)
    and the resulting one-hot encoding of the byte that was copied from memory.
    """
    def __init__(self, input_dim, hamming):
        if hamming:
            self.num_outputs = 9
        else:  # Full byte
            self.num_outputs = 256
        self.model = Sequential()
        #self.model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
        self.model.add(Dense(256, activation='linear', input_dim=input_dim))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='linear'))
        #self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_outputs, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train_set(self, x, y):
        one_hot_labels = keras.utils.to_categorical(y, num_classes=self.num_outputs)
        self.model.fit(x, one_hot_labels, epochs=10, batch_size=256, shuffle=True)

    def test(self, x):
        pass


def correlation_loss(y_true_raw, y_pred_raw):
    """
    Custom loss function that calculates the Pearson correlation of the prediction with
    the true values over a number of batches.
    """
    # y_true_raw = K.print_tensor(y_true_raw, message='y_true_raw = ')  # Note: print truncating is incorrect in the print_tensor function
    # y_pred_raw = K.print_tensor(y_pred_raw, message='y_pred_raw = ')
    y_true = (y_true_raw - K.mean(y_true_raw, axis=0, keepdims=True))  # We are taking correlation over columns, so normalize columns
    y_pred = (y_pred_raw - K.mean(y_pred_raw, axis=0, keepdims=True))

    loss = K.variable(0.0)
    for key_col in range(AICORRNET_KEY_LOW, AICORRNET_KEY_HIGH):  # 0 - 16
        y_key = K.expand_dims(y_true[:,key_col], axis=1)  # [?, 16] -> [?, 1]
        y_keypred = K.expand_dims(y_pred[:,key_col-AICORRNET_KEY_LOW], axis=1)  # [?, 16] -> [?, 1]
        denom = K.sqrt(K.dot(K.transpose(y_keypred), y_keypred)) * K.sqrt(K.dot(K.transpose(y_key), y_key))
        denom = K.maximum(denom, K.epsilon())
        correlation = K.dot(K.transpose(y_key), y_keypred) / denom
        loss += 1.0 - correlation

    return loss


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class SavingHistory(History):
    def __init__(self, path):
        self.path = path
        super(SavingHistory, self).__init__()

    def on_train_end(self, logs={}):
        pickle.dump(self.history, open("%s-history.p" % self.path, "wb"))


class Clip(keras.constraints.Constraint):
    """
    Custom kernel constraint, limiting their values between a certain range.
    """
    def __init__(self):
        self.weight_range = [-1.0, 1.0]

    def __call__(self, w):
        return K.clip(w, self.weight_range[0], self.weight_range[1])
keras.constraints.Clip = Clip  # Register custom constraint in Keras


class LastLoss(keras.callbacks.Callback):
    """
    Callback to keep last loss.
    """
    def on_train_begin(self, logs={}):
        self.value = None

    def on_batch_end(self, batch, logs={}):
        self.value = logs.get('loss')


class SaveLowestValLoss(keras.callbacks.Callback):
    def __init__(self, path):
        super(SaveLowestValLoss, self).__init__()
        self.lowest = None
        self.path = path
        self.lowest_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = float(logs.get('val_loss'))

        if self.lowest is None:
            self.lowest = val_loss
        else:
            if val_loss < self.lowest:
                self.lowest = val_loss
                self.lowest_epoch = epoch
                self.model.save(self.path)

    def on_train_end(self, logs=None):
        print("Found lowest val_loss of %f at epoch %d" % (self.lowest, self.lowest_epoch))
        print("This model is saved at %s" % self.path)


class CustomTensorboard(keras.callbacks.TensorBoard):
    """
    Extension of the standard Tensorboard callback that uses Matplotlib to
    plot graphs to Tensorboard.
    """
    def __init__(self, freq=10, *args, **kwargs):
        self.freq = freq
        super(CustomTensorboard, self).__init__(*args, **kwargs)

    def _plt_to_tf(self, plot, tag='plot'):
        """
        Convert Matplotlib plot to Tensorboard summary.
        """
        # Write to PNG buffer
        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)

        # Add to TensorBoard summary
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0) # Add the batch dimension
        return tf.summary.image(tag, image, 1)

    def _plot_fft_weights(self, samp_rate):
        # Get weights of first layer
        weights = self.model.layers[0].get_weights()[0]  # Assumes Dense layer with shape (input, output)
        input_size = weights.shape[0]
        output_size = weights.shape[1]

        # Plot weights
        fig = plt.figure()
        axis = fig.add_subplot(111)
        labels = np.fft.fftfreq(input_size, d=1.0/samp_rate)
        plt.title('Weight values')

        x = np.arange(input_size)

        for i in range(0, output_size):
            y = weights[:,i]
            axis.plot(x, y)

        #axis.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: "%.2f kHz" % (labels[int(val)] / 1000.0) if int(val) in x else ""))
        #plt.xticks(rotation=15.0)

    def on_epoch_end(self, epoch, logs=None):
        super(CustomTensorboard, self).on_epoch_end(epoch, logs)
        if epoch % self.freq == 0:
            try:
                self._plot_fft_weights(80000000)  # TODO: hardcoded sample rate

                # Generate plot summary
                images = [self._plt_to_tf(plt, tag='plot'+str(epoch))]
                summary_images = tf.summary.merge(images, collections=None, name=None)
                summary_result = K.get_session().run(summary_images)
                self.writer.add_summary(summary_result)
            except Exception as e:
                print("Exception in image generation: %s" % str(e))
                pass


def spec_reg(weight_matrix):
    return 0.001 * K.sum(K.abs(1.0 - weight_matrix))


def str_to_reg(string, reg_lambda):
    if string == 'l1':
        return regularizers.l1(reg_lambda)
    elif string == 'l2':
        return regularizers.l2(reg_lambda)
    elif string == 'l1l2':
        return regularizers.l1_l2(l1=reg_lambda, l2=reg_lambda)
    else:
        return None


def str_to_activation(string):
    if string == 'leakyrelu':
        return LeakyReLU()
    else:
        if string is None:
            return None
        else:
            return Activation(string)


class AICorrNet(AI):
    def __init__(self, input_dim, name="aicorrnet", n_hidden_layers=1, use_bias=True, activation='leakyrelu', batch_norm=True, momentum=0.1, reg=None, regfinal=None, reg_lambda=0.001, cnn=False, ptinput=False, nomodel=False, metric_freq=10, suffix=None, path=None):
        # Get name based on config
        if nomodel:
            name += "-nomodel"
        name += "-h" + str(n_hidden_layers)
        if not cnn:
            if not use_bias:
                name += "-bias"
            if not activation is None:
                name += "-" + str(activation)
            if batch_norm:
                name += "-bn"
            if not reg is None:
                name += "-reg" + str(reg)
            if not regfinal is None:
                name += "-regfinal" + str(regfinal)
        else:
            name += '-cnn'
        super(AICorrNet, self).__init__(name, suffix=suffix, path=path)

        # Configure regularizer
        self.using_regularization = (not reg is None) or (not regfinal is None)

        #optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)
        if cnn:
            optimizer = keras.optimizers.Nadam(lr=0.00001)
        else:
            optimizer = keras.optimizers.Nadam(lr=0.0001)
        #optimizer = keras.optimizers.Adadelta()

        if not cnn:
            self.model = Sequential()
            #initializer = keras.initializers.Constant(value=1.0/input_dim)
            #initializer = keras.initializers.Constant(value=0.5)
            #initializer = keras.initializers.Constant(value=1.0)
            #initializer = keras.initializers.RandomUniform(minval=0, maxval=1.0, seed=None)
            #initializer = keras.initializers.RandomUniform(minval=0, maxval=0.001, seed=None)
            initializer = 'glorot_uniform'
            #constraint = Clip()
            constraint = None

            # Hidden layers
            for i in range(0, n_hidden_layers):
                hidden_nodes = 256
                self.model.add(Dense(hidden_nodes, input_dim=input_dim, use_bias=use_bias, activation=None, kernel_initializer=initializer, kernel_regularizer=str_to_reg(reg, reg_lambda)))
                input_dim=hidden_nodes
                if batch_norm:
                    self.model.add(BatchNormalization(momentum=momentum))
                self.model.add(str_to_activation(activation))

            # Output layer
            self.model.add(Dense(AICORRNET_KEY_HIGH - AICORRNET_KEY_LOW, input_dim=input_dim, use_bias=use_bias, activation=None, kernel_initializer=initializer, kernel_constraint=constraint, kernel_regularizer=str_to_reg(regfinal, reg_lambda)))
            if batch_norm:
                self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(str_to_activation(activation))
        else:
            from ASCAD_train_models import cnn_best_nosoftmax
            self.model = cnn_best_nosoftmax(input_shape=(input_dim,1), classes=AICORRNET_KEY_HIGH - AICORRNET_KEY_LOW)

        # Compile model
        self.model.compile(optimizer=optimizer, loss=correlation_loss, metrics=[])

        # Custom callbacks
        self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id, freq=metric_freq)
        self.callbacks['rank'] = CorrRankCallback('/tmp/keras/' + self.name + '-' + self.id + '/rank/', save_best=True, save_path=self.model_path, cnn=cnn, freq=metric_freq, ptinput=ptinput, nomodel=nomodel)

    def train_set(self, x, y, save=False, epochs=1, extra_callbacks=[]):
        """
        Train entire training set with model.fit()

        Used in qa_emma
        """

        # Callbacks
        last_loss = LastLoss()
        tensorboard_callback = TensorBoard(log_dir='/tmp/keras/' + self.id)
        history = LossHistory()

        # Fit model
        self.model.fit(x, y, epochs=epochs, batch_size=999999999, shuffle=False, verbose=2, callbacks=[last_loss, history] + extra_callbacks)

        # Get loss from callback
        self.last_loss = last_loss.value

        self._post_train(save)


class AISHACPU(AI):
    def __init__(self, input_shape, name="aishacpu", hamming=True, subtype='vgg16', suffix=None, path=None):
        super(AISHACPU, self).__init__(name + ('-hw' if hamming else ''), suffix=suffix, path=path)
        assert(K.image_data_format() == 'channels_last')
        input_tensor = Input(shape=input_shape)  # Does not include batch size

        self.model = None
        if subtype == 'vgg16':
            self.model = VGG16(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling='avg', classes=9 if hamming else 256)
        elif subtype == 'custom':
            #reg = regularizers.l2(0.01)
            reg = None
            self.model = Sequential()
            self.model.add(Dense(1024, input_shape=input_shape, kernel_regularizer=reg))
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(Activation('relu'))
            input_shape = (None, 1024)
            self.model.add(Dense(256, input_shape=input_shape, kernel_regularizer=reg))
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(Activation('relu'))
            input_shape = (None, 256)
            self.model.add(Dense(9 if hamming else 256, use_bias=True, input_shape=input_shape, kernel_regularizer=reg))
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(Activation('softmax'))

            # Extra callbacks
            #self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id)

        optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


class AISHACC(AI):
    def __init__(self, input_shape, name="aishacc", hamming=True, suffix=None, path=None):
        super(AISHACC, self).__init__(name + ('-hw' if hamming else ''), suffix=suffix, path=path)
        input_tensor = Input(shape=input_shape)  # Does not include batch size

        """
        kernel_initializer = 'glorot_uniform'
        cc_args = {
            'filters': 9 if hamming else 256,
            'kernel_size': 15,
            'dilation_rate': 1,
            'padding': 'valid',
            'kernel_initializer': kernel_initializer,
            'use_bias': True,
            'activation': 'relu',
        }

        reg = None
        self.model = Sequential()
        #self.model.add(Dense(1024, input_shape=input_shape, kernel_regularizer=reg))
        #self.model.add(BatchNormalization(momentum=0.1))
        #self.model.add(Activation('relu'))
        #input_shape = (1024,)
        self.model.add(Reshape(input_shape + (1,), input_shape=input_shape))
        self.model.add(CCLayer(**cc_args))
        self.model.add(Dense(9 if hamming else 256))
        self.model.add(BatchNormalization(momentum=0.1))
        self.model.add(Activation('relu'))
        self.model.add(Dense(9 if hamming else 256))
        self.model.add(BatchNormalization(momentum=0.1))
        self.model.add(Activation('softmax'))
        """

        """
        self.model = Sequential()
        self.model.add(Reshape(input_shape + (1,), input_shape=input_shape))
        self.model.add(Conv1D(filters=9, kernel_size=1023, activation='relu', padding='same'))
        self.model.add(Conv1D(filters=9 if hamming else 256, kernel_size=15, activation='relu', padding='same'))
        self.model.add(MaxPool1D(pool_size=input_shape[0]))
        #self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(9 if hamming else 256))
        self.model.add(BatchNormalization(momentum=0.99))
        self.model.add(Activation('relu'))
        self.model.add(Dense(9 if hamming else 256))
        self.model.add(BatchNormalization(momentum=0.99))
        self.model.add(Activation('softmax'))
        """

        self.model = Sequential()
        self.model.add(Reshape(input_shape + (1,), input_shape=input_shape))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='tanh', padding='same'))
        #self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same'))
        #self.model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(Conv1D(filters=256, kernel_size=3, activation='tanh', padding='same'))
        #self.model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
        #self.model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(Conv1D(filters=512, kernel_size=3, activation='tanh', padding='same'))
        self.model.add(Conv1D(filters=512, kernel_size=3, activation='tanh', padding='same'))
        #self.model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
        #self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(9 if hamming else 256))
        self.model.add(BatchNormalization(momentum=0.99))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(9 if hamming else 256))
        self.model.add(BatchNormalization(momentum=0.99))
        self.model.add(Activation('softmax'))

        print(self.model.summary())

        # Extra callbacks
        #self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id)

        optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.0)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def cc_loss(y_true, y_pred):
    # y_true: [batch, 256]
    # y_pred: [batch, correlations, filter_nr]
    loss = K.variable(0.0)

    # A higher correlation for the filter at the true class is good
    #filter_score = tf.reduce_mean(y_pred, axis=1, keepdims=False) * y_true
    filter_score = y_pred * y_true
    filter_loss = tf.reduce_sum(-filter_score, axis=1)
    #loss += tf.reduce_sum(filter_loss, axis=0, keepdims=False)
    #loss += tf.reduce_max(filter_loss, axis=0, keep_dims=False)
    loss += tf.reduce_mean(filter_loss, axis=0, keep_dims=False)  # TODO try me

    return loss


def cc_catcross_loss(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


class AIASCAD(AI):
    def __init__(self, input_shape, name="aiascad", suffix=None, path=None):
        super(AIASCAD, self).__init__(name, suffix=suffix, path=path)
        from ASCAD_train_models import cnn_best

        self.model = cnn_best(input_shape=input_shape)


class CCLayer(Conv1D):
    def __init__(self, epsilon=1e-7, normalize_inputs=False, **kwargs):
        self.epsilon = epsilon
        self.normalize_inputs = normalize_inputs
        super(CCLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CCLayer, self).build(input_shape)
        kernel_mean = tf.reduce_mean(self.kernel, axis=0, keep_dims=True)
        kernel_l2norm = tf.norm(self.kernel, ord=2, axis=0, keep_dims=True)

        self.zn_kernel = tf.divide(tf.subtract(self.kernel, kernel_mean), kernel_l2norm + self.epsilon)

    def call(self, inputs):
        if self.normalize_inputs:
            '''
            TODO: This will not result in a true ZN correlation because we cannot set a stride for the reduce_mean operator. Can this behavior be enforced using
            something like https://www.tensorflow.org/api_docs/python/tf/strided_slice?
            '''
            inputs_mean = tf.reduce_mean(inputs, axis=1, keep_dims=True)
            inputs_l2norm = tf.norm(inputs, ord=2, axis=1, keep_dims=True)
            inputs = tf.divide(tf.subtract(inputs, inputs_mean), inputs_l2norm + self.epsilon)

        outputs = K.conv1d(
            inputs,
            self.zn_kernel,
            strides=self.strides[0],
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        outputs = K.max(outputs, axis=1, keepdims=False)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.filters)
