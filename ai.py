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
import visualizations
import lossfunctions
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Conv1D, Reshape, MaxPool1D, Flatten, LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import TensorBoard, History
from keras.applications.vgg16 import VGG16
from keras import regularizers
from leakagemodels import LeakageModel

K.set_epsilon(1e-15)


def softmax_np(inputs):
    return np.exp(inputs) / sum(np.exp(inputs))


def softmax(inputs):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=inputs.shape)
        result = sess.run(tf.nn.softmax(x), feed_dict={x: inputs})
    return result


class AI:
    """
    Base class for the models.
    """
    def __init__(self, conf, model_type="unknown"):
        """
        Initialize AI based on a configuration.
        :param conf:
        :param model_type:
        """
        # Set parameters
        self.conf = conf
        self.last_loss = None
        self.model_type = model_type
        self.n_hidden_layers = conf.n_hidden_layers
        self.use_bias = conf.use_bias
        self.batch_norm = conf.batch_norm
        self.activation = conf.activation
        self.cnn = conf.cnn
        self.metric_freq = conf.metric_freq
        self.reg = conf.regularizer
        self.regfinal = conf.regularizer
        self.reg_lambda = conf.reglambda
        self.momentum = 0.1
        self.hamming = conf.hamming
        self.key_low = conf.key_low
        self.key_high = conf.key_high
        self.loss = lossfunctions.get_loss(conf)

        self.suffix = "" if conf.model_suffix is None else '-' + conf.model_suffix  # Added to name later
        self.name = self.conf_to_name(model_type, conf)

        # ID
        self.id = str(int(time.time()))

        # Get path
        models_dir = os.path.join(os.getcwd(), 'models', emutils.conf_to_id(conf))
        self.models_dir = os.path.abspath(models_dir)
        if not os.path.isdir(self.models_dir):  # TODO only do this when saving. (don't forget callbacks)
            os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, "%s.h5" % self.name)
        self.base_path = self.model_path.rpartition('.')[0]

        # Internal model
        self.model = None

        # Some additional properties
        self.using_regularization = (not self.reg is None) or (not self.regfinal is None)

        # Callbacks during training
        self.callbacks = {
            'lastloss': LastLoss(),
            'tensorboard': TensorBoard(log_dir='/tmp/keras/' + self.name + '-' + self.id),
            'save': SaveLowestValLoss(self.model_path),
        }

    def _debug_batch(self, iterator, name):
        print("Saving plot of debug batch %s" % name)
        example_batch = next(iterator)
        signals, values = example_batch
        predictions = self.predict(signals)
        loss = self.model.evaluate(signals, values, verbose=0)
        pickle.dump(predictions, open("/tmp/predictions-%s.p" % name, "wb"))
        pickle.dump(values, open("/tmp/values-%s.p" % name, "wb"))
        pickle.dump(loss, open("/tmp/loss-%s.p" % name, "wb"))
        visualizations.plot_correlations(predictions, values, label1="Predictions", label2="True values", show=False)
        visualizations.plt_save_pdf("/tmp/correlations-plot-%s.pdf" % name)

    def train_generator(self, training_iterator, validation_iterator, epochs=2000, workers=1, save=True):
        validation_batch = validation_iterator.next()  # Get one mini-batch from validation set to quickly test validation error

        # If we have a RankCallback set, pass our supplied validation set to it
        if 'rank' in self.callbacks:
            all_validation_trace_set = validation_iterator.get_all_as_trace_set(limit=80)
            self.callbacks['rank'].set_trace_set(all_validation_trace_set)

        steps_per_epoch = int(training_iterator.num_total_examples / training_iterator.batch_size)

        # Train model
        print("Starting training. Training set: %s" % training_iterator.trace_set_paths)
        self.model.fit_generator(training_iterator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_batch,
                                 workers=workers,
                                 callbacks=list(self.callbacks.values()) + [SavingHistory(self.base_path)],
                                 verbose=1,
                                 shuffle=True)
        # WARNING: This shuffle=True (default) has no effect because we set steps_per_epoch. See
        # https://keras.io/models/sequential/. For this reason, it is imperative that the augment
        # _shuffle option is always set to True, so that batches are shuffled at iterator level.

        # Temporary debug stuff to validate models
        # self._debug_batch(training_iterator, "train")
        # self._debug_batch(validation_iterator, "test")

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

        print(inputs.shape)
        print(labels.shape)

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
            self.model.fit(shuffled_inputs_train,
                           shuffled_labels_train,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=1,
                           callbacks=None,
                           validation_data=(shuffled_inputs_val, shuffled_labels_val))

            # Now, evaluate the rank for increasing number of traces from the validation set (steps of 10)
            validation_traces = shuffled_traces[num_train_traces:]
            for j in range(0, int(num_validation_traces / rank_trace_step)):
                validation_traces_subset = validation_traces[0:(j+1)*rank_trace_step]
                x = np.array([trace.signal for trace in validation_traces_subset])
                if conf.cnn:
                    x = np.expand_dims(x, axis=-1)
                encodings = self.model.predict(x)  # Output: [?, 16]
                keys = np.array([trace.key for trace in validation_traces_subset])
                plaintexts = np.array([trace.plaintext for trace in validation_traces_subset])
                fake_ts = traceset.TraceSet(traces=encodings, plaintexts=plaintexts, keys=keys, name="fake_ts")
                fake_ts.window = emutils.Window(begin=0, end=encodings.shape[1])
                fake_ts.windowed = True
                r, c = rank.calculate_traceset_rank(fake_ts, 2, keys[0][2], conf)
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
            r, c = rank.calculate_traceset_rank(fake_ts, 2, keys[0][2], conf)
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
        # TODO can we move this to child classes instead? i.e. in this case AICorrNet
        if self.model_type == 'autoenc':
            get_encode_layer_output = K.function([self.model.layers[0].input],
                                                 [self.model.layers[1].output])  # TODO hardcoded "encode" layer index for autoenc
            return get_encode_layer_output([x])[0]
        else:
            # return self.model.predict(x, batch_size=10000, verbose=0)
            outputs = self.model.predict(x, batch_size=self.conf.batch_size, verbose=0)
            if self.conf.loss_type == 'correlation_special':
                num_encodings = self.model.output.shape[1]-1
                encodings = outputs[:, 0:num_encodings]
                weights = np.mean(outputs[:, num_encodings], axis=0)
                return np.multiply(encodings, weights)
            else:
                return outputs

    def conf_to_name(self, model_type, conf):
        name = model_type

        name += "-" + conf.leakage_model.replace("_", "-")
        name += "-" + conf.input_type.replace("_", "-")
        name += "-" + conf.loss_type.replace("_", "-")
        name += "-e" + str(conf.epochs)
        name += "-h" + str(conf.n_hidden_layers)
        name += "-n" + str(conf.n_hidden_nodes)
        name += "-lr" + str(conf.lr).replace(".", "-")
        if not conf.cnn:
            if not conf.use_bias:
                name += "-nobias"
            if not conf.activation is None:
                name += "-" + str(conf.activation).replace("_", "-")
            if conf.batch_norm:
                name += "-bn"
            if not conf.regularizer is None:
                name += "-reg" + str(conf.regularizer)
            if conf.hamming:
                name += '-hw'
        else:
            name += '-cnn'
        name = name + self.suffix

        return name

    def load(self):
        print("Loading model %s" % self.model_path)
        custom_objects = dict()
        if not isinstance(self.loss, str):
            custom_objects[self.loss.__name__] = self.loss
        custom_objects['CCLayer'] = CCLayer

        self.model = load_model(self.model_path, custom_objects=custom_objects)

    def get_output_gradients(self, neuron_index, examples_batch, mean_of_gradients=False, square_gradients=False):
        """
        Gets the gradients of the neuron at neuron_index in the output layer of the model, with respect to a given batch of inputs.
        :param neuron_index:
        :param examples_batch:
        :param mean_of_gradients: Take mean of the gradients and return as a numpy array of the same size. Useful for visualizations.
        :param square_gradients: Square the gradients.
        :return:
        """
        # Define tensors
        gradients_tensor = K.gradients(self.model.output[:, neuron_index], self.model.input)[0]
        get_gradients = K.function([self.model.input], [gradients_tensor])

        # Get gradients of this batch
        gradients = get_gradients([examples_batch])[0]

        # Square gradient (if we don't care about the sign or about low values)
        if square_gradients:
            gradients = np.square(gradients)

        # Replace with mean
        if mean_of_gradients:
            gradients_mean = np.mean(gradients, axis=0)
            gradients = []
            for i in range(0, examples_batch.shape[0]):
                gradients.append(gradients_mean)
            gradients = np.array(gradients)

        return gradients

    def info(self):
        result = ""
        result += "Model    : %s (%s)\n" % (self.name, self.__class__.__name__)
        result += "Loss     : %s\n" % self.model.loss if isinstance(self.model.loss, str) else self.model.loss.__name__.strip()
        result += "Optimizer: %s\n" % self.model.optimizer if isinstance(self.model.optimizer, str) else self.model.optimizer.__class__.__name__.strip()
        result += "Inputs   : %d\n" % self.model.input.shape[1]
        result += "Outputs  : %d\n" % self.model.output.shape[1]
        return result


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

        if val_loss <= self.lowest:
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
    elif string == 'prelu':
        return PReLU(alpha_initializer='uniform')
    else:
        if string is None:
            return None
        else:
            return Activation(string)


class AICorrNet(AI):
    def __init__(self, conf, input_dim, name="aicorrnet"):
        super(AICorrNet, self).__init__(conf, name)

        #optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0.0)
        if self.cnn:
            optimizer = keras.optimizers.Nadam(lr=conf.lr / 10.0)
        else:
            optimizer = keras.optimizers.Nadam(lr=conf.lr)
        #optimizer = keras.optimizers.Adadelta()

        if not self.cnn:
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
            for i in range(0, self.n_hidden_layers):
                hidden_nodes = conf.n_hidden_nodes
                self.model.add(Dense(hidden_nodes, input_dim=input_dim, use_bias=self.use_bias, activation=None, kernel_initializer=initializer, kernel_regularizer=str_to_reg(self.reg, self.reg_lambda)))
                input_dim=hidden_nodes
                if self.batch_norm:
                    self.model.add(BatchNormalization(momentum=self.momentum))
                self.model.add(str_to_activation(self.activation))

            # Output layer
            extra_outputs = 1 if conf.loss_type == 'correlation_special' else 0
            self.model.add(Dense(LeakageModel.get_num_outputs(conf) + extra_outputs, input_dim=input_dim, use_bias=self.use_bias, activation=None, kernel_initializer=initializer, kernel_constraint=constraint, kernel_regularizer=str_to_reg(self.regfinal, self.reg_lambda)))
            if self.batch_norm:
                self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(str_to_activation(self.activation))
        else:
            from ASCAD_train_models import cnn_best_nosoftmax
            self.model = cnn_best_nosoftmax(input_shape=(input_dim, 1), classes=conf.key_high - conf.key_low)

        # Compile model
        self.model.compile(optimizer=optimizer, loss=self.loss, metrics=[])

        # Custom callbacks
        self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id, freq=self.metric_freq)

        if not conf.norank:
            self.callbacks['rank'] = rank.CorrRankCallback(conf, '/tmp/keras/' + self.name + '-' + self.id + '/rank/', save_best=True, save_path=self.model_path)

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
    def __init__(self, conf, input_shape, name="aishacpu", subtype='vgg16'):
        super(AISHACPU, self).__init__(conf, name)

        assert(K.image_data_format() == 'channels_last')
        input_tensor = Input(shape=input_shape)  # Does not include batch size

        self.model = None
        if subtype == 'vgg16':
            self.model = VGG16(include_top=True, weights=None, input_tensor=input_tensor, input_shape=None, pooling='avg', classes=9 if self.hamming else 256)
        elif subtype == 'custom':
            self.model = Sequential()
            self.model.add(Dense(1024, input_shape=input_shape, kernel_regularizer=self.reg))
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(Activation('relu'))
            input_shape = (None, 1024)
            self.model.add(Dense(256, input_shape=input_shape, kernel_regularizer=self.reg))
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(Activation('relu'))
            input_shape = (None, 256)
            self.model.add(Dense(9 if self.hamming else 256, use_bias=True, input_shape=input_shape, kernel_regularizer=self.reg))
            self.model.add(BatchNormalization(momentum=0.1))
            self.model.add(Activation('softmax'))

            # Extra callbacks
            #self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id)

        optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


class AISHACC(AI):
    def __init__(self, conf, input_shape, name="aishacc"):
        super(AISHACC, self).__init__(conf, name)

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
        self.model.add(Dense(9 if self.hamming else 256))
        self.model.add(BatchNormalization(momentum=0.99))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(9 if self.hamming else 256))
        self.model.add(BatchNormalization(momentum=0.99))
        self.model.add(Activation('softmax'))

        print(self.model.summary())

        # Extra callbacks
        #self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id)

        optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.0)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


class AIASCAD(AI):
    def __init__(self, conf, input_shape, name="aiascad"):
        super(AIASCAD, self).__init__(conf, name)
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


class AutoEncoder(AI):
    def __init__(self, conf, input_dim, name="autoenc"):
        super(AutoEncoder, self).__init__(conf, name)
        self.name = "autoenc"  # Override conf name

        self.model = Sequential()
        self.model.add(Dense(256, input_dim=input_dim, activation=None))
        self.model.add(str_to_activation('leakyrelu'))
        self.model.add(Dense(input_dim, activation='linear'))

        # Compile model
        optimizer = keras.optimizers.Nadam(lr=conf.lr)
        #optimizer = 'adadelta'
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[])

        # Custom callbacks
        self.callbacks['tensorboard'] = CustomTensorboard(log_dir='/tmp/keras/' + self.name + '-' + self.id, freq=self.metric_freq)

    def train_set(self, x, y, epochs=1):
        """
        Train entire training set with model.fit()

        Used in qa_emma
        """

        # Fit model
        self.model.fit(x, y, epochs=epochs, batch_size=999999999, shuffle=False, verbose=2, callbacks=[])
