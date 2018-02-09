import numpy as np
np.random.seed(1)  # Make results reproducible
import keras.backend as K
import keras
import pickle
import time
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.callbacks import TensorBoard

class AI():
    def __init__(self, name="unknown"):
        self.id = str(int(time.time()))
        self.name = name
        self.models_dir = os.path.join(os.getcwd(), 'models')
        if not os.path.isdir(self.models_dir):
            os.makedirs(self.models_dir)
        self.model_path = os.path.join(self.models_dir, "%s.h5" % self.name)

class AIMemCopyDirect():
    '''
    Extremely simple NN that attempts to find a relation between the power consumption (input)
    and the resulting one-hot encoding of the byte that was copied from memory.
    '''
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

def correlation_loss(y_true, y_pred):
    loss = K.variable(0.0)
    for key_col in range(0, 16):  # 0 - 16
        y_key = K.expand_dims(y_true[:,key_col], axis=1)  # [?, 16] -> [?, 1]
        denom = K.sqrt(K.dot(K.transpose(y_pred), y_pred)) * K.sqrt(K.dot(K.transpose(y_key), y_key))
        denom = K.maximum(denom, K.epsilon())
        correlation = K.dot(K.transpose(y_key), y_pred) / denom
        loss += 1.0 - K.square(correlation)
    return loss

class Clip(keras.constraints.Constraint):
    def __init__(self):
        self.weight_range = [0.0, 1.0]

    def __call__(self, w):
        return K.clip(w, self.weight_range[0], self.weight_range[1])
keras.constraints.Clip = Clip

class LastLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.value = None

    def on_batch_end(self, batch, logs={}):
        self.value = logs.get('loss')

class AICorrNet(AI):
    def __init__(self, input_dim, name="aicorrnet"):
        super(AICorrNet, self).__init__(name)
        self.model = Sequential()
        self.use_bias = False
        #initializer = keras.initializers.Constant(value=1.0/input_dim)
        #initializer = keras.initializers.Constant(value=0.5)
        #initializer = keras.initializers.Constant(value=1.0)
        #initializer = keras.initializers.RandomUniform(minval=0, maxval=1.0, seed=None)
        initializer = keras.initializers.RandomUniform(minval=0, maxval=0.001, seed=None)
        #initializer = 'glorot_uniform'
        #constraint = Clip()
        constraint = None
        #optimizer = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.0)
        activation = None
        #activation = 'relu'

        #self.model.add(Dense(256, input_dim=input_dim, activation='tanh'))
        #input_dim=256
        self.model.add(Dense(1, use_bias=self.use_bias, kernel_initializer=initializer, kernel_constraint=constraint, input_dim=input_dim))
        self.model.add(BatchNormalization())  # Required for correct correlation calculation
        if not activation is None:
            self.model.add(Activation(activation))
        self.model.compile(optimizer=optimizer, loss=correlation_loss, metrics=[])

    def train_set(self, x, y, save=True, epochs=1):
        y = y - np.mean(y, axis=0) # Required for correct correlation calculation! Note that x is normalized using batch normalization. In Keras, this function also remembers the mean and variance from the training set batches. Therefore, there's no need to normalize before calling model.predict

        # Callbacks
        last_loss = LastLoss()
        tensorboard_callback = TensorBoard(log_dir='/tmp/keras/' + self.id)

        # Fit model
        self.model.fit(x, y, epochs=epochs, batch_size=999999999, shuffle=False, verbose=2, callbacks=[last_loss, tensorboard_callback])

        # Get loss from callback
        self.last_loss = last_loss.value

        self._post_train(save)

    def train_generator(self, generator, epochs=2000, workers=1, save=True):
        # Callbacks
        last_loss = LastLoss()
        tensorboard_callback = TensorBoard(log_dir='/tmp/keras/' + self.id)

        # Train model
        self.model.fit_generator(generator,
                                epochs=epochs,
                                steps_per_epoch=1,
                                #validation_data=(x_test, y_test),
                                workers=workers, callbacks=[last_loss, tensorboard_callback], verbose=2)

        # Get loss from callback
        self.last_loss = last_loss.value

        self._post_train(save)

    def _post_train(self, save=True):
        '''
        Do some post-train actions like getting the model weights and saving the model.
        '''
        activations = self.model.get_weights()[0]
        print(activations)
        if self.use_bias:
            bias = self.model.get_weights()[1]
            print(bias)

        top = 10
        activations = np.reshape(activations, -1)
        best_indices = np.argsort(activations)[-top:]
        print("Best indices: %s" % str(best_indices))
        print("Max weights: %s" % str([activations[i] for i in best_indices]))

        # Save progress
        if save:
            pickle.dump(activations, open("/tmp/weights.p", "wb"))  # TODO remove me later. Use Tensorboard instead
            self.model.save(self.model_path)

    def predict(self, x):
        return self.model.predict(x, batch_size=999999999, verbose=0)

    def load(self):
        self.model = load_model(self.model_path, custom_objects={'correlation_loss': correlation_loss})
