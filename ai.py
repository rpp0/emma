import keras.backend as K
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

    def train(self, x, y):
        one_hot_labels = keras.utils.to_categorical(y, num_classes=self.num_outputs)
        self.model.fit(x, one_hot_labels, epochs=10, batch_size=256, shuffle=True)

    def test(self, x):
        pass

def correlation_loss(y_true, y_pred):
    loss = K.variable(0.0)
    denom = K.sqrt(K.dot(K.transpose(y_pred), y_pred)) * K.sqrt(K.dot(K.transpose(y_true), y_true))
    denom = K.maximum(denom, K.epsilon())
    correlation = K.dot(K.transpose(y_true), y_pred) / denom
    loss += 1 - K.abs(correlation)
    return loss

class Clip(keras.constraints.Constraint):
    def __init__(self, weight_range):
        self.weight_range = weight_range

    def __call__(self, w):
        return K.clip(w, self.weight_range[0], self.weight_range[1])

class LastLoss(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.value = None

    def on_batch_end(self, batch, logs={}):
        self.value = logs.get('loss')

class AICorrNet():
    def __init__(self, input_dim):
        self.model = Sequential()
        self.use_bias = False
        #initializer = keras.initializers.Constant(value=1.0/input_dim)
        #initializer = keras.initializers.Constant(value=0.5)
        initializer = keras.initializers.RandomUniform(minval=0, maxval=1.0, seed=None)
        constraint = Clip([0.0, 1.0])
        #constraint = None
        optimizer = keras.optimizers.SGD(lr=100.0, decay=1e-6, momentum=0.9, nesterov=True)
        #optimizer = 'adam'

        self.model.add(Dense(1, activation=None, use_bias=self.use_bias, kernel_initializer=initializer, input_dim=input_dim, kernel_constraint=constraint))
        self.model.compile(optimizer=optimizer, loss=correlation_loss, metrics=['accuracy'])

    def train(self, x, y):
        last_loss = LastLoss()
        self.model.fit(x, y, epochs=10, batch_size=256, shuffle=False, verbose=0, callbacks=[last_loss])

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
        print("Loss: %f" % last_loss.value)

        return activations
