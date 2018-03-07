#!/usr/bin/python3

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.utils.test_utils import layer_test
from keras.utils.generic_utils import CustomObjectScope
import tensorflow as tf
import keras.backend as K
import keras
import ai
import numpy as np

with CustomObjectScope({'CCLayer': ai.CCLayer}):
    input_data = np.array([
        [7, 8, 9, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 2, 3],
        [1, 5, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.float32)
    labels = np.array([
        [1, 0] + [0] * 254,
        [0, 1] + [0] * 254,
    ])
    input_data = np.expand_dims(input_data, axis=2)
    cclayer_expected_output = []
    kwargs = {
        'filters': 256,
        'kernel_size': 6,
        'dilation_rate': 1,
        'padding': 'valid',
        'kernel_initializer': 'glorot_uniform',
        'use_bias': False,
        'activation': 'softmax'
    }
    layer = ai.CCLayer(**kwargs)
    x = Input(batch_shape=input_data.shape)
    y = layer(x)
    model = Model(x, y)

    '''
    with tf.Session() as sess:
        print("SIM")
        sess.run(tf.global_variables_initializer())
        y_pred = tf.placeholder(tf.float32, shape=cclayer_actual_output.shape)
        print(y_pred.shape)
        y_true = tf.placeholder(tf.float32, shape=labels.shape)
        print(y_true.shape)
        filter_score = tf.reduce_max(y_pred, axis=1, keepdims=False) * y_true
        print(filter_score.shape)

        filter_loss = tf.reduce_sum(-filter_score, axis=1)
        print(filter_loss.shape)

        loss = tf.reduce_sum(filter_loss, axis=0, keepdims=False)

        print(sess.run(tf.square(y_pred), feed_dict={y_pred: cclayer_actual_output}))
        print(sess.run(tf.reduce_max(tf.square(y_pred), axis=1, keepdims=False), feed_dict={y_pred: cclayer_actual_output}))
        print(sess.run(y_true, feed_dict={y_true: labels}))
        print(sess.run(filter_loss, feed_dict={y_pred: cclayer_actual_output, y_true: labels}))
        print(sess.run(loss, feed_dict={y_pred: cclayer_actual_output, y_true: labels}))
        print("MODEL")
        model.compile(optimizer='adam', loss=ai.cc_loss)
        closs = model.train_on_batch(input_data, labels)
        print(closs)
    exit(0)
    '''

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    for i in range(0, 6000):
        loss = model.train_on_batch(input_data, labels)
        if i % 100 == 0:
            print(loss)

    kernel = np.array(layer.get_weights()[0])
    print("KERNEL")
    print(kernel.shape)
    print((kernel - np.mean(kernel, axis=0)) / np.std(kernel, axis=0))
    cclayer_actual_output = model.predict(input_data)
    print("OUTPUT")
    print(cclayer_actual_output.shape)
    print(cclayer_actual_output)
    #best_points = np.amax(cclayer_actual_output, axis=1)
    #print(best_points)
    predicted_classes = np.argmax(cclayer_actual_output, axis=1)
    print(predicted_classes)


    # Use generic Keras checks (loading, saving, etc.) as extra test
    #layer_test(ai.CCLayer, input_data=input_data, kwargs=kwargs, expected_output=expected_output)
