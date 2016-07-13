#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
numpy.random.seed(10)


def create_weight_variables(shape, seed):
    """
    Create gaussian random neurons with mean 0 and std 0.1

    **Paramters**

      shape: Shape of the layer
    """

    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)


def create_bias_variables(shape):
    """
    Create the bias term
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def create_conv2d(x, W):
    """
    Create a convolutional kernel with 1 pixel of stride

    **Parameters**
        x: input layer
        W: Neurons

    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def create_max_pool(x):
    """
    Create max pooling using a patch of 2x2

    **Parameters**
        x: input layer
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def create_relu(x, bias):
    """
    Create the ReLU activations

    **Parameters**
        x: input layer
        bias: bias term

    """

    return tf.nn.relu(tf.nn.bias_add(x, bias))


def scale_mean_norm(data, scale=0.00390625):
    mean = numpy.mean(data)
    data = (data - mean) * scale

    return data


def evaluate(data, labels, session, network, data_node):

    """
    Evaluate the network using the validation set and compute the accuracy
    """

    predictions = numpy.argmax(session.run(
        network,
        feed_dict={data_node: data[:]}), 1)

    return 100. * numpy.sum(predictions == labels) / predictions.shape[0]



def load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/"):

    import bob.db.mnist
    db = bob.db.mnist.Database(data_dir)
    raw_data = db.data()

    # data  = raw_data[0].astype(numpy.float64)
    data = raw_data[0]
    labels = raw_data[1]

    return data, labels

