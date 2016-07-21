#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy
import tensorflow as tf
numpy.random.seed(10)


def create_weight_variables(shape, seed, name, use_gpu=False):
    """
    Create gaussian random neurons with mean 0 and std 0.1

    **Paramters**

      shape: Shape of the layer
    """

    #import ipdb; ipdb.set_trace()

    if len(shape) == 4:
        in_out = shape[0] * shape[1] * shape[2] + shape[3]
    else:
        in_out = shape[0] + shape[1]

    import math
    stddev = math.sqrt(3.0 / in_out) # XAVIER INITIALIZER (GAUSSIAN)

    initializer = tf.truncated_normal(shape, stddev=stddev, seed=seed)
    
    if use_gpu:
        with tf.device("/gpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    else:
        with tf.device("/cpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)


def create_bias_variables(shape, name, use_gpu=False):
    """
    Create the bias term
    """
    initializer = tf.constant(0.1, shape=shape)
    
    if use_gpu:
        with tf.device("/gpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    else:
        with tf.device("/cpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    
    
   


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


def create_sigmoid(x, bias):
    """
    Create the Sigmoid activations

    **Parameters**
        x: input layer
        bias: bias term

    """

    return tf.nn.sigmoid(tf.nn.bias_add(x, bias))


def scale_mean_norm(data, scale=0.00390625):
    mean = numpy.mean(data)
    data = (data - mean) * scale

    return data


def evaluate_softmax(data, labels, session, network, data_node):

    """
    Evaluate the network assuming that the output layer is a softmax
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


def plot_embedding_pca(features, labels):
    """

    Trains a PCA using bob, reducing the features to dimension 2 and plot it the possible clusters

    :param features:
    :param labels:
    :return:
    """

    import bob.learn.linear
    import matplotlib.pyplot as mpl

    colors = ['#FF0000', '#FFFF00', '#FF00FF', '#00FFFF', '#000000',
             '#AA0000', '#AAAA00', '#AA00AA', '#00AAAA', '#330000']

    # Training PCA
    trainer = bob.learn.linear.PCATrainer()
    machine, lamb = trainer.train(features.astype("float64"))

    # Getting the first two most relevant features
    projected_features = machine(features.astype("float64"))[:, 0:2]

    # Plotting the classes
    n_classes = max(labels)+1
    fig = mpl.figure()

    for i in range(n_classes):
        indexes = numpy.where(labels == i)[0]

        selected_features = projected_features[indexes,:]
        mpl.scatter(selected_features[:, 0], selected_features[:, 1],
                 marker='.', c=colors[i], linewidths=0, label=str(i))
    mpl.legend()
    return fig

def plot_embedding_lda(features, labels):
    """

    Trains a LDA using bob, reducing the features to dimension 2 and plot it the possible clusters

    :param features:
    :param labels:
    :return:
    """

    import bob.learn.linear
    import matplotlib.pyplot as mpl

    colors = ['#FF0000', '#FFFF00', '#FF00FF', '#00FFFF', '#000000',
             '#AA0000', '#AAAA00', '#AA00AA', '#00AAAA', '#330000']
    n_classes = max(labels)+1


    # Training PCA
    trainer = bob.learn.linear.FisherLDATrainer(use_pinv=True)
    lda_features = []
    for i in range(n_classes):
        indexes = numpy.where(labels == i)[0]
        lda_features.append(features[indexes, :].astype("float64"))

    machine, lamb = trainer.train(lda_features)

    #import ipdb; ipdb.set_trace();


    # Getting the first two most relevant features
    projected_features = machine(features.astype("float64"))[:, 0:2]

    # Plotting the classes
    fig = mpl.figure()

    for i in range(n_classes):
        indexes = numpy.where(labels == i)[0]

        selected_features = projected_features[indexes,:]
        mpl.scatter(selected_features[:, 0], selected_features[:, 1],
                 marker='.', c=colors[i], linewidths=0, label=str(i))
    mpl.legend()
    return fig


def compute_eer(data_train, labels_train, data_validation, labels_validation, n_classes):
    import bob.measure
    from scipy.spatial.distance import cosine

    # Creating client models
    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(numpy.mean(data_train[indexes, :], axis=0))

    # Probing
    positive_scores = numpy.zeros(shape=0)
    negative_scores = numpy.zeros(shape=0)

    for i in range(n_classes):
        # Positive scoring
        indexes = labels_validation == i
        positive_data = data_validation[indexes, :]
        p = [cosine(models[i], positive_data[j]) for j in range(positive_data.shape[0])]
        positive_scores = numpy.hstack((positive_scores, p))

        # negative scoring
        indexes = labels_validation != i
        negative_data = data_validation[indexes, :]
        n = [cosine(models[i], negative_data[j]) for j in range(negative_data.shape[0])]
        negative_scores = numpy.hstack((negative_scores, n))

    # Computing performance based on EER
    negative_scores = (-1) * negative_scores
    positive_scores = (-1) * positive_scores

    threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
    far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
    eer = (far + frr) / 2.

    return eer


def compute_accuracy(data_train, labels_train, data_validation, labels_validation, n_classes):
    from scipy.spatial.distance import cosine

    # Creating client models
    models = []
    for i in range(n_classes):
        indexes = labels_train == i
        models.append(numpy.mean(data_train[indexes, :], axis=0))

    # Probing
    tp = 0
    for i in range(data_validation.shape[0]):

        d = data_validation[i,:]
        l = labels_validation[i]

        scores = [cosine(m, d) for m in models]
        predict = numpy.argmax(scores)

        if predict == l:
            tp += 1

    return (float(tp) / data_validation.shape[0]) * 100
