#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains a triplet net (Lenet as basis) with MNIST data

Usage:
  train_mnist_triplet.py [--batch-size=<arg> --iterations=<arg> --validation-interval=<arg>]
  train_mnist_triplet.py -h | --help
Options:
  -h --help     Show this screen.
  --batch-size=<arg>  [default: 1]
  --iterations=<arg>  [default: 30000]
  --validation-interval=<arg>  [default: 100]
"""

from docopt import docopt
import tensorflow as tf
from .. import util
from ..DataShuffler import DataShuffler
from ..lenet import Lenet

SEED = 10


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    d = tf.square(tf.sub(x, y))
    d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
    return d


def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):

    """
    Compute the contrastive loss as in


    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m

    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin

    **Returns**
     Return the loss operation

    """

    d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
    d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

    loss = d_p_squared - d_n_squared + margin

    return loss


def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    perc_train = 0.9
    MARGIN = 0.2

    data, labels = util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
    data_shuffler = DataShuffler(data, labels)

    # Siamease place holders
    train_anchor_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name="anchor")
    train_positive_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name="positive")
    train_negative_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1), name="negative")
    labels_anchor = tf.placeholder(tf.int32, shape=BATCH_SIZE)
    labels_positive = tf.placeholder(tf.int32, shape=BATCH_SIZE)
    labels_negative = tf.placeholder(tf.int32, shape=BATCH_SIZE)

    validation_data = tf.placeholder(tf.float32, shape=(data_shuffler.validation_data.shape[0], 28, 28, 1))

    # Creating the architecture
    lenet_architecture = Lenet(seed=SEED)
    lenet_train_anchor = lenet_architecture.create_lenet(train_anchor_data)
    lenet_train_positive = lenet_architecture.create_lenet(train_positive_data)
    lenet_train_negative = lenet_architecture.create_lenet(train_negative_data)
    lenet_validation = lenet_architecture.create_lenet(validation_data, train=False)

    # Defining the triplet loss
    anchor_output = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lenet_train_anchor, labels_anchor))
    positive_output = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lenet_train_positive, labels_positive))
    negative_output = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lenet_train_negative, labels_negative))

    loss = compute_triplet_loss(anchor_output, positive_output, negative_output, MARGIN)

    #regularizer = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
    #                tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    #loss += 5e-4 * regularizer

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001, # Learning rate
        batch * BATCH_SIZE,
        data_shuffler.train_data.shape[0],
        0.95 # Decay step
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    validation_prediction = tf.nn.softmax(lenet_validation)

    print("Initializing")
    # Training
    with tf.Session() as session:

        tf.initialize_all_variables().run()

        for step in range(ITERATIONS):

            batch_anchor, batch_positive, batch_negative, \
            batch_labels_anchor, batch_labels_positive, batch_labels_negative = data_shuffler.get_triplet(n_labels=10)

            feed_dict = {train_anchor_data: batch_anchor,
                         train_positive_data: batch_positive,
                         train_negative_data: batch_negative,
                         labels_anchor: batch_labels_anchor,
                         labels_positive: batch_labels_positive,
                         labels_negative: batch_labels_negative
            }

            _, l, lr = session.run([optimizer, loss, learning_rate],
                                                feed_dict=feed_dict)

            if step % VALIDATION_TEST == 0:
                batch_validation_data, batch_validation_labels = data_shuffler.get_batch(data_shuffler.validation_data.shape[0],
                                                                             train_dataset=False)
                accuracy = util.evaluate(batch_validation_data, batch_validation_labels, session, validation_prediction,
                                    validation_data)
                print("Step {0}. Loss = {1}, Lr={2}, Accuracy validation = {3}".format(step, l, lr, accuracy))

        print("Step {0}. Loss = {1}, Lr={2}, Accuracy validation = {3}".format(step, l, lr, accuracy))
        print("End !!")
