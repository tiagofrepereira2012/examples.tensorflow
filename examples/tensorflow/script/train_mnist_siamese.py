#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains MNIST with LENET using Tensor flow

Usage:
  train_mnist.py [--batch-size=<arg> --iterations=<arg> --validation-interval=<arg> --use-gpu]
  train_mnist.py -h | --help
Options:
  -h --help     Show this screen.
  --batch-size=<arg>  [default: 1]
  --iterations=<arg>  [default: 30000]
  --validation-interval=<arg>  [default: 100]
"""

from docopt import docopt
import tensorflow as tf
from .. import util
from ..DataShuffler import *
from ..lenet import Lenet
from matplotlib.backends.backend_pdf import PdfPages
import sys
import bob.measure

SEED = 10
from ..DataShuffler import *


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    with tf.name_scope('euclidean_distance') as scope:
        #d = tf.square(tf.sub(x, y))
        #d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), 1))
        return d


def compute_contrastive_loss(left_feature, right_feature, label, margin, is_target_set_train=True):

    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2

    OR MAYBE THAT

    L = 0.5 * (1-Y) * D^2 + 0.5 * (Y) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation

    """

    with tf.name_scope("contrastive_loss"):
        label = tf.to_float(label)
        one = tf.constant(1.0)

        d = compute_euclidean_distance(left_feature, right_feature)
        # first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)
        first_part = tf.mul(one-label, tf.square(d))  # (1-Y)*(d^2)

        max_part = tf.square(tf.maximum(d - margin, 0))
        second_part = tf.mul(label, max_part)  # (Y) * max((margin - d)^2, 0)

        loss = 0.5 * tf.reduce_mean(first_part + second_part)

        return loss, tf.reduce_mean(first_part), tf.reduce_mean(second_part)




def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    BATCH_SIZE_TEST = int(args['--batch-size'])
    #BATCH_SIZE_TEST = 300
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    perc_train = 0.9
    CONTRASTIVE_MARGIN = 0.1
    USE_GPU = args['--use-gpu']


    data, labels = util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
    data_shuffler = DataShuffler(data, labels, scale=True)

    # Creating the variables
    lenet_architecture = Lenet(seed=SEED, use_gpu=USE_GPU)

    # Siamease place holders - Training
    train_left_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE*2, 28, 28, 1), name="left")
    train_right_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE * 2, 28, 28, 1), name="right")
    labels_data = tf.placeholder(tf.int32, shape=BATCH_SIZE * 2)

    # Creating the graphs for training
    lenet_train_left = lenet_architecture.create_lenet(train_left_data)
    lenet_train_right = lenet_architecture.create_lenet(train_right_data)
    #lenet_validation = lenet_architecture.create_lenet(validation_data, train=False)


    loss, between_class, within_class = compute_contrastive_loss(lenet_train_left, lenet_train_right, labels_data, CONTRASTIVE_MARGIN)
    distances = compute_euclidean_distance(lenet_train_left, lenet_train_right)


    #regularizer = (tf.nn.l2_loss(lenet_architecture.W_fc1) + tf.nn.l2_loss(lenet_architecture.b_fc1) +
    #                tf.nn.l2_loss(lenet_architecture.W_fc2) + tf.nn.l2_loss(lenet_architecture.b_fc2))
    #loss += 5e-4 * regularizer

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.0001, # Learning rate
        batch * BATCH_SIZE,
        data_shuffler.train_data.shape[0],
        0.95 # Decay step
    )

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.99, use_locking=False, name='Momentum').minimize(loss, global_step=batch)

    # Training

    with tf.Session() as session:

        #Trying to write things on tensor board
        train_writer = tf.train.SummaryWriter('./logs_tensorboard/siamese/train',
                                              session.graph)

        test_writer = tf.train.SummaryWriter('./logs_tensorboard/siamese/test',
                                              session.graph)

        tf.scalar_summary('loss', loss)
        tf.scalar_summary('between_class', between_class)
        tf.scalar_summary('within_class', within_class)
        tf.scalar_summary('lr', learning_rate)
        merged = tf.merge_all_summaries()

        tf.initialize_all_variables().run()

        for step in range(ITERATIONS):

            batch_left, batch_right, labels = data_shuffler.get_pair(BATCH_SIZE)

            feed_dict = {train_left_data: batch_left,
                         train_right_data: batch_right,
                         labels_data: labels}

            _, l, lr, summary = session.run([optimizer, loss, learning_rate, merged], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)

            if step % VALIDATION_TEST == 0:

                batch_left, batch_right, labels = data_shuffler.get_pair(n_pair=BATCH_SIZE_TEST,
                                                                         is_target_set_train=False)
                feed_dict = {train_left_data: batch_left,
                             train_right_data: batch_right,
                             labels_data: labels}

                d, lv, summary = session.run([distances, loss, merged], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)

                positive_scores = d[numpy.where(labels == 1)[0]].astype("float")
                negative_scores = d[numpy.where(labels == 0)[0]].astype("float")

                threshold = bob.measure.eer_threshold(negative_scores, positive_scores)
                far, frr = bob.measure.farfrr(negative_scores, positive_scores, threshold)
                eer = ((far + frr) / 2.) * 100.
                print("EER = {0}".format(eer))
                print("Loss Validation {0}".format(lv))

        print("End !!")
        train_writer.close()
        test_writer.close()


