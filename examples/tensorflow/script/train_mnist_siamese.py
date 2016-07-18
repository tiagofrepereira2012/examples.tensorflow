#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 


"""
Simple script that trains MNIST with LENET using Tensor flow

Usage:
  train_mnist.py [--batch-size=<arg> --iterations=<arg> --validation-interval=<arg>]
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

SEED = 10
from ..DataShuffler import *


def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    #d = tf.square(tf.sub(x, y))
    #d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
    d = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), 1))
    return d


def compute_contrastive_loss(left_feature, right_feature, label, margin):

    """
    Compute the contrastive loss as in

    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    L = 0.5 * (Y) * D^2 + 0.5 * (1-Y) * {max(0, margin - D)}^2

    **Parameters**
     left_feature: First element of the pair
     right_feature: Second element of the pair
     label: Label of the pair (0 or 1)
     margin: Contrastive margin

    **Returns**
     Return the loss operation

    """

    #label = tf.to_float(label)
    #one = tf.constant(1.0)
    #zero = tf.constant(0.0)
    #half = tf.constant(0.5)
    #m = tf.constant(margin)

    #d = compute_euclidean_distance(left_feature, right_feature)
    #first_part = tf.mul(label, tf.square(d))# (Y)*(d^2)

    #max_part = tf.square(tf.maximum(m-d, zero))
    #second_part = tf.mul(one-label, max_part)  # (1-Y) * max(margin - d, 0)
    #loss = half * tf.reduce_sum(first_part + second_part)

    #return loss


    # Stack overflow "fix"

    label = tf.to_float(label)
    one = tf.constant(1.0)

    d = compute_euclidean_distance(left_feature, right_feature)
    first_part = tf.mul(one - label, tf.square(d))  # (Y-1)*(d^2)

    max_part = tf.square(tf.maximum(margin - d, 0))
    second_part = tf.mul(label, max_part)  # (Y) * max((margin - d)^2, 0)

    loss = 0.5 * tf.reduce_mean(first_part + second_part)

    return loss






def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    perc_train = 0.9
    CONTRASTIVE_MARGIN = 0.2

    data, labels = util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
    data_shuffler = DataShuffler(data, labels, scale=True)

    # Siamease place holders
    train_left_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE*2, 28, 28, 1), name="left")
    train_right_data = tf.placeholder(tf.float32, shape=(BATCH_SIZE * 2, 28, 28, 1), name="right")
    labels_data = tf.placeholder(tf.int32, shape=BATCH_SIZE*2)

    validation_data = tf.placeholder(tf.float32, shape=(data_shuffler.validation_data.shape[0], 28, 28, 1))

    # Creating the architecture
    lenet_architecture = Lenet(seed=SEED)
    lenet_train_left = lenet_architecture.create_lenet(train_left_data)
    lenet_train_right = lenet_architecture.create_lenet(train_right_data)
    lenet_validation = lenet_architecture.create_lenet(validation_data, train=False)

    # Defining the constrastive loss
    #left_output = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lenet_train_left, labels_data))
    #right_output = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lenet_train_right, labels_data))
    loss = compute_contrastive_loss(lenet_train_left, lenet_train_right, labels_data, CONTRASTIVE_MARGIN)

    #regularizer = (tf.nn.l2_loss(lenet_architecture.W_fc1) + tf.nn.l2_loss(lenet_architecture.b_fc1) +
    #                tf.nn.l2_loss(lenet_architecture.W_fc2) + tf.nn.l2_loss(lenet_architecture.b_fc2))
    #loss += 5e-4 * regularizer

    # Defining training parameters
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01, # Learning rate
        batch * BATCH_SIZE,
        data_shuffler.train_data.shape[0],
        0.95 # Decay step
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    #validation_prediction = tf.nn.softmax(lenet_validation)

    print("Initializing")
    # Training

    with tf.Session() as session:

        tf.initialize_all_variables().run()

        pp = PdfPages("groups.pdf")
        for step in range(ITERATIONS):

            batch_left, batch_right, labels = data_shuffler.get_pair(BATCH_SIZE)

            feed_dict = {train_left_data: batch_left,
                         train_right_data: batch_right,
                         labels_data: labels}

            _, l, lr = session.run([optimizer, loss, learning_rate], feed_dict=feed_dict)

            if step % VALIDATION_TEST == 0:

                batch_train_data, batch_train_labels = data_shuffler.get_batch(
                    data_shuffler.validation_data.shape[0],
                    train_dataset=True)

                features_train = session.run(lenet_validation,
                                       feed_dict={validation_data: batch_train_data[:]})

                batch_validation_data, batch_validation_labels = data_shuffler.get_batch(data_shuffler.validation_data.shape[0],
                                                                                         train_dataset=False)

                features_validation = session.run(lenet_validation,
                                       feed_dict={validation_data: batch_validation_data[:]})

                #eer = util.compute_eer(features_train, batch_train_labels, features_validation, batch_validation_labels, 10)
                #print("Step {0}. Loss = {1}, Lr={2}, EER = {3}".
                #      format(step, l, lr, eer))

                accuracy = util.compute_accuracy(features_train, batch_train_labels, features_validation, batch_validation_labels, 10)
                print("Step {0}. Loss = {1}, Lr={2}, Acc = {3}".
                      format(step, l, lr, accuracy))

                fig = util.plot_embedding_lda(features_validation, batch_validation_labels)

                pp.savefig(fig)



                #accuracy_train = util.evaluate_softmax(batch_train_data, batch_train_labels, session,
                #                               tf.nn.softmax(lenet_validation),
                #                               validation_data)

                #accuracy_validation = util.evaluate_softmax(batch_validation_data, batch_validation_labels, session,
                #                                    tf.nn.softmax(lenet_validation), validation_data)
                #
                #print("Step {0}. Loss = {1}, Lr={2}, Accuracy train = {3}, Accuracy validation = {4}".
                #      format(step, l, lr, accuracy_train, accuracy_validation))

        #print("EER = {0}".format(eer))
        #print("Step {0}. Loss = {1}, Lr={2}, Accuracy train = {3}, Accuracy validation = {4}".
        #      format(step, l, lr, accuracy_train, accuracy_validation))
        pp.close()
        print("End !!")


