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
SEED = 10
from ..DataShuffler import *
from ..lenet import Lenet


def main():
    args = docopt(__doc__, version='Mnist training with TensorFlow')

    BATCH_SIZE = int(args['--batch-size'])
    ITERATIONS = int(args['--iterations'])
    VALIDATION_TEST = int(args['--validation-interval'])
    perc_train = 0.9

    # Loading data
    data, labels = util.load_mnist(data_dir="./src/bob.db.mnist/bob/db/mnist/")
    data_shuffler = DataShuffler(data, labels)

    # Defining place holders for train and validation
    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 28, 28, 1))
    train_labels_node = tf.placeholder(tf.int64, shape=BATCH_SIZE)
    validation_data_node = tf.placeholder(tf.float32, shape=(data_shuffler.validation_data.shape[0], 28, 28, 1))

    # Creating the architecture for train and validation
    lenet_architecture = Lenet(seed=SEED)
    lenet_train = lenet_architecture.create_lenet(train_data_node)
    lenet_validation = lenet_architecture.create_lenet(validation_data_node, train=False)

    # Simple loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(lenet_train, train_labels_node))

    #regularizer = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
    #                tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
    #loss += 5e-4 * regularizer

    # Learning rate
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001, # Learning rate
        batch * BATCH_SIZE,
        data_shuffler.train_data.shape[0],
        0.95 # Decay step
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)
    train_prediction = tf.nn.softmax(lenet_train)
    validation_prediction = tf.nn.softmax(lenet_validation)

    print("Initializing !!")
    # Training
    with tf.Session() as session:
        tf.initialize_all_variables().run()
        for step in range(ITERATIONS):

            train_data, train_labels = data_shuffler.get_batch(BATCH_SIZE)

            feed_dict = {train_data_node: train_data,
                         train_labels_node: train_labels}

            _, l, lr, predictions = session.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
            if step % VALIDATION_TEST == 0:
                validation_data, validation_labels = data_shuffler.get_batch(data_shuffler.validation_data.shape[0],
                                                                             train_dataset=False)

                accuracy = util.evaluate(validation_data, validation_labels, session, validation_prediction, validation_data_node)
                print("Step {0}. Loss = {1}, Lr={2}, Accuracy validation = {3}".format(step, l, lr, accuracy))
    print("Step {0}. Loss = {1}, Lr={2}, Accuracy validation = {3}".format(step, l, lr, accuracy))


print("End !!")
