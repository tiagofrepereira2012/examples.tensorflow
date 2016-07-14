#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# @date: Wed 11 May 2016 09:39:36 CEST 

import numpy


def scale_mean_norm(data, scale=0.00390625):
    mean = numpy.mean(data)
    data = (data - mean) * scale

    return data, mean


"""
Data
"""


class DataShuffler(object):
    def __init__(self, data, labels, perc_train=0.9, scale=True):
        """
         Some base functions for neural networks

         **Parameters**
           data:
        """
        scale_value = 0.00390625

        total_samples = data.shape[0]

        indexes = numpy.array(range(total_samples))
        numpy.random.shuffle(indexes)

        # Spliting train and validation
        train_samples = round(total_samples * perc_train)
        validation_samples = total_samples - train_samples
        data = numpy.reshape(data, (data.shape[0], 28, 28, 1))

        self.train_data = data[indexes[0:train_samples], :, :, :]
        self.train_labels = labels[indexes[0:train_samples]]

        self.validation_data = data[indexes[train_samples:train_samples + validation_samples], :, :, :]
        self.validation_labels = labels[indexes[train_samples:train_samples + validation_samples]]
        self.total_labels = 10

        if scale:
            # data = scale_minmax_norm(data,lower_bound = -1, upper_bound = 1)
            self.train_data, self.mean = scale_mean_norm(self.train_data)
            self.mean = (self.validation_data - self.mean) * scale_value

    def get_batch(self, n_samples, train_dataset=True):

        if train_dataset:
            data = self.train_data
            label = self.train_labels
        else:
            data = self.validation_data
            label = self.validation_labels

        # Shuffling samples
        indexes = numpy.array(range(data.shape[0]))
        numpy.random.shuffle(indexes)

        selected_data = data[indexes[0:n_samples], :, :, :]
        selected_labels = label[indexes[0:n_samples]]

        return selected_data.astype("float32"), selected_labels

    def get_pair(self, n_pair=1, is_target_set_train=True):
        """
        Get a random pair of samples

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        def get_genuine_or_not(input_data, input_labels, genuine=True):

            if genuine:
                # TODO: THIS KEY SELECTION NEEDS TO BE MORE EFFICIENT

                # Getting a client
                index = numpy.random.randint(self.total_labels)

                # Getting the indexes of the data from a particular client
                indexes = numpy.where(input_labels == index)[0]
                numpy.random.shuffle(indexes)

                # Picking a pair
                data = input_data[indexes[0], :, :, :]
                data_p = input_data[indexes[1], :, :, :]

            else:
                # Picking a pair from different clients
                index = numpy.random.choice(self.total_labels, 2, replace=False)

                # Getting the indexes of the two clients
                indexes = numpy.where(input_labels == index[0])[0]
                indexes_p = numpy.where(input_labels == index[1])[0]
                numpy.random.shuffle(indexes)
                numpy.random.shuffle(indexes_p)

                # Picking a pair
                data = input_data[indexes[0], :, :, :]
                data_p = input_data[indexes_p[0], :, :, :]

            return data, data_p

        if is_target_set_train:
            target_data = self.train_data
            target_labels = self.train_labels
        else:
            target_data = self.validation_data
            target_labels = self.validation_labels

        total_data = n_pair * 2
        c = target_data.shape[3]
        w = target_data.shape[1]
        h = target_data.shape[2]

        data = numpy.zeros(shape=(total_data, w, h, c), dtype='float32')
        data_p = numpy.zeros(shape=(total_data, w, h, c), dtype='float32')
        labels_siamese = numpy.zeros(shape=total_data, dtype='float32')

        genuine = True
        for i in range(total_data):
            data[i, :, :, :], data_p[i, :, :, :] = get_genuine_or_not(target_data, target_labels, genuine=genuine)
            labels_siamese[i] = genuine
            genuine = not genuine

        return data, data_p, labels_siamese

    def get_triplet(self, n_labels, n_triplets=1, is_target_set_train=True):
        """
        Get a triplet

        **Parameters**
            is_target_set_train: Defining the target set to get the batch

        **Return**
        """

        def get_one_triplet(input_data, input_labels):

            # Getting a pair of clients
            index = numpy.random.choice(n_labels, 2, replace=False)
            label_positive = index[0]
            label_negative = index[1]

            # Getting the indexes of the data from a particular client
            indexes = numpy.where(input_labels == index[0])[0]
            numpy.random.shuffle(indexes)

            # Picking a positive pair
            data_anchor = input_data[indexes[0], :, :, :]
            data_positive = input_data[indexes[1], :, :, :]

            # Picking a negative sample
            indexes = numpy.where(input_labels == index[1])[0]
            numpy.random.shuffle(indexes)
            data_negative = input_data[indexes[0], :, :, :]

            return data_anchor, data_positive, data_negative, label_positive, label_positive, label_negative

        if is_target_set_train:
            target_data = self.train_data
            target_labels = self.train_labels
        else:
            target_data = self.validation_data
            target_labels = self.validation_labels

        c = target_data.shape[3]
        w = target_data.shape[1]
        h = target_data.shape[2]

        data_a = numpy.zeros(shape=(n_triplets, w, h, c), dtype='float32')
        data_p = numpy.zeros(shape=(n_triplets, w, h, c), dtype='float32')
        data_n = numpy.zeros(shape=(n_triplets, w, h, c), dtype='float32')
        labels_a = numpy.zeros(shape=n_triplets, dtype='float32')
        labels_p = numpy.zeros(shape=n_triplets, dtype='float32')
        labels_n = numpy.zeros(shape=n_triplets, dtype='float32')

        for i in range(n_triplets):
            data_a[i, :, :, :], data_p[i, :, :, :], data_n[i, :, :, :], \
            labels_a[i], labels_p[i], labels_n[i] = \
                get_one_triplet(target_data, target_labels)

        return data_a, data_p, data_n, labels_a, labels_p, labels_n
