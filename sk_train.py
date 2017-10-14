#!/usr/bin/env python
""" Implements S-K Algorithm for training SVMs on Zener Cards. """

import sys
import os
import re
import numpy as np
from PIL import Image


class SVM(object):
    """An SVM model

    Attirbutes:
        epsilon:
        max_updates
        class_letter
        zener_card_letters
        pos_input
        neg_input
        pos_centroid
        neg_centroid

    """

    def __init__(self, epsilon, max_updates, class_letter):
        """ Return a new SVM object

        :param epsilon:
        :param max_updates: No. of maximum updates
        :param class_letter: Image class for which SVM is trained
        :return: returns nothing
        """

        self.epsilon = epsilon
        self.max_updates = max_updates
        self.class_letter = class_letter
        self.lambda_max = 0.0
        # TODO: Convert to list
        self.pos_input = np.zeros(625)
        self.neg_input = np.zeros(625)
        self.pos_centroid = np.zeros(625)
        self.neg_centroid = np.zeros(625)

    def set_training_inputs(self, train_folder):
        """Sets positive and negative training data arrays.

        :param train_folder: Folder name where training inputs are stored
        "returns: returns nothing
        """
        # Check if folder exists
        if os.path.isdir(train_folder) is not True:
            print >> sys.stderr, "NO DATA"
            exit(1)

        is_empty = True

        # Training class filename pattern (positive input pattern)
        pos_pattern = re.compile("[0-9]+_" + self.class_letter + ".png")

        class_letter_set = set(list(self.class_letter))

        zener_card_letters = set(['O', 'P', 'Q', 'S', 'W'])

        # Get class letters of negative inputs
        neg_class_letter_set = zener_card_letters - class_letter_set

        neg_class_letters = "".join(list(neg_class_letter_set))

        # Non-training class filename pattern (negative input pattern)
        neg_pattern = re.compile("[0-9]+_[" + neg_class_letters + "].png")

        pos_input_sum = np.zeros(625)
        neg_input_sum = np.zeros(625)

        # TODO: Remove this
        files = []
        for filename in os.listdir(train_folder):
            files.append(filename)

        files.sort()

        # Convert images to numpy array of pixels
        for filename in files:
            # Get absolute path
            abs_path = os.path.abspath(train_folder) + "/" + filename
            # TODO: Remove duplicate code
            # Check if filename matches training class pattern
            if pos_pattern.match(filename):
                is_empty = False
                # Open image using PIL
                image = Image.open(abs_path)
                # Convert to numpy array
                img_array = np.array(image)
                # Reshape array to one dimension
                img_array = img_array.reshape(-1)
                # img_array /= 255
                # Add to tpos_input_sum to calculate the centroid
                pos_input_sum = np.add(pos_input_sum, img_array)
                # Append to positive input collection array
                self.pos_input = np.vstack((self.pos_input, img_array))

            # Check if filename matches negative input class pattern
            elif neg_pattern.match(filename):
                is_empty = False
                # Open image using PIL
                image = Image.open(abs_path)
                # Convert to numpy array
                img_array = np.array(image)
                # Reshape array to one dimension
                img_array = img_array.reshape(-1)
                # img_array /= 255
                # Add to neg_input_sum to calculate the centroid
                neg_input_sum = np.add(neg_input_sum, img_array)
                # Append to positive input collection array
                self.neg_input = np.vstack((self.neg_input, img_array))

        if is_empty is True:
            print >> sys.stderr, "NO DATA"
            exit(1)

        self.pos_input = np.delete(self.pos_input, 0, 0)
        self.neg_input = np.delete(self.neg_input, 0, 0)

        self.__set_input_class_centroids(pos_input_sum, neg_input_sum)

    def __set_input_class_centroids(self, pos_sum, neg_sum):
        """Sets positive and negative training input centroids

        :param pos_sum: Array sum of positive inputs
        :param neg_sum: Array sum of negative inputs
        :return: returns nothing
        """
        self.pos_centroid = pos_sum / self.pos_input.shape[0]
        self.neg_centroid = neg_sum / self.neg_input.shape[0]

    def train(self, output_file):
        self.__scale_convex_hull()

        init_params = dict.fromkeys(['sk_a', 'sk_b', 'sk_c', 'sk_pos_d',
                                     'sk_neg_d', 'sk_pos_e', 'sk_neg_e'])

        self.__sk_initialize(init_params)

        print self.__sk_stop(init_params)

        print init_params['sk_a'], init_params['sk_b'], init_params['sk_c']

        self.__sk_update(init_params)

    def __scale_convex_hull(self):
        """Scale convex hull of inputs

        Calculate maximum value of scaling factor (lambda) and
        scales the inputs using the lambda value
        """

        radius = np.linalg.norm(self.pos_centroid - self.neg_centroid)

        pos_radius = max([np.linalg.norm(p - self.pos_centroid)
                          for p in self.pos_input])

        neg_radius = max([np.linalg.norm(p - self.neg_centroid)
                          for p in self.neg_input])

        self.lambda_max = radius / (pos_radius + neg_radius)

        self.__scale_inputs()

    def __scale_inputs(self):
        """Scale inputs"""
        # print self.pos_input[0]
        # print self.pos_centroid
        self.pos_input *= self.lambda_max
        self.pos_input += ((1 - self.lambda_max) * self.pos_centroid)

        self.neg_input *= self.lambda_max
        self.neg_input += ((1 - self.lambda_max) * self.neg_centroid)

        # print self.pos_input[0]
        # print self.neg_input[0]

    def __sk_initialize(self, params):
        """Initialize parameters for S-K algorithms

        :param params:
        """

        pos_alpha = np.zeros(self.pos_input.shape[0])
        neg_alpha = np.zeros(self.neg_input.shape[1])

        pos_alpha[0] = 1
        neg_alpha[0] = 1

        params['sk_a'] = self.__polynomial_kernal(
            self.pos_input[0], self.pos_input[0])

        params['sk_b'] = self.__polynomial_kernal(
            self.neg_input[0], self.neg_input[0])

        params['sk_c'] = self.__polynomial_kernal(
            self.pos_input[0], self.neg_input[0])

        params['sk_pos_d'] = [self.__polynomial_kernal(
            p, self.pos_input[0]) for p in self.pos_input]

        params['sk_neg_d'] = [self.__polynomial_kernal(
            p, self.pos_input[0]) for p in self.neg_input]

        params['sk_pos_e'] = [self.__polynomial_kernal(
            p, self.neg_input[0]) for p in self.pos_input]

        params['sk_neg_e'] = [self.__polynomial_kernal(
            p, self.neg_input[0]) for p in self.neg_input]

    def __polynomial_kernal(self, vector_a, vector_b):
        """Return result of degree four polynomial kernel function

        :param a:
        :param b:
        :returns:
        """

        result = np.dot(vector_a, vector_b) + 1
        result = result ** 4

        return result

    def __sk_stop(self, init_param):
        """Check if the stopping condition is true

        :param init_param:
        """

        pos_m_array = [self.__calculate_m(init_param, i, 1)
                       for i, p in enumerate(self.pos_input)]

        neg_m_array = [self.__calculate_m(init_param, i, -1)
                       for i, p in enumerate(self.neg_input)]

        pos_min, pindex = min((val, idx) for idx, val in enumerate(pos_m_array))
        neg_min, nindex = min((val, idx) for idx, val in enumerate(neg_m_array))

        point_t = {}

        if pos_min > neg_min:
            point_t['class'] = 'pos'
            point_t['index'] = pindex
            min_m = pos_min
        else:
            point_t['class'] = 'neg'
            point_t['index'] = nindex
            min_m = neg_min

        sum_sqrt = init_param['sk_a'] + \
            init_param['sk_b'] - 2 * init_param['sk_c']
        sum_sqrt **= 0.5

        return (point_t, sum_sqrt - min_m < self.epsilon)

    @staticmethod
    def __calculate_m(params, index, input_class):
        """Calculate m

        :param params:
        :param point:
        :input_class:
        """

        if input_class < 0:
            numerator = params['sk_neg_e'][index] - params['sk_neg_d'][index]
            numerator += params['sk_a'] - params['sk_c']
        else:
            numerator = params['sk_pos_d'][index] - params['sk_pos_e'][index]
            numerator += params['sk_b'] - params['sk_c']

        denominator = params['sk_a'] + params['sk_b'] - 2 * params['sk_c']
        denominator **= 0.5

        return numerator / denominator

    def __sk_update(self, init_params, ):


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 6:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sk_train.py epsilon max_updates",
        print >> sys.stderr, "class_letter model_file_name train_folder_name\""
        exit(1)

    svm = SVM(sys.argv[1], sys.argv[2], sys.argv[3])

    svm.set_training_inputs(sys.argv[5])
    svm.train(sys.argv[4])
