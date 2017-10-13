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
        self.zener_card_letters = set(['O', 'P', 'Q', 'S', 'W'])
        self.pos_input = np.zeros(625)
        self.neg_input = np.zeros(625)

    def get_training_inputs(self, train_folder):
        """Get training images and convert them to numpy array.

        :param train_folder: Folder name where training inputs are stored
        "returns: Numpy array of arrays of training images' pixel values
        """
        # Check if folder exists
        if os.path.isdir(train_folder) is not True:
            print >> sys.stderr, "NO DATA"
            exit(1)

        is_empty = True

        # Training class filename pattern (positive input pattern)
        pos_pattern = re.compile("[0-9]+_" + self.class_letter + ".png")

        class_letter_set = set(list(self.class_letter))

        # Get class letters of negative inputs
        neg_class_letter_set = self.zener_card_letters - class_letter_set

        neg_class_letters = "".join(list(neg_class_letter_set))

        # Non-training class filename pattern (negative input pattern)
        neg_pattern = re.compile("[0-9]+_[" + neg_class_letters + "].png")

        pos_input_sum = np.zeros(625)
        neg_input_temp = np.zeros(625)

        # Convert images to numpy array of pixels
        for filename in os.listdir(train_folder):

            # Get absolute path
            abs_path = os.path.abspath(train_folder) + "/" + filename

            # Check if filename matches training class pattern
            if pos_pattern.match(filename):
                is_empty = False
                # Open image using PIL
                image = Image.open(abs_path)
                # Convert to numpy array
                img_array = np.array(image)
                # Reshape array to one dimension
                img_array = img_array.reshape(-1)
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
                # Add to neg_input_temp to calculate the centroid
                neg_input_temp = np.add(neg_input_temp, img_array)
                # Append to positive input collection array
                self.neg_input = np.vstack((self.neg_input, img_array))

        self.pos_input = np.delete(self.pos_input, 0, 0)
        self.neg_input = np.delete(self.neg_input, 0, 0)

        print self.pos_input.shape
        print self.neg_input.shape

        if is_empty is True:
            print >> sys.stderr, "NO DATA"
            exit(1)

        # print img_array

    def polynomial_kernal(self, vector_a, vecotr_b):
        """Return result of degree four polynomial kernel function

        :param a:
        :param b:
        :returns:
        """
        degree = 4 * self.epsilon

        return degree


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 6:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sk_train.py epsilon max_updates",
        print >> sys.stderr, "class_letter model_file_name train_folder_name\""
        exit(1)

    svm = SVM(sys.argv[1], sys.argv[2], sys.argv[3])

    svm.get_training_inputs(sys.argv[5])
