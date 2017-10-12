#!/usr/bin/env python
""" Implements S-K Algorithm for training SVMs on Zener Cards. """

import sys
import numpy


class SVM(object):
    """An SVM model

    Attirbutes:
        epsilon:

    """

    def __init__(self, epsilon):
        """ Return a new SVM object

        :param epsilon:
        :return: returns nothing
        """

        self.epsilon = epsilon

    def polynomial_kernal(self, vector_a, vecotr_b):
        """Return computational result of degree four polynomial kernel function

        :param a:
        :param b:
        :returns:
        :raises:

        """
        degree = 4


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 8:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sk_train.py epsilon max_updates",
        print >> sys.stderr, "class_letter model_file_name train_folder_name\""
        exit(1)
