#!/usr/bin/env python
""" Tests SVM model trained on Zener Cards generated using S-K Algorithm."""

import sys
import pickle
import os
import re
import numpy as np
from PIL import Image


class SVMTEST(object):
    """Implements the SVM model tester"""

    def __init__(self, param_dict, test_input, test_classes):
        self.params = param_dict
        self.test_input = test_input
        self.test_classes = test_classes

    def test(self):

        correct = 0.0
        false_positive = 0.0
        false_negative = 0.0

        for index, test_vector in enumerate(self.test_input):
            result = self.__svm_test(test_vector)

            if result > 0 and self.params['class_letter'] == self.test_classes[index]:
                print str(index) + " Correct"
                correct += 1
            elif result > 0 and self.params['class_letter'] != self.test_classes[index]:
                print str(index) + " False Positive"
                false_positive += 1
            elif result < 0 and self.params['class_letter'] != self.test_classes[index]:
                print str(index) + " Correct"
                correct += 1
            elif result < 0 and self.params['class_letter'] == self.test_classes[index]:
                print str(index) + " False Negative"
                false_negative += 1

        print "Fraction Correct: " + str(correct/self.test_input.shape[0])
        print "Fraction False Positive: " + str(false_positive / self.test_input.shape[0])
        print "Fraction False Negative: " + str(false_negative / self.test_input.shape[0])



    def __svm_test(self, test_vector):

        score = 0.0
        for i in range(self.params['pos_input'].shape[0]):
            score += (self.params['pos_alpha'][i] *
                      self.__polynomial_kernal(test_vector,
                                               self.params['pos_input'][i]))

        for i in range(self.params['neg_input'].shape[0]):
            score += (self.params['neg_alpha'][i] * -1 *
                      self.__polynomial_kernal(test_vector,
                                               self.params['neg_input'][i]))

        score += ((self.params['A'] - self.params['B']) / 2)

        return score

    def __polynomial_kernal(self, vector_a, vector_b):
        """Return result of degree four polynomial kernel function

        :param a:
        :param b:
        :returns:
        """

        result = np.dot(vector_a, vector_b) + 1
        result = result ** 4

        return result


def get_test_input(train_folder, test_folder):
    """ Get test inputs and convert to array"""
    # Check for training data
    if os.path.isdir(train_folder) is not True:
        print >> sys.stderr, "NO TRAINING DATA"
        exit(1)

    # Check for testing data
    if os.path.isdir(test_folder) is not True:
        print >> sys.stderr, "NO TEST DATA"
        exit(1)

    train_empty = True
    test_empty = True

    generator_pattern = re.compile("[0-9]+_" + "[OPWSQ]" + ".png")

    test_input = np.zeros(625)
    test_classes = []

    for filename in os.listdir(test_folder):
        abs_path = os.path.abspath(train_folder) + "/" + filename

        if generator_pattern.match(filename):
            classname = filename.split("_")[-1].split(".")[0]
            test_classes.append(classname)
            test_empty = False
            # Open image using PIL
            image = Image.open(abs_path)
            # Convert to numpy array
            img_array = np.array(image)
            # Reshape array to one dimension
            img_array = img_array.reshape(-1)
            # Append to test input collection array
            test_input = np.vstack((test_input, img_array))

    for filename in os.listdir(train_folder):
        if generator_pattern.match(filename):
            train_empty = False
            break

    if train_empty is True:
        print >> sys.stderr, "NO TRAINING DATA"
        exit(1)

    if test_empty is True:
        print >> sys.stderr, "NO TEST DATA"
        exit(1)

    test_input = np.delete(test_input, 0, 0)

    return test_classes, test_input


if __name__ == '__main__':
    # Check if correct number of arguments passed
    if len(sys.argv) < 4:
        print >> sys.stderr, "Some arguments are missing!",
        print >> sys.stderr, "Please make sure the command is in format:"
        print >> sys.stderr, "\"python sk_model_tester.py model_file_name",
        print >> sys.stderr, "train_folder_name test_folder_name \""
        exit(1)

    with open(sys.argv[1], 'rb') as handle:
        model_params = pickle.loads(handle.read())

    test_classes, test_input = get_test_input(sys.argv[2], sys.argv[3])

    test_model = SVMTEST(model_params, test_input, test_classes)

    test_model.test()

