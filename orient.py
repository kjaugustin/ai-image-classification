#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Augustine Joseph
# Course: CS-B551-Fall2017
# Assignment 4
#

import time
import sys
import pickle
from nnet_classifier import *
from nearest_classifier import *
from adaboost_classifier import *
import numpy as np
import shutil

'''
A detailed discussion of the results are provided in a file named A4_final_report.pdf
    
'''
normalize = lambda x: float(x)/255.0
class_0 = lambda x: 1 if x == 0 else -1
class_90 = lambda x: 1 if x == 90 else -1
class_180 = lambda x: 1 if x == 180 else -1
class_270 = lambda x: 1 if x == 270 else -1
estimate = lambda x: ['0', '90', '180', '270'][x]

def main():

    global switch
    switch = sys.argv[1]
    model_file = sys.argv[3]        # Use this later
    algo = sys.argv[4]
    output_file = "output.txt"
    # set model parameters
    if algo == 'nearest':
        k = 3
        model_file = "nearest_model.txt"
    elif algo == 'nnet':
        no_hidden = 15
        learning_rate = 0.3
        no_iterations = 5
        no_inputs = 192
        no_outputs = 4
        model_file = "nnet_model.txt"
    elif algo == 'adaboost':
        no_it = 15  # Number of iterations
        model_file = "adaboost_model.txt"
    elif algo == 'best':
        no_hidden = 30
        learning_rate = 0.1
        no_iterations = 10
        no_inputs = 192
        no_outputs = 4
        model_file = "best_model.txt"

    if switch == 'train':
        train_file = sys.argv[2]

        if algo == 'nearest':

            model_nearest_file = open(model_file, 'wb')

            model_nearest_file.close()

            print "Training Data for knn classifier is complete. Please run the model on test data using the following command: "
            print "python ./orient.py test test-data.txt nearest_model.txt nearest"
        elif algo == 'nnet':
            # storing train data in a dict
            train_dict = {}
            for line in map(lambda x: x.strip().split(), open(train_file)):
                train_dict[(line[0], line[1])] = (line[1], line[2:])
            nnet_train(train_dict, no_inputs, no_hidden, no_outputs, no_iterations, learning_rate, model_file)
        elif algo == 'adaboost':
            print "Training Data for adaboost classifier..."
            Name = []
            label_0 = []
            label_90 = []
            label_180 = []
            label_270 = []
            train_data = []

            for line in map(lambda x: x.strip().split(), open(train_file)):
                Name.append(line[0])
                label_0.append(class_0(int(line[1])))
                label_90.append(class_90(int(line[1])))
                label_180.append(class_180(int(line[1])))
                label_270.append(class_270(int(line[1])))
                train_data.append(map(normalize, line[2:]))

            model_0 = adaboost_train(train_data,label_0, no_it)
            model_0_file = open("model_0", 'wb')
            pickle.dump(model_0, model_0_file)
            model_0_file.close()
            model_90 = adaboost_train(train_data, label_90, no_it)
            model_90_file = open("model_90", 'wb')
            pickle.dump(model_90, model_90_file)
            model_90_file.close()
            model_180 = adaboost_train(train_data, label_180, no_it)
            model_180_file = open("model_180", 'wb')
            pickle.dump(model_180, model_180_file)
            model_180_file.close()
            model_270 = adaboost_train(train_data, label_270, no_it)
            model_270_file = open("model_270", 'wb')
            pickle.dump(model_270, model_270_file)
            model_270_file.close()
            #
            print "Training Data for adaboost classifier is complete. Please run the model on test data using the following command: "
            print "python ./orient.py test test-data.txt adaboost_model.txt adaboost"

        elif algo == 'best':
            # storing train data in a dict
            train_dict = {}
            for line in map(lambda x: x.strip().split(), open(train_file)):
                train_dict[(line[0], line[1])] = (line[1], line[2:])
            nnet_train(train_dict, no_inputs, no_hidden, no_outputs, no_iterations, learning_rate, model_file)

    elif switch == 'test':
        test_file = sys.argv[2]

        if algo == 'nearest':

            train_file = "train-data.txt"
            train_dict = {}
            for line in map(lambda x: x.strip().split(), open(train_file)):
                train_dict[(line[0], line[1])] = (line[1], np.array(map(normalize, line[2:])))

            test_dict = {}
            for line in map(lambda x: x.strip().split(), open(test_file)):
                test_dict[line[0]] = (line[1], np.array(map(normalize, line[2:])))



            knn_test(train_dict, test_dict, k, output_file)

        elif algo == 'nnet':
            # storing test data in a dict
            test_dict = {}
            for line in map(lambda x: x.strip().split(), open(test_file)):
                test_dict[line[0]] = (line[1], line[2:])

            nnet_test(test_dict, no_inputs, no_hidden, no_outputs, model_file, output_file)

        elif algo == 'adaboost':
            Name = []
            true_label = []  # label from test data
            test_data = []  #
            for line in map(lambda x: x.strip().split(), open(test_file)):
                Name.append(line[0])
                true_label.append(line[1])
                test_data.append(map(normalize, line[2:]))

            # reload model from file
            model_0_file = open("model_0", 'rb')
            model_0 = pickle.load(model_0_file)
            model_0_file.close()
            model_90_file = open("model_90", 'rb')
            model_90 = pickle.load(model_90_file)
            model_90_file.close()
            model_180_file = open("model_180", 'rb')
            model_180 = pickle.load(model_180_file)
            model_180_file.close()
            model_270_file = open("model_270", 'rb')
            model_270 = pickle.load(model_270_file)
            model_270_file.close()

            output_0 = adaboost_test(test_data, model_0)
            output_90 = adaboost_test(test_data, model_90)
            output_180 = adaboost_test(test_data, model_180)
            output_270 = adaboost_test(test_data, model_270)
            count = 0.0
            no_test_data = len(output_0)
            output_file = open(output_file, "w")
            for i in range(no_test_data):
                adaboost_out = [output_0[i], output_90[i], output_180[i], output_270[i]]
                max_index = adaboost_out.index(max(adaboost_out))
                estimated_label = estimate(max_index)

                output_file.write(Name[i] + " " + estimated_label + "\n")
                if (int(estimated_label) == int(true_label[i])):
                    count += 1

            output_file.close()

            print "Final Classification Accuracy of adaboost classifier: " + "%.2f" % (count / no_test_data * 100.0) + " %"
            # print estimated_label

        elif algo == 'best':
            # storing test data in a dict
            test_dict = {}
            for line in map(lambda x: x.strip().split(), open(test_file)):
                test_dict[line[0]] = (line[1], line[2:])

            nnet_test(test_dict, no_inputs, no_hidden, no_outputs, model_file, output_file)


start_time = time.time()
if __name__ == '__main__':
    main()
if switch == 'test':
    print "Total time taken to classify the " + switch + " data in seconds: %0.2fs"  % (time.time() - start_time)

