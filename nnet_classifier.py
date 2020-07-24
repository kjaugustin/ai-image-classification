#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Augustine Joseph
# Course: CS-B551-Fall2017
# Assignment A4
#

import sys
import math
from random import randrange
import numpy as np

out_vector = lambda x: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]][int(x) / 90]
init_w = 0.01 # Initial weight
normalize = lambda x:float(x)/255.0


# Activation function for nnet, sigmoid function
def activ_fn(x):
    return 1.0/(1.0 + math.exp(-x))

# Derivative of activation function
def deriv_afn(x):
    return activ_fn(x)*(1.0 - activ_fn(x))


# Sum of inputs x weights for each hidden node

def inp_sum(w, inp_dict, node, start, end):
    total = 0
    for x in range(start, end):
        total += w[(x, node)]*inp_dict[x]
    return total


# Sum of outputs and weights for each hidden node

def out_sum(w, inp_dict, node, start, end):
    total = 0
    for x in range(start, end):
        total += w[(node, x)]*inp_dict[x]
    return total


# Randomly initialize weights

def build_weights(w, no_inputs, no_hidden, no_outputs):
    for x in range(no_inputs):
        for y in range(no_hidden):
            w[(x, y + no_inputs)] = randrange(-5, 5)*init_w

    for x in range(no_hidden):
        for y in range(no_outputs):
            w[(x + no_inputs, y + no_inputs + no_hidden)] = randrange(-5, 5)*init_w
    return w


# Updating weights in the network using deltas or error compensation

def weight_update(w, alpha, a, err, no_inputs, no_hidden, no_outputs):
    for x in range(no_inputs):
        for y in range(no_hidden):
            w[(x, y + no_inputs)] += a[x] * err[y + no_inputs] * alpha
    for x in range(no_hidden):
        for y in range(no_outputs):
            w[(x + no_inputs, y + no_inputs + no_hidden)] += a[x + no_inputs] * err[y + no_inputs + no_hidden] * alpha

    return w



# Save trained model parameters to  file
def save_model_param(w, file_name):
    with open(file_name, "w") as model:
        for key, val in w.items():
            model.write(str(key[0]) + " " + str(key[1]) + " " + str(val) + "\n")


# To train the nnet model

def train_model(train_dict, no_inputs, no_hidden, no_outputs, no_iterations, learning_rate):

    w = {}
    w = build_weights(w, no_inputs, no_hidden, no_outputs)
    # Loop iterations
    for k in range(no_iterations):
        true_count = 0.0
        counter = 0
        # Loop over each training data
        for i in train_dict.keys():
            #print i[1]
            counter += 1

            # Actual target output
            target_output = out_vector(i[1])

            # Activation ai dict
            a = {}
            # Input dictionary
            inp_dict = {}
            # Error dictionary
            err = {}

            # Feed Forward
            for x in range(no_inputs):
                a[x] = float(train_dict[i][1][x])/255
                #print a[x] # use lambda
            for node in range(no_inputs, no_inputs + no_hidden):
                inp_dict[node] = inp_sum(w, a, node, 0, no_inputs) + 1
                a[node] = activ_fn(inp_dict[node])
            #print(inp_dict[193])
            output_vector = []
            for node in range(no_inputs + no_hidden, no_inputs + no_hidden + no_outputs):
                inp_dict[node] = inp_sum(w, a, node, no_inputs, no_inputs + no_hidden) + 1
                a[node] = activ_fn(inp_dict[node])
                output_vector.append(a[node])

            max_indx = output_vector.index(max(output_vector))
            true_label = target_output.index(1.0)
            # Estimated label is same as true label label if max_indx == true_label
            if max_indx == true_label:
                true_count += 1

            # Back Propagation
            for node in range(no_inputs + no_hidden, no_inputs + no_hidden + no_outputs):
                err[node] = deriv_afn(inp_dict[node])*(target_output[node - no_inputs - no_hidden] - a[node])

            for node in range(no_inputs, no_inputs + no_hidden):
                err[node] = deriv_afn(inp_dict[node])*out_sum(w, err, node, no_inputs + no_hidden, no_inputs + no_hidden + no_outputs)

            # Update Weights
            w = weight_update(w, learning_rate, a, err, no_inputs, no_hidden, no_outputs)
        print "After iteration " + str(k+1) + ", model accuracy is: " + "%.2f" % (true_count / counter * 100.0) + " %"
    return w


def test_model(test_dict, no_inputs, no_hidden, no_outputs, w, output_file):
    true_count = 0.0
    counter = 0
    a = {}

    # Save estimated label to a file
    output_file = open(output_file, "w")

    # Loop over each test data
    for i in test_dict.keys():
        inp_dict = {}
        target_output = out_vector(test_dict[i][0])
        # Estimate label using trained model
        for x in range(no_inputs):
            a[x] = float(test_dict[i][1][x])/255
        for node in range(no_inputs, no_inputs + no_hidden):
            inp_dict[node] = inp_sum(w, a, node, 0, no_inputs) + 1
            a[node] = activ_fn(inp_dict[node])
        output_vector = []
        for node in range(no_inputs + no_hidden, no_inputs + no_hidden + no_outputs):
            inp_dict[node] = inp_sum(w, a, node, no_inputs, no_inputs + no_hidden) + 1
            a[node] = activ_fn(inp_dict[node])
            output_vector.append(a[node])

        max_indx = output_vector.index(max(output_vector))
        true_label = target_output.index(1.0)
        counter += 1
        # output_file.write(str(i) + " " + str(max_indx*90) + " " + test_dict[i][0] + "\n") #Need to change the output file format
        output_file.write(str(i) + " " + str(max_indx * 90) + "\n")
        # Estimated label is same as true label label if max_indx == true_label
        if max_indx == true_label:
            true_count += 1

    output_file.close()

    print "Final Classification Accuracy of Nnet classifier: " + "%.2f" % (true_count/counter*100.0) + " %"

def nnet_train(train_dict, no_inputs, no_hidden, no_outputs, no_iterations, learning_rate, model_file):

    print "Training Data for nnet classifier..."
    w = train_model(train_dict, no_inputs, no_hidden, no_outputs, no_iterations, learning_rate)
    save_model_param(w, model_file)
    print "NN Model training complete"
    print "Run the following command to run the NN classifier on test data"
    print "python ./orient.py test test-data.txt nnet_model.txt nnet"

def nnet_test(test_dict, no_inputs, no_hidden, no_outputs, model_file, output_file):

    #Load weights from model parameters file.
    weights = {}
    for line in map(lambda x: x.strip().split(), open(model_file)):
        weights[(int(line[0]), int(line[1]))] = float(line[2])

    print "Testing Data using nnet model..."
    test_model(test_dict, no_inputs, no_hidden, no_outputs, weights, output_file)
