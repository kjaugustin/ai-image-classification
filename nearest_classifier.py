#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Augustine Joseph
# Course: CS-B551-Fall2017
# Assignment A4
#
from numpy import *
import numpy as np
from numpy import linalg as LA
from collections import Counter

# normalize = lambda x: float(x)/255.0

def dist_euclidean(x, y):

    dist = LA.norm(x-y)
    #return np.linalg.norm(x-y)
    #return np.sqrt(np.sum((x-y)**2))
    return dist



def knn_test(train_data, test_data, k, outfile):
    count = 0.0
    t_count = 0
    no_test_data = len(test_data)
    k = int(k)
    output_file = open(outfile, "w")

    for j in test_data.keys():
        result = []
        # Euclidean distance of each test data with every training data

        for i in train_data.keys():
            result.append([train_data[i][0], dist_euclidean(train_data[i][1], test_data[j][1])])

        srt_result = sorted(result, key=lambda x: x[1])
        nearest = srt_result[:k]
        label = Counter([item[0] for item in nearest]).most_common(1)[0][0]
        true_label = test_data[j][0]
        if label == true_label:
            count += 1

        output_file.write(str(j) + " " + label + "\n")
        t_count += 1
        print "Classified Test data: " + str(t_count)
    output_file.close()

    print "Final Classification Accuracy of KNN classifier: " + "%.2f" % (count / no_test_data * 100.0) + " %"

