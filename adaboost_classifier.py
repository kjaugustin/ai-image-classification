#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Augustine Joseph
# Course: CS-B551-Fall2017
# Assignment A4
#

from numpy import *
import pickle

# weak stump function
def build_weak_stump(d,l,D):
    data_matrix = mat(d)
    label_matrix = mat(l).T
    m,n = shape(data_matrix)

    numstep = 10.0

    best_stump = {}
    best_class = mat(zeros((192,1)))
    min_err = inf
    for i in range(n):
        datamin = data_matrix[:,i].min()
        datamax = data_matrix[:,i].max()
        step_size = (datamax - datamin) / numstep
        for j in range(-1,int(numstep)+1):
            for inequal in ['lt','gt']:
                threshold = datamin + float(j) * step_size
                predict = stump_classify(data_matrix,i,threshold,inequal)
                err = mat(ones((m,1)))
                err[predict == label_matrix] = 0
                weighted_err = D.T * err;
                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class = predict.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['ineq'] = inequal
    return best_stump, min_err, best_class

# classify training data with weak stump
def stump_classify(datamat,dim,threshold,inequal):
    res = ones((shape(datamat)[0],1))
    if inequal == 'lt':
        res[datamat[:,dim] <= threshold] = -1.0
    else:
        res[datamat[:,dim] > threshold] = -1.0
    return res

# Adaboost Training
def adaboost_train(data,label,numIt = 1000):
    weak_classifiers = []
    m = shape(data)[0]
    D = mat(ones((m,1))/m)
    Ensemble_class_Estimate = mat(zeros((m,1)))
    for i in range(numIt):
        best_stump, error, class_estimate = build_weak_stump(data,label,D)
        alpha = float(0.5*log((1.0-error) / (error+1e-15)))
        best_stump['alpha'] = alpha
        weak_classifiers.append(best_stump)
        weightD = multiply((-1*alpha*mat(label)).T,class_estimate)
        D = multiply(D,exp(weightD))
        D = D/D.sum()
        Ensemble_class_Estimate += class_estimate*alpha
        Ensemble_errors = multiply(sign(Ensemble_class_Estimate)!=mat(label).T,\
                                  ones((m,1)))  #Converte to float
        error_rate = Ensemble_errors.sum()/m
        #print "Sum of error:  ",error_rate
        if error_rate == 0.0:
            break

    return weak_classifiers

# Applying adaboost classifier for a single data sample
def adaboost_classify(dataTest,classifier):
    data_matrix = mat(dataTest)
    m = shape(data_matrix)[0]
    Ensemble_class_Estimate = mat(zeros((m,1)))
    for i in range(len(classifier)):
        class_estimate = stump_classify(data_matrix,classifier[i]['dim'],classifier[i]['threshold'],classifier[i]['ineq'])
        Ensemble_class_Estimate += classifier[i]['alpha']*class_estimate
        #print Ensemble_class_Estimate
    return Ensemble_class_Estimate.item()


# Testing
def adaboost_test(dataSet,classifier):
    label = []

    for i in range(shape(dataSet)[0]):
        label.append(adaboost_classify(dataSet[i], classifier))

    return label