#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:45:14 2018

@author: wsw
"""

"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.
"""
import operator
import numpy as np

'''
something wrong not use
'''

def ErrorRateAt95Recall(labels, scores):
    recall_point = 0.95
    # Sort label-score tuples by the score in descending order.
    sorted_scores = zip(labels, scores)
    sorted_scores = sorted(sorted_scores,key=operator.itemgetter(1), reverse=False)

    # Compute error rate
    n_match = sum(1 for x in sorted_scores if x[0] == 1)
    n_thresh = recall_point * n_match

    tp = 0
    count = 0
    for label, score in sorted_scores:
        count += 1
        if label == 1:
            tp += 1
        if tp >= n_thresh:
            break

    return float(count - tp) / count


if __name__ == '__main__':
    np.random.seed(0)
    label = np.random.randint(2,size=1000)
    score = np.random.uniform(0,30,size=1000)
    fpr95 = ErrorRateAt95Recall(label,score)
    print(fpr95)