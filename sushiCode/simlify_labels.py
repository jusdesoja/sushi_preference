#!/usr/bin/env python
# encoding: utf-8

import numpy as np
def simplify_labels(labels):
    n = len(labels)
    sorted_ind = np.argsort(labels)
    label_sorted = np.sort(labels)
    simple_labels = np.zeros((n), dtype = int)
    simple_labels[sorted_ind[0]] = 0
    for i in range(1, n):
        if label_sorted[i] == label_sorted[i-1]:
            simple_labels[sorted_ind[i]] = simple_labels[sorted_ind[i-1]]
        else:
            simple_labels[sorted_ind[i]] = simple_labels[sorted_ind[i-1]]+1
    return simple_labels
"""
labels = [1,5,4,4,3,5,2]
print(simplify_labels(labels))
"""
