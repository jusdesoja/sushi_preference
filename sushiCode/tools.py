#!/usr/bin/env python
# encoding: utf-8

"""
    File:   tools.py
    Author: Yiru Zhang (yiru.zhang@irisa.fr)
    Date:   Sep 14, 2017

    About
    -----------------------------
        This file contains methods for mass functions initialisation
"""

import numpy as np
#import math
from exceptions import *

# Global variable for mass vector size
vector_size = 16
#vector_size = 8
order_a_vector_size = 4
def _score_2_partial_order_mat(scoreMap):
    nbItem = len(scoreMap)
    ptOrderMat = np.zeros((nbItem,nbItem))
    itemSeq = sorted(scoreMap.keys())
    for k1 in itemSeq:
        for k2 in itemSeq:
            if scoreMap[k1] >= scoreMap[k2]:
                ptOrderMat[itemSeq.index(k1)][itemSeq.index(k2)] = 1
    return ptOrderMat

def _seq_2_partial_order_mat(seq):
    nbItem = len(seq)
    ptOrderMat = np.zeros((nbItem,nbItem))
    sortedSeq = sorted(seq)
    for i in sortedSeq:
        for j in sortedSeq:
            #ind1 = seq.index(i)
            #ind2 = seq.index(j)
            #print(i," ", j," ", ind1," ", ind2)
            if seq.index(i) <= seq.index(j):
                ptOrderMat[sortedSeq.index(i)][sortedSeq.index(j)] = 1
            #print(i,' ', j)
            #print(ptOrderMat)
    return ptOrderMat


def _score_2_order_mat(scoreMap, nbItem = 100):
    shape = (nbItem,nbItem)
    orderMat = np.full(shape, -1, dtype = int)
    #np.fill_diagonal(orderMat,1)
    for k1,v1 in scoreMap.items():
        for k2,v2 in scoreMap.items():
            if k1 <= k2:
                if v1 > v2:
                    orderMat[k1][k2] = 1
                    orderMat[k2][k1] = 0
                elif v1 < v2:
                    orderMat[k1][k2] = 0
                    orderMat[k2][k1] = 1
                else:
                    orderMat[k1][k2] = 1
                    orderMat[k2][k1] = 1
    return orderMat


def _seq_2_order_mat(seq, nbItem = 100):
    if max(seq)>nbItem:
        raise IllegalItemNumberError("Preference Sequence size is larger than the given item numbers. (should be smaller or equal)")
    else:
        shape = (nbItem, nbItem)
        orderMat = np.full(shape, -1, dtype = int)
        #sortedSeq = sorted(seq)
        for i in seq:
            for j in seq:
                ind1 = seq.index(i)
                ind2 = seq.index(j)
                if ind1 < ind2:
                    orderMat[i][j] = 1
                    orderMat[j][i] = 0
                elif ind1 > ind2:
                    orderMat[i][j] = 0
                    orderMat[j][i] = 1
                else:
                    orderMat[i][j] = 1
    return orderMat

def fargin_distance(scoreMat, seqMat, tiePenalty = 0.5):
    """
    Calculate Fargin distance with ties.
    The score and sequence are on the same group of items, so is an complete order.
    By matricial calculating, each pair is counted once, so we use 0.5 as default penalty value.
    """
    if scoreMat.shape == seqMat.shape:
        m = scoreMat - seqMat
        n_pairs = m.shape[1] ** 2
        #print(m)
        discoTie = np.count_nonzero(m + np.transpose(m)) / 2
        #print("Tie Mat: \n",discoTie)
        discoInv = (np.count_nonzero(m) - discoTie) / 2
        #print("Inverse Mat: \n", discoInv)
        faginDistance = discoInv + discoTie * tiePenalty
        #print("m shape: \n", m.shape)
        normalized_fargin_distance = faginDistance / n_pairs
        if normalized_fargin_distance > 1:
            print(m)
            raise ValueError("normalized fargin distance %f is bigger than 1 (Tie: %d, inverse: %d, pair number:%d)" % (normalized_fargin_distance, discoTie, discoInv, m.shape[1]))
        return normalized_fargin_distance
    else:
        raise IllegalMatrixShapeError("Given score shape and sequence shape are not the same (%s, %s)" % (scoreMat.shape, seqMat.shape))

def _one_mass_init(r1, r2, ignorance = 1.0):
    """
    """
    if r1 > 4 or r2 > 4:
        raise ValueError("r1 and r2 should be smaller than 4. However, %d and %d are given." % (r1,r2))
    #elif r1<4 and r1==r2:

        #raise ValueError("r1 and r2 representing two conflictual relations should not be identical. However %d and %d are given " %(r1, r2))
    if ignorance <0 or ignorance > 1:
        raise ValueError("Ignorance value should be betwwen 0 and 1. Hoever, %f is given (and r1=%d r2=%d)" % (ignorance,r1,r2))
    else:
        mass = np.zeros(vector_size)
        if r1==r2:
            if r1==4:
                pass
            else:
                mass[int(2 ** r1)] = 1 - ignorance
        else:
            mass_value = (1 - ignorance) / (float(r1 < 4) + float(r2 < 4))
            if r1 <= 3:   #relation  = 0,1,2,3
                mass[int(2 ** r1) ] = mass_value
            if r2 <= 3:
                mass[int(2 ** r2) ] = mass_value
        mass[-1] = ignorance
        return mass

def all_mass_init(scoreMap, seq, candList, nbItem = 100):
    """
    initiate mass function values for a voter. if the candidate space is not in [1,100], we convert the candidate by candList to avoid sparse matrix.
    Parameters:
    -----------
    scoreMap: dict of size 10.
        score on given candidate
    seq: list of size 10.
        preference sequence
    candList: list
        candidate space of all given voters.
    nbItem: int
        number of global candidate, 100 by default
    Return:
    -------
    massMat: matrix of 3 dimensions
    """
    farginDist = fargin_distance(_score_2_partial_order_mat(scoreMap), _seq_2_partial_order_mat(seq))
    #convert_dict = {x:i for i,x in enumerate(candList)}
    given_cand_nb = len(candList)
    massMat = np.zeros((given_cand_nb, given_cand_nb, vector_size), dtype = object)
    for i in range(given_cand_nb):
        for j in range(i+1, given_cand_nb):
            r1, r2 = 3,3
            if candList[i] not in seq or candList[j] not in seq:
                massMat[i][j] = _one_mass_init(r1, r2, 1.0)
            else:
                if seq.index(candList[i]) > seq.index(candList[j]):
                    r1 = 1
                else:
                    r1 = 0
                if scoreMap[candList[i]] > scoreMap[candList[j]]:
                    r2 = 0
                elif scoreMap[candList[i]] < scoreMap[candList[j]]:
                    r2 = 1
                else:
                    r2 = 2

                massMat[i][j] = _one_mass_init(r1, r2, farginDist)
    return massMat
"""
def all_mass_init(scoreMap, seq, nbItem = 100):

    farginDist = fargin_distance(_score_2_partial_order_mat(scoreMap), _seq_2_partial_order_mat(seq))
    massMat = np.zeros((nbItem, nbItem,vector_size), dtype = object)
    for i in range(nbItem):
        for j in range(i + 1, nbItem):
            r1, r2 = 4,4
            if i not in seq or j not in seq:
                massMat[i][j] = _one_mass_init(r1, r2, 1.0)
            else:
                if seq.index(i) > seq.index(j):
                    r1=1
                else:
                    r1=0
                if scoreMap[i] > scoreMap[j]:
                    r2=0
                elif scoreMap[i] < scoreMap[j]:
                    r2=1
                elif scoreMap[i] == scoreMap[j]:
                    r2=2
                massMat[i][j] = _one_mass_init(r1, r2, farginDist)
    return massMat
"""

def _one_certain_mass_init(r):
    #mass = np.zeros(vector_size)
    mass = np.zeros(order_a_vector_size)
    mass[int(2 ** r)] = 1
    return mass

def all_certain_mass_init(seq, nbItem = 100):
    #massMat = np.zeros((nbItem, nbItem, vector_size), dtype = object)
    massMat = np.zeros((nbItem, nbItem, order_a_vector_size), dtype = object)
    for i in range(nbItem):
        for j in range(i+1, nbItem):
            r = 2
            if i in seq and j in seq:
                if seq.index(i) < seq.index(j):
                    r = 0
                else:
                    r = 1
            massMat[i][j] = _one_certain_mass_init(r)
    return massMat

def flatten_pref_mass_mat(square_mat):
    """
    Flatten a preference square matrix into a vector

    Parameter
    ----------------------------
    square_mat: Cartesian product of items, elements in upper triangle with setoff=1 represent preference pairs by mass function.

    Return
    ----------------------------
    flattened_mass: flattened matrix from square mat.
    """
    size = square_mat.shape[1]
    iu = np.triu_indices(size, 1) # diagnoal offset = 1
    flattened_mass = square_mat[iu]
    return flattened_mass

"""test codette/ Examples"""
"""
score_map = {1:0, 3:4, 4:2, 12:1, 44: 1, 58: 4, 60: 2, 67:0, 74: 0, 87: 2 }
pref_seq = [58,4,3,44,87,60,67,1,12,74]
cand_list = [0, 1, 3, 5, 6, 2, 4, 7, 8, 10, 9, 11, 12, 13, 14, 15, 17, 16, 18, 19, 21, 20, 22, 23, 25, 24, 26, 27, 28, 30, 29, 31, 33, 34, 35, 37, 32, 38, 36, 41]
#score_map = {3:4, 4:2, 44:1, 58:4, 87:2}
#pref_seq = [58,4,3,44,87]


np.set_printoptions(precision = 3)

print("---------score2pmat---------")
print(_score_2_partial_order_mat(score_map))
print("---------seq2pmat-----------")
print(_seq_2_partial_order_mat(pref_seq))
print("--------farginDistance:partial------")
print(fargin_distance(_score_2_partial_order_mat(score_map), _seq_2_partial_order_mat(pref_seq)))
print("-------farginDistance: complete-------")
print(fargin_distance(_score_2_order_mat(score_map), _seq_2_order_mat(pref_seq)))
mass_mat = all_mass_init(score_map, pref_seq, cand_list)
print(flatten_pref_mass_mat(mass_mat))
print("shape:%s" %((mass_mat.shape,)))
"""
