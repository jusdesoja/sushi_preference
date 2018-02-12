#!/usr/bin/env python
# encoding: utf-8

"""
    File:   center_cal.py
    Author: Yiru Zhang(yiru.zhang@irisa.fr)
    Date:   Oct 19, 2017

    About
    ---------
    This file contains methods for cluster center calculate represented by mass functions in metric of Jousselme distance.
    The center is calculated with condition that the sum of intra distaces(Jousselme) is minimal.
    The optimisation process is realised by Lagrange multiplier.
"""

import numpy as np
from scipy.linalg import lu, inv
from iBelief.Dcalculus import Dcalculus

def _calculate_B(Jacc, m_N):
    n_samples = m_N.shape[0]
    n_discernment = Jacc.shape[0]
    #print(n_samples, n_discernment)
    B = np.zeros((n_discernment + 1 , 1))
    for k in range(n_discernment):
        for n in range(n_samples):
            B[k] += np.dot(Jacc[k],m_N[n])
    B /= n_samples
    B[-1] = 1
    return B

def _calculate_A(Jacc, N):
    n_discernment = Jacc.shape[0]
    coe_normalize = np.array([[- 1 / (2 * N)] * n_discernment])
    A = np.concatenate((Jacc,coe_normalize.T), axis=1)
    I = np.array([[1] * n_discernment +[0]])
    A = np.concatenate((A,I),axis = 0)
    return A

def _gausse_elimination(A, B):
    pl, u = lu(A, permute_l=True)
    y = np.zeros(B.size)
    for m,b in enumerate(B.flatten()):
        y[m] = b
        if m:
            for n in range(m):
                y[m] -= y[n] * pl[m,n]
        y[m] /= pl[m,m]

    x = np.zeros(B.size)
    #lastidx = B.size-1
    for midx in range(B.size):
        m = B.size - 1 -midx
        x[m] = y[m]
        if midx:
            for nidx in range(midx):
                n = B.size -1 -nidx
                x[m] -= x[n] * u[m,n]
        x[m] /= u[m,m]
    return x

def one_mass_center_calculate(m_N):
    #print(m_N.shape)
    n_samples, n_discernment = m_N.shape
    #print (m_N.shape)
    Jacc = Dcalculus(n_discernment)
    center = _gausse_elimination(_calculate_A(Jacc, n_samples),_calculate_B(Jacc,m_N))
    return center[:-1]
    #return center
def all_mass_center_calculate(X):
    X_temp = X.transpose((1,0,2))
    n_samples, n_features, n_discernment = X.shape
    center = np.zeros((n_features, n_discernment), dtype = object)
    for fidx in range(n_features):
        center[fidx] = one_mass_center_calculate(X_temp[fidx])
    return center

m1 = np.array([0,0.3,0.4,0.3])
m2 = np.array([0,0.5,0.1,0.4])
m3 = np.array([0,0.6,0.2,0.2])
#m2 = np.array([0,0.3, 0.4, 0.3])

"""
m31 = np.array([0, 0, 0.2, 0.8, 0, 0, 0, 0])
m32 = np.array([0, 0, 0.3, 0, 0, 0, 0, 0.7])
m33 = np.array([0, 0, 0, 0, 0, 0, 0.3, 0.7])
"""
m_N = np.vstack((m1, m2, m3))
Jacc = Dcalculus(4)
print(Dcalculus(4))
#print(_calculate_B(Jacc,m_N))
#m_N = np.concatenate([m1,m2,m3] , axis= 0)
center = one_mass_center_calculate(m_N)
print("center:%s" % center)
#print(center.sum())
#A = np.array([[6,0,0,0,-1],[0,6,0,3,-1],[0,0,6,3,-1],[0,3,3,6,-1],[1,1,1,1,0]])
#B = np.array([0,3.7,5,3.9,1])
#print(_gausse_elimination(A,B))
from iBelief.distance import JousselmeDistance
"""
print("distance 1: %f" % JousselmeDistance(center, m31))
print("distance 2: %f" % JousselmeDistance(center, m32))
print("distance 3: %f" % JousselmeDistance(center, m33))
def test_distances(center, m_N):
    n_samples = m_N.shape[0]
    distance = 0.0
    for i in range(n_samples):
        distance += JousselmeDistance(center, m_N[i])
    return distance

print(test_distances(center,m_N))
c2 = np.array([0,0,0.1666,0.2666,0,0,0.1,0.466])
print(test_distances(c2, m_N))
"""
