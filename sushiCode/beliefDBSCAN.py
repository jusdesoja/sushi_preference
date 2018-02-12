#!/usr/bin/env python
# encoding: utf-8

"""A brutal and original implementation of DBSCAN on belief functions """

from beliefKMeans import _user_conflict, _user_Jousselme_distance
import numpy as np
#import itertools
def range_query(data_size, distances, P, eps):
    """Search for neighbor points of P within distance epsilon (eps)
    Parameters:
    data_size: number of all data. used for point pair set calculation
    dist_dic: distance dictionary of all pair and distances
    P: index of point P
    eps: radius epsilon
    Return:
    Ns: list of point index
    """
    Ns = []
    #import pdb; pdb.set_trace()
    for i in range(data_size):
        if distances[i][P] < eps:
            Ns.append(i)
    Ns.append(P)
    return set(Ns)

def cal_distances(X, metric="conflict"):
    """
    Generate a distance dictionary from all instances to all instances
    Parameter:
    X:
    Return:
    dist_dic
    """
    #import pdb; pdb.set_trace()
    #print(X)
    n_sample = X.shape[0]
    distances = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        for j in range(n_sample):
            if (j>i):
                if metric == "conflict":
                    distances[i][j] = _user_conflict(X[i], X[j])
                elif metric == 'jousselme':
                    distances[i][j] = _user_Jousselme_distance(X[i], X[j])
    distances = distances + distances.T
    return distances
    """
    dist_dic = dict()
    for e in itertools.product(range(n_sample), range(n_sample)):
        set_e = frozenset(e)
        #print(type(set_e))
        if len(set_e) == 2 and set_e not in dist_dic.keys():
            if metric == "conflict":
                dist_dic[set_e] = _user_conflict(X[e[0]], X[e[1]])
            elif metric == "Jousselme":
                dist_dic[set_e] = _user_Jousselme_distance(X[e[0]], X[e[1]])
    return dist_dic
    """
class BeliefDBSCAN(object):
    def __init__(self, min_samples, eps, metric = 'conflict'):
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric
    def fit(self, X):
        self.distances = cal_distances(X, self.metric)
    def predict(self, X):
        #import pdb; pdb.set_trace()
        if hasattr(self, 'distances'):
            return self._belief_DBSCAN(X, self.distances, self.eps, self.min_samples, self.metric)
        else:
            raise ValueError("Model is empty, please fit the model first.")
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
    def _belief_DBSCAN(self, X, distances, eps, minPts, metric = "conflict"):
    #dist_dic = gen_dist_dic(X, metric)
        C = 0
        #import pdb; pdb.set_trace()
        n_samples = X.shape[0]
        labels = -np.ones(n_samples, dtype=np.intp)
        for P_idx in range(n_samples):
            if labels[P_idx] != -1:
                continue
            neighbors = range_query(n_samples, distances, P_idx, eps)
            #import pdb; pdb.set_trace()
            if len(neighbors) < minPts:
                labels[P_idx] = -2
                continue
            C += 1
            labels[P_idx] = C
            seeds = neighbors.copy()
            seeds.remove(P_idx)
            for Q_idx in neighbors:
                if labels[Q_idx] == -2:
                    labels[Q_idx] = C
                elif labels[Q_idx] != -1:
                    continue
                labels[Q_idx] = C
                neighbors = range_query(n_samples, distances, Q_idx, eps)
                if len(neighbors) >= minPts:
                    seeds = seeds.union(neighbors)
        return labels
