#!/usr/bin/env python
# encoding: utf-8

"""
    ----------------------------------------------------
    Methods for order feature extraction and clustering
    ----------------------------------------------------
"""

import math
import random

def spearman_distance(o1, o2):
    """
    Calculate spearman's distance between two orders o1, o2.
    o1, o2 can be either complete or incomplete
    We assume that both o1 and o2 are legal, which means that the elements are unique in each order, and no tie is allowed in the ranks.
    """
    commonList = list(set(o1).intersection(set(o2)))
    L = len(commonList)

    new_o1 = []
    new_o2 = []
    for element in o1:
        if element in commonList:
            new_o1.append(element)
    for element in o2:
        if element in commonList:
            new_o2.append(element)
    diff_sum = 0
    for ele in commonList:
        diff_sum = diff_sum + math.pow((new_o1.index(ele)-new_o2.index(ele)),2)
    d_rho = ((6 * diff_sum)/(math.pow(L,3)-L))

    return d_rho


def k_o_means(S, n_clusters, max_iter):
    """
    k-o'means-TMSE algorithm

    Parameters
    ------------------------------------
    S: k-v dictionary
        a dictionary of orders.
        with k: voter number, v: order
    n_clusters: integer
        the number of clusters
    max_iter: integer
        the limit of iteration times

    output
    ------------------------------------
        pi: k-v dictionary
            clusters. k: cluster number, v: list of order numbers
    """
    #initialisation
    #---------------------

    #initialise the universal items
    univ_items = set()
    for v in S.values():
        univ_items = univ_items.union(set(v))

    pi = {k:[] for k in range(n_clusters)}
    t = 0
    centroids = {}
    #randomly partition S into n clusters
    for Oi in S.keys():
        k = random.randint(0, n_clusters - 1)
        print(k)
        pi[k].append(Oi)

    pi2 = pi.copy()
    while t < max_iter:
        t = t+1
        for k in pi.keys():
            Ck = [S[o] for o in pi[k]]
            centroids[k] = mean_rank_in_Ck(Ck, univ_items)
        pi = {k:[] for k in range(n_clusters)} #initialise pi to update clsterting
        for Oi in S.keys():
            pi[argmin_Ck_distance(S[Oi], centroids, univ_items)].append(Oi)
        if pi == pi2:
            break
        else:
            pi2 = pi.copy()
    return pi

def argmin_Ck_distance(oi, centroids, univ_items):
    min_value = -1
    for c in centroids.keys():
        if min_value == -1:
            min_value = spearman_distance(oi,centroids[c])
            arg_min = c
        else:
            distance = spearman_distance(oi,centroids[c])
            if min_value > distance:
                min_value = distance
                arg_min = c
    return arg_min

def mean_rank_in_Ck(Ck, univ_items):
    """
    Get mean rank of each item in cluster Ck

    Parameters
    ------------------------------------
        Ck: cluster indexed by k of orders
        univ_items: all items to be ranked
    output
    ------------------------------------
        mean_rank: mean rank in Ck
    """
    numCk = len(Ck)
    print(numCk)
    L = len(univ_items)
    rk_dict = dict((k, 0) for k in univ_items)
    for Oi in Ck:
        Li = len(Oi)
        for xj in univ_items:
            if xj in Oi:  #if xj belongs to Xi
                E_rj_Oi = Oi.index(xj) * ((L + 1) / (Li + 1))
            else:
                E_rj_Oi = 0.5 *  (L + 1)
            rk_dict[xj] = rk_dict[xj] + E_rj_Oi
    for rj in rk_dict.keys():
        rk_dict[rj] = rk_dict[rj] / numCk
    mean_rank = sorted(rk_dict, key = rk_dict.get)
    #print(mean_rank)
    return mean_rank



"""
Method tests
"""
"""
print("testing:")
o1 = [1,3, 4, 6]
o2 = [5,4,3,2,6]

print(spearman_distance(o1,o2))
"""
