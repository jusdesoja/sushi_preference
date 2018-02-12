#!/usr/bin/env python
# encoding: utf-8
import math
import numpy as np
from beliefDBSCAN import cal_distances
#from simplify_labels import simplify_labels
"""
def cal_distances_common(X, metric="conflict"):
    n_sample = X.shape[0]
    distances = np.zeros((n_sample, n_sample))
    common_pair_mat = np.zeros((n_sample, n_sample))

    total_ign = np.zeros(X.shape[2])
    total_ign[-1] = 1
    for i in range(n_sample):
        for j in range(n_sample):
            if (j>i):
                if metric == "conflict":
                    distances[i][j] = _user_conflict(X[i], X[j])
                elif metric == 'jousselme':
                    distances[i][j] = _user_Jousselme_distance(X[i], X[j])
    distances = distances + distances.T
    return distances,
"""
def get_KNN(dist_mat, K):
    """

    """
    n_sample = dist_mat.shape[0]
    knn_ind = np.zeros((n_sample,K), dtype = int)
    knn_dist = np.zeros((n_sample,K))
    for i in range(n_sample):
        ind = dist_mat[i].argsort()[:(K+1)]
        #import pdb;pdb.set_trace()
        knn_ind[i] = np.delete(ind, np.where(ind==i))
        knn_dist[i] = dist_mat[i][knn_ind[i].astype(int)]
    return knn_ind, knn_dist

def phi_func(dist, alpha=1, gamma=1):
    """
    a non-increasing mapping function. Map distance to phi
    """
    return alpha * math.exp(-gamma * dist)

def cal_alpha_mat(common_pair_mat, maximum_pair):
    """Calculate alpha0 in phi function
    alpha0 between two individuls has a positive relation with the number of their preference pairs in common
    """
    return common_pair_mat / maximum_pair

def init_labels(n, c):
    """
    initialize a label matrix s of size n*c in boolean type
    if object i (0~n-1) belongs to cluster k (0~c-1), s[i][k] = True, False otherwise
    Parameters:
    ----------
    c: number of clusters
    n: number of objects
    Return:
    ---------
    s: matrix of size (n*c) in boolean type
        label matrix
    """

    s = np.random.choice(a = [False, True], size = (n, c))
    init_check = True
    while init_check:
        init_check = False
        for i in range(n):
            if np.equal(s[i],False).all():
                init_check = True
                s[i] = np.random.choice([True, False], c)

    #s = np.ones((n,c), dtype = bool)
    return s

def cal_alpha_v(dist, s, alpha_mat, gamma,  verbose = False):
    """
    Calculate alpha matrix and v matrix based on distance matrix and label matrix.

    Parameters:
    ----------
    dist: distance matrix of size n*n (n: n_samples)
    s: label matrix of size n*c
    Return:
    ----------
    a: alpha matrix
    v: v matrix
    """
    #if verbose:
    #    print("calculate alpha and v matrix with parameter alpha0=%f, gamma=%f in phi function"%(alpha, gamma))
    n_samples = dist.shape[0]
    a = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for k in np.where(s[i]==True)[0]:
            for j in np.where(s.T[k] == True)[0]:
                # modified
                #if j!=i:
                a[i][j]=min(phi_func(dist[i][j], alpha_mat[i][j],gamma), 0.6) #avoid alpha equal to 1
                    #print("alpha of individuls pair (%d,%d) : %s" % (i,j, alpha_mat[i][j]))
                    ###############
                    #if a[i][j]==1:
                     #   print(i,j)
    v = -np.log(1-a)
    return a, v

def cal_pl(v, s, sample_order=None, verbose = False):
    """
    Calculate the plausibility matrix u of belonging to each cluster.

    Parameters:
    ----------
    v: v matrix calculated from distance and labels
    s: label matrix
    sample_order: a list of samples in random order.

    Return:
    u: plausibility matrix
    """
    n, c = s.shape
    u = np.zeros((n,c))
    #import pdb; pdb.set_trace()
    if type(sample_order) == type(None):
        sample_order = np.arange(n)
        np.random.shuffle(sample_order)
    if verbose:
        print("sample order:%s" % (sample_order))
    for i in sample_order:
        for k in np.where(s[i]==True)[0]:
            u_temp=0
            for j in np.where(s.T[k]== True)[0]:
                u_temp += v[i][j] #* s[j][k]
                #print(s[j][k])
            u[i][k] = u_temp
    return u
def update_labels(u, s1, verbose = False):
    """
    update label matrix s based on u matrix. Each object is assigned to the cluster with the highest plausibility.
    Check if any modification is done after updating

    Parameters:
    -----------
    u: plausibility matrix
    s1: original labels matrix

    Returns:
    --------
    s1: new label matrix
    change: boolean
        if any chagement exists, return True, otherwise False
    """
    n, c = s1.shape
    s2 = np.zeros(s1.shape, dtype = bool)
    for i in range(n):
        #import pdb; pdb.set_trace()
        s2[i][np.where(u[i]==np.amax(u[i]))]=True

    if verbose:
        print("number of different elements:%d" % (np.count_nonzero(s1==s2)==True))
    check = not (s1==s2).all()
    if verbose:
        print("check=",check)
    return s2, check

def EKNNclus(X, K=None, dist_mat = None, common_pair_mat = None, maximum_pair = 10, dist_metric = "conflict", jobs = 1, verbose = False):
    n_samples = X.shape[0]
    if K == None:
        if verbose:
            print('cluster number not given, set to objects number')
        K = n_samples
    elif K > n_samples:
        if verbose:
            print("K=%d too large, reset to n_sample = %d"%(K, n_samples))
        K = n_samples

    #initialization
    gamma = 1/np.percentile(dist_mat, 50)
    #gamma = 1
    #Distance matrix preparation
    if type(dist_mat) == type(None):
        if verbose:
            print("distance matrix not given, calculating...")
        # TODO: distance calculation
        dist_mat = cal_distances(X, metric = dist_metric)
    #initialize s, v, alpha
    if verbose:
        print('start initialization')
    s = init_labels(n_samples, K)
    alpha, v = cal_alpha_v(dist_mat, s, alpha_mat = cal_alpha_mat(common_pair_mat, maximum_pair) ,gamma = gamma, verbose = verbose)
    import pdb;pdb.set_trace()
    if verbose:
        print("initialization finished. \nStart iteration")
    if jobs == 1:
        #single thread
        change = True
        cpt = 0
        while change:
            u = cal_pl(v, s, verbose = verbose)
            s, change = update_labels(u, s, verbose = verbose)
            cpt += 1
            pdb.set_trace()
            if verbose:
                print("%dth iteration" % (cpt))


        return s
    #TODO: multiple missions

#from simplify_labels import simplify_labels

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
def EKNNclus_Th(X,K,y0,alpha_mat = None,D=None, ntrials=1, q=50, p=1, verbose=True, tr=False):
    if type(D) == type(None):
        if verbose:
            print("distance matrix not given, calculating...")
            D = cal_distances(X)
    knn_ind, knn_dist = get_KNN(D,K)
    n= knn_ind.shape[0]

    if type(alpha_mat) != type(None):
        knn_alpha = np.zeros(knn_ind.shape)
        for i in range(n):
            knn_alpha[i] = alpha_mat[i][knn_ind[i]]

    g = 1/ np.percentile(knn_dist ** p, q)
    if type(alpha_mat) == type(None):
        alpha = np.minimum(np.maximum(np.exp(-g*(knn_dist**p)), np.full(knn_dist.shape, 1e-4)), np.full(knn_dist.shape, 0.999)) # max
    else:
        alpha = np.maximum(knn_alpha * np.exp(-g*(knn_dist**p)), np.full(knn_dist.shape, 1e-4))
    W = -np.log(1-alpha)

    critmax = -1
    Trace = None
    for trial in range(ntrials):
        change = 1
        k=0
        y=np.array(y0)
        c = np.amax(y)+1
        I = np.eye(c)
        crit  = 0
        #import pdb

        for i in range(n):
            #pdb.set_trace()
            #print(i)
            crit = crit + 0.5 * np.sum(W[i]*(y[i]==y[knn_ind[i]]))
        if tr:
            if type(Trace) == type(None):
                Trace = np.array([trial, k, crit, c])
            else:
                Trace = np.vstack((Trace, np.array([trial, k, crit, c])))
        while(change> 0):
            k = k+1
            ii = np.arange(n)
            np.random.shuffle(ii)
            change = 0
            for i in range(n):
                #import pdb; pdb.set_trace()
                S = I[y[knn_ind[ii[i]]]]
                u = np.dot(W[ii[i]], S)
                kstar = np.argmax(u)
                #import pdb;pdb.set_trace()
                #print(kstar, y[ii[i]], u.shape)
                crit = crit + u[kstar] - u[y[ii[i]]]
                if y[ii[i]] != kstar:
                    y[ii[i]] = kstar
                    change += 1
                if tr:
                    np.vstack((Trace, np.array([trial, k, crit, c])))
            y = simplify_labels(y)
            c = np.amax(y)+1
            I = np.eye(c)
            if verbose:
                #import pdb; pdb.set_trace()
                print(trial, k, change, c)
        #pdb.set_trace()
        if crit > critmax:
            critmax = crit
            ybest = y

    #alpha = 1-np.exp(-W)


    return ybest
