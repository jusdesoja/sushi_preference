#!/usr/bin/env python
# encoding: utf-8

"""
    File:   beliefKMeans.py
    Author: Yiru Zhang (yiru.zhang@irisa.fr)
    Date:   Sep 19, 2017

    About
    -------------------------------------
        This file contains methods for k-means between users.
"""

import numpy as np
from iBelief.combinationRules import DST
from iBelief.Dcalculus import Dcalculus
from iBelief.distance import JousselmeDistance
from iBelief.conflict import conflict, func_distance
from exceptions import *
from center_cal import one_mass_center_calculate
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

import pdb

########################Fusion Operations#############################
def _mat_same_shape_check(mats):
    u_shape = mats[0].shape
    for mat in mats[1:]:
        if mat.shape != u_shape:
            return False
    return True




def _fusion(X, strategy = 2, verbose = False):
    """
    Fusion process in "Preference fusion and Condorcet's paradox under uncertainty, Y. Zhang, T. Bouati & A. Martin"
    stragety implies the strategy applied in the fusion.
    The strategy is implemented in an iterative way.

    Parameters
    ----------------------------
    X: array of mass functions, shape (n_sample, n_pref_pair)
    strategy: fusion strategy. 0 for stragety A, 1 for strategy B and 2 for strategy C


    Return
    ----------------------------
    fusion_mass_vector: vector of mass functions, shape(1, n_preference_pairs)
    """
    n_pref_pair = X[0].shape[0]
    #print(X[0].shape)
    fusion_mass_vector = np.empty(X[0].shape, dtype = object)
    X_temp = X.transpose((1,0,2))
    #print(X.shape, X_temp.shape)
    #pdb.set_trace()
    for i in range(n_pref_pair):
        #print(X_temp[i])
        if strategy == 0:
            fusion_mass_vector[i] = _strategy_A(X_temp[i]).T[0]
        if strategy == 2:
            fusion_mass_vector[i] = _strategy_C(X_temp[i])
        if strategy == 4:
            fusion_mass_vector[i] = _strategy_D(X_temp[i]).T[0]
    if verbose:
        print(fusion_mass_vector)
    return fusion_mass_vector


    """
    if not _mat_same_shape_check(prefBFMats):
        raise MatrixShapeMatchError("Preference Belief function matrix do have the same shapes")
    else:
        shape = prefBFMats[0].shape
        fusion_mat = np.zeros(shape)
        mass_shape = prefBFMats[0][0][shape[1]].shape   #the shape of upper right corner mass function
        mass_function_mat = np.empty(mass_shape, dtype = float)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if i < j:
                    for mat in prefBFMats:
                        mass_function_mat = np.vstack((mass_function_mat, mat[i][j]))
                    if strategy ==  0:
                        fusion_mat[i][j] = _strategy_A(mass_function_mat)
                    elif strategy == 1:
                        fusion_mat[i][j] = _strategy_B(mass_function_mat)
                    else:
                        raise IllegalParameterError("Parameter strategy should be either 0 or 1")

        return mass_function_mat
        """
def _strategy_A(mass_function_mat):
    """
        Fusion stragety A.

        Parameter
        ------------------------
            mass_function_mat: matrix of mass function vectors.
                Each vector is from one user on one pair of objects.
    """
    #print(mass_function_mat.shape)
    return DST(mass_function_mat.T, 1)


def _strategy_B(mass_function_mat):
    """
        Fusion strategy B.

        Parameter
        ------------------------
            mass_function_mat: matrix of mass function vectors.
                Each vector is from one user on one pair of objects.
    """

def _strategy_C(mass_function_mat):
    #pdb.set_trace()
    return one_mass_center_calculate(mass_function_mat)

def _strategy_D(mass_function_mat):
    return DST(mass_function_mat.T, 14)
######################Jusselme distance operations###########################


def _user_Jousselme_distance(flat_BFMat1, flat_BFMat2):
    if not flat_BFMat1.shape == flat_BFMat2.shape:
        raise MatrixShapeMatchError("Global Jusselme distance need two matrix with same shape. matrix with size %d and %d is given" % (flat_BFMat1.size, flat_BFMat2.size))
    else:
        J_size = flat_BFMat1.shape[0]
        D = Dcalculus(flat_BFMat1[0].size)
        J_distance = 0
        #print(J_size)
        for i in range(J_size):
            J_distance = J_distance + JousselmeDistance(flat_BFMat1[i], flat_BFMat2[i], D)
        return J_distance / J_size

def _user_conflict(flat_BFMat1, flat_BFMat2):
    if flat_BFMat1.shape != flat_BFMat2.shape:
        raise ValueError("Global Conflict need two matrix with same shape; matrix with size %s and %s is given" % (flat_BFMat1.size, flat_BFMat2.size))
    else:
        C_size = flat_BFMat1.shape[0]
        D = Dcalculus(flat_BFMat1[0].size)
        U_conflict = 0
        for i in range(C_size):
            U_conflict = U_conflict + conflict(flat_BFMat1[i], flat_BFMat2[i], D)
        return U_conflict

def _user_function_distance(flat_BFMat1, flat_BFMat2, f):
    if flat_BFMat1.shape != flat_BFMat2.shape:
        raise ValueError("Global Conflict need two matrix with same shape; matrix with size %s and %s is given" % (flat_BFMat1.size, flat_BFMat2.size))
    else:
        size=flat_BFMat1.shape[0]
        dist=0
        for i in range(size):
            dist = dist + func_distance(flat_BFMat1[i], flat_BFMat2[i], f)
        return dist/size
###################### K-means #####################################

def _init_centroids(X, k, init, random_state=None, x_squared_norms=None, init_size=None):
    """
    Compute the initial centroids

    Parameters
    -------------
    X : array of mass functions, shape (n_samples, n_pref_pair)

    k : integer
        The number of centroids.

    x_squared_norms: array, shape(n_samples,)
        Squared Euclidean norm of each data point.

    random_state : numpy.RandomState
        The generator used to initialize the centers.

    init_size: int, optional
        Number of samples to randomly sample for speeding up the
        initialization

    Returns:
    ---------
    centers: array, shape(k, n_pref_pair)

    """

    # For instance, we only implement center selection process by random sampling.
    #import random
    random_state = check_random_state(random_state)


    interval = X.shape[0]
    centers = X[np.random.choice(range(interval), k, replace = False)]
    print("centers initialized, shape:%s" % (centers.shape,))
    return centers

def _tolerance(X, tol):
    "Returen a tolerance value which is independent of the data set"

    # We modified this method from scikit learn library.
    # in sklearn, the tolerence is based on the variance of the dataset
    # However, the variance of mass function is not defined,
    # We return directly tolerence

    #TODO define a variance for mass function
    return tol

def _k_means_single(X, n_clusters, max_iter, init, verbose = False, x_squared_norms = None, random_state = None, tol = 1e-3, precompute_distance=True):

    best_labels, best_inertia, best_centers = None, None, None
    centers = _init_centroids(X, n_clusters, init, random_state = random_state, x_squared_norms =x_squared_norms)
    if verbose:
        print("Initialization complete")
    distances = np.zeros(shape=(X.shape[0]), dtype = float)

    #iterations
    for i in range(max_iter):
        centers_old = centers.copy()
        # Label assignment is also called the E-step of EM
        print("E step:")
        labels, inertia, distances = _labels_inertia(X, x_squared_norms, centers, precompute_distance, distances)
        #pdb.set_trace()
        # Computation of means(we use "fusion" here) is also called M-step of EM
        print("M step:")
        centers = _centers_dense(X, labels, n_clusters, distances)
        if verbose:
            print("Iteration %2d, inertia %.3f \nBest inertia: %s" % (i, inertia, best_inertia))
        if best_inertia is None or inertia < best_inertia:
            print("best result updated")
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_distances = distances
        center_shift_total = 0.0
        for center_idx in range(centers.shape[0]):
            center_shift_total = center_shift_total + _user_function_distance(centers_old[center_idx], centers[center_idx], 'q')
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break
    #print("best inertia:%f" % best_inertia)
    """
    if center_shift_total > 0:
        # return E-step in case of non-convergence so that predicted labels match cluster centers
        best_labels, best_inertia, best_distances = \
            _labels_inertia(X, x_squared_norms, best_centers, precompute_distance=precompute_distance, distances = distances)
    """
    return best_labels, best_inertia, best_centers, best_distances, i + 1


def _centers_dense(X, labels, n_clusters, distances):
    """
    M step of the k-means EM algorithm

    Computation of cluster centers

    Parameters
    ---------------
    X : data array, shape(n_samples, n_pref_pair)

    labels : array of integers, shape (n_samples)
        Current label assignment

    n_clusters : int
        Number of desired clusters

    distances : array-like, shape(n_samples)
        Distance to the closest cluster for each sample

    Returns
    --------
    centers : array, shape(n_cluster, n_preference_pairs)
        The resulting centers
    """

    #n_samples = X.shape[0]
    #n_features = X.shape[1]
    centers_shape = (n_clusters, X.shape[1], X.shape[2])
    centers = np.zeros(centers_shape, dtype = object)

    n_samples_in_cluster = np.bincount(labels, minlength = n_clusters)
    print(n_samples_in_cluster)
    empty_clusters = np.where(n_samples_in_cluster==0)[0]
    non_empty_clusters = np.where(n_samples_in_cluster != 0)[0]
    #print("empty cluster nb:%d" % empty_clusters.size)
    if len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]
        #print("instance far from centers: %s" % far_from_centers)
        for i, cluster_id in enumerate(empty_clusters):
            print("empty cluster id: %d" % cluster_id)
            new_center = X[far_from_centers[i]]
            centers[cluster_id] = new_center
            n_samples_in_cluster[cluster_id] = 1
            labels[far_from_centers[i]] = cluster_id

    for i in non_empty_clusters:
        #pdb.set_trace()
        centers[i] = _fusion(X[np.where(labels == i)[0]], strategy = 4)
        print("new center for %d calculated" % i)
    return centers



def _labels_inertia(X, x_squared_norms, centers, precompute_distance= True, distances = None):
    """
    E step of the k-means EM algorithm.
    Compute the labels and the inertia of the given samples and centers.

    Parameters
    -------------
    X :
    """
    n_clusters = centers.shape[0]
    n_samples = X.shape[0]

    inertia = 0.0

    labels = -np.ones(n_samples, np.int32)
    #Assign distances array for intra inertia calculating
    if distances is None:
        distances = np.zeros(shape=(n_samples, ), dtype = object)
    for sample_idx in range(n_samples):
        min_dist = -1
        for center_idx in range(n_clusters):
            ##### inertia evaluated by Jousselme distance
            #dist = _user_Jousselme_distance(X[sample_idx], centers[center_idx])
            #######inertia evaluated by conflict measure
            #dist = _user_conflict(X[sample_idx], centers[center_idx])
            ######"inertia evaluated by function distance
            dist = _user_function_distance(X[sample_idx], centers[center_idx], 'q')

            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                labels[sample_idx] = center_idx
        distances[sample_idx] = min_dist
        inertia = inertia + min_dist
    print("labels after step E: %s" % labels)

    return labels, inertia, distances

def k_means(X, n_clusters, init = 'k_means', precompute_distance = 'auto', n_init = 10, max_iter = 300, verbose = False, tol = 1e-4, random_state= None, copy_x = True, n_jobs = 1, algorithm = "auto", return_n_iter=False):
    """
    k-means clustering method for preferences


    Parameters
    ------------------------------
    X: array of mass functions, shape (n_sample, n_pref_pair)
    n_init: number of clusters
    max_iter: optional, maxmum iteration time
    """

    if n_init <= 0:
        raise ValueError("Invalid value for initialization."
                         " n_init=%d must bigger than zero." % n_init)
    random_state = check_random_state(random_state)
    if max_iter <=0:
        raise ValueError("Number of iteration should be a positive integer value"
                         "got %d instead" % max_iter)

    tol = _tolerance(X, tol)

    # init
    #centers = _init_centroids(X, n_clusters, init,random_state = random_state)

    # Diss...............
    # Allocate memory to store the distance for each sample to its
    # closer center for reallocation in case of ties
    # distances = np.zeros(shape = (X.shape[0],), dtype = X.dtype)

    if precompute_distance == 'auto':
        n_samples = X.shape[0]
        precompute_distance = (n_clusters * n_samples) < 12e6
    elif isinstance(precompute_distance, bool):
        pass
    else:
        raise ValueError('precompute_distance should be "auto" or True/False'
                         ', but a value of %r was passed' % precompute_distance)

    #TODO Validate init array


    #TODO substract of mean of x for more accurate distance computations


    #TODO precompute squared norms of data points
    # square norm of mass function has not been defined


    best_labels, best_inertia, best_centers = None, None, None
    x_squared_norms = None
    # TODO complete different k-means calculation
    #for instance, we use _k_means_single
    kmeans_single = _k_means_single


    if n_jobs == 1:
        # single thread
        #iteration

        for it in range(n_init):
            # run a k-means once
            labels, inertia, centers, distances, n_iter_ = kmeans_single(X, n_clusters, max_iter = max_iter, init = init, verbose = verbose, precompute_distance = precompute_distance, tol=tol, x_squared_norms = x_squared_norms, random_state = random_state)
            #determine if these results are the best so far
            #print("best inertia: %.3f, current inertia: %.3f"%(best_inertia, inertia))
            if best_inertia is None or inertia < best_inertia:
                #print("Better inertia, update new labels")
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_distances = distances
                best_n_iter = n_iter_
    else:
        # TODO multiple thread for parallelisation of k-means run
        print("parallelisation calculation is not supported yet")


    if return_n_iter:
        return best_centers, best_labels, best_inertia, best_distances, best_n_iter
    else:
        return best_centers, best_labels, best_inertia, best_distances


######################## clustering evaluation

def _inter_intra_inertia(centers, labels, distances):
    n_clusters = centers.shape[0]
    n_samples = labels.shape[0]
    centers_fusion = _fusion(centers, 4)
    n_samples_in_cluster = np.bincount(labels, minlength=n_clusters)

    inter, intra = 0, 0
    for c in range(n_clusters):
        #inter += (_user_Jousselme_distance(centers[c], centers_fusion) ** 2) * float(n_samples_in_cluster[c]) / n_samples
        inter += (_user_function_distance(centers[c], centers_fusion, 'q') ** 2) * float(n_samples_in_cluster[c]) / n_samples
        for s in range(n_samples):
            intra += distances[s] **2 / n_samples_in_cluster[c]

    return inter, intra
def _ratio_inter_intra_inertia(centers, labels, distances):
    inter, intra = _inter_intra_inertia(centers, labels, distances)
    return inter / intra
def _pairwise_distances(X,metric = "jousselme",  n_jobs=1):
    """

    """
    # TODO parallelisation

    n_samples = X.shape[0]
    distances = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if (j > i ):
                if metric == "jousselme":
                    distances[i][j] = _user_Jousselme_distance(X[i], X[j])
                elif metric == "conflict":
                    distances[i][j] = _user_conflict(X[i], X[j])
    distances = distances + distances.T
    return distances



def check_number_of_labels(n_labels, n_samples):
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
"to n_samples - 1 (inclusive)" % n_labels)


def silhouette_samples(X, labels, metric='jousselme', distances = None):#, **kwds):
    """Compute the Silhouette Coefficient for each sample.
    The Silhouette Coefficient is a measure of how well samples are clustered
    with samples that are similar to themselves. Clustering models with a high
    Silhouette Coefficient are said to be dense, where samples in the same
    cluster are similar to each other, and well separated, where samples in
    different clusters are not very similar to each other.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 <= n_labels <= n_samples - 1.
    This function returns the Silhouette Coefficient for each sample.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array [n_samples_a, n_features] otherwise
        Belief function feature array.
    labels : array, shape = [n_samples]
             label values for each sample
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array.
        We define a Jousselme distance metric for calculating using _pairwise_distances
    **kwds : optional keyword parameters

    Not used for the moment!!!!
    Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : array, shape = [n_samples]
        Silhouette Coefficient for each samples.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
       <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    #X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    check_number_of_labels(len(le.classes_), X.shape[0])
    if type(distances) == type(None):
        distances = _pairwise_distances(X, metric)#, metric=metric, **kwds)
    unique_labels = le.classes_
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))

    # For sample i, store the mean distance of the cluster to which
    # it belongs in intra_clust_dists[i]
    intra_clust_dists = np.zeros(distances.shape[0], dtype=distances.dtype)

    # For sample i, store the mean distance of the second closest
    # cluster in inter_clust_dists[i]
    inter_clust_dists = np.inf + intra_clust_dists

    for curr_label in range(len(unique_labels)):

        # Find inter_clust_dist for all samples belonging to the same
        # label.
        mask = labels == curr_label
        current_distances = distances[mask]

        # Leave out current sample.
        n_samples_curr_lab = n_samples_per_label[curr_label] - 1
        if n_samples_curr_lab != 0:
            intra_clust_dists[mask] = np.sum(
                current_distances[:, mask], axis=1) / n_samples_curr_lab

        # Now iterate over all other labels, finding the mean
        # cluster distance that is closest to every sample.
        for other_label in range(len(unique_labels)):
            if other_label != curr_label:
                other_mask = labels == other_label
                other_distances = np.mean(
                    current_distances[:, other_mask], axis=1)
                inter_clust_dists[mask] = np.minimum(
                    inter_clust_dists[mask], other_distances)

    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # score 0 for clusters of size 1, according to the paper
    sil_samples[n_samples_per_label.take(labels) == 1] = 0
    return sil_samples

def silhouette_score(X, labels, metric='euclidean', sample_size=None,
                     random_state=None, distances = None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples.
    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is 2 <= n_labels <= n_samples - 1.
    This function returns the mean Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`silhouette_samples`.
    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.
    Read more in the :ref:`User Guide <silhouette_coefficient>`.
    Parameters
    ----------
    X : array [n_samples_a, n_samples_a] if metric == "precomputed", or, \
             [n_samples_a, n_features] otherwise
        Array of pairwise distances between samples, or a feature array.
    labels : array, shape = [n_samples]
         Predicted labels for each sample.
    metric : string, or callable
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`metrics.pairwise.pairwise_distances
        <sklearn.metrics.pairwise.pairwise_distances>`. If X is the distance
        array itself, use ``metric="precomputed"``.
    sample_size : int or None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    random_state : int, RandomState instance or None, optional (default=None)
        The generator used to randomly select a subset of samples.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`. Used when ``sample_size is not None``.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.
    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_
    """
    if sample_size is not None:
        #X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])
        random_state = check_random_state(random_state)
        indices = random_state.permutation(X.shape[0])[:sample_size]
        if metric == "precomputed":
            X, labels = X[indices].T[indices].T, labels[indices]
        else:
            X, labels = X[indices], labels[indices]
    return np.mean(silhouette_samples(X, labels, metric=metric, distances = distances,**kwds))



