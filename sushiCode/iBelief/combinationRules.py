## Combination rules for masses
##' @param MassIn The matrix containing the masses. Each column represents a
##' piece of mass.
##' @param criterion The combination criterion:
##'
##' criterion=1 Smets criterion (conjunctive combination rule)
##'
##' criterion=2 Dempster-Shafer criterion (normalized)
##'
##' criterion=3 Yager criterion
##'
##' criterion=4 Disjunctive combination criterion
##'
##' criterion=5 Dubois criterion (normalized and disjunctive combination)
##'
##' criterion=6 Dubois and Prade criterion (mixt combination), only for Bayesian masses whose focal elements are singletons
##'
##' criterion=7 Florea criterion
##'
##' criterion=8 PCR6
##'
##' criterion=9 Cautious Denoeux Min for functions non-dogmatics
##'
##' criterion=10 Cautious Denoeux Max for separable masses
##'
##' criterion=11 Hard Denoeux for functions sub-normal
##'
##' criterion=12 Mean of the bbas
##'
##' criterion=13 LNS rule, for separable masses
##'
##' criterion=131 LNSa rule, for separable masses
##' @param TypeSSF If TypeSSF = 0, it is not a SSF, the general case. If TypeSSF = 1, a SSF with a singleton as a focal element. If TypeSSF = 2, a SSF with any subset of \eqn{\Theta} as a focal element.
##' @return The combined mass vector. One column.

import numpy as np
from .DST_fmt_functions import *

def DST(massIn, criterion, TypeSSF=0):
    """
    Combination rules for multiple masses.

    Parameters:
    ----------
    massIn: ndarray
        Masses to be combined, represented by a 2D matrix
    criterion: integer
        Combination rule to be applied
        The criterion values represented respectively the following rules:
            criterion=1 Smets criterion (conjunctive combination rule)
            criterion=2 Dempster-Shafer criterion (normalized)
            criterion=3 Yager criterion
            criterion=4 Disjunctive combination criterion
            criterion=5 Dubois criterion (normalized and disjunctive combination)
            criterion=6 Dubois and Prade criterion (mixt combination), only for Bayesian masses whose focal elements are singletons
            criterion=7 Florea criterion
            criterion=8 PCR6
            criterion=9 Cautious Denoeux Min for functions non-dogmatics
            criterion=10 Cautious Denoeux Max for separable masses
            criterion=11 Hard Denoeux for functions sub-normal
            criterion=12 Mean of the bbas
            criterion=13 LNS rule, for separable masses
            criterion=131 LNSa rule, for separable masses
    TypeSSF: integer
        If TypeSSF = 0, it is not a SSF (the general case).
        If TypeSSF = 1, it is a SSF with a singleton as a focal element.
        If TypeSSF = 2, it is a SSF with any subset of \Theta as a focal element.

    Return:
    ----------
    Mass: ndarray
        a final mass vector combining all masses
    """
    n, m = massIn.shape
    if criterion in (4,5,6,7):
        b_mat = np.apply_along_axis(mtob, axis = 0, arr = massIn)
        b = np.apply_along_axis(np.prod, axis = 1, arr = b_mat )
    if criterion in (1,2,3,6,7, 14):

        q_mat = np.apply_along_axis(mtoq, axis = 0, arr = massIn) #apply on column. (2 in R)
        q = np.apply_along_axis(np.prod, axis = 1,arr = q_mat) # apply on row (1 in R)
    if criterion == 1:
        #Smets criterion
        Mass = qtom(q)
        Mass[0] = 1.0 - np.sum(Mass[1:])
    elif criterion == 2:
        #Dempster-Shafer criterion (normalized)
        Mass = qtom(q)
        Mass = Mass/(1-Mass[0])
        Mass[0] = 0
    #elif criterion == 3:
        #Yager criterion
    #elif criterion == 4:
        #disjunctive combination
    #elif criterion == 5:
    #elif criterion == 6:
    #elif criterion == 7:
    #elif criterion == 8:
    elif criterion == 9:
        #Cautious Denoeux min for fonctions non-dogmatic
        wtot = np.apply_along_axis(mtow, axis = 0, arr = massIn)
        w = np.apply_along_axis(np.ndarray.min , axis = 1, arr = wtot )
        Mass = wtom(w)
    elif criterion == 10:
        #Cautious Denoeux max only for separable fonctions
        wtot = np.apply_along_axis(mtow, axis = 0, arr = massIn)
        w = np.apply_along_axis(np.ndarray.max, axis = 1, arr = wtot)
        Mass = wtom(w)
    #elif criterion == 11:
    elif criterion == 12:
        # mean of the masses
        Mass = np.apply_along_axis(np.mean, axis = 1, arr = massIn)
    #elif criterion == 13:

    elif criterion == 14:
        Mass = qtom(np.apply_along_axis(np.mean, axis=1, arr=q_mat))
    return Mass[np.newaxis].transpose()



