#!/usr/bin/env python
# encoding: utf-8

"""Calculate distance between two BBA on the same frame of discernment.
The distance is represented by Jousselme Distance
"""

import numpy as np
import math
from .Dcalculus import Dcalculus
#from exceptions import IllegalMassSizeError
def JousselmeDistance(mass1,mass2, D = "None"):
    m1 = np.array(mass1).reshape((1,mass1.size))
    m2 = np.array(mass2)
    if m1.size != m2.size:
        raise ValueError("mass vector should have the same size, given %d and %d" % (m1.size, m2.size))
    else:
        if type(D)== str:
            D = Dcalculus(m1.size)
        m_diff = m1 - m2
        #print(D)
        #print(m_diff.shape,D.shape)
        #print(np.dot(m_diff,D))


        #----JousselmeDistance modified for testing, don't forget to correct back------#

        out = math.sqrt(np.dot(np.dot(m_diff,D),m_diff.T)/2.0)
        #out = np.dot(np.dot(m_diff,D),m_diff.T)
        return out

#m1 = np.array([0., 0.31,0.59,0., 0.1, 0., 0. ,0.])

#m2 = np.array([0., 0.5,0.3,0., 0.2, 0., 0. ,0.])
#m2 = np.array([0., 0.28, 0.22, 0., 0.5, 0., 0., 0.])
#m3 = np.array([0.,1.,0.,0.,0.,0.,0.,0.])
#D = Dcalculus(8)
#print(JousselmeDistance(m1, m3,D), JousselmeDistance(m2,m3,D))
