#!/usr/bin/env python
# encoding: utf-8

from sushiCode.iBelief.combinationRules import DST
import numpy as np

mN=np.array([[0, 0.2, 0.1, 0, 0.3, 0,0,0.4],[0, 0.1, 0.1, 0.1, 0.3, 0,0,0.4],[0, 0.2, 0.1, 0, 0.1, 0.1,0.1,0.4]])
print(DST(mN.T, criterion=14))
print(DST(mN.T, criterion=2))

