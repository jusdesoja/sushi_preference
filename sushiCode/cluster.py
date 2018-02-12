#!/usr/bin/env python
# encoding: utf-8

import read
import featureExtraction

vMap = read.read_voter_file("../data/sushi3-2016/sushi3.udata")
read.read_order_file("../data/sushi3-2016/sushi3b.5000.10.order", vMap)
read.read_order_file("../data/sushi3-2016/sushi3a.5000.10.order", vMap)

orderMap = {}
for k,v in vMap.items():
    orderMap[k] = v.get_order_b()

pi = featureExtraction.k_o_means(orderMap, 10, 50)

for k,v in pi.items():
    print(k, len(v))



