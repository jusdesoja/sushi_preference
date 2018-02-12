#!/usr/bin/env python
# encoding: utf-8

'''
    File:   io.py
    Author: Yiru Zhang (yiru.zhang@irisa.fr)
    Date:   Aug 15, 2017

    About
    --------------------------------
        This file contains a set of modules for reading, writing preference file from sushi preference data
'''

from sushi import *
from voter import *

def read_cand_file(inputFile):
    candMap = dict()
    with open(inputFile) as f:
        for line in f:

            cand = line.split('\t')
            candMap[int(cand[0])] = Sushi(int(cand[0]), cand[1],int(cand[2]), int(cand[3]), int(cand[4]),float(cand[5]), float(cand[6]), float(cand[7]), float(cand[8]))

    return candMap

def read_voter_file(inputFile, n_samples = -1):
    voterMap = dict()
    with open(inputFile) as f:
        voterInd = 0
        for line in f:
            voter = line.split('\t')
            voter = list(map(int,voter))
            voterMap[voterInd] = Voter(voter[0], voter[1], voter[2],voter[3],voter[4], voter[5], voter[6], voter[7], voter[8], voter[9], voter[10])
            voterInd = voterInd + 1
            if n_samples > 0 and voterInd >= n_samples:
                break
    return voterMap

def read_score_file(inputFile, voterMap, n_samples = -1):
    with open(inputFile) as f:
        voterInd = 0
        for line in f:
            scores = line.split()
            scores = list(map(int,scores))
            voterMap[voterInd].set_scores(scores)
            voterInd = voterInd+1
            if n_samples >0 and voterInd >= n_samples:
                break
    return scores

def read_order_file(inputFile, voterMap, n_samples = -1):
    voterInd = 0
    with open(inputFile) as f:
        orderType = f.readline().split()[0]
        for line in f:
            order = list(map(int,line.split()))[2:]
            #print (orderType)
            if orderType == "10":
                voterMap[voterInd].set_order_a(order)
            elif orderType == "100":
                voterMap[voterInd].set_order_b(order)
            voterInd = voterInd + 1
            if n_samples > 0 and voterInd>= n_samples:
                #print("enough samples: voterInd:%d, samples:%d" % (voterInd, n_samples) )
                break


def is_descend(scoreList):
    res = True
    for i in range(len(scoreList)-1):
        if scoreList[i] < scoreList[i+1]:
            res = False
    return res

def check_rationality(voter):
    order = voter.get_order_b()
    scores = voter.get_scores()
    sc_in_ord = list(scores[i] for i in order)
    return is_descend(sc_in_ord)


def count_candidate(voterMap):
    cand_count =[0] * 100
    for v in voterMap.values():
        for cand_idx in v.get_order_b():
            cand_count[cand_idx] += 1
    return cand_count



def copy_lines_in_file(inputFile, outputFile, lineList,order=0):
    f = open(inputFile, "r")
    copy = open(outputFile, "w")
    for i,l in enumerate(f):
        if i in lineList:
            print(i,l)
            copy.write(l)
    f.close()
    copy.close()


"""
a = [5,5,4,3,2]
b = [5,5,4,1,2]
print(is_descend(a))
print(is_descend(b))
"""

vMap = read_voter_file("../data/sushi3-2016/sushi3.udata", 5000)
read_score_file("../data/sushi3-2016/sushi3b.5000.10.score", vMap, 5000)
read_order_file("../data/sushi3-2016/sushi3b.5000.10.order", vMap, 5000)
cand_count_list=count_candidate(vMap)
sorted_cand_index = sorted(range(len(cand_count_list)), key=lambda k:cand_count_list[k],reverse=True)
print(cand_count_list)
print(sorted_cand_index[:18])
most_selected_cand = set(sorted_cand_index[:18])
cpt = 0
voter_list = []
for k,v in vMap.items():
    if set(v.get_order_b()).issubset(most_selected_cand):
        cpt += 1
        voter_list.append(k)
        #print(v.ID)
print(cpt)
print(voter_list)
#copy_lines_in_file("../data/sushi3-2016/orderb_for_copy", "../data/sushi3-2016/sushi3b.com25.10.order", voter_list)
#copy_lines_in_file("../data/sushi3-2016/sushi3b.5000.10.score", "../data/sushi3-2016/sushi3b.com25.10.score", voter_list)
#copy_lines_in_file("../data/sushi3-2016/sushi3.udata", "../data/sushi3-2016/sushi3.com25.udata", voter_list)
"""
print(len(vMap))
cpt = 0

for v in vMap.values():

    if check_rationality(v):
        cpt  = cpt + 1
print(cpt, len(vMap))
print(float(cpt)/len(vMap))
print("scores:",vMap[0].get_scores())
"""
