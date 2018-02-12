#!/usr/bin/env python
# encoding: utf-8

#from tools import all_mass_init


class Voter(object):
    def __init__(self, ID, gender, age, fillTime, prefectureYoung, regionYoung, eastWestYoung, prefectureCur,regionCur, eastWestCur, youngCurEq):
        self.ID = ID
        self.gender = gender
        self.age = age
        self.fillTime = fillTime
        self.prefectureYoung = prefectureYoung
        self.regionYoung = regionYoung
        self.eastWestYoung = eastWestYoung
        self.prefectureCur = prefectureCur
        self.regionCur = regionCur
        self.eastWestCur = eastWestCur
        self.youngCurEq = youngCurEq

    def __repr__(self):
        return str(self.ID) + "; " + str(self.gender) + "; " +str(self.age) + "; " + str(self.fillTime) + "; " \
            + str(self.prefectureYoung) + "; " + str(self.regionYoung) + "; " + str(self.eastWestYoung) + "; " \
            + str(self.prefectureCur) + "; " + str(self.regionCur) + "; " + str(self.eastWestCur) + "; " + str(self.youngCurEq)

    def get_ID(self):
        return self.ID

    def get_gender(self):
        return self.gender

    def get_age(self):
        return self.age

    def get_fillTime(self):
        return self.fillTime

    def get_prefecture_young(self):
        return self.prefectureYoung

    def get_region_young(self):
        return self.regionYoung

    def get_east_west_young(self):
        return self.eastWestYoung

    def get_prefecture_current(self):
        return self.prefectureCur

    def get_regrion_current(self):
        return self.regionCur

    def get_east_west_current(self):
        return self.eastWestCur

    def get_young_current_equal(self):
        return self.youngCurEq

    def get_scores(self):
        return self.scoreMap

    def get_order_a(self):
        return self.orderA

    def get_order_b(self):
        return self.orderB

    def set_scores(self, scoreList):
        self.scoreMap = dict()
        for i,score in enumerate(scoreList):
            if score != -1:
                self.scoreMap[i] = score
        #self.preferenceScore = scoreList

    def set_order_a(self, orderList):
        if len(orderList) == 10:
            self.orderA = orderList
        else:
            print('order size error')

    def set_order_b(self, orderList):
        if len(orderList) == 10:
            self.orderB = orderList
        else:
            print('order size error')


    #def calculate_pref_mass(self):
    #    if hasattr(self, orderB) and hasattr(self, preferenceScore):


