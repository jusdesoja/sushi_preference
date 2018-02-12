#!/usr/bin/env python
# encoding: utf-8

class Sushi():
    def __init__(self, ID, name, style, majGr, minGr, oiliness, eatFre, price, soldFre):
        self.ID = ID
        self.name = name
        self.style = style
        self.majGr = majGr
        self.minGr = minGr
        self.oiliness = oiliness
        self.eatFre = eatFre
        self.price = price
        self.soldFre = soldFre

    def __repr__(self):
        return self.ID + "; " + self.name + "; " + self.style + "; " +self.majGr + "; "+ self.minGr + "; " \
            + self.oiliness +"; "+ self.eatFre + "; " + self.price + "; " + self.soldFre
    def get_ID(self):
        return self.ID

    def get_name(self):
        return self.name

    def get_style(self):
        return self.style

    def get_majour_group(self):
        return self.majGr

    def get_minor_group(self):
        return self.minGr

    def get_oiliness(self):
        return self.oiliness

    def get_eat_fre(self):
        return self.eatFre

    def get_price(self):
        return self.price

    def get_sold_fre(self):
        return self.soldFre

