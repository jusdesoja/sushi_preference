#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import random
import re

def DMS2decimal(DMSstring):
    try:
        DMS = re.search('(\d+)°(\d+)′(\d+)″', DMSstring)
        if DMS:
            decimal = float(DMS.group(1))+float(DMS.group(2))/60+float(DMS.group(3))/3600
            return decimal
        else:
            return DMSstring
    except TypeError:
        print("Error: ",DMSstring)

def random_gps_gen_from_range(s_lat,n_lat, e_lon, w_lon):
    """Generate random gps position in range of given positions
    Parameters:
    -----------
    s_lat: latitude of south point.
    n_lat: latitude of north point.
    e_lon: longitude of east point.
    w_lon: longitude of west point.

    Returns:
    --------
    latitude, longitude
    """
    #print(s_lat, n_lat, e_lon, w_lon)
    latitude = random.uniform(s_lat, n_lat)
    longitude = random.uniform(e_lon, w_lon)
    return latitude, longitude
def random_gps_gen_from_center(c_lat, c_lon, r):
    latitude = random.uniform(c_lat-r, c_lat+r)
    longitude = random.uniform(c_lon-r, c_lon+r)
    return latitude, longitude

def gps_gen_from_no(geo_info, region_no, mode=1):
    """Generate latitude and longitude position in a given aera range from region numbers

    """
    #col_list=["south_lat","north_lat","east","west"]
    s_lat,n_lat, e_lon, w_lon , c_lat,c_lon= list(geo_info.loc[region_no, ["south_lat","north_lat","east","west", "center_lat", "center"]])
    if mode == 1:
        return random_gps_gen_from_range(s_lat,n_lat, e_lon, w_lon)
    elif mode == 2:
        r = min(abs(n_lat-c_lat), abs(s_lat-c_lat), abs(e_lon-c_lon), abs(w_lon-c_lon)) / 2
        return random_gps_gen_from_center(c_lat, c_lon, r)

def read_geo_info(file_name):
    geo_info = pd.read_csv(file_name)
    geo_info["center_lat"] = geo_info.center_lat.apply(DMS2decimal)
    geo_info["east_lat"] = geo_info.east_lat.apply(DMS2decimal)
    geo_info["west_lat"] = geo_info.west_lat.apply(DMS2decimal)
    geo_info["south_lat"] = geo_info.south_lat.apply(DMS2decimal)
    geo_info["north_lat"] = geo_info.north_lat.apply(DMS2decimal)
    geo_info["center"] = geo_info.center.apply(DMS2decimal)
    geo_info["east"] = geo_info.east.apply(DMS2decimal)
    geo_info["west"] = geo_info.west.apply(DMS2decimal)
    geo_info["south"] = geo_info.south.apply(DMS2decimal)
    geo_info["north"] = geo_info.north.apply(DMS2decimal)
    return geo_info

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def scatter_on_map(vMap, labels, geo_info, title=None, mode =1):
    """
    Plot users on Japan's map. Users in the same cluster are plotted with identical color.

    Parameters:
    vMap: User's information map.
    labels: cluster labels of all users
    geo_info: prefectures' geographical information
    title: Title of the graph
    mode: geographical position generation method.
        1: randomly generate position in the range of the prefectures' east, west north south extreme border
        2: randomly generate position around the center of the prefecture
    """
    classes = np.unique(labels)
    colors = []
    for i in range(0,len(classes)):
        colors.append(generate_new_color(colors,pastel_factor = 0.9))

    m = Basemap(llcrnrlon=128.,llcrnrlat=30,urcrnrlon=147.,urcrnrlat=46., projection='lcc',lat_1=20.,lat_2=40.,lon_0=135.,
                resolution ='h',area_thresh=1000.)
    m.drawmapboundary(fill_color='#99ffff')
    #m.drawlsmask(land_color='coral', ocean_color='white', lakes=True)
    m.fillcontinents(color='#cc9966',lake_color='#99ffff')
    for c in classes:
        voters = np.where(labels == c)[0]
        #print('voters:',voters)
        lats = []
        lons = []
        for k in voters:
            #print(k)
            gps = gps_gen_from_no(geo_info, vMap[k].get_prefecture_young(), mode)
            lats.append(gps[0])
            lons.append(gps[1])
        #import pdb; pdb.set_trace()
        m.scatter(lons,lats,marker='.',color=colors[np.where(classes==c)[0][0]], latlon=True, zorder=10)
    if title == None:
        title = "Sushi Consumer Geo Info"
    plt.title(title, fontsize=12)
    plt.show()



