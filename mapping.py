#!/usr/bin/env python3

#================================================================================
# This program generate gas distribution maps including mean map and variance map
# using kernel DM+V algorithm.
#================================================================================

__author__      = "Luong Duc Nhat"
__copyright__   = "Copyright 2021, The Chemical Plume Tracing (CPT) Robot Project"
__credits__     = ["Luong Duc Nhat"]
__license__     = "GPL"
__version__     = "2.0.0"
__maintainer__  = "Luong Duc Nhat"
__email__       = "luong.d.aa@m.titech.ac.jp"
__status__      = "Production"

import matplotlib.pyplot as plt
import numpy as np
from grid_map import Gridmap

"""
Create new grid map object in form of numpy.array with the size based on the input data file
Param data: Input data extract from csv file
"""
def get_plain_map(data):
    column_x = data['x']
    column_y = data['y']
    x_min = column_x.min() 
    x_max = column_x.max()
    y_min = column_y.min()
    y_max = column_y.max()
    return Gridmap(x_min, x_max, y_min, y_max)


"""
Indicates the likelihood that the measurement represents the concentration at
a given distance from the point of measurement. The readings were convolved 
using the two dimensional normalised Gaussian.
Param distance: A matrix of distances from a measurement points to all cells
Param sigma: Kernel width
"""
def weight_cal(distance, sigma):
    exp_term =  np.exp(-0.5*distance*distance/(sigma**2))
    weight = (0.5*exp_term)/(np.pi*sigma**2)
    return weight 


"""
Calculate the distances of a measument point to other cells in the map
Param data: Input data extract from csv file
Param index: Index of measurement point
Param map: The gas distribution map
"""
def get_distance(data, index, map):
    xs_, ys_ = map.get_grid() 
    dx = data.at[index,'x'] - xs_
    dy = data.at[index,'y'] - ys_
    distance = np.sqrt(dx**2 + dy**2)
    return distance


"""
To calculate thr estimate v0 of the distribution variance
Param data: Input data extract from csv file
"""
def cal_mean_variance(data):
    mean_val = data['sensor_value'].mean()
    sum = 0
    for i in range(len(data.index)):
        sensor_reading = data.at[i,'sensor_value']
        sum += (sensor_reading-mean_val)**2
    mean_variance = sum/(len(data.index)-1)
    return mean_variance


"""
Plot the NLPD fuction. 
Param sigma_list: list of tested kernel width
Param nlpd_val_list: list of calculated nlpd value with respected to sigma_list
"""
def plot_nlpd(list_nlpd):
    plt.plot(*zip(*list_nlpd))
    plt.title('NLPD function')
    plt.xlabel('Kernel width (mm)')
    plt.ylabel('NLPD value')
    plt.show()


"""
Plot nlpd graph to get the smallest value of nlpd => best kernel width
Param data: Input data extract from csv file
"""
def nlpd_test(data):
    list_nlpd = []
    for sigma in range(10,300,10):        
        map = get_plain_map(data)
        mean_map, variacne_map = get_maps(data, sigma)
        
        sum = 0
        for i in range(len(data.index)):
            distance = get_distance(data, i, map)
            ind = np.unravel_index(np.argmin(distance, axis=None), distance.shape)

            v_hat = variacne_map[ind]
            r_hat = mean_map[ind]
            value = data.at[i,'sensor_value']

            temp = ((value-r_hat)**2)/v_hat
            sum += np.log(v_hat) + temp

        nlpd_val = sum/(2*len(data.index)) + np.log(2*np.pi)
        list_nlpd.append((sigma, nlpd_val))
    
    plot_nlpd(list_nlpd)
    best_sigma = min(list_nlpd, key = lambda t: t[1])[0]
    print("Best sigma is: ", best_sigma)
    return best_sigma


def get_mean_map(data, map):
    estimate_mean = data['sensor_value'].mean()
    mean_map = map.confidence *(map.acc_reading/map.acc_weight) + (1-map.confidence)*estimate_mean
    return mean_map


def get_variance_map(data, map, mean_map, sigma):
    xs_, ys_ = map.get_grid() 
    estimate_mean = data['sensor_value'].mean()
    sum = 0
    for i in range(len(data.index)):
        sensor_reading = data.at[i,'sensor_value']
        sum += (sensor_reading - estimate_mean)**2
    mean_val = sum/(len(data.index)-1)

    for i in range(len(data.index)):
        dx = data.at[i,'x'] - xs_
        dy = data.at[i,'y'] - ys_
        distance = np.sqrt(dx**2 + dy**2)
        
        weight = weight_cal(distance, sigma)
        map.acc_variance += weight*(data.at[i,'sensor_value']-mean_map)**2
   
    variance_map = map.confidence*(map.acc_variance/map.acc_weight) + (1-map.confidence)*mean_val
    return variance_map


"""
Get the mean map and variance map in form of numpy.array
Param data: data extract from csv file
Param sigma: kernel width. Could be set automaticaly or manually in config file
"""
def get_maps(data, sigma):
    map = get_plain_map(data)
    xs_, ys_ = map.get_grid()   

    # Adding accumulated weight and accumulated reading
    for i in range(len(data.index)):
        dx = data.at[i,'x'] - xs_
        dy = data.at[i,'y'] - ys_
        distance = np.sqrt(dx**2 + dy**2)
        
        weight = weight_cal(distance, sigma)
        map.acc_weight += weight
        map.acc_reading += weight*data.at[i,'sensor_value']  

    scaling_param = weight_cal(0, sigma)
    exp_term = (map.acc_weight/scaling_param)**2
    map.confidence = 1 - np.exp(-1*exp_term)

    mean_map = get_mean_map(data, map)
    variacne_map = get_variance_map(data, map, mean_map, sigma)

    return mean_map, variacne_map