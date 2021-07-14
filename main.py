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

import sys
import pandas as pd 
from mapping import get_maps, nlpd_test
import matplotlib.pyplot as plt
from config import sigma
import numpy as np


"""
Normalize value of all cells in the map. The range is [0,1]
Param matrx: the grid map
"""
def normalize_map(matrix):
    max = np.max(matrix)
    min = np.min(matrix)
    matrix = (matrix - min)/(max-min)
    return matrix


"""
Plot gas distribution maps. One using mean and one using variance 
Param mean_map: grid map in form of numpy.array of mean value
Param variance_map: grid map in form of numpy.array of variance value
"""
def plot_maps(mean_map, variance_map):
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Gas Distribution Mapping')
    
    mean_map     = normalize_map(mean_map)
    variance_map = normalize_map(variance_map)

    axs[0].imshow(mean_map.T, cmap='hot')
    axs[0].set_title('Mean map')
    axs[1].imshow(variance_map.T, cmap='hot')
    axs[1].set_title('Variance map')

    for ax in axs.flat:
        ax.set(xlabel='x (cells)', ylabel='y (cells)')
    plt.show()


"""
Syntax: $ python3 main.py <arg1> <arg2>
arg1: gas data file in csv form (e.g data.csv)
arg2: option to use NLPD test (e.g y or Y)
"""
if __name__ == "__main__":
    try:
        csv_name = sys.argv[1]  # Get csv file name of data in command line
        use_nlpd = sys.argv[2]  # Get option to use NLPD test or not
    except Exception as e:
        print("missing csv file name or option of using nlpd test as arguments")
        sys.exit()

    data = pd.read_csv(csv_name)

    # True: Automaticaly find the best kernel, Else: setup manualy in config.py
    if (use_nlpd == 'y' or use_nlpd == 'Y'):
        sigma = nlpd_test(data)
    
    mean_map, variance_map = get_maps(data, sigma) 
    plot_maps(mean_map, variance_map)  
    print("Done!!")