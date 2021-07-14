#!/usr/bin/env python3

#=============================================================
# This class generate a grid map object in form of numpy.array
#=============================================================

__author__      = "Luong Duc Nhat"
__copyright__   = "Copyright 2021, The Chemical Plume Tracing (CPT) Robot Project"
__credits__     = ["Luong Duc Nhat"]
__license__     = "GPL"
__version__     = "2.0.0"
__maintainer__  = "Luong Duc Nhat"
__email__       = "luong.d.aa@m.titech.ac.jp"
__status__      = "Production"

import config
import numpy as np

class Gridmap:       
    # The init method or constructor
    def __init__(self, x_min, x_max, y_min, y_max):
        self._x_min = x_min - config.extend               
        self._x_max = x_max + config.extend
        self._y_min = y_min - config.extend
        self._y_max = y_max + config.extend
        self._res = config.resolution

        self._n_cells_x = int((x_max - x_min)/self._res) + 1
        self._n_cells_y = int((y_max - y_min)/self._res) + 1

        shape = (self._n_cells_x, self._n_cells_y)
        self.acc_weight  = np.zeros(shape)
        self.acc_reading = np.zeros(shape)
        self.confidence  = np.zeros(shape)
        self.acc_variance = np.zeros(shape)

    """
    Get the coordinate of x and y in form of 2 separate matries
    """
    def get_grid(self):
        xs = np.linspace(self._x_min, self._x_max, self._n_cells_x)
        ys = np.linspace(self._y_min, self._y_max, self._n_cells_y)
        xs_, ys_ = np.meshgrid(xs, ys, indexing='ij')  
        return xs_, ys_