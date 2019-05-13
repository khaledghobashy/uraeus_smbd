#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:16:46 2019

@author: khaledghobashy
"""

from smbd.utilities.numerics.euler_parameters import A, B, E, G, dcm2ep
from smbd.utilities.numerics.spatial_alg import (skew_matrix, triad, mirrored,
                                                  oriented, centered)
from smbd.utilities.numerics.geometries import (cylinder_geometry, 
                                                 composite_geometry,
                                                 triangular_prism)
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

###############################################################################

class lookup_table(object):
    
    def __init__(self, name):
        self.name = name
        
    def read_csv(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None, names=['x', 'y'])
        self.x = self.data.x
        self.y = self.data.y
    
    def draw(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.x, self.y)
        plt.legend()
        plt.grid()
        plt.show()
    
    def get_interpolator(self):
        self.interp = f = interp1d(self.x, self.y, kind='cubic')
        return f
        
###############################################################################
###############################################################################
