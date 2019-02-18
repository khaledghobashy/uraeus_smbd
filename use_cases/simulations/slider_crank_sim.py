# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

import use_cases.generated_templates.assemblies.slider_crank_assm as f
from source.solvers.python_solver import solver

f.SG.config.load_from_csv('slider_points_v1.csv')
f.SG.config.ax2_jcs_uni = f.SG.config.ax2_jcs_uni*-1
f.TR.config.load_from_csv('slider_crank_testrig_base_cfg_v1.csv')

f.TR.config.AF_jcs_rev = lambda t : np.deg2rad(50)*t

assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)
soln = solver(assm)

time_array = np.linspace(0,1,250)
soln.solve_kds(time_array)


plt.figure(figsize=(8,4))
plt.plot(time_array,soln.pos_dataframe['SG.rbs_slider.z'])
plt.grid()
plt.show()

