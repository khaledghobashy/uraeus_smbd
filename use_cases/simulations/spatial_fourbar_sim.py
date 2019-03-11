# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:47:07 2019

@author: khaled.ghobashy
"""

import numpy as np
import matplotlib.pyplot as plt

import use_cases.generated_templates.assemblies.spatial_fourbar_assm as f
from source.solvers.python_solver import solver

f.FB.config.load_from_csv('spatial_fourbar_mod.csv')

f.FB.config.AF_jcs_rev_crank = lambda t: -np.pi*t #-np.deg2rad(360)*t

assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)
soln = solver(assm)

time_array = np.linspace(0, 2.5, 250)
soln.solve_kds(time_array)
soln.pos_dataframe.to_csv('spatial_fourbar_temp.csv', index=True)


plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.pos_dataframe['FB.rbs_coupler.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.pos_dataframe['FB.rbs_rocker.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.vel_dataframe['FB.rbs_rocker.z'])
plt.legend()
plt.grid()
plt.show()