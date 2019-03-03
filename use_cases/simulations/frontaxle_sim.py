# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

import use_cases.generated_templates.assemblies.front_axle as f
from source.solvers.python_solver import solver

f.SU.config.load_from_csv('dwb_st500_axletech_2.csv')
f.ST.config.load_from_csv('steer_st500_axletech_2.csv')
f.TR.config.load_from_csv('test_rig_v1_mod.csv')

f.TR.config.AF_jcs_steer_gear = lambda t : np.deg2rad(15)*np.sin(t)
f.TR.config.AF_mcr_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_mcl_ver_act = lambda t : 0*170*np.sin(t)
f.TR.config.AF_jcr_rev = lambda t : np.deg2rad(360)*t
f.TR.config.AF_jcl_rev = lambda t : -np.deg2rad(360)*t


assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)
soln = solver(assm)

time_array = np.linspace(0,2*np.pi,100)
soln.solve_kds(time_array)

vertical_travel = np.array(list(map(f.TR.config.AF_mcr_ver_act,time_array)))

plt.figure(figsize=(8,4))
plt.plot(vertical_travel,soln.pos_dataframe['SU.rbl_hub.y'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array,soln.pos_dataframe['ST.rbr_rocker.y'])
plt.grid()
plt.show()

soln.pos_dataframe.to_csv('sim_dwb_st500_axletech_temp.csv',index=True)
