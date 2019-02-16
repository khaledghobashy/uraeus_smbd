# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

import use_cases.generated_templates.assemblies.suspension_assm as f
from source.solvers.python_solver import solver

f.SU.config.load_from_csv('dwb_bc_points_mod.csv')
f.TR.config.load_from_csv('sus_test_rig_base_cfg_mod.csv')

f.TR.config.AF_mcr_ver_act = lambda t : 25.4*np.sin(t)
f.TR.config.AF_mcl_ver_act = lambda t : 25.4*np.sin(t)
f.TR.config.AF_jcr_rev = lambda t : 0*np.deg2rad(360)*t

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
plt.plot(vertical_travel,soln.pos_dataframe['SU.rbr_hub.y'])
plt.grid()
plt.show()

