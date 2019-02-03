# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:10:55 2019

@author: khale
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import use_cases.generated_templates.front_axle as f
from source.solvers.python_solver import solver

f.SU1.config.load_from_csv('temp_front_axle/dwb_v1_mod.csv')
f.ST.config.load_from_csv('temp_front_axle/steer_v1_mod.csv')
f.TR.config.load_from_csv('temp_front_axle/test_rig_v1_mod.csv')

f.TR.config.F_jcs_steer_gear = lambda t : np.deg2rad(15)*np.sin(t)
f.TR.config.F_mcr_ver_act = lambda t : 170*np.sin(t)
f.TR.config.F_jcr_rev = lambda t : np.deg2rad(360)*t

assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)
soln = solver(assm)

time_array = np.linspace(0,5,150)
soln.solve_kds(time_array)

pos = pd.DataFrame(np.concatenate(list(soln.pos_history.values()),1))

vertical_travel = np.array(list(map(f.TR.config.F_mcr_ver_act,time_array)))
plt.figure(figsize=(8,4))
plt.plot(vertical_travel,pos.T[8])
plt.grid()
plt.show()

