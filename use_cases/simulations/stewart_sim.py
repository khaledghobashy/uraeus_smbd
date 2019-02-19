# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

import use_cases.generated_templates.assemblies.stewart_assm as f
from source.solvers.python_solver import solver

f.SG.config.load_from_csv('stewart_mod_points_v3.csv')
f.TR.config.load_from_csv('stewart_testrig_base_cfg_v1.csv')

f.TR.config.AF_jcs_rev_1 = lambda t : np.deg2rad(360)*t
f.TR.config.AF_jcs_rev_2 = lambda t : np.deg2rad(-360)*t
f.TR.config.AF_jcs_rev_3 = lambda t : np.deg2rad(360)*t

assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)
soln = solver(assm)

time_array = np.linspace(0,2,250)
soln.solve_kds(time_array)


plt.figure(figsize=(8,4))
plt.plot(time_array,soln.pos_dataframe['SG.rbs_table.z'])
plt.grid()
plt.show()

