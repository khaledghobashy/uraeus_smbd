# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt
from source.solvers.python_solver import solver

import use_cases.generated_templates.assemblies.stewart_assm as f
from use_cases.generated_templates.configurations import stewart_points
from use_cases.generated_templates.configurations import stewart_testrig_cfg

f.SG.config = stewart_points.configuration()
f.TR.config = stewart_testrig_cfg.configuration()

f.SG.config.load_from_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\csv_files\stewart_points_mod1.csv')
f.TR.config.load_from_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\csv_files\stewart_testrig_cfg_mod.csv')

f.TR.config.AF_jcs_rev_1 = lambda t: np.deg2rad(360)*t
f.TR.config.AF_jcs_rev_2 = lambda t: 0*np.deg2rad(360)*t
f.TR.config.AF_jcs_rev_3 = lambda t: 0*np.deg2rad(360)*t

assm = f.numerical_assembly()
soln = solver(assm)

soln.set_time_array(1, 100)
soln.solve_kds('stewart_sim_data_temp', save=True)

time_array = soln.time_array

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.pos_dataframe['SG.rbs_table.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.pos_dataframe['SG.rbs_table.x'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.vel_dataframe['SG.rbs_table.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.acc_dataframe['SG.rbs_table.z'])
plt.legend()
plt.grid()
plt.show()


# Solving For Joints Reactions
soln.eval_reactions()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.reactions_dataframe['SG.F_vbs_ground_jcs_rev_1.x'])
plt.plot(time_array, soln.reactions_dataframe['SG.F_vbs_ground_jcs_rev_1.y'])
plt.plot(time_array, soln.reactions_dataframe['SG.F_vbs_ground_jcs_rev_1.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.reactions_dataframe['SG.T_vbs_ground_jcs_rev_1.x'])
plt.plot(time_array, soln.reactions_dataframe['SG.T_vbs_ground_jcs_rev_1.y'])
plt.plot(time_array, soln.reactions_dataframe['SG.T_vbs_ground_jcs_rev_1.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_1_jcs_rev_1.x'])
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_1_jcs_rev_1.y'])
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_1_jcs_rev_1.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_2_jcs_rev_2.x'])
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_2_jcs_rev_2.y'])
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_2_jcs_rev_2.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_3_jcs_rev_3.x'])
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_3_jcs_rev_3.y'])
plt.plot(time_array, soln.reactions_dataframe['TR.T_vbs_rocker_3_jcs_rev_3.z'])
plt.legend()
plt.grid()
plt.show()
