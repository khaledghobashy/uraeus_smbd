# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:37:05 2019

@author: khaled.ghobashy
"""

import numpy as np
import matplotlib.pyplot as plt

from source.solvers.python_solver import solver

import use_cases.generated_templates.assemblies.pendulum_assm as assm
from use_cases.generated_templates.configurations import pendulum_cfg

assm.PD.config = pendulum_cfg.configuration()
assm.PD.config.load_from_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\csv_files\pendulum_cfg_mod.csv')

assm.PD.config.AF_jcs_rev_crank = lambda t: -2*np.pi*t

assembled = assm.numerical_assembly()
soln = solver(assembled)

soln.set_time_array(1, 100)
soln.solve_kds('pendulum_temp', save=True)

time_array = soln.time_array

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.pos_dataframe['PD.rbs_crank.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.vel_dataframe['PD.rbs_crank.z'])
plt.plot(time_array, np.gradient(soln.pos_dataframe['PD.rbs_crank.z'],soln.step_size))
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(time_array, soln.acc_dataframe['PD.rbs_crank.z'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['PD.rbs_crank.z'],soln.step_size))
plt.legend()
plt.grid()
plt.show()
