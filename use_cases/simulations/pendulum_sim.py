# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:37:05 2019

@author: khaled.ghobashy
"""

import numpy as np
import matplotlib.pyplot as plt

from source.solvers.python_solver import solver, dynamic_solver

import use_cases.generated_templates.assemblies.pendulum_assm as assm
from use_cases.generated_templates.configurations import pendulum_cfg

assm.PD.config = pendulum_cfg.configuration()

#asurt_path = r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic'
asurt_path = r'E:\Main\asurt_cdt_symbolic'

assm.PD.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\pendulum_cfg_mod_dyn.csv')

#assm.PD.config.AF_jcs_rev_crank = lambda t: -2*np.pi*t
#assm.PD.config.AF_mcs_act = lambda t: 0


assembled = assm.numerical_assembly()

dynamic_soln = dynamic_solver(assembled)
dynamic_soln.set_time_array(2,1000)
time_array = dynamic_soln.time_array

try:
    dynamic_soln.solve_dds('pendulum_temp_dyn', save=True)
except np.linalg.LinAlgError:
    dynamic_soln._creat_results_dataframes()
    dynamic_soln.save_results('pendulum_temp_dyn')
    time_array = time_array[:len(dynamic_soln.acc_dataframe)]


plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.pos_dataframe['PD.rbs_crank.z'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.vel_dataframe['PD.rbs_crank.z'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.acc_dataframe['PD.rbs_crank.z'])
plt.grid()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.pos_dataframe['PD.rbs_crank.y'])
plt.grid()
plt.show()



#soln = solver(assembled)
#soln.set_time_array(5, 500)
#soln.solve_kds('pendulum_temp', save=True)
#
#time_array = soln.time_array
#
#plt.figure(figsize=(8, 4))
#plt.plot(time_array, soln.pos_dataframe['PD.rbs_crank.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#plt.figure(figsize=(8, 4))
#plt.plot(time_array, soln.vel_dataframe['PD.rbs_crank.z'])
#plt.plot(time_array, np.gradient(soln.pos_dataframe['PD.rbs_crank.z'],soln.step_size))
#plt.legend()
#plt.grid()
#plt.show()
#
#plt.figure(figsize=(8, 4))
#plt.plot(time_array, soln.acc_dataframe['PD.rbs_crank.z'])
#plt.plot(time_array, np.gradient(soln.vel_dataframe['PD.rbs_crank.z'],soln.step_size))
#plt.legend()
#plt.grid()
#plt.show()
