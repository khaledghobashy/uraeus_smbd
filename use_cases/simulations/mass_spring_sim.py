# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:37:05 2019

@author: khaled.ghobashy
"""

import numpy as np
import matplotlib.pyplot as plt

from source.solvers.python_solver import solver, dynamic_solver

import use_cases.generated_templates.assemblies.mass_spring_assm as assm
from use_cases.generated_templates.configurations import mass_spring_cfg

assm.MS.config = mass_spring_cfg.configuration()

#asurt_path = r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic'
asurt_path = r'E:\Main\asurt_cdt_symbolic'


def stiffness_func(x):
    print(x)
    return 200000*x

def damping_func(x):
    return -5*1e3*x

assm.MS.config.Fs_fas_spring = stiffness_func
assm.MS.config.Fd_fas_spring = damping_func

assm.MS.config.T_rbs_dummy_fas_spring = 0


assm.MS.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\mass_spring_cfg_mod2.csv')



assembled = assm.numerical_assembly()

dynamic_soln = dynamic_solver(assembled)
dynamic_soln.set_time_array(2,2/5e-3)
time_array = dynamic_soln.time_array

try:
    dynamic_soln.solve_dds('mass_spring_temp_dyn', save=True)
except np.linalg.LinAlgError:
    dynamic_soln._creat_results_dataframes()
    dynamic_soln.save_results('mass_spring_temp_dyn')
    time_array = time_array[:len(dynamic_soln.acc_dataframe)]
    raise np.linalg.LinAlgError
    


plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.pos_dataframe['MS.rbs_block.z'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.vel_dataframe['MS.rbs_block.z'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.acc_dataframe['MS.rbs_block.z'])
plt.grid()
plt.show()


