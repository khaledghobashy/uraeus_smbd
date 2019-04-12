# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

from source import pkg_path
from source.solvers.python_solver import solver

import use_cases.generated_templates.assemblies.front_axle_assembly as f

from use_cases.generated_templates.configurations import dwb_cfg
from use_cases.generated_templates.configurations import steer_cfg
from use_cases.generated_templates.configurations import front_axle_testrig_bcfg

f.SU.config = dwb_cfg.configuration()
f.ST.config = steer_cfg.configuration()
f.TR.config = front_axle_testrig_bcfg.configuration()


f.SU.config.load_from_csv(pkg_path + r'\use_cases\generated_templates\configurations\csv_files\dwb_st500_axletech.csv')
f.ST.config.load_from_csv(pkg_path + r'\use_cases\generated_templates\configurations\csv_files\steer_st500_axletech.csv')
f.TR.config.load_from_csv(pkg_path + r'\use_cases\generated_templates\configurations\csv_files\front_axle_testrig_bcfg_1.csv')

f.TR.config.AF_jcs_steer_gear = lambda t : 0*np.deg2rad(15)*np.sin(t)
f.TR.config.AF_mcr_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_mcl_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_jcr_rev = lambda t :  0*np.deg2rad(360)*t
f.TR.config.AF_jcl_rev = lambda t : 0*-np.deg2rad(360)*t

assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)

soln = solver(assm)
soln.set_time_array(2*np.pi,100)
soln.solve_kds('sim_dwb_st500_axletech_temp', save=True)
time_array = soln.time_array

vertical_travel = np.array(list(map(f.TR.config.AF_mcr_ver_act, time_array)))

plt.figure(figsize=(8,4))
plt.plot(vertical_travel, soln.pos_dataframe['SU.rbl_hub.y'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(vertical_travel, soln.pos_dataframe['SU.rbl_hub.z'])
plt.plot(vertical_travel, vertical_travel)
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(time_array, soln.pos_dataframe['ST.rbr_rocker.y'])
plt.grid()
plt.show()


# System Accelerations
plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbl_hub.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbl_lower_strut.y'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['SU.rbl_lower_strut.y'],soln.step_size))
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbr_hub.y'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['SU.rbr_hub.y'],soln.step_size))
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(time_array, soln.pos_dataframe['SU.rbl_lower_strut.z'])
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(8,4))
plt.plot(time_array, soln.vel_dataframe['SU.rbl_lower_strut.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbl_lower_strut.z'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['SU.rbl_lower_strut.z'],soln.step_size))
plt.legend()
plt.grid()
plt.show()

