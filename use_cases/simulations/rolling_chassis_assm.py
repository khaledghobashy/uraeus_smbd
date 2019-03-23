# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:35:54 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

from source.solvers.python_solver import solver, dynamic_solver
from source.solvers.py_numerical_functions import lookup_table

import use_cases.generated_templates.assemblies.rolling_chassis as model

from use_cases.generated_templates.configurations import chassis_bcfg
from use_cases.generated_templates.configurations import dwb_da_cfg
from use_cases.generated_templates.configurations import steer_simple_points
from use_cases.generated_templates.configurations import front_axle_testrig_bcfg
from use_cases.generated_templates.configurations import rolling_chassis_trg_bcfg


model.SU1.config = dwb_da_cfg.configuration()
model.TR1.config = front_axle_testrig_bcfg.configuration()

model.SU2.config = dwb_da_cfg.configuration()
model.TR2.config = rolling_chassis_trg_bcfg.configuration()

model.ST.config = steer_simple_points.configuration()
model.CH.config = chassis_bcfg.configuration()

#asurt_path = r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic'
asurt_path = r'E:\Main\asurt_cdt_symbolic'

model.SU1.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\dwb_st500_axletech_front.csv')
model.TR1.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\front_axle_testrig_bcfg_1.csv')

model.SU2.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\dwb_st500_axletech_rear.csv')
model.TR2.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\rear_axle_testrig_bcfg_1.csv')

model.ST.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\steer_st500_axletech.csv')
model.CH.config.load_from_csv(asurt_path + r'\use_cases\generated_templates\configurations\csv_files\chassis_bcfg_mod.csv')


model.TR1.config.AF_jcs_steer_gear = lambda t : 0*np.deg2rad(15)*np.sin(t)

model.TR1.config.AF_jcr_rev = lambda t :  0*np.deg2rad(360)*t
model.TR1.config.AF_jcl_rev = lambda t : 0*-np.deg2rad(360)*t
model.TR2.config.AF_jcr_rev = lambda t :  0*np.deg2rad(360)*t
model.TR2.config.AF_jcl_rev = lambda t : 0*-np.deg2rad(360)*t

model.TR1.config.AF_mcr_ver_act = lambda t : 0
model.TR1.config.AF_mcl_ver_act = lambda t : 0
model.TR2.config.AF_mcr_ver_act = lambda t : 0
model.TR2.config.AF_mcl_ver_act = lambda t : 0



model.CH.config.m_rbs_chassis = 8*1e3*1e3
model.CH.config.Jbar_rbs_chassis = 1e24*np.eye(3)

damping_data = lookup_table('damping')
damping_data.read_csv(r'E:\Main\ST500_struts_data\ST500_Damping.csv')
damping_func = damping_data.get_interpolator()
def damping_func_mod(x):
    return damping_func(x)*1e6

stiffness_data = lookup_table('stiffness')
stiffness_data.read_csv(r'E:\Main\ST500_struts_data\ST500_GasSpring_IsoThermal.csv')
stiffness_func = stiffness_data.get_interpolator()
def stiffness_func_mod(x):
    print(x)
    x = (x if x>=0 else 0)
    return stiffness_func(x)*1e6

model.SU1.config.Fd_fal_strut = damping_func_mod
model.SU1.config.Fd_far_strut = damping_func_mod
model.SU2.config.Fd_fal_strut = damping_func_mod
model.SU2.config.Fd_far_strut = damping_func_mod

model.SU1.config.Fs_fal_strut = stiffness_func_mod
model.SU1.config.Fs_far_strut = stiffness_func_mod
model.SU2.config.Fs_fal_strut = stiffness_func_mod
model.SU2.config.Fs_far_strut = stiffness_func_mod

model.SU1.config.far_strut_FL = 794
model.SU1.config.fal_strut_FL = 794
model.SU2.config.far_strut_FL = 794
model.SU2.config.fal_strut_FL = 794


assm = model.numerical_assembly()
assm.set_gen_coordinates(assm.q0)

dynamic_soln = dynamic_solver(assm)
dynamic_soln.set_time_array(20,400)
dynamic_soln.solve_dds('sim_dwb_st500_axletech_temp_dyn', save=True)

#try:
#    dynamic_soln.solve_dds('sim_dwb_st500_axletech_temp_dyn', save=True)
#except:
#    dynamic_soln._creat_results_dataframes()
#    dynamic_soln.save_results('sim_dwb_st500_axletech_temp_dyn')

time_array = dynamic_soln.time_array


plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.pos_dataframe['SU1.rbl_hub.z'])
plt.plot(time_array, dynamic_soln.pos_dataframe['CH.rbs_chassis.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.vel_dataframe['SU1.rbl_hub.z'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.pos_dataframe['SU1.rbr_hub.z'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.vel_dataframe['SU1.rbr_hub.z'])
plt.grid()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(time_array, dynamic_soln.pos_dataframe['SU1.rbr_hub.z'])
plt.grid()
plt.show()

