# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

from source.solvers.python_solver import solver
from source.solvers.py_numerical_functions import lookup_table

import use_cases.generated_templates.assemblies.front_axle as f

from use_cases.generated_templates.configurations import dwb_simple_points
from use_cases.generated_templates.configurations import steer_simple_points
from use_cases.generated_templates.configurations import frontaxle_testrig

f.SU.config = dwb_simple_points.configuration()
f.ST.config = steer_simple_points.configuration()
f.TR.config = frontaxle_testrig.configuration()


f.SU.config.load_from_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\csv_files\dwb_st500_axletech.csv')
f.ST.config.load_from_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\csv_files\steer_st500_axletech.csv')
f.TR.config.load_from_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\csv_files\frontaxle_testrig_mod.csv')

f.TR.config.AF_jcs_steer_gear = lambda t : 0*np.deg2rad(15)*np.sin(t)
f.TR.config.AF_mcr_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_mcl_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_jcr_rev = lambda t :  0*np.deg2rad(360)*t
f.TR.config.AF_jcl_rev = lambda t : 0*-np.deg2rad(360)*t

damping_data = lookup_table('damping')
damping_data.read_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Data\ST500_Damping.csv')
damping_func = damping_data.get_interpolator()
def damping_func_mod(x):
    return damping_func(x)*1e6
f.SU.config.Fd_fal_strut = damping_func_mod
f.SU.config.Fd_far_strut = damping_func_mod


stiffness_data = lookup_table('stiffness')
stiffness_data.read_csv(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Data\ST500_GasSpring_IsoThermal.csv')
stiffness_func = stiffness_data.get_interpolator()
def stiffness_func_mod(x):
    x = (x if x>=0 else 0)
    return stiffness_func(x)*1e6
f.SU.config.Fs_fal_strut = stiffness_func_mod
f.SU.config.Fs_far_strut = stiffness_func_mod


#f.SU.config.far_strut_FL = 794
#f.SU.config.fal_strut_FL = 794

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

plt.figure(figsize=(8,4))
plt.plot(vertical_travel, soln.acc_dataframe['SU.rbl_hub.y'])
plt.plot(vertical_travel, np.gradient(soln.vel_dataframe['SU.rbl_lower_strut.y'],soln.step_size))
plt.legend()
plt.grid()
plt.show()


# System Accelerations
plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbl_hub.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbl_hub.y'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['SU.rbl_lower_strut.y'],soln.step_size))
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array, soln.acc_dataframe['SU.rbr_hub.y'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['SU.rbr_lower_strut.y'],soln.step_size))
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(8,4))
plt.plot(time_array,soln.pos_dataframe['SU.rbl_lower_strut.z'])
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(8,4))
plt.plot(time_array,soln.vel_dataframe['SU.rbl_lower_strut.z'])
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(8,4))
plt.plot(time_array,soln.acc_dataframe['SU.rbl_lower_strut.z'])
plt.plot(time_array, np.gradient(soln.vel_dataframe['SU.rbl_lower_strut.z'],soln.step_size))
plt.legend()
plt.grid()
plt.show()



# Solving For Joints Reactions
#soln.eval_reactions()
#
#plt.figure(figsize=(8,4))
#plt.plot(time_array, soln.inr_dataframe['SU.rbl_hub.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#
#plt.figure(figsize=(8,4))
#plt.plot(vertical_travel, soln.reactions_dataframe['TR.F_vbr_hub_mcr_ver_act.x'])
#plt.plot(vertical_travel, soln.reactions_dataframe['TR.F_vbr_hub_mcr_ver_act.y'])
#plt.plot(vertical_travel, soln.reactions_dataframe['TR.F_vbr_hub_mcr_ver_act.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#
#plt.figure(figsize=(8,4))
#plt.plot(time_array,soln.con_dataframe['SU.rbl_lower_strut.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#
#plt.figure(figsize=(8,4))
#plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbr_upper_strut_jcr_strut_chassis.x'])
#plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbr_upper_strut_jcr_strut_chassis.y'])
#plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbr_upper_strut_jcr_strut_chassis.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#plt.figure(figsize=(8,4))
#plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbl_upper_strut_jcl_strut_chassis.x'])
#plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbl_upper_strut_jcl_strut_chassis.y'])
#plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbl_upper_strut_jcl_strut_chassis.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#
#plt.figure(figsize=(8,4))
#plt.plot(time_array,soln.reactions_dataframe['SU.F_rbr_lower_strut_jcr_strut_lca.x'])
#plt.plot(time_array,soln.reactions_dataframe['SU.F_rbr_lower_strut_jcr_strut_lca.y'])
#plt.plot(time_array,soln.reactions_dataframe['SU.F_rbr_lower_strut_jcr_strut_lca.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#plt.figure(figsize=(8,4))
#plt.plot(time_array,soln.reactions_dataframe['SU.F_rbl_lower_strut_jcl_strut_lca.x'])
#plt.plot(time_array,soln.reactions_dataframe['SU.F_rbl_lower_strut_jcl_strut_lca.y'])
#plt.plot(time_array,soln.reactions_dataframe['SU.F_rbl_lower_strut_jcl_strut_lca.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
#
#plt.figure(figsize=(8,4))
#plt.plot(time_array,soln.reactions_dataframe['TR.F_vbl_hub_mcl_ver_act.x'])
#plt.plot(time_array,soln.reactions_dataframe['TR.F_vbl_hub_mcl_ver_act.y'])
#plt.plot(time_array,soln.reactions_dataframe['TR.F_vbl_hub_mcl_ver_act.z'])
#plt.legend()
#plt.grid()
#plt.show()
#
