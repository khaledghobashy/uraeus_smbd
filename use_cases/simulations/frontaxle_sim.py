# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
import matplotlib.pyplot as plt

import use_cases.generated_templates.assemblies.front_axle as f
from source.solvers.python_solver import solver

f.SU.config.load_from_csv('dwb_st500_axletech_2.csv')
f.ST.config.load_from_csv('steer_st500_axletech_2.csv')
f.TR.config.load_from_csv('test_rig_v1_mod.csv')

f.TR.config.AF_jcs_steer_gear = lambda t : 0*np.deg2rad(15)*np.sin(t)
f.TR.config.AF_mcr_ver_act = lambda t : 130*np.sin(t)
f.TR.config.AF_mcl_ver_act = lambda t : 130*np.sin(t)
f.TR.config.AF_jcr_rev = lambda t : 0*np.deg2rad(360)*t
f.TR.config.AF_jcl_rev = lambda t : 0*-np.deg2rad(360)*t

strut_damping   = lambda v : -36*(1e6)*v
strut_stiffness = lambda x : (407*(1e6)*x if x>0 else 0*20*1e3*1e6)

f.SU.config.Fd_fal_strut = strut_damping
f.SU.config.Fd_far_strut = strut_damping
f.SU.config.Fs_fal_strut = strut_stiffness
f.SU.config.Fs_far_strut = strut_stiffness
f.SU.config.far_strut_FL = 794
f.SU.config.fal_strut_FL = 794


assm = f.numerical_assembly()
assm.set_gen_coordinates(assm.q0)
soln = solver(assm)

time_array = np.linspace(0,2*np.pi,200)
soln.solve_kds(time_array)

vertical_travel = np.array(list(map(f.TR.config.AF_mcr_ver_act,time_array)))

plt.figure(figsize=(8,4))
plt.plot(vertical_travel,soln.pos_dataframe['SU.rbl_hub.y'])
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array,soln.pos_dataframe['ST.rbr_rocker.y'])
plt.grid()
plt.show()

soln.pos_dataframe.to_csv('sim_dwb_st500_axletech_temp.csv',index=True)


# Solving For Joints Reactions
soln.eval_reactions_eq()

plt.figure(figsize=(8,4))
plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbr_upper_strut_jcr_strut_chassis.x'])
plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbr_upper_strut_jcr_strut_chassis.y'])
plt.plot(vertical_travel,soln.reactions_dataframe['SU.F_rbr_upper_strut_jcr_strut_chassis.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(time_array,soln.reactions_dataframe['SU.F_rbr_lower_strut_jcr_strut_lca.x'])
plt.plot(time_array,soln.reactions_dataframe['SU.F_rbr_lower_strut_jcr_strut_lca.y'])
plt.plot(time_array,soln.reactions_dataframe['SU.F_rbr_lower_strut_jcr_strut_lca.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(vertical_travel,soln.reactions_dataframe['TR.F_vbl_hub_mcl_ver_act.x'])
plt.plot(vertical_travel,soln.reactions_dataframe['TR.F_vbl_hub_mcl_ver_act.y'])
plt.plot(vertical_travel,soln.reactions_dataframe['TR.F_vbl_hub_mcl_ver_act.z'])
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(vertical_travel,soln.reactions_dataframe['TR.F_vbr_hub_mcr_ver_act.x'])
plt.plot(vertical_travel,soln.reactions_dataframe['TR.F_vbr_hub_mcr_ver_act.y'])
plt.plot(vertical_travel,soln.reactions_dataframe['TR.F_vbr_hub_mcr_ver_act.z'])
plt.legend()
plt.grid()
plt.show()

