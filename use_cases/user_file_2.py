# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:54:31 2019

@author: khaled.ghobashy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:59:39 2019

@author: khale
"""

from front_axle_1 import inputs, numerical_assembly
from source.solvers.python_solver import solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


config = inputs()
#config.F_jcr_rocker_ch = lambda t : np.deg2rad(90)*np.sin(5*t)+np.deg2rad(45)
config.F_jcr_rocker_ch = lambda t : 0#np.deg2rad(30) + np.deg2rad(30)*np.sin(2*t)
config.F_mcl_zact = lambda t : 170*np.sin(2*t)
config.F_mcr_zact = lambda t : 170*np.sin(2*t)
config.F_jcl_hub_bearing = config.F_jcr_hub_bearing = lambda t : 0

config.eval_constants()
assembled = numerical_assembly(config)

def solve_system(time_array):
    try:
        soln = solver(assembled)
        soln.solve_kds(time_array)
        return soln
    except np.linalg.LinAlgError:
        return soln

time_array = np.arange(0,3.14,0.1)
soln = solve_system(time_array)

pos_history = pd.DataFrame(np.concatenate(list(soln.pos_history.values()),1).T,index=soln.pos_history.keys(),columns=range(126))
shape = pos_history.shape
time_array_mod = time_array[:shape[0]-1]

plt.figure(figsize=(10,6))
plt.plot(time_array_mod,pos_history[7][:-1])
plt.plot(time_array_mod,pos_history[8][:-1])
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
#plt.plot(pos_history[14][:-1],pos_history[15][:-1])
plt.plot(time_array_mod,pos_history[14][:-1])
plt.plot(time_array_mod,pos_history[15][:-1])
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
#plt.plot(pos_history[21][:-1],pos_history[22][:-1])
plt.plot(pos_history[93][:-1],pos_history[92][:-1])
#plt.plot(time_array_mod,pos_history[22][:-1])
plt.grid()
plt.show()

