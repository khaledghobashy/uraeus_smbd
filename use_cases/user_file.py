# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:59:39 2019

@author: khale
"""

from assm4 import inputs, numerical_assembly
from source.solvers.python_solver import solver
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


config = inputs()
config.F_jcr_rocker_ch = lambda t : np.deg2rad(60)*np.sin(t)
config.F_jcr_rocker_ch = lambda t : np.deg2rad(330)*t
config.eval_constants()
assembled = numerical_assembly(config)

def solve_system(time_array):
    try:
        soln = solver(assembled)
        soln.solve_kds(time_array)
        return soln
    except np.linalg.LinAlgError:
        return soln

time_array = np.arange(0,5,0.005)
soln = solve_system(time_array)

pos_history = pd.DataFrame(np.concatenate(list(soln.pos_history.values()),1).T,index=soln.pos_history.keys(),columns=range(28))
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
plt.plot(time_array_mod,pos_history[21][:-1])
plt.plot(time_array_mod,pos_history[22][:-1])
plt.grid()
plt.show()

