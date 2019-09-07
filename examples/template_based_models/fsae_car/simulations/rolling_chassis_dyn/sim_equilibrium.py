import sys
import os

import numpy as np
import pandas as pd

pkg_path = '/home/khaledghobashy/Documents/smbd'
sys.path.append(pkg_path)

database_directory = '/home/khaledghobashy/Documents/smbd/examples/notebooks'

try:
    import fsae_car
except ModuleNotFoundError:
    sys.path.append(database_directory)


from smbd.numenv.python.numerics.systems import simulation
from smbd.utilities.numerics.euler_parameters import A

import num_assm
from num_assm import num_model

dt = num_assm.dt

TR = 254

# ==============================================================
#                 Creating Simulation Subroutines
# ==============================================================

def FR_Torque():
    return 0

def FL_Torque():
    return 0

def RR_Torque():
    return 0

def RL_Torque():
    return 0


def steering_function(t):
    
    t_dur = 2
    t_str = 4
    t_end = t_str + t_dur
    
    rack_travel = 0
    amplitude = 20
    if t >= t_str and t <= t_end:
        rack_travel = amplitude*np.sin((2*np.pi/t_dur)*(t-t_str))
    
    return rack_travel


# ==============================================================
#                  Assigning Driving Functions
# ==============================================================

num_assm.AX1_config.UF_far_drive_T = FR_Torque
num_assm.AX1_config.UF_fal_drive_T = FL_Torque
num_assm.AX2_config.UF_far_drive_T = RR_Torque
num_assm.AX2_config.UF_fal_drive_T = RL_Torque

num_assm.AX1_config.UF_far_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX1_config.UF_fal_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_far_drive_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.AX2_config.UF_fal_drive_F = lambda : np.zeros((3,1), dtype=np.float64)

num_assm.ST1_config.UF_mcs_rack_act = lambda t : 0

num_assm.CH_config.UF_fas_aero_drag_F = lambda : np.zeros((3,1), dtype=np.float64)
num_assm.CH_config.UF_fas_aero_drag_T = lambda : np.zeros((3,1), dtype=np.float64)

# ==============================================================
#                   Starting Simulation
# ==============================================================

sim = simulation('sim', num_model, 'dds')

sim.set_time_array(5, dt)
sim.solve()

sim.save_results('results', 'equlibrium')


# =============================================================================
#                       Plotting Simulation Results
# =============================================================================

import matplotlib.pyplot as plt

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e0', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e1', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e2', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e3', grid=True)

plt.show()
