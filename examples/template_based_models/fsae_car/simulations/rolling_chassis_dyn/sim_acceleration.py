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

def drive_torque(P_hub, factor):
    local_torque = np.array([[0],
                             [-factor*(75*9.81)*1e6*TR],
                             [0]])
    global_torque = A(P_hub).dot(local_torque)
    return global_torque


def FR_Torque():
    return 0

def FL_Torque():
    return 0

def RR_Torque():
    factor = 1.2 # if num_model.topology.t <= 4 else 0.8
    return drive_torque(num_model.Subsystems.AX2.P_rbr_hub, factor)

def RL_Torque():
    factor = 1.2 # if num_model.topology.t <= 4 else 0.8
    return drive_torque(num_model.Subsystems.AX2.P_rbl_hub, factor)


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

equlibrium_results = pd.read_csv('results/equlibrium.csv', index_col=0)
q0 = np.array(equlibrium_results.iloc[-1][:-1]).reshape((num_model.topology.n, 1))

sim = simulation('sim', num_model, 'dds')

sim.soln.set_initial_states(q0, 0*q0)
sim.set_time_array(5, dt)
sim.solve()

sim.save_results('results', 'acc_2')


# =============================================================================
#                       Plotting Simulation Results
# =============================================================================

import matplotlib.pyplot as plt

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.z', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.vel_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)
sim.soln.acc_dataframe.plot(x='time', y='CH.rbs_chassis.x', grid=True)

sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e0', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e1', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e2', grid=True)
sim.soln.pos_dataframe.plot(x='time', y='CH.rbs_chassis.e3', grid=True)

sim.soln.pos_dataframe.plot(x='time', 
                            y=['AX1.rbr_hub.x', 'AX1.rbl_hub.x',
                               'AX2.rbr_hub.x', 'AX2.rbl_hub.x'], 
                            grid=True)

sim.soln.vel_dataframe.plot(x='time', 
                            y=['AX1.rbr_hub.x', 'AX1.rbl_hub.x',
                               'AX2.rbr_hub.x', 'AX2.rbl_hub.x'], 
                            grid=True)

sim.soln.pos_dataframe.plot(x='time', 
                            y=['AX2.rbr_hub.e0', 'AX2.rbr_hub.e1',
                               'AX2.rbr_hub.e2', 'AX2.rbr_hub.e3'], 
                            grid=True)

plt.show()
