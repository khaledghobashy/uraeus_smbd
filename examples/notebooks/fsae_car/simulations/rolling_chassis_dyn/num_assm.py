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


from smbd.numenv.python.numerics.systems import multibody_system, simulation
from smbd.utilities.numerics.euler_parameters import A

from fsae_car.numenv.python.assemblies import rolling_chassis as assembly

num_model = multibody_system(assembly)


# ==============================================================
#                  Assigning Configuration Data
# ==============================================================

from front_axle import AX1_config
from rear_axle import AX2_config
from steering import ST1_config
from chassis import CH_config

num_model.Subsystems.AX1.config = AX1_config
num_model.Subsystems.AX2.config = AX2_config
num_model.Subsystems.ST1.config = ST1_config
num_model.Subsystems.CH.config  = CH_config


# ==============================================================
#                  Assigning Vehicle Tire Models
# ==============================================================

from smbd.numenv.python.numerics.utilities.vehicle_dynamics.tire_models import \
brush_model

tire_model = brush_model

dt = 1e-2

TR = 254

# ====================================================
#               Front Right Tire
# ====================================================

front_right_tire = tire_model(AX1_config.P_rbr_upright)

front_right_tire.mu = 1.3

front_right_tire.nominal_radius = TR
front_right_tire.kz = 250*1e6
front_right_tire.cz = 0.3*1e6

front_right_tire.a  = 100
front_right_tire.Iyy = 1*1e9

front_right_tire.cp   = 1500*1e3
front_right_tire.C_Fk = 400*1e9
front_right_tire.C_Fa = 400*1e9


def fr_tire_force():
    print('FR_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX1.R_rbr_hub
    P  = num_model.Subsystems.AX1.P_rbr_hub
    Rd = num_model.Subsystems.AX1.Rd_rbr_hub
    Pd = num_model.Subsystems.AX1.Pd_rbr_hub
    
    drive_torque = np.linalg.norm(AX1_config.UF_far_drive_T())
    if drive_torque == 0:
        front_right_tire.driven = 0

    front_right_tire.Eval_Forces(t, dt, R, P, Rd, Pd, drive_torque)
    force = front_right_tire.F
    torque_ratio = drive_torque / front_right_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force



# ====================================================
#               Front Left Tire
# ====================================================

front_left_tire = tire_model(AX1_config.P_rbl_upright)

front_left_tire.mu = 1.3

front_left_tire.nominal_radius = TR
front_left_tire.kz = 250*1e6
front_left_tire.cz = 0.3*1e6

front_left_tire.a  = 100
front_left_tire.Iyy = 1*1e9

front_left_tire.cp   = 1500*1e3
front_left_tire.C_Fk = 400*1e9
front_left_tire.C_Fa = 400*1e9


def fl_tire_force():
    print('FL_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX1.R_rbl_hub
    P  = num_model.Subsystems.AX1.P_rbl_hub
    Rd = num_model.Subsystems.AX1.Rd_rbl_hub
    Pd = num_model.Subsystems.AX1.Pd_rbl_hub
    
    drive_torque = np.linalg.norm(AX1_config.UF_fal_drive_T())
    if drive_torque == 0:
        front_left_tire.driven = 0

    front_left_tire.Eval_Forces(t, dt, R, P, Rd, Pd, drive_torque)
    force = front_left_tire.F
    torque_ratio = drive_torque / front_left_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force


# ====================================================
#               Rear Right Tire
# ====================================================

rear_right_tire = tire_model(AX2_config.P_rbr_upright)

rear_right_tire.mu = 1.3

rear_right_tire.nominal_radius = TR
rear_right_tire.kz = 250*1e6
rear_right_tire.cz = 0.3*1e6

rear_right_tire.a  = 100
rear_right_tire.Iyy = 1*1e9

rear_right_tire.cp   = 1500*1e3
rear_right_tire.C_Fk = 400*1e9
rear_right_tire.C_Fa = 400*1e9


def rr_tire_force():
    print('RR_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX2.R_rbr_hub
    P  = num_model.Subsystems.AX2.P_rbr_hub
    Rd = num_model.Subsystems.AX2.Rd_rbr_hub
    Pd = num_model.Subsystems.AX2.Pd_rbr_hub
    
    drive_torque = np.linalg.norm(AX2_config.UF_far_drive_T())
    if drive_torque == 0:
        rear_right_tire.driven = 0

    rear_right_tire.Eval_Forces(t, dt, R, P, Rd, Pd, drive_torque)
    force = rear_right_tire.F
    torque_ratio = drive_torque / rear_right_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force

# ====================================================
#               Rear Left Tire
# ====================================================

rear_left_tire = tire_model(AX2_config.P_rbl_upright)

rear_left_tire.mu = 1.3

rear_left_tire.nominal_radius = TR
rear_left_tire.kz = 250*1e6
rear_left_tire.cz = 0.3*1e6

rear_left_tire.a  = 100
rear_left_tire.Iyy = 1*1e9

rear_left_tire.cp   = 1500*1e3
rear_left_tire.C_Fk = 400*1e9
rear_left_tire.C_Fa = 400*1e9


def rl_tire_force():
    print('RL_Tire : \n=========')
    t  = num_model.topology.t
    R  = num_model.Subsystems.AX2.R_rbl_hub
    P  = num_model.Subsystems.AX2.P_rbl_hub
    Rd = num_model.Subsystems.AX2.Rd_rbl_hub
    Pd = num_model.Subsystems.AX2.Pd_rbl_hub
    
    drive_torque = np.linalg.norm(AX2_config.UF_fal_drive_T())
    if drive_torque == 0:
        rear_left_tire.driven = 0

    rear_left_tire.Eval_Forces(t, dt, R, P, Rd, Pd, drive_torque)
    force = rear_left_tire.F
    torque_ratio = drive_torque / rear_left_tire.My
    
    print('Drive Torque = %s'%drive_torque)
    print('Wheel Torque = %s'%torque_ratio)
    print('\n\n')
    return force

# =========================================================
# =========================================================

AX1_config.UF_far_tire_F = fr_tire_force
AX1_config.UF_far_tire_T = lambda : front_right_tire.M

AX1_config.UF_fal_tire_F = fl_tire_force
AX1_config.UF_fal_tire_T = lambda : front_left_tire.M

AX2_config.UF_far_tire_F = rr_tire_force
AX2_config.UF_far_tire_T = lambda : rear_right_tire.M

AX2_config.UF_fal_tire_F = rl_tire_force
AX2_config.UF_fal_tire_T = lambda : rear_left_tire.M

