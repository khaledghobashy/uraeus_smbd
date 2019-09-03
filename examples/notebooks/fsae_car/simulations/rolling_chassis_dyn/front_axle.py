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

from fsae_car.numenv.python.templates.dwb_bc_pushrod import dwb_bc_pushrod_cfg


AX1_config = dwb_bc_pushrod_cfg.configuration()

config_dir  = os.path.join(database_directory, 'fsae_car', 'symenv', 'templates', 'config_inputs')
config_file = os.path.join(config_dir, 'dwb_bc_pushrod_cfg.csv')

AX1_df = pd.read_csv(config_file, index_col=0)


# Tire Radius
TR = 254


# Upper Control Arms
AX1_df.loc['hpr_ucaf'] = [-235, 213, 89 + TR, 0]
AX1_df.loc['hpr_ucar'] = [ 170, 262, 61 + TR, 0]
AX1_df.loc['hpr_ucao'] = [   7, 466, 80 + TR, 0]

# Lower Control Arms
AX1_df.loc['hpr_lcaf'] = [-235, 213, -90 + TR, 0]
AX1_df.loc['hpr_lcar'] = [ 170, 262, -62 + TR, 0]
AX1_df.loc['hpr_lcao'] = [  -7, 483, -80 + TR, 0]

# Tie-Rod
AX1_df.loc['hpr_tri'] = [-122, 227, -122 + TR, 0]
AX1_df.loc['hpr_tro'] = [-122, 456, -132 + TR, 0]

# Push-Rod
AX1_df.loc['hpr_pushrod_rocker'] = [ 6.6, 347, 341 + TR, 0]
AX1_df.loc['hpr_pushrod_uca'] = [ 6.5, 412, 106 + TR, 0]

# Bell-Crank
AX1_df.loc['hpr_rocker_chassis'] = [ 6.5, 280, 320 + TR, 0]

# Struts
AX1_df.loc['hpr_strut_chassis'] = [ 6.5, 22.5, 377 + TR, 0]
AX1_df.loc['hpr_strut_rocker']  = [ 6.5, 263, 399 + TR, 0]
AX1_df.loc['far_strut_FL'] = [ 255, 0, 0, 0]


# Wheel-Hub
AX1_df.loc['hpr_wc']  = [0, 575, 0 + TR, 0]
AX1_df.loc['hpr_wc1'] = [0, 600, 0 + TR, 0]
AX1_df.loc['hpr_wc2'] = [0, 550, 0 + TR, 0]
AX1_df.loc['pt1_far_tire'] = AX1_df.loc['hpr_wc']


AX1_df.loc['vcs_x'] = [1, 0, 0, 0]
AX1_df.loc['vcs_y'] = [0, 1, 0, 0]
AX1_df.loc['vcs_z'] = [0, 0, 1, 0]

# Geometry Scalars
AX1_df.loc['s_tire_radius'] = [TR, 0, 0, 0]
AX1_df.loc['s_links_ro']    = [8, 0, 0, 0]
AX1_df.loc['s_strut_inner'] = [15, 0, 0, 0]
AX1_df.loc['s_strut_outer'] = [22, 0, 0, 0]
AX1_df.loc['s_thickness']   = [8, 0, 0, 0]

AX1_df.to_csv('config_inputs/front_axle.csv', index=True)

# Loading data into the configuration instance
AX1_config.load_from_dataframe(AX1_df)


# Overriding some configuration data
# ==================================

wheel_inertia =  np.array([[1*1e4, 0,      0 ],
                           [0    , 1*1e9, 0 ],
                           [0    , 0, 1*1e4  ]])


AX1_config.Jbar_rbr_hub = wheel_inertia
AX1_config.Jbar_rbl_hub = wheel_inertia

AX1_config.m_rbr_hub = 11*1e3
AX1_config.m_rbl_hub = 11*1e3


# =====================================================
#                   Coil-Over Data
# =====================================================

def strut_spring(x):
    k = 75*1e6
    force = k * x if x >0 else 0
    return force

def strut_damping(v):
    force = 3*1e6 * v
    return force


AX1_config.UF_far_strut_Fs = strut_spring
AX1_config.UF_fal_strut_Fs = strut_spring
AX1_config.UF_far_strut_Fd = strut_damping
AX1_config.UF_fal_strut_Fd = strut_damping


