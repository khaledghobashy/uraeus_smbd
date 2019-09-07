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


AX2_config = dwb_bc_pushrod_cfg.configuration()

config_dir  = os.path.join(database_directory, 'fsae_car', 'symenv', 'templates', 'config_inputs')
config_file = os.path.join(config_dir, 'dwb_bc_pushrod_cfg.csv')

AX2_df = pd.read_csv(config_file, index_col=0)


# Tire Radius
TR = 254

# Wheel Base
WB = 1600


# Upper Control Arms
AX2_df.loc['hpr_ucaf'] = [ -60 + WB, 250, 52 + TR, 0]
AX2_df.loc['hpr_ucar'] = [-247 + WB, 276, 25 + TR, 0]
AX2_df.loc['hpr_ucao'] = [   0 + WB, 462, 90 + TR, 0]

# Lower Control Arms
AX2_df.loc['hpr_lcaf'] = [ -60 + WB, 250, -84 + TR, 0]
AX2_df.loc['hpr_lcar'] = [-202 + WB, 269, -79 + TR, 0]
AX2_df.loc['hpr_lcao'] = [   0 + WB, 462, -90 + TR, 0]

# Tie-Rod
AX2_df.loc['hpr_tri'] = [-247 + WB, 276, -77 + TR, 0]
AX2_df.loc['hpr_tro'] = [-200 + WB, 462, -90 + TR, 0]

# Push-Rod
AX2_df.loc['hpr_pushrod_rocker'] = [-65 + WB, 336, 207 + TR, 0]
AX2_df.loc['hpr_pushrod_uca']    = [-65 + WB, 381,  90 + TR, 0]

# Bell-Crank
AX2_df.loc['hpr_rocker_chassis'] = [-65 + WB, 289, 190 + TR, 0]

# Struts
AX2_df.loc['hpr_strut_chassis'] = [-65 + WB,  34, 202 + TR, 0]
AX2_df.loc['hpr_strut_rocker']  = [-65 + WB, 270, 241 + TR, 0]
AX2_df.loc['far_strut_FL'] = [ 255, 0, 0, 0]


# Wheel-Hub
AX2_df.loc['hpr_wc']  = [0 + WB, 525, 0 + TR, 0]
AX2_df.loc['hpr_wc1'] = [0 + WB, 550, 0 + TR, 0]
AX2_df.loc['hpr_wc2'] = [0 + WB, 500, 0 + TR, 0]
AX2_df.loc['pt1_far_tire'] = AX2_df.loc['hpr_wc']


AX2_df.loc['vcs_x'] = [1, 0, 0, 0]
AX2_df.loc['vcs_y'] = [0, 1, 0, 0]
AX2_df.loc['vcs_z'] = [0, 0, 1, 0]

# Geometry Scalars
AX2_df.loc['s_tire_radius'] = [TR, 0, 0, 0]
AX2_df.loc['s_links_ro']    = [8, 0, 0, 0]
AX2_df.loc['s_strut_inner'] = [15, 0, 0, 0]
AX2_df.loc['s_strut_outer'] = [22, 0, 0, 0]
AX2_df.loc['s_thickness']   = [8, 0, 0, 0]

AX2_df.to_csv('config_inputs/rear_axle.csv', index=True)

# Loading data into the configuration instance
AX2_config.load_from_dataframe(AX2_df)


# Overriding some configuration data
# ==================================

wheel_inertia =  np.array([[1*1e4, 0,      0 ],
                           [0    , 1*1e9, 0 ],
                           [0    , 0, 1*1e4  ]])


AX2_config.Jbar_rbr_hub = wheel_inertia
AX2_config.Jbar_rbl_hub = wheel_inertia

AX2_config.m_rbr_hub = 11*1e3
AX2_config.m_rbl_hub = 11*1e3


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


AX2_config.UF_far_strut_Fs = strut_spring
AX2_config.UF_fal_strut_Fs = strut_spring
AX2_config.UF_far_strut_Fd = strut_damping
AX2_config.UF_fal_strut_Fd = strut_damping


