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


from fsae_car.numenv.python.templates.chassis import chassis_cfg


CH_config = chassis_cfg.configuration()

config_dir  = os.path.join(database_directory, 'fsae_car', 'symenv', 'templates', 'config_inputs')
config_file = os.path.join(config_dir, 'chassis_cfg.csv')

CH_df = pd.read_csv(config_file, index_col=0)


# Tire Radius
TR = 254

# Wheel Base
WB = 1600


CH_df.loc['hps_CG'] = [WB/2, 0, 300, 0]
CH_df.loc['s_CG_radius'] = [50, 0, 0, 0]
CH_df.loc['m_rbs_chassis'] = [280*1e3, 0, 0, 0]

CH_df.to_csv('config_inputs/chassis.csv', index=True)
# Loading data into the configuration instance
CH_config.load_from_dataframe(CH_df)


# Overriding some configuration data
# ==================================

CH_config.Jbar_rbs_chassis = np.array([[120*1e9 , 0, 0   ],
                                       [0   , 150*1e9 ,0 ],
                                       [0   , 0, 150*1e9 ]])


