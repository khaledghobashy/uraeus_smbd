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

from fsae_car.numenv.python.templates.steering_rack import steering_rack_cfg


ST1_config = steering_rack_cfg.configuration()

config_dir  = os.path.join(database_directory, 'fsae_car', 'symenv', 'templates', 'config_inputs')
config_file = os.path.join(config_dir, 'steering_rack_cfg.csv')

ST1_df = pd.read_csv(config_file, index_col=0)

# =====================================================
#                   Numerical Data
# =====================================================

TR = 254

ST1_df.loc['hpr_rack_end'] = [-122, 227, -122 + TR, 0]
ST1_df.loc['vcs_y'] = [0, 1, 0, 0]
ST1_df.loc['s_rack_radius'] = [12, 0, 0, 0]

ST1_df.to_csv('config_inputs/steering.csv', index=True)
# Loading data into the configuration instance
ST1_config.load_from_dataframe(ST1_df)



