# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:34:03 2019

@author: khale
"""

import numpy as np
from source.interfaces.scripting_interfaces import simulation

soln = simulation('dwb_st500_axletech', assm, 'kds')
soln.set_time_array(2*np.pi, 100)
soln.solve()
time_array = soln.soln.time_array

vertical_travel = f.TR.config.AF_mcr_ver_act(time_array)

soln.plot(['SU.rbl_hub.y'])
soln.plot(['SU.rbl_hub.z'], vertical_travel)
soln.plot(['ST.rbr_rocker.y'])

soln.plot(['SU.rbl_hub.z'], level='acc')
soln.plot(['SU.rbl_lower_strut.y'], level='acc')


