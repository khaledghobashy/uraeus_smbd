# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:17:44 2019

@author: khaled.ghobashy
"""

from source.interfaces.scripting_interfaces import assembly

import use_cases.parametric_templates.templates.double_wishbone_direct_acting as dwb
import use_cases.parametric_templates.templates.parallel_link_steering as steer
import use_cases.parametric_templates.templates.front_axle_testrig as testrig

assembled = assembly(__file__)

assembled.add_subsystem('SU', dwb.template)
assembled.add_subsystem('ST', steer.template)
assembled.add_subsystem('TR', testrig.template)

assembled.assign_virtual_body('SU.vbr_steer', 'ST.rbr_rocker')
assembled.assign_virtual_body('TR.vbr_upright', 'SU.rbr_upright')
assembled.assign_virtual_body('TR.vbr_hub', 'SU.rbr_hub')
assembled.assign_virtual_body('TR.vbs_steer_gear', 'ST.rbl_rocker')


assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

assembled.write_python_code()

