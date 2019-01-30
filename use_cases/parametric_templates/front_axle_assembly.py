# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:17:44 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.double_wishbone_direct_acting as dwb
import use_cases.parametric_templates.parallel_link_steering as steer
import use_cases.parametric_templates.front_axle_testrig as testrig


SU1_sym = subsystem('SU1',dwb.template)
ST_sym  = subsystem('ST',steer.template)
TR_sym  = subsystem('TR',testrig.template)


assembled = assembly('front_axle')
assembled.add_subsystem(SU1_sym)
assembled.add_subsystem(ST_sym)
assembled.add_subsystem(TR_sym)

assembled.assign_virtual_body('SU1.vbr_steer','ST.rbr_rocker')
assembled.assign_virtual_body('TR.vbr_upright','SU1.rbr_upright')
assembled.assign_virtual_body('TR.vbr_hub','SU1.rbr_hub')
assembled.assign_virtual_body('TR.vbs_steer_gear','ST.rbl_rocker')


assembled.assemble_model()
assembled.draw_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()

