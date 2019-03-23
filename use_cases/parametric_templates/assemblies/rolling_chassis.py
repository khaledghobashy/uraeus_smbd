# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:30:20 2019

@author: khale
"""
import os

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.templates.double_wishbone_direct_acting as dwb
import use_cases.parametric_templates.templates.parallel_link_steering as steer
import use_cases.parametric_templates.templates.chassis as chassis
import use_cases.parametric_templates.templates.front_axle_testrig as front_trg
import use_cases.parametric_templates.templates.rolling_chassis_trg as rear_trg

assembly_name = os.path.basename(__file__).split('.')[0]

SU1_sym = subsystem('SU1',dwb.template)
TR1_sym = subsystem('TR1',front_trg.template)

SU2_sym = subsystem('SU2',dwb.template)
TR2_sym = subsystem('TR2',rear_trg.template)

ST_sym = subsystem('ST',steer.template)
CH_sym = subsystem('CH',chassis.template)

assembled = assembly(assembly_name)

assembled.add_subsystem(SU1_sym)
assembled.add_subsystem(TR1_sym)

assembled.add_subsystem(SU2_sym)
assembled.add_subsystem(TR2_sym)

assembled.add_subsystem(ST_sym)
assembled.add_subsystem(CH_sym)


assembled.assign_virtual_body('SU1.vbr_steer','ST.rbr_rocker')
assembled.assign_virtual_body('SU1.vbs_chassis','CH.rbs_chassis')
assembled.assign_virtual_body('SU2.vbs_chassis','CH.rbs_chassis')
assembled.assign_virtual_body('SU2.vbr_steer','CH.rbs_chassis')
assembled.assign_virtual_body('ST.vbs_chassis','CH.rbs_chassis')

assembled.assign_virtual_body('TR1.vbr_upright','SU1.rbr_upright')
assembled.assign_virtual_body('TR1.vbr_hub','SU1.rbr_hub')

assembled.assign_virtual_body('TR2.vbr_upright','SU2.rbr_upright')
assembled.assign_virtual_body('TR2.vbr_hub','SU2.rbr_hub')

assembled.assign_virtual_body('TR1.vbs_steer_gear','ST.rbl_rocker')
assembled.assign_virtual_body('TR1.vbs_chassis','CH.rbs_chassis')


assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()


