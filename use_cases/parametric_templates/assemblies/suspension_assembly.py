# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:17:44 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.templates.double_wishbone_bellcranck as dwb
import use_cases.parametric_templates.templates.suspension_testrig as testrig


SU_sym = subsystem('SU',dwb.template)
TR_sym = subsystem('TR',testrig.template)

assembled = assembly('suspension_assm')

assembled.add_subsystem(SU_sym)
assembled.add_subsystem(TR_sym)

assembled.assign_virtual_body('TR.vbr_upright','SU.rbr_upright')
assembled.assign_virtual_body('TR.vbr_hub','SU.rbr_hub')

assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()

