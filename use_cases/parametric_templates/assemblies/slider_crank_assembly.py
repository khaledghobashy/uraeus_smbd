# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:45:49 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.templates.slider_crank as slider
import use_cases.parametric_templates.templates.slider_crank_testrig as testrig


SG_sym = subsystem('SG',slider.template)
TR_sym = subsystem('TR',testrig.template)

assembled = assembly('slider_crank_assm')

assembled.add_subsystem(SG_sym)
assembled.add_subsystem(TR_sym)

assembled.assign_virtual_body('TR.vbs_rocker','SG.rbs_rocker')

assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()


