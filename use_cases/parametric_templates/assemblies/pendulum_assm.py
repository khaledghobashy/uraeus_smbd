# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:35:16 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.templates.pendulum as pendulum


PD = subsystem('PD',pendulum.template)

assembled = assembly('pendulum_assm')

assembled.add_subsystem(PD)

assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()
