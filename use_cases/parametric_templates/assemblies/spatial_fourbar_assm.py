# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:55:57 2019

@author: khaled.ghobashy
"""

from source.interfaces.scripting_interfaces import assembly

import use_cases.parametric_templates.templates.spatial_fourbar as fourbar

assembled = assembly(__file__)
assembled.add_subsystem('FB', fourbar.template)

assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

assembled.write_python_code()

