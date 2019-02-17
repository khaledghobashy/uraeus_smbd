# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 11:45:49 2019

@author: khaled.ghobashy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 09:17:44 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import assembly_code_generator
from source.mbs_creators.topology_classes import subsystem, assembly

import use_cases.parametric_templates.templates.stewart_gough_3dof as stewart
import use_cases.parametric_templates.templates.stewart_gough_testrig as testrig


SG_sym = subsystem('SG',stewart.template)
TR_sym = subsystem('TR',testrig.template)

assembled = assembly('stewart_assm')

assembled.add_subsystem(SG_sym)
assembled.add_subsystem(TR_sym)

assembled.assign_virtual_body('TR.vbs_rocker_1','SG.rbs_rocker_1')
assembled.assign_virtual_body('TR.vbs_rocker_2','SG.rbs_rocker_2')
assembled.assign_virtual_body('TR.vbs_rocker_3','SG.rbs_rocker_3')

assembled.assemble_model()
assembled.draw_constraints_topology()
assembled.draw_interface_graph()

numerical_code = assembly_code_generator(assembled)
numerical_code.write_code_file()


