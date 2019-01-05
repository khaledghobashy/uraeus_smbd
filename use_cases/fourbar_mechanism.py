# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:21:42 2019

@author: khale
"""

from source.code_generators.code_generators import python_generator

from source.symbolic_classes.spatial_joints import (revolute, universal,
                                                    spherical, rotational_actuator)

from source.mbs_creators.topology_classes import (template_based_topology, 
                                                  subsystem, assembly)

fourbar_template = template_based_topology('fourbar')

fourbar_template.add_body('l1')
fourbar_template.add_body('l2')
fourbar_template.add_body('l3')
fourbar_template.add_virtual_body('ground')

fourbar_template.add_joint(revolute,'a','vbs_ground','rbs_l1')
fourbar_template.add_joint(revolute,'d','vbs_ground','rbs_l3')
fourbar_template.add_joint(spherical,'b','rbs_l1','rbs_l2')
fourbar_template.add_joint(universal,'c','rbs_l2','rbs_l3')
fourbar_template.add_joint_actuator(rotational_actuator,'act','jcs_a')

fourbar_subsystem = subsystem('sub',fourbar_template)
fourbar_subsystem.assemble_model()

fourbar_assembly = assembly('assm')
fourbar_assembly.add_subsystem(fourbar_subsystem)
fourbar_assembly.assemble_model()
fourbar_assembly.generate_equations()
fourbar_assembly.map_coordinates()
#fourbar_assembly.q_maped

fourbar_assembly.draw_topology()

import source
source.symbolic_classes.abstract_matrices.ccode_print = True
source.symbolic_classes.abstract_matrices.enclose = True
fourbar_generator = python_generator(fourbar_assembly)
code = fourbar_generator.dump_code()
#print(code)
with open('fourbar_v6_temp.py','w') as file:
    file.write(code)

