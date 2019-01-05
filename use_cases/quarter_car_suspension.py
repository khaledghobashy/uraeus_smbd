# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 10:31:23 2019

@author: khaled.ghobashy
"""

from source.code_generators.code_generators import python_generator

from source.symbolic_classes.spatial_joints import (revolute, universal,
                                                    spherical, rotational_actuator,
                                                    cylinderical)

from source.mbs_creators.topology_classes_2 import (template_based_topology, 
                                                  subsystem, assembly)

dwb_template = template_based_topology('dwb')

dwb_template.add_body('uca')
dwb_template.add_body('lca')
dwb_template.add_body('upright')
dwb_template.add_body('upper_strut')
dwb_template.add_body('lower_strut')
dwb_template.add_body('tie_rod')
dwb_template.add_body('hub')
dwb_template.add_virtual_body('ground')

dwb_template.add_joint(spherical,'uca_upright','rbs_uca','rbs_upright')
dwb_template.add_joint(spherical,'lca_upright','rbs_lca','rbs_upright')
dwb_template.add_joint(spherical,'tie_upright','rbs_tie_rod','rbs_upright')
dwb_template.add_joint(revolute,'uca_chassis','rbs_uca','vbs_ground')
dwb_template.add_joint(revolute,'lca_chassis','rbs_lca','vbs_ground')
dwb_template.add_joint(revolute,'hub_bearing','rbs_upright','rbs_hub')
dwb_template.add_joint(universal,'strut_chassis','rbs_upper_strut','vbs_ground')
dwb_template.add_joint(universal,'strut_lca','rbs_lower_strut','rbs_lca')
dwb_template.add_joint(universal,'tie_steering','rbs_tie_rod','vbs_ground')
dwb_template.add_joint(cylinderical,'strut','rbs_upper_strut','rbs_lower_strut')
dwb_template.add_absolute_actuator('zact','rbs_upright','z')
dwb_template.add_joint_actuator(rotational_actuator,'rot_act','jcs_hub_bearing')



dwb_subsystem = subsystem('sub',dwb_template)
dwb_subsystem.assemble_model()

dwb_assembly = assembly('assm')
dwb_assembly.add_subsystem(dwb_subsystem)
dwb_assembly.assemble_model()
dwb_assembly.generate_equations()
dwb_assembly.map_coordinates()

dwb_assembly.draw_topology()

import source
source.symbolic_classes.abstract_matrices.ccode_print = True
source.symbolic_classes.abstract_matrices.enclose = True
dwb_generator = python_generator(dwb_assembly)
code = dwb_generator.dump_code()
#print(code)
with open('dwb_quarter_car_3.py','w') as file:
    file.write(code)

