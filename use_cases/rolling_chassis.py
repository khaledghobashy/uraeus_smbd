# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:37:15 2019

@author: khale
"""

from source.code_generators.code_generators import python_code_generator, assembly_generator

from source.symbolic_classes.spatial_joints import (revolute, universal,
                                                    spherical, rotational_actuator,
                                                    cylinderical)

from source.mbs_creators.topology_classes import (template_based_topology, 
                                                  subsystem, assembly)

#chassis_template = template_based_topology('chassis')
#chassis_template.add_body('chassis')
#
#chassis_subsystem = subsystem('CH',chassis_template)
#chassis_subsystem.assemble_model()


dwb_template = template_based_topology('dwb')

dwb_template.add_body('uca',mirrored=True)
dwb_template.add_body('lca',mirrored=True)
dwb_template.add_body('upright',mirrored=True)
dwb_template.add_body('upper_strut',mirrored=True)
dwb_template.add_body('lower_strut',mirrored=True)
dwb_template.add_body('tie_rod',mirrored=True)
dwb_template.add_body('hub',mirrored=True)
dwb_template.add_virtual_body('steer',mirrored=True)
dwb_template.add_virtual_body('chassis')

dwb_template.add_joint(spherical,'uca_upright','rbr_uca','rbr_upright',mirrored=True)
dwb_template.add_joint(spherical,'lca_upright','rbr_lca','rbr_upright',mirrored=True)
dwb_template.add_joint(spherical,'tie_upright','rbr_tie_rod','rbr_upright',mirrored=True)
dwb_template.add_joint(revolute,'uca_chassis','rbr_uca','vbs_chassis',mirrored=True)
dwb_template.add_joint(revolute,'lca_chassis','rbr_lca','vbs_chassis',mirrored=True)
dwb_template.add_joint(revolute,'hub_bearing','rbr_upright','rbr_hub',mirrored=True)
dwb_template.add_joint(universal,'strut_chassis','rbr_upper_strut','vbs_chassis',mirrored=True)
dwb_template.add_joint(universal,'strut_lca','rbr_lower_strut','rbr_lca',mirrored=True)
dwb_template.add_joint(universal,'tie_steering','rbr_tie_rod','vbr_steer',mirrored=True)
dwb_template.add_joint(cylinderical,'strut','rbr_upper_strut','rbr_lower_strut',mirrored=True)
#dwb_template.add_joint_actuator(rotational_actuator,'rot_act','jcr_hub_bearing',mirrored=True)
#dwb_template.add_absolute_actuator('zact','rbr_hub','z',mirrored=True)

dwb_template.assemble_model()
dwb_code = python_code_generator(dwb_template)
dwb_code.write_code_file()

front_axle = subsystem('SU1',dwb_template)
#front_axle.assemble_model()

rear_axle = subsystem('SU2',dwb_template)
#rear_axle.assemble_model()


steering_template = template_based_topology('steering')
steering_template.add_body('coupler')
steering_template.add_body('rocker',mirrored=True)
steering_template.add_virtual_body('chassis')
steering_template.add_joint(revolute,'rocker_ch','rbr_rocker','vbs_chassis',mirrored=True)
steering_template.add_joint(spherical,'rc_sph','rbr_rocker','rbs_coupler')
steering_template.add_joint(cylinderical,'rc_cyl','rbl_rocker','rbs_coupler')
#steering_template.add_joint_actuator(rotational_actuator,'rot_act','jcr_rocker_ch')

steering_template.assemble_model()
steering_code = python_code_generator(steering_template)
steering_code.write_code_file()


steering_subsystem = subsystem('ST',steering_template)
#steering_subsystem.assemble_model()


rolling_chassis = assembly('rolling_chassis')
rolling_chassis.add_subsystem(front_axle)
rolling_chassis.add_subsystem(rear_axle)
rolling_chassis.add_subsystem(steering_subsystem)
#rolling_chassis.add_subsystem(chassis_subsystem)
#rolling_chassis.assign_virtual_body('SU1_vbs_chassis','CH_rbs_chassis')
#rolling_chassis.assign_virtual_body('ST_vbs_chassis','CH_rbs_chassis')
rolling_chassis.assign_virtual_body('SU1.vbr_steer','ST.rbr_rocker')
#rolling_chassis.assign_virtual_body('SU2_vbr_steer','CH_rbs_chassis')
#rolling_chassis.assign_virtual_body('SU2_vbs_chassis','CH_rbs_chassis')


rolling_chassis.assemble_model(full=False)
rolling_chassis.draw_topology()
rolling_chassis.draw_interface_graph()

