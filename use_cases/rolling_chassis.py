# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:37:15 2019

@author: khale
"""

from source.code_generators.python_code_generators import template_code_generator, assembly_code_generator

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

test_rig = template_based_topology('test_rig')
test_rig.add_virtual_body('hub',mirrored=True)
test_rig.add_virtual_body('upright',mirrored=True)
test_rig.add_joint(revolute,'rev','vbr_hub','vbr_upright',mirrored=True)
test_rig.add_joint_actuator(rotational_actuator,'act','jcr_rev',True)
test_rig.assemble_model()
test_rig_code = template_code_generator(test_rig)
test_rig_code.write_code_file()
test_rig_sub = subsystem('TS',test_rig)



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

dwb_template.assemble_model()
dwb_code = template_code_generator(dwb_template)
dwb_code.write_code_file()
SU1_sym = subsystem('SU1',dwb_template)
SU2_sym = subsystem('SU2',dwb_template)


steering_template = template_based_topology('steering')
steering_template.add_body('coupler')
steering_template.add_body('rocker',mirrored=True)
steering_template.add_virtual_body('chassis')
steering_template.add_joint(revolute,'rocker_ch','rbr_rocker','vbs_chassis',mirrored=True)
steering_template.add_joint(spherical,'rc_sph','rbr_rocker','rbs_coupler')
steering_template.add_joint(cylinderical,'rc_cyl','rbl_rocker','rbs_coupler')

steering_template.assemble_model()
steering_code = template_code_generator(steering_template)
steering_code.write_code_file()
ST_sym = subsystem('ST',steering_template)


rolling_chassis = assembly('rolling_chassis_assm')
rolling_chassis.add_subsystem(SU1_sym)
#rolling_chassis.add_subsystem(SU2_sym)
rolling_chassis.add_subsystem(ST_sym)

rolling_chassis.add_subsystem(test_rig_sub)
rolling_chassis.assign_virtual_body('TS.vbr_hub','SU1.rbr_hub')
rolling_chassis.assign_virtual_body('TS.vbr_upright','SU1.rbr_upright')
#rolling_chassis.add_subsystem(chassis_subsystem)
#rolling_chassis.assign_virtual_body('SU1_vbs_chassis','CH_rbs_chassis')
#rolling_chassis.assign_virtual_body('ST_vbs_chassis','CH_rbs_chassis')
rolling_chassis.assign_virtual_body('SU1.vbr_steer','ST.rbr_rocker')
#rolling_chassis.assign_virtual_body('SU2_vbr_steer','CH_rbs_chassis')
#rolling_chassis.assign_virtual_body('SU2_vbs_chassis','CH_rbs_chassis')


rolling_chassis.assemble_model(full=False)
rolling_chassis.draw_topology()
rolling_chassis.draw_interface_graph()

rolling_code = assembly_code_generator(rolling_chassis)
rolling_code.write_code_file()

