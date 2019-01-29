# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:21:51 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.spatial_joints import (revolute,spherical,cylinderical)

from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator

template = template_based_topology('steer')

template.add_body('coupler')
template.add_body('rocker',mirrored=True)
template.add_virtual_body('chassis')

template.add_joint(revolute,'rocker_ch','rbr_rocker','vbs_chassis',mirrored=True)
template.add_joint(spherical,'rc_sph','rbr_rocker','rbs_coupler')
template.add_joint(cylinderical,'rc_cyl','rbl_rocker','rbs_coupler')

template.assemble_model()

numerical_code = template_code_generator(template)
numerical_code.write_code_file()
