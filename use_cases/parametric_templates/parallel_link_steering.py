# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:21:51 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.symbolic_classes.spatial_joints import (revolute,spherical,cylinderical)

from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator

template = template_based_topology('steer')

template.add_body('coupler')
template.add_body('rocker',mirrored=True)
template.add_body('chassis',virtual=True)

template.add_joint(revolute,'rocker_ch','rbr_rocker','vbs_chassis',mirrored=True)
template.add_joint(spherical,'rc_sph','rbr_rocker','rbs_coupler')
template.add_joint(cylinderical,'rc_cyl','rbl_rocker','rbs_coupler')

template.assemble_model()

template.param_config.add_point('rocker_chassis',mirror=True)
template.param_config.add_point('rocker_coupler',mirror=True)

template.param_config.add_relation(CR.Equal_to,'pt1_jcr_rocker_ch',['hpr_rocker_chassis'],True)
template.param_config.add_relation(CR.Equal_to,'pt1_jcs_rc_sph',['hpr_rocker_coupler'])
template.param_config.add_relation(CR.Equal_to,'pt1_jcs_rc_cyl',['hpl_rocker_coupler'])

template.param_config.add_relation(CR.Centered,'R_rbr_rocker',['hpr_rocker_chassis','hpr_rocker_coupler'],True)
template.param_config.add_relation(CR.Centered,'R_rbs_coupler',['hpr_rocker_coupler','hpl_rocker_coupler'])

template.param_config.add_relation(CR.Oriented,'ax1_jcs_rc_cyl',['hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis'])
template.param_config.add_relation(CR.Oriented,'ax1_jcr_rocker_ch',['hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis'],True)


numerical_code = template_code_generator(template)
numerical_code.write_code_file()

