# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:18:17 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.symbolic_classes.spatial_joints import (revolute, universal, spherical,
                                                    cylinderical)

from source.code_generators.python_code_generators import template_code_generator
from source.mbs_creators.topology_classes import template_based_topology

template = template_based_topology('dwb')

template.add_body('uca',mirrored=True)
template.add_body('lca',mirrored=True)
template.add_body('upright',mirrored=True)
template.add_body('upper_strut',mirrored=True)
template.add_body('lower_strut',mirrored=True)
template.add_body('tie_rod',mirrored=True)
template.add_body('hub',mirrored=True)
template.add_virtual_body('steer',mirrored=True)
template.add_virtual_body('chassis')

template.add_joint(spherical,'uca_upright','rbr_uca','rbr_upright',mirrored=True)
template.add_joint(spherical,'lca_upright','rbr_lca','rbr_upright',mirrored=True)
template.add_joint(spherical,'tie_upright','rbr_tie_rod','rbr_upright',mirrored=True)
template.add_joint(revolute,'uca_chassis','rbr_uca','vbs_chassis',mirrored=True)
template.add_joint(revolute,'lca_chassis','rbr_lca','vbs_chassis',mirrored=True)
template.add_joint(revolute,'hub_bearing','rbr_upright','rbr_hub',mirrored=True)
template.add_joint(universal,'strut_chassis','rbr_upper_strut','vbs_chassis',mirrored=True)
template.add_joint(universal,'strut_lca','rbr_lower_strut','rbr_lca',mirrored=True)
template.add_joint(universal,'tie_steering','rbr_tie_rod','vbr_steer',mirrored=True)
template.add_joint(cylinderical,'strut','rbr_upper_strut','rbr_lower_strut',mirrored=True)

template.assemble_model()


#template.param_config.add_point('ucaf',mirror=True)
#template.param_config.add_point('ucar',mirror=True)
#template.param_config.add_point('ucao',mirror=True)
#template.param_config.add_point('lcaf',mirror=True)
#template.param_config.add_point('lcar',mirror=True)
#template.param_config.add_point('lcao',mirror=True)
#template.param_config.add_point('tro',mirror=True)
#template.param_config.add_point('tri',mirror=True)
#template.param_config.add_point('strut_chassis',mirror=True)
#template.param_config.add_point('strut_mid',mirror=True)
#template.param_config.add_point('strut_lca',mirror=True)
#template.param_config.add_point('wc',mirror=True)
#
#template.param_config.add_relation(CR.Centered,'hpr_strut_mid',['hpr_strut_chassis','hpr_strut_lca'],True)
#
#template.param_config.add_relation(CR.Centered,'R_rbr_uca',['hpr_ucao','hpr_ucaf','hpr_ucar'],True)
#template.param_config.add_relation(CR.Centered,'R_rbr_lca',['hpr_lcao','hpr_lcaf','hpr_lcar'],True)
#template.param_config.add_relation(CR.Centered,'R_rbr_upright',['hpr_ucao','hpr_lcao','hpr_wc'],True)
#template.param_config.add_relation(CR.Centered,'R_rbr_upper_strut',['hpr_strut_chassis','hpr_strut_mid'],True)
#template.param_config.add_relation(CR.Centered,'R_rbr_lower_strut',['hpr_strut_lca','hpr_strut_mid'],True)
#template.param_config.add_relation(CR.Centered,'R_rbr_tie_rod',['hpr_tro','hpr_tri'],True)
#template.param_config.add_relation(CR.Centered,'R_rbr_hub',['hpr_wc','R_rbr_upright'],True)
#
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_uca_upright',['hpr_ucao'],True)
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_lca_upright',['hpr_lcao'],True)
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_tie_upright',['hpr_tro'],True)
#
#template.param_config.add_relation(CR.Centered,'pt1_jcr_uca_chassis',['hpr_ucaf','hpr_ucar'],True)
#template.param_config.add_relation(CR.Oriented,'ax1_jcr_uca_chassis',['hpr_ucaf','hpr_ucar'],True)
#
#template.param_config.add_relation(CR.Centered,'pt1_jcr_lca_chassis',['hpr_lcaf','hpr_lcar'],True)
#template.param_config.add_relation(CR.Oriented,'ax1_jcr_lca_chassis',['hpr_lcaf','hpr_lcar'],True)
#
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_hub_bearing',['hpr_wc'],True)
#
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_strut_chassis',['hpr_strut_chassis'],True)
#template.param_config.add_relation(CR.Oriented,'ax1_jcr_strut_chassis',['hpr_strut_chassis','hpr_strut_lca'],True)
#template.param_config.add_relation(CR.Oriented,'ax2_jcr_strut_chassis',['hpr_strut_lca','hpr_strut_chassis'],True)
#
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_strut_lca',['hpr_strut_lca'],True)
#template.param_config.add_relation(CR.Oriented,'ax1_jcr_strut_lca',['hpr_strut_chassis','hpr_strut_lca'],True)
#template.param_config.add_relation(CR.Oriented,'ax2_jcr_strut_lca',['hpr_strut_lca','hpr_strut_chassis'],True)
#
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_tie_steering',['hpr_tri'],True)
#template.param_config.add_relation(CR.Oriented,'ax1_jcr_tie_steering',['hpr_tri','hpr_tro'],True)
#template.param_config.add_relation(CR.Oriented,'ax2_jcr_tie_steering',['hpr_tro','hpr_tri'],True)
#
#template.param_config.add_relation(CR.Equal_to,'pt1_jcr_strut',['hpr_strut_mid'],True)
#template.param_config.add_relation(CR.Oriented,'ax1_jcr_strut',['hpr_strut_lca','hpr_strut_chassis'],True)
#

numerical_code = template_code_generator(template)
numerical_code.write_code_file()
