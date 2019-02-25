# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:54:31 2019

@author: khale
"""
from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  simple_geometry, 
                                                  composite_geometry,
                                                  cylinder_geometry)

import use_cases.parametric_templates.templates.stewart_gough_3dof as model


def main():
    global config
    
    name = 'stewart_points'
    config = parametric_configuration(model.template)
    config.assemble_base_layer()
    config.name = name
    model.template.cfg_file = name
    
    # HARD POINTS:
    config.add_point('bottom_1')
    config.add_point('bottom_2')
    config.add_point('bottom_3')
    config.add_point('middle_1')
    config.add_point('middle_2')
    config.add_point('middle_3')
    config.add_point('upper_1')
    config.add_point('upper_2')
    config.add_point('upper_3')
    config.add_point('tripod')

    # BODIES LOCATIONS
    config.add_relation('R_rbs_rocker_1',CR.Centered,'hps_bottom_1','hps_middle_1')
    config.add_relation('R_rbs_rocker_2',CR.Centered,'hps_bottom_2','hps_middle_2')
    config.add_relation('R_rbs_rocker_3',CR.Centered,'hps_bottom_3','hps_middle_3')
    
    config.add_relation('R_rbs_link_1',CR.Centered,'hps_middle_1','hps_upper_1')
    config.add_relation('R_rbs_link_2',CR.Centered,'hps_middle_2','hps_upper_2')
    config.add_relation('R_rbs_link_3',CR.Centered,'hps_middle_3','hps_upper_3')
    
    config.add_relation('R_rbs_table',CR.Centered,'hps_upper_1','hps_upper_2','hps_upper_3')

    # JOINTS CONFIGURATIONS
    
    # Revolute Joints:
    config.add_relation('pt1_jcs_rev_1',CR.Equal_to,'hps_bottom_1')
    config.add_relation('ax1_jcs_rev_1',CR.Oriented,'hps_bottom_1','hps_middle_1','hps_upper_1')
    
    config.add_relation('pt1_jcs_rev_2',CR.Equal_to,'hps_bottom_2')
    config.add_relation('ax1_jcs_rev_2',CR.Oriented,'hps_bottom_2','hps_middle_2','hps_upper_2')
    
    config.add_relation('pt1_jcs_rev_3',CR.Equal_to,'hps_bottom_3')
    config.add_relation('ax1_jcs_rev_3',CR.Oriented,'hps_bottom_3','hps_middle_3','hps_upper_3')
    
    # Bottom Spherical Joints:
    config.add_relation('pt1_jcs_bottom_sph_1',CR.Equal_to,'hps_middle_1')

    config.add_relation('pt1_jcs_bottom_sph_2',CR.Equal_to,'hps_middle_2')
    config.add_relation('ax1_jcs_bottom_sph_2',CR.Equal_to,'ax1_jcs_bottom_sph_1')

    config.add_relation('pt1_jcs_bottom_sph_3',CR.Equal_to,'hps_middle_3')
    config.add_relation('ax1_jcs_bottom_sph_3',CR.Equal_to,'ax1_jcs_bottom_sph_1')
    
    # Upper Universal Joints:
    config.add_relation('pt1_jcs_upper_uni_1',CR.Equal_to,'hps_upper_1')
    config.add_relation('ax1_jcs_upper_uni_1',CR.Oriented,'hps_middle_1','hps_upper_1')
    config.add_relation('ax2_jcs_upper_uni_1',CR.Equal_to,'ax1_jcs_bottom_sph_1')

    config.add_relation('pt1_jcs_upper_uni_2',CR.Equal_to,'hps_upper_2')
    config.add_relation('ax1_jcs_upper_uni_2',CR.Oriented,'hps_middle_2','hps_upper_2')
    config.add_relation('ax2_jcs_upper_uni_2',CR.Equal_to,'ax1_jcs_bottom_sph_1')
    
    config.add_relation('pt1_jcs_upper_uni_3',CR.Equal_to,'hps_upper_3')
    config.add_relation('ax1_jcs_upper_uni_3',CR.Oriented,'hps_middle_3','hps_upper_3')
    config.add_relation('ax2_jcs_upper_uni_3',CR.Equal_to,'ax1_jcs_bottom_sph_1')
    
    # Upper Tripod Joint:
    config.add_relation('pt1_jcs_tripod',CR.Equal_to,'hps_tripod')
    config.add_relation('ax1_jcs_tripod',CR.Equal_to,'ax1_jcs_bottom_sph_1')
    
    # Testing Geometries
    config.add_scalar('links_ro')
    config.add_scalar('rockers_ro')
    
    config.add_geometry('rocker_1')
    config.add_relation('gms_rocker_1',cylinder_geometry,'hps_bottom_1','hps_middle_1','s_rockers_ro')
    config.assign_geometry_to_body('rbs_rocker_1','gms_rocker_1')

    config.add_geometry('rocker_2')
    config.add_relation('gms_rocker_2',cylinder_geometry,'hps_bottom_2','hps_middle_2','s_rockers_ro')
    config.assign_geometry_to_body('rbs_rocker_2','gms_rocker_2')

    config.add_geometry('rocker_3')
    config.add_relation('gms_rocker_3',cylinder_geometry,'hps_bottom_3','hps_middle_3','s_rockers_ro')
    config.assign_geometry_to_body('rbs_rocker_3','gms_rocker_3')

    config.add_geometry('link_1')
    config.add_relation('gms_link_1',cylinder_geometry,'hps_upper_1','hps_middle_1','s_links_ro')
    config.assign_geometry_to_body('rbs_link_1','gms_link_1')

    config.add_geometry('link_2')
    config.add_relation('gms_link_2',cylinder_geometry,'hps_upper_2','hps_middle_2','s_links_ro')
    config.assign_geometry_to_body('rbs_link_2','gms_link_2')

    config.add_geometry('link_3')
    config.add_relation('gms_link_3',cylinder_geometry,'hps_upper_3','hps_middle_3','s_links_ro')
    config.assign_geometry_to_body('rbs_link_3','gms_link_3')
    
#    config.add_geometry('link_3')
#    config.add_relation('gms_link_3',cylinder_geometry,'hps_upper_3','hps_middle_3','s_links_ro')
#    config.add_sub_relation('R_rbs_link_3',CR.Equal_to,'gms_link_3.R')
#    config.add_sub_relation('m_rbs_link_3',CR.Equal_to,'gms_link_3.m')
#    config.add_sub_relation('Jbar_rbs_link_3',CR.Equal_to,'gms_link_3.J')
    
    model.template.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()


if __name__ == '__main__':
    main()

