# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:54:31 2019

@author: khale
"""
from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.stewart_gough_3dof as model


def main():
    global config
    
    name = 'stewart_mod_points'
    config = model.template.param_config
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

    # BODIES LOCATIONS
    config.add_relation(CR.Centered,'R_rbs_rocker_1',['hps_bottom_1','hps_middle_1'])
    config.add_relation(CR.Centered,'R_rbs_rocker_2',['hps_bottom_2','hps_middle_2'])
    config.add_relation(CR.Centered,'R_rbs_rocker_3',['hps_bottom_3','hps_middle_3'])
    
    config.add_relation(CR.Centered,'R_rbs_link_1',['hps_middle_1','hps_upper_1'])
    config.add_relation(CR.Centered,'R_rbs_link_2',['hps_middle_2','hps_upper_2'])
    config.add_relation(CR.Centered,'R_rbs_link_3',['hps_middle_3','hps_upper_3'])
    
    config.add_relation(CR.Centered,'R_rbs_table',['hps_upper_1','hps_upper_2','hps_upper_3'])

    # JOINTS CONFIGURATIONS
    
    # Revolute Joints:
    config.add_relation(CR.Equal_to,'pt1_jcs_rev_1',['hps_bottom_1'])
    config.add_relation(CR.Oriented,'ax1_jcs_rev_1',['hps_bottom_1','hps_middle_1','hps_upper_1'])
    
    config.add_relation(CR.Equal_to,'pt1_jcs_rev_2',['hps_bottom_2'])
    config.add_relation(CR.Oriented,'ax1_jcs_rev_2',['hps_bottom_2','hps_middle_2','hps_upper_2'])
    
    config.add_relation(CR.Equal_to,'pt1_jcs_rev_3',['hps_bottom_3'])
    config.add_relation(CR.Oriented,'ax1_jcs_rev_3',['hps_bottom_3','hps_middle_3','hps_upper_3'])
    
    # Bottom Cylinderical Joints:
    config.add_relation(CR.Equal_to,'pt1_jcs_bottom_cyl_1',['hps_middle_1'])

    config.add_relation(CR.Equal_to,'pt1_jcs_bottom_cyl_2',['hps_middle_2'])
    config.add_relation(CR.Equal_to,'ax1_jcs_bottom_cyl_2',['ax1_jcs_bottom_cyl_1'])

    config.add_relation(CR.Equal_to,'pt1_jcs_bottom_cyl_3',['hps_middle_3'])
    config.add_relation(CR.Equal_to,'ax1_jcs_bottom_cyl_3',['ax1_jcs_bottom_cyl_1'])
    
    # Upper Universal Joints:
    config.add_relation(CR.Equal_to,'pt1_jcs_upper_uni_1',['hps_upper_1'])
    config.add_relation(CR.Oriented,'ax1_jcs_upper_uni_1',['hps_middle_1','hps_upper_1'])
    config.add_relation(CR.Equal_to,'ax2_jcs_upper_uni_1',['ax1_jcs_bottom_cyl_1'])

    config.add_relation(CR.Equal_to,'pt1_jcs_upper_uni_2',['hps_upper_2'])
    config.add_relation(CR.Oriented,'ax1_jcs_upper_uni_2',['hps_middle_2','hps_upper_2'])
    config.add_relation(CR.Equal_to,'ax2_jcs_upper_uni_2',['ax1_jcs_bottom_cyl_1'])
    
    config.add_relation(CR.Equal_to,'pt1_jcs_upper_uni_3',['hps_upper_3'])
    config.add_relation(CR.Oriented,'ax1_jcs_upper_uni_3',['hps_middle_3','hps_upper_3'])
    config.add_relation(CR.Equal_to,'ax2_jcs_upper_uni_3',['ax1_jcs_bottom_cyl_1'])
    
    # Upper Tripod Joint:
    config.add_relation(CR.Equal_to,'pt1_jcs_tripod',['R_rbs_table'])
    config.add_relation(CR.Equal_to,'ax1_jcs_tripod',['ax1_jcs_bottom_cyl_1'])
    
    config.topology.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()


if __name__ == '__main__':
    main()

