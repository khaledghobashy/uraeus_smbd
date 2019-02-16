# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:54:31 2019

@author: khale
"""
from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.parallel_link_steering as model


def main():
    
    name = 'steer_simple_points'
    config = model.template.param_config
    config.name = name
    model.template.cfg_file = name
    
    config.add_point('rocker_chassis',mirror=True)
    config.add_point('rocker_coupler',mirror=True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_rocker_ch',['hpr_rocker_chassis'],True)
    config.add_relation(CR.Equal_to,'pt1_jcs_rc_sph',['hpr_rocker_coupler'])
    config.add_relation(CR.Equal_to,'pt1_jcs_rc_cyl',['hpl_rocker_coupler'])
    
    config.add_relation(CR.Centered,'R_rbr_rocker',['hpr_rocker_chassis','hpr_rocker_coupler'],True)
    config.add_relation(CR.Centered,'R_rbs_coupler',['hpr_rocker_coupler','hpl_rocker_coupler'])
    
    config.add_relation(CR.Oriented,'ax1_jcs_rc_cyl',['hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis'])
    config.add_relation(CR.Oriented,'ax1_jcr_rocker_ch',['hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis'],True)
    
    config.topology.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()


if __name__ == '__main__':
    main()

