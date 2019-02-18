# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:54:31 2019

@author: khale
"""
from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.slider_crank as model


def main():
    global config
    
    name = 'slider_points'
    config = model.template.param_config
    config.name = name
    model.template.cfg_file = name
    
    # HARD POINTS:
    config.add_point('A')
    config.add_point('B')
    config.add_point('C')

    # BODIES LOCATIONS
    config.add_relation(CR.Centered,'R_rbs_rocker',['hps_A','hps_B'])
    config.add_relation(CR.Centered,'R_rbs_rod',['hps_B','hps_C'])
    config.add_relation(CR.Equal_to,'R_rbs_slider',['hps_C'])
    
    # JOINTS CONFIGURATIONS
    config.add_relation(CR.Equal_to,'pt1_jcs_rev',['hps_A'])
    config.add_relation(CR.Oriented,'ax1_jcs_rev',['hps_A','hps_B','hps_C'])
    
    config.add_relation(CR.Equal_to,'pt1_jcs_cyl',['hps_B'])
    config.add_relation(CR.Equal_to,'ax1_jcs_cyl',['ax1_jcs_rev'])
    
    config.add_relation(CR.Equal_to,'pt1_jcs_uni',['hps_B'])
    config.add_relation(CR.Oriented,'ax1_jcs_uni',['hps_B','hps_C'])
    config.add_relation(CR.Equal_to,'ax2_jcs_uni',['ax1_jcs_trans'])
    
    config.add_relation(CR.Equal_to,'pt1_jcs_trans',['hps_C'])
        
    config.topology.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()


if __name__ == '__main__':
    main()

