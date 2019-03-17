# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:54:31 2019

@author: khale
"""
from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.parallel_link_steering as model
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  composite_geometry,
                                                  triangular_prism, 
                                                  cylinder_geometry)

def main():
    
    global config
    
    name = 'steer_simple_points'
    config = parametric_configuration(name,model.template)
    config.assemble_base_layer()
    
    config.add_point('rocker_chassis',mirror=True)
    config.add_point('rocker_coupler',mirror=True)
    
    config.add_relation('pt1_jcr_rocker_ch',CR.Equal_to,'hpr_rocker_chassis',mirror=True)
    config.add_relation('pt1_jcs_rc_sph',CR.Equal_to,'hpr_rocker_coupler')
    config.add_relation('pt1_jcs_rc_cyl',CR.Equal_to,'hpl_rocker_coupler')
    
    config.add_relation('R_rbr_rocker',CR.Centered,'hpr_rocker_chassis','hpr_rocker_coupler',mirror=True)
    config.add_relation('R_rbs_coupler',CR.Centered,'hpr_rocker_coupler','hpl_rocker_coupler')
    
    config.add_relation('ax1_jcs_rc_cyl',CR.Oriented,'hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis')
    config.add_relation('ax1_jcr_rocker_ch',CR.Oriented,'hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis',mirror=True)
    
    # GEOMETRIES
    config.add_scalar('links_ro')
    config.add_scalar('thickness')
    
    config.add_geometry('rocker',mirror=True)
    config.add_relation('gmr_rocker',cylinder_geometry,'hpr_rocker_chassis','hpr_rocker_coupler','s_links_ro',mirror=True)
    config.assign_geometry_to_body('rbr_rocker','gmr_rocker',mirror=True)
    
    config.add_geometry('coupler')
    config.add_relation('gms_coupler',cylinder_geometry,'hpr_rocker_coupler','hpl_rocker_coupler','s_links_ro')
    config.assign_geometry_to_body('rbs_coupler','gms_coupler')
    
    # Saving Topology and Writing Code
    config.topology.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

    from source.post_processors.blender.scripter import scripter
    geo_code = scripter(config)
    geo_code.write_code_file()


if __name__ == '__main__':
    main()

