# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:37:32 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.spatial_fourbar as model
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  cylinder_geometry)

def main():
    
    global config
    
    name = 'spatial_fourbar_cfg'
    config = parametric_configuration(model.template)
    config.name = name
    config.assemble_base_layer()
    
    config.add_point('rev_crank')
    config.add_point('rev_rocker')
    config.add_point('coupler_crank')
    config.add_point('coupler_rocker')
    
    config.add_relation('pt1_jcs_rev_crank',CR.Equal_to,'hps_rev_crank')

    config.add_relation('pt1_jcs_rev_rocker',CR.Equal_to,'hps_rev_rocker')
    
    config.add_relation('pt1_jcs_sph_coupler_crank',CR.Equal_to,'hps_coupler_crank')
    
    config.add_relation('pt1_jcs_uni_coupler_rocker',CR.Equal_to,'hps_coupler_rocker')
    config.add_relation('ax1_jcs_uni_coupler_rocker',CR.Oriented,'hps_coupler_rocker','hps_coupler_crank')
    config.add_relation('ax2_jcs_uni_coupler_rocker',CR.Oriented,'hps_coupler_crank','hps_coupler_rocker')
    
    # GEOMETRIES
    config.add_scalar('links_ro')
    
    config.add_geometry('rocker')
    config.add_relation('gms_rocker',cylinder_geometry,'hps_rev_rocker','hps_coupler_rocker','s_links_ro')
    config.assign_geometry_to_body('rbs_rocker','gms_rocker')
    
    config.add_geometry('crank')
    config.add_relation('gms_crank',cylinder_geometry,'hps_rev_crank','hps_coupler_crank','s_links_ro')
    config.assign_geometry_to_body('rbs_crank','gms_crank')

    config.add_geometry('coupler')
    config.add_relation('gms_coupler',cylinder_geometry,'hps_coupler_crank','hps_coupler_rocker','s_links_ro')
    config.assign_geometry_to_body('rbs_coupler','gms_coupler')
    
    # Writing Code Files
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

    from source.post_processors.blender.scripter import scripter
    geo_code = scripter(config)
    geo_code.write_code_file()


if __name__ == '__main__':
    main()

