# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:32:59 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.mass_spring as model
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  cylinder_geometry)

def main():
    
    global config
    
    name = 'mass_spring_cfg'
    config = parametric_configuration(name,model.template)
    config.assemble_base_layer()
    
    config.add_point('origin')
    config.add_point('top')
    config.add_point('bottom')
    config.add_point('block_center')
    
    config.add_relation('hps_block_center',CR.Centered,'hps_top','hps_bottom')
    config.add_relation('pt1_jcs_cyl_joint',CR.Centered,'hps_origin','hps_block_center')
        
    # GEOMETRIES
    config.add_scalar('outer_raduis')
        
    config.add_geometry('block')
    config.add_relation('gms_block',cylinder_geometry,'hps_top','hps_bottom','s_outer_raduis')
    config.assign_geometry_to_body('rbs_block','gms_block')
    
    # Writing Code Files
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

    from source.post_processors.blender.scripter import scripter
    geo_code = scripter(config)
    geo_code.write_code_file()


if __name__ == '__main__':
    main()
