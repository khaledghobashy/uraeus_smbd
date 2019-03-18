# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:32:59 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.pendulum as model
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  cylinder_geometry)

def main():
    
    global config
    
    name = 'pendulum_cfg'
    config = parametric_configuration(name,model.template)
    config.assemble_base_layer()
    
    config.add_point('rev_crank')
    config.add_point('end_point')
    
    config.add_relation('pt1_jcs_rev_crank',CR.Equal_to,'hps_rev_crank')
        
    # GEOMETRIES
    config.add_scalar('links_ro')
        
    config.add_geometry('crank')
    config.add_relation('gms_crank',cylinder_geometry,'hps_rev_crank','hps_end_point','s_links_ro')
    config.assign_geometry_to_body('rbs_crank','gms_crank')
    
    # Writing Code Files
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

    from source.post_processors.blender.scripter import scripter
    geo_code = scripter(config)
    geo_code.write_code_file()


if __name__ == '__main__':
    main()
