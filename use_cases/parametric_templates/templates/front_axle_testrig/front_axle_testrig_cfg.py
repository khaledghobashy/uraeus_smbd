# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:11:46 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.front_axle_testrig as model
from source.mbs_creators.topology_helpers import parametric_configuration

def main():
    
    global config
    
    name = 'frontaxle_testrig'
    config = parametric_configuration(name,model.template)
    config.assemble_base_layer()
        
    # Writing Code    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

if __name__ == '__main__':
    main()
