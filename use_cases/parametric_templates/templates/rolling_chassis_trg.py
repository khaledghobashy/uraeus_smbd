# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:31:45 2019

@author: khale
"""

import os
import pickle

from source.symbolic_classes.joints import (revolute,rotational_actuator)

from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator


topology_name = os.path.basename(__file__).split('.')[0]

def load():
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path,topology_name),'rb') as f:
        template = pickle.load(f)

def create():
    template = template_based_topology(topology_name)
    
    template.add_body('hub',mirrored=True,virtual=True)
    template.add_body('upright',mirrored=True,virtual=True)
    template.add_joint(revolute,'rev','vbr_upright','vbr_hub',mirrored=True,virtual=True)
    template.add_joint_actuator(rotational_actuator,'act','jcr_rev',mirrored=True)
    template.add_absolute_actuator('ver_act','vbr_upright','z',mirrored=True)
        
    template.assemble_model()
    template.save()
    
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()

    
if __name__ == '__main__':
    create()
else:
    load()

