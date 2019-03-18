# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:08:34 2019

@author: khaled.ghobashy
"""

import os
import pickle

from source.symbolic_classes.joints import (revolute,rotational_actuator)

from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator


topology_name = 'test_rig'

def load():
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path,topology_name),'rb') as f:
        template = pickle.load(f)

def create():
    template = template_based_topology('test_rig')
    
    template.add_body('hub',mirrored=True,virtual=True)
    template.add_body('upright',mirrored=True,virtual=True)
    template.add_joint(revolute,'rev','vbr_upright','vbr_hub',mirrored=True,virtual=True)
    template.add_joint_actuator(rotational_actuator,'act','jcr_rev',mirrored=True)
    template.add_absolute_actuator('ver_act','vbr_hub','z',mirrored=True)
    
    template.add_body('steer_gear',virtual=True)
    template.add_body('chassis',virtual=True)
    template.add_joint(revolute,'steer_gear','vbs_steer_gear','vbs_chassis',virtual=True)
    template.add_joint_actuator(rotational_actuator,'steer_act','jcs_steer_gear')
    
    template.assemble_model()
    template.save()
    
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()

    
if __name__ == '__main__':
    create()
else:
    load()

