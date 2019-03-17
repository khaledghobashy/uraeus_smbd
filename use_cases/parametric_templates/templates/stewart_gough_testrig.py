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


topology_name = 'stewart_testrig'

def load():
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path,topology_name),'rb') as f:
        template = pickle.load(f)

def create():
    global template
    template = template_based_topology(topology_name)
    
    template.add_body('rocker_1',virtual=True)
    template.add_body('rocker_2',virtual=True)
    template.add_body('rocker_3',virtual=True)
    
    template.add_joint(revolute,'rev_1','vbs_rocker_1','vbs_ground',virtual=True)
    template.add_joint(revolute,'rev_2','vbs_rocker_2','vbs_ground',virtual=True)
    template.add_joint(revolute,'rev_3','vbs_rocker_3','vbs_ground',virtual=True)
    
    template.add_joint_actuator(rotational_actuator,'act_1','jcs_rev_1')
    template.add_joint_actuator(rotational_actuator,'act_2','jcs_rev_2')
    template.add_joint_actuator(rotational_actuator,'act_3','jcs_rev_3')
        
    template.assemble_model()
    template.save()
    
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()

    
if __name__ == '__main__':
    create()
else:
    load()

