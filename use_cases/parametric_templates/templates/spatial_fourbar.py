# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:23:45 2019

@author: khaled.ghobashy
"""

import os
import pickle

from source.symbolic_classes.spatial_joints import (revolute, universal, spherical,
                                                    rotational_actuator)
from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator


topology_name = 'spatial_fourbar'

def load():
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path,topology_name),'rb') as f:
        template = pickle.load(f)

def create():
    global template
    global numerical_code
    
    template = template_based_topology(topology_name)
    
    # Adding System Bodies
    template.add_body('crank')
    template.add_body('rocker')
    template.add_body('coupler')
    
    # Adding System Joints
    template.add_joint(revolute,'rev_crank','rbs_crank','vbs_ground')
    template.add_joint(revolute,'rev_rocker','rbs_rocker','vbs_ground')
    template.add_joint(spherical,'sph_coupler_crank','rbs_coupler','rbs_crank')
    template.add_joint(universal,'uni_coupler_rocker','rbs_coupler','rbs_rocker')
    template.add_joint_actuator(rotational_actuator,'act','jcs_rev_crank')

    # Assembling System
    template.assemble_model()
    template.draw_constraints_topology()
    template.save()
    
    # Writing Code Files
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()
    

if __name__ == '__main__':
    create()
else:
    load()