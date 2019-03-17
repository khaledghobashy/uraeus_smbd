# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:21:51 2019

@author: khaled.ghobashy
"""
import os
import pickle

from source.symbolic_classes.joints import (revolute,spherical,cylinderical)

from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator


topology_name = 'steer'

def load():
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path,topology_name),'rb') as f:
        template = pickle.load(f)

def create():
    global template
    
    template = template_based_topology(topology_name)
    
    template.add_body('coupler')
    template.add_body('rocker',mirrored=True)
    template.add_body('chassis',virtual=True)
    
    template.add_joint(revolute,'rocker_ch','rbr_rocker','vbs_chassis',mirrored=True)
    template.add_joint(spherical,'rc_sph','rbr_rocker','rbs_coupler')
    template.add_joint(cylinderical,'rc_cyl','rbl_rocker','rbs_coupler')
    
    template.assemble_model()
    template.save()
    
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()

    
if __name__ == '__main__':
    create()
else:
    load()
