# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:18:17 2019

@author: khaled.ghobashy
"""
import os
import pickle

from source.symbolic_classes.joints import (revolute, universal, spherical,
                                                    cylinderical)
from source.symbolic_classes.forces import internal_force
from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator
from source.code_generators.python_code_generators import configuration_code_generator


topology_name = 'dwb'

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
    template.add_body('uca',mirrored=True)
    template.add_body('lca',mirrored=True)
    template.add_body('upright',mirrored=True)
    template.add_body('upper_strut',mirrored=True)
    template.add_body('lower_strut',mirrored=True)
    template.add_body('tie_rod',mirrored=True)
    template.add_body('hub',mirrored=True)
    template.add_body('steer',mirrored=True,virtual=True)
    template.add_body('chassis',virtual=True)
    
    # Adding System Joints
    template.add_joint(spherical,'uca_upright','rbr_uca','rbr_upright',mirrored=True)
    template.add_joint(spherical,'lca_upright','rbr_lca','rbr_upright',mirrored=True)
    template.add_joint(spherical,'tie_upright','rbr_tie_rod','rbr_upright',mirrored=True)
    template.add_joint(revolute,'uca_chassis','rbr_uca','vbs_chassis',mirrored=True)
    template.add_joint(revolute,'lca_chassis','rbr_lca','vbs_chassis',mirrored=True)
    template.add_joint(revolute,'hub_bearing','rbr_upright','rbr_hub',mirrored=True)
    template.add_joint(universal,'strut_chassis','rbr_upper_strut','vbs_chassis',mirrored=True)
    template.add_joint(universal,'strut_lca','rbr_lower_strut','rbr_lca',mirrored=True)
    template.add_joint(universal,'tie_steering','rbr_tie_rod','vbr_steer',mirrored=True)
    template.add_joint(cylinderical,'strut','rbr_upper_strut','rbr_lower_strut',mirrored=True)

    # Adding System Forces
    template.add_force(internal_force,'strut','rbr_upper_strut','rbr_lower_strut',mirrored=True)

    template.assemble_model()
    template.draw_constraints_topology()
    template.save()
    
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()
    
    config_code = configuration_code_generator(template.param_config)
    config_code.write_code_file()
    

if __name__ == '__main__':
    create()
else:
    load()
    


