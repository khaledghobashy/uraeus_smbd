# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:53:50 2019

@author: khale
"""

tempalte_script = '''
import os
import pickle

import source.symbolic_classes.joints
from source.symbolic_classes.forces import internal_force
from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator

topology_name = os.path.basename(__file__).split('.')[0]

def load():
    """
    Loading the template form the binary format in case of using it from other
    script.
    """
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path, topology_name), 'rb') as f:
        template = pickle.load(f)

def create():
    """
    Creating or Re-Creating a template if running the file directly. 
    """
    global template
    global numerical_code
    
    template = template_based_topology(topology_name)
    
    # Adding System Bodies:
    # ---------------------
    # args : body_name
    # kwargs : mirrored (defalut = False), virtual (default = False)
    # >>> template.add_body('body_name', mirrored=True)
    
    
    # Adding System Joints:
    # ---------------------
    # args : joint_type, joint_name, 1st body, 2nd body
    # kwargs : mirrored (defalut = False), virtual (default = False)
    # >>> template.add_joint(joints.spherical, 'joint_A', 'body_1', 'body_2', mirrored=True)


    # Adding System Forces:
    # ---------------------
    # args : force_type, force_name, 1st body, 2nd body
    # kwargs : mirrored (defalut = False)
    # >>> template.add_force(internal_force, 'strut', 'rbr_upper_strut', 'rbr_lower_strut', mirrored=True)

    # Visualizing the System Connectivity:
    # ------------------------------------
    template.draw_constraints_topology()


    # Assembling the System:
    # ----------------------
    template.assemble_model()
    template.save()
    
    # Generating Python Code Files of the System:
    # -------------------------------------------
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()
    

if __name__ == '__main__':
    create()
else:
    load()

''' 

def create_template_script():
    name = ''
    while not name.isidentifier():
        name = input('Please enter a valid script name.\n')
    with open('%s.py'%name, 'w') as f:
        f.write(tempalte_script)

create_template_script()
