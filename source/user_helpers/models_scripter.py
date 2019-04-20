# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:01:19 2019

@author: khaled.ghobashy
"""

symbolic_template_script = '''
import os
import pickle

import source.symbolic_classes.joints as joints
import source.symbolic_classes.forces as forces
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
    Creating/Updating a template if running the script directly. 
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
    # >>> template.add_force(forces.internal_force, 'strut', 'rbr_upper_strut', 'rbr_lower_strut', mirrored=True)

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


symbolic_configuration_script = '''
from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  cylinder_geometry)

import use_cases.parametric_templates.templates.mass_spring as model

config_name = os.path.basename(__file__).split('.')[0]

def main():
    
    global config
    
    config = parametric_configuration(config_name,model.template)
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

'''

#def create_template_script():
#    name = ''
#    while not name.isidentifier():
#        name = input('Please enter a valid script name.\n')
#    with open('%s.py'%name, 'w') as f:
#        f.write(tempalte_script)
#
#create_template_script()