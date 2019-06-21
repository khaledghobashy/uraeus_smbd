# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:07:05 2019

@author: khaled.ghobashy
"""

from smbd.interfaces.scripting import standalone_topology, configuration
from smbd.numenv.python.codegen import generators
from smbd.numenv.cpp_eigen.codegen import projects

model = standalone_topology('fourbar')

model.add_body('l1')
model.add_body('l2')
model.add_body('l3')

model.add_joint.revolute('a', 'ground', 'rbs_l1')
model.add_joint.spherical('b', 'rbs_l1', 'rbs_l2')
model.add_joint.universal('c', 'rbs_l2', 'rbs_l3')
model.add_joint.revolute('d', 'rbs_l3', 'ground')

model.add_actuator.rotational_actuator('act', 'jcs_a')


model.assemble()


config = configuration('%s_cfg'%model._name, model)

config.add_point.UserInput('a')
config.add_point.UserInput('b')
config.add_point.UserInput('c')
config.add_point.UserInput('d')

config.add_vector.UserInput('x')
config.add_vector.UserInput('y')
config.add_vector.UserInput('z')

config.add_relation.Equal_to('pt1_jcs_a', ('hps_a',))
config.add_relation.Equal_to('pt1_jcs_b', ('hps_b',))
config.add_relation.Equal_to('pt1_jcs_c', ('hps_c',))
config.add_relation.Equal_to('pt1_jcs_d', ('hps_d',))

config.add_relation.Oriented('ax1_jcs_c', ('hps_b', 'hps_c'))
config.add_relation.Oriented('ax2_jcs_c', ('hps_c', 'hps_b'))

config.add_relation.Equal_to('ax1_jcs_a', ('vcs_x',))
config.add_relation.Equal_to('ax1_jcs_b', ('vcs_z',))
config.add_relation.Equal_to('ax1_jcs_d', ('vcs_y',))


config.add_scalar.UserInput('links_ro')

config.add_geometry.Cylinder_Geometry('l1', ('hps_a','hps_b','s_links_ro'))
config.assign_geometry_to_body('rbs_l1', 'gms_l1')

config.add_geometry.Cylinder_Geometry('l2', ('hps_b','hps_c','s_links_ro'))
config.assign_geometry_to_body('rbs_l2', 'gms_l2')

config.add_geometry.Cylinder_Geometry('l3', ('hps_c','hps_d','s_links_ro'))
config.assign_geometry_to_body('rbs_l3', 'gms_l3')

config.assemble()

#config_code = generators.configuration_codegen(config._config)
#config_code.write_code_file()
#
#code = generators.template_codegen(model._mbs)
#code.write_code_file()


file_path = '/home/khaledghobashy/Documents/smbd/smbd/numenv/cpp_eigen/numerics/tests'

project = projects.project_generator(model._mbs, config._config)
project.generate_project(parent_dir='', dir_name='fourbar', overwrite=True)


