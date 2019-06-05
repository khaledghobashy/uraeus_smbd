# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:07:05 2019

@author: khaled.ghobashy
"""

#from smbd.symbolic.symbolic_classes import bodies
#from smbd.symbolic.symbolic_classes import joints
#
#ground = bodies.ground()
#
#l1 = bodies.body('l1')
#l2 = bodies.body('l2')
#l3 = bodies.body('l3')
#
#a = joints.revolute('a', ground, l1)
#b = joints.spherical('b', l1, l2)
#c = joints.universal('c', l2, l3)
#d = joints.revolute('d', l3, ground)
#
#ground.global_frame.draw_tree()

from smbd.interfaces.scripting import standalone_topology, configuration
from smbd.numenv.python.codegen import generators
from smbd.numenv.cpp_eigen.codegen import generators as cppgen

model = standalone_topology('fourbar')

model.add_body('crank')
model.add_body('conct')
model.add_body('rockr')

model.add_joint.revolute('a', 'ground', 'rbs_crank')
model.add_joint.spherical('b', 'rbs_crank', 'rbs_conct')
model.add_joint.universal('c', 'rbs_conct', 'rbs_rockr')
model.add_joint.revolute('d', 'rbs_rockr', 'ground')

#model.add_actuator.absolute_locator('abs','rbs_crank', 'x')
#model.add_actuator.rotational_actuator('act', 'jcs_a')


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

config.assemble()

config_code = generators.configuration_codegen(config._config)
config_code.write_code_file()

code = generators.template_codegen(model._mbs)
code.write_code_file()

cppcode = cppgen.template_codegen(model._mbs)
cppcode.write_header_file()
cppcode.write_source_file()




