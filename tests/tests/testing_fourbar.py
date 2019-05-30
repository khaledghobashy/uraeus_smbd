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

from smbd.interfaces.scripting import standalone_topology
from smbd.numenv.python.codegen import generators

model = standalone_topology('fourbar')

model.add_body('crank')
model.add_body('conct')
model.add_body('rockr')

model.add_joint.revolute('a', 'ground', 'rbs_crank')
model.add_joint.spherical('b', 'rbs_crank', 'rbs_conct')
model.add_joint.universal('c', 'rbs_conct', 'rbs_rockr')
model.add_joint.revolute('d', 'rbs_rockr', 'ground')

model.add_actuator.absolute_locator('abs','rbs_crank', 'x')
model.add_actuator.rotational_actuator('act', 'jcs_a')


model.assemble()
model.save()

code = generators.template_codegen(model._mbs)
code.write_code_file()
