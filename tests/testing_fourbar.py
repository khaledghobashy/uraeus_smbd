# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:07:05 2019

@author: khaled.ghobashy
"""

from asurt.interfaces.scripting import standalone_topology

model = standalone_topology('fourbar')

model.add_body('crank')
model.add_body('conct')
model.add_body('rockr')

model.add_joint.revolute('a', 'ground', 'rbs_crank')
model.add_joint.spherical('b', 'rbs_crank', 'rbs_conct')
model.add_joint.universal('c', 'rbs_conct', 'rbs_rockr')
model.add_joint.revolute('d', 'rbs_rockr', 'ground')

model.add_actuator.rotational_actuator('act','jcs_a')


model.assemble_model()
