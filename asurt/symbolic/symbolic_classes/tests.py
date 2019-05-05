#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:41:26 2019

@author: khaledghobashy
"""

from asurt.symbolic.symbolic_classes import matrices
from asurt.symbolic.symbolic_classes import bodies
from asurt.symbolic.symbolic_classes import joints
from asurt.symbolic.symbolic_classes import forces

frame = matrices.global_frame('global')

a = bodies.body('a')
b = bodies.body('b')

j = joints.revolute('j', a, b)

f = forces.internal_force('f', a, b)



