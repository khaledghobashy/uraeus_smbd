#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:41:26 2019

@author: khaledghobashy
"""

from asurt.symbolic.symbolic_classes import matrices
from asurt.symbolic.symbolic_classes import algebraic_constraints as cons
from asurt.symbolic.symbolic_classes import bodies
from asurt.symbolic.symbolic_classes import joints

frame = matrices.global_frame('global')
#matrices.reference_frame.global_frame = frame
#cons.reference_frame.global_frame = frame
#bodies.reference_frame.global_frame = frame
#
a = bodies.body('a')
b = bodies.body('b')

j = joints.revolute('j')
j.body_i = a
j.body_j = b
j.construct()

