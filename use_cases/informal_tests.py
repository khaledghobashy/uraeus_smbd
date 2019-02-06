# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:38:17 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import global_frame, reference_frame
from source.symbolic_classes.bodies import body
from source.symbolic_classes.spatial_joints import (revolute, universal, spherical,
                                                    cylinderical)

global_instance = global_frame('test')
reference_frame.set_global_frame(global_instance)

a = body('a')
b = body('b')

j = revolute('J',a,b)
j._create_reactions_args()
