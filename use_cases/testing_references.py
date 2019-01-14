# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:36:42 2019

@author: khaled.ghobashy
"""

from IPython.display import display

from source.symbolic_classes.abstract_matrices import global_frame, reference_frame, mbs_string
from source.symbolic_classes.bodies import body
from source.symbolic_classes.spatial_joints import revolute

global_instance = global_frame()
reference_frame.set_global_frame(global_instance)

a = body(mbs_string('a','s_0','rbl'))
b = body(mbs_string('b'))
j = revolute('J',a,b)

reference_frame.global_frame.draw_tree()
j.mi_bar.global_frame.draw_tree()

display(j.pos_level_equations)
display(j.vel_level_equations)
display(j.acc_level_equations)
display(j.jacobian_i)
display(j.jacobian_j)


