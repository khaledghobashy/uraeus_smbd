# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:36:42 2019

@author: khaled.ghobashy
"""

from IPython.display import display
import sympy as sm

from source.symbolic_classes.abstract_matrices import global_frame, reference_frame
from source.symbolic_classes.bodies import body
from source.symbolic_classes.spatial_joints import cylinderical,revolute, universal, rotational_actuator

global_instance = global_frame()
reference_frame.set_global_frame(global_instance)

a = body('s0.a')
b = body('s1.b')
j = cylinderical('s0.J',a,b)
j_act = rotational_actuator('s0.jact',j)

reference_frame.global_frame.draw_tree()
j.mi_bar.global_frame.draw_tree()

display(j.pos_level_equations)
display(j.vel_level_equations)
display(j.acc_level_equations)
display(j.jacobian_i)
display(j.jacobian_j)
display(sm.MutableSparseMatrix([j.jacobian_i.blocks,j.jacobian_j.blocks]))

display(j_act.pos_level_equations)
display(j_act.vel_level_equations)
display(j_act.acc_level_equations)
display(j_act.jacobian_i)
display(j_act.jacobian_j)

