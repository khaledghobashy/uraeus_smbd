# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:11:10 2019

@author: khaled ghobashy
"""

# 3rd parties library imports
import sympy as sm

# Local application imports
from .algebraic_constraints import (abstract_joint, abstract_actuator,
                                    absolute_actuator, joint_actuator, 
                                    joint_constructor)
                                    
from .constraints_equations import (spehrical_constraint, 
                                    dot_product_1, dot_product_2,
                                    angle_constraint, coordinate_constraint)


###############################################################################

class fixed(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_1('i', 'j')]
    
class fixed_orientation(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 0
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_1('i', 'j')]


class spherical(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [spehrical_constraint()]


class revolute(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i', 'k'),
                        dot_product_1('j', 'k')]


class cylinderical(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_2('i'),
                        dot_product_2('j')]
    

class translational(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_2('i'),
                        dot_product_2('j'),
                        dot_product_1('i', 'j')]


class universal(abstract_joint, metaclass=joint_constructor):
    def_axis = 2
    def_locs = 1
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i', 'i')]
    

class tripod(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_1('i', 'j'),
                        dot_product_2('i'),
                        dot_product_2('j')]


class rotational_actuator(joint_actuator, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 0
    vector_equations = [angle_constraint()]
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._pos_level_equations[0]])



class translational_actuator(joint_actuator, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_2('k')]
    

class absolute_locator(absolute_actuator, metaclass=joint_constructor):
    def_axis = 0
    def_locs = 2
    vector_equations = [coordinate_constraint()]


class absolute_rotator(abstract_actuator, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 0
    vector_equations = [angle_constraint()]
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._pos_level_equations[0]])


class inline(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_2('i'),
                        dot_product_2('j')]


class dummy_cylinderical(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 2
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_2('i'),
                        dot_product_2('j')]
    
