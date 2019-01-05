# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:11:10 2019

@author: khale
"""

import sympy as sm


from source.symbolic_classes.abstract_matrices import (reference_frame, vector,
                                                       zero_matrix)

from source.symbolic_classes.algebraic_constraints import (algebraic_constraints,
                                                           joint_constructor,
                                                           joint_actuator,
                                   spehrical_constraint, dot_product_1, dot_product_2,
                                   angle_constraint, coordinate_constraint,
                                   absolute_actuator)


class fixed(algebraic_constraints,metaclass=joint_constructor):
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i','k'),
                        dot_product_1('j','k'),
                        dot_product_1('i','j')]


class spherical(algebraic_constraints,metaclass=joint_constructor):
    vector_equations = [spehrical_constraint()]

class revolute(algebraic_constraints,metaclass=joint_constructor):
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i','k'),
                        dot_product_1('j','k')]

class cylinderical(algebraic_constraints,metaclass=joint_constructor):
    vector_equations = [dot_product_1('i','k'),
                        dot_product_1('j','k'),
                        dot_product_2('i'),
                        dot_product_2('j')]
    
class translational(algebraic_constraints,metaclass=joint_constructor):
    vector_equations = [dot_product_1('i','k'),
                        dot_product_1('j','k'),
                        dot_product_2('i'),
                        dot_product_2('j'),
                        dot_product_1('i','j')]

class universal(algebraic_constraints,metaclass=joint_constructor):
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i','i')]
    
    def configuration_constants(self):
        loc  = vector('pt_%s'%self.name)
        axis_1 = vector('ax_%s'%self.name)
        axis_2 = vector('ax2_%s'%self.name)

        ui_bar_eq = sm.Eq(self.ui_bar, loc.express(self.body_i) - self.Ri.express(self.body_i))
        uj_bar_eq = sm.Eq(self.uj_bar, loc.express(self.body_j) - self.Rj.express(self.body_j))

        marker_1 = reference_frame('M1_%s'%self.name,format_as=r'{{M1}_{%s}}'%self.name)
        marker_1.orient_along(axis_1)
        
        marker_2 = reference_frame('M2_%s'%self.name,format_as=r'{{M2}_{%s}}'%self.name)
        marker_2.orient_along(axis_2,marker_1.A[:,1])

        mi_bar      = marker_1.express(self.body_i)
        mi_bar_eq   = sm.Eq(self.mi_bar.A, mi_bar)

        mj_bar      = marker_2.express(self.body_j)
        mj_bar_eq   = sm.Eq(self.mj_bar.A, mj_bar)

        assignments = [ui_bar_eq,uj_bar_eq,mi_bar_eq,mj_bar_eq]
        return assignments
    
    def numerical_arguments(self):
        axis_2 = vector('ax2_%s'%self.name)
        axis_2 = sm.Eq(axis_2,sm.MutableDenseMatrix([0,0,1]))
        return super().numerical_arguments() + [axis_2]
        


class rotational_actuator(joint_actuator,metaclass=joint_constructor):
    vector_equations = [angle_constraint()]
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix(self._pos_level_equations)


class absolute_locator(absolute_actuator,metaclass=joint_constructor):
    vector_equations = [coordinate_constraint()]


    
