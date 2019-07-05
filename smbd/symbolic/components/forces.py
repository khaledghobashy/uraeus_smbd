# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:43:54 2019

@author: khaled.ghobashy
"""

# Standard library imports
import itertools

# 3rd parties libraries imports
import sympy as sm

# Local application imports
from .matrices import (vector, G, Skew, zero_matrix,
                       matrix_function_constructor, Force)
from .helpers import body_setter, name_setter
from .joints import dummy_cylinderical


class abstract_force(object):
    
    n   = 0
    nc  = 0
    nve = 0
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self, name, body_i=None, body_j=None):
        name_setter(self, name)
        if body_i : self.body_i = body_i 
        if body_j : self.body_j = body_j
        
        self._create_arguments()
        self._create_local_equalities()
        
    @property
    def name(self):
        return self._name
    @property
    def id_name(self):
        splited_name = self.name.split('.')
        return ''.join(splited_name[-1])

    @property
    def body_i(self):
        return self._body_i
    @body_i.setter
    def body_i(self, body_i):
        body_setter(self, body_i, 'i')
        self.Gi  = body_i.G
        self._construct_force_i()
    
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self, body_j):
        body_setter(self, body_j, 'j')
        self.Gj  = body_j.G
        self._construct_force_j()
        
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [self.Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[self.Fj], [self.Tj_e]])
    
    @property
    def arguments_symbols(self):
        return self._arguments
    @property
    def runtime_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return self._sym_constants
    @property
    def constants_numeric_expr(self):
        return []
    @property
    def constants_symbols(self):
        constants_expr = itertools.chain(self.constants_symbolic_expr,
                                         self.constants_numeric_expr)
        return [expr.lhs for expr in constants_expr]
    
    
    def _create_def_axis(self, i):
        format_ = (self.prefix, i, self.id_name)
        v = vector('%sax%s_%s'%format_)
        setattr(self, 'axis_%s'%i, v)
    
    def _create_def_loc(self, i):
        format_ = (self.prefix, i, self.id_name)
        u = vector('%spt%s_%s'%format_)
        setattr(self, 'loc_%s'%i, u)
        
    def _create_arguments(self):
        
        for i in range(self.def_axis):
            self._create_def_axis(i+1)
        for i in range(self.def_locs):
            self._create_def_loc(i+1)
            
        l = []
        for i in range(self.def_axis):
            n = i+1
            v = getattr(self, 'axis_%s'%n)
            l.append(v)
        for i in range(self.def_locs):
            n = i+1
            u = getattr(self, 'loc_%s'%n)
            l.append(u)
        self._arguments = l


    def _create_local_equalities(self):
        self._sym_constants = []
        
        if self.def_axis == 0:
            axis_equalities = []
        
        elif self.def_axis == 1:
            axis   = self.axis_1
            axis_bar  = axis.express(self.body_i)
            axis_bar_eq = sm.Eq(self.vi_bar, axis_bar/sm.sqrt(axis_bar.T*axis_bar))
            axis_equalities = [axis_bar_eq]
        else: 
            raise NotImplementedError
        self._sym_constants += axis_equalities
        
        if self.def_locs == 0:
            location_equalities = []

        elif self.def_locs == 1:
            loc  = self.loc_1
            ui_bar_eq = sm.Eq(self.ui_bar, loc.express(self.body_i) - self.Ri.express(self.body_i))
            location_equalities = [ui_bar_eq]
        else: 
            raise NotImplementedError
        self._sym_constants += location_equalities

        
    def _construct_force_i(self):
        bname = self.body_i.id_name
        F_format = (self.prefix, 'F', bname, self.id_name)
        T_format = (self.prefix, 'T', bname, self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        T_raw_name, T_frm_name = self._formatter(*T_format)
        self.Fi   = vector(F_raw_name, format_as=F_frm_name)
        self.Ti   = vector(T_raw_name, format_as=T_frm_name)
        self.Ti_e = 2*G(self.Pi).T * (self.Ti + Skew(self.ui).T*self.Fi)
    
    def _construct_force_j(self):
        bname = self.body_j.id_name
        F_format = (self.prefix, 'F', bname, self.id_name)
        T_format = (self.prefix, 'T', bname, self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        T_raw_name, T_frm_name = self._formatter(*T_format)
        self.Fj   = vector(F_raw_name, format_as=F_frm_name)
        self.Tj   = vector(T_raw_name, format_as=T_frm_name)
        self.Tj_e = 2*G(self.Pj).T * (self.Tj + Skew(self.uj).T*self.Fj)
    
    @staticmethod
    def _formatter(*args):
        raw_name = '%s%s_%s_%s'%(*args,)
        frm_name = r'{%s{%s}^{%s}_{%s}}'%(*args,)
        return (raw_name, frm_name)

###############################################################################
###############################################################################

class gravity_force(abstract_force):
    
    def_axis = 0
    def_locs = 0
    
    def __init__(self, name, body, *args):
        name = 'gravity'
        super().__init__(name, body, *args)
    
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [zero_matrix(4, 1)]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return []
    @property
    def constants_numeric_expr(self):
        gravity = sm.Eq(self.Fi, self.body_i.m*sm.Matrix([0, 0, -9.81e3]))
        return [gravity]

###############################################################################
###############################################################################

class centrifugal_force(abstract_force):
    
    def_axis = 0
    def_locs = 0
    
    def __init__(self, name, body ,*args):
        name = 'centrifugal'
        super().__init__(name, body, *args)
    
    @property
    def Qi(self):
        Ti_e = 8*G(self.Pdi).T*self.body_i.Jbar*G(self.Pdi)*self.Pi
        return sm.BlockMatrix([[zero_matrix(3,1)], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3,1)], [zero_matrix(4,1)]])
    
    @property
    def arguments_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return []

###############################################################################
###############################################################################

class generic_force(abstract_force):
    
    def_axis = 0
    def_locs = 1
    
    def __init__(self, name, body, *args):
        super().__init__(name, body, *args)
        
        self.Fi = matrix_function_constructor('UF_%s_F'%name, (3, 1))
        self.Ti = matrix_function_constructor('UF_%s_T'%name, (3, 1))
        
        self._Fi_alias = sm.Function('UF_%s_F'%name)
        self._Ti_alias = sm.Function('UF_%s_T'%name)
        
    @property
    def Qi(self):
        Ti_e = 2*G(self.Pi).T * (self.Ti() + Skew(self.ui).T*self.Fi())
        return sm.BlockMatrix([[self.Fi()], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        forces_args = [self._Fi_alias, self._Ti_alias, self.loc_1]
        return forces_args

###############################################################################
###############################################################################

class force(abstract_force):
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self, name, body, *args):
        super().__init__(name, body, *args)
        self.t  = sm.symbols('t', real=True)
        self.Fi = sm.Function('UF_%s'%name)

    
    @property
    def Qi(self):
        force = self.Fi(self.t) * self.vi
        Ti_e = 2*G(self.Pi).T * (self.Ti + Skew(self.ui).T*force)
        return sm.BlockMatrix([[force], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        forces_args = [self.Fi, self.axis_1, self.loc_1]
        return forces_args
    @property
    def constants_numeric_expr(self):
        eq1 = sm.Eq(self.Ti, zero_matrix(3, 1), evaluate=False)
        return [eq1]

###############################################################################
###############################################################################

class torque(abstract_force):
    
    def_axis = 1
    def_locs = 0
    
    def __init__(self, name, body, *args):
        super().__init__(name, body, *args)
        self.t  = sm.symbols('t', real=True)
        self.Ti = sm.Function('UF_%s'%name)

    @property
    def Qi(self):
        torque = self.Ti(self.t) * self.vi
        Ti_e = 2*G(self.Pi).T * torque 
        return sm.BlockMatrix([[self.Fi], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        forces_args = [self.Ti, self.axis_1]
        return forces_args
    @property
    def constants_numeric_expr(self):
        eq1 = sm.Eq(self.Fi, zero_matrix(3, 1), evaluate=False)
        return [eq1]

###############################################################################
###############################################################################

class internal_force(abstract_force):
    
    def_axis = 0
    def_locs = 0
    
    def __init__(self, name, body_i=None, body_j=None):
        super().__init__(name, body_i, body_j)
        self.joint = dummy_cylinderical(name, body_i, body_j)
        format_ = (self.prefix, self.id_name)
        self.LF = sm.symbols('%s%s_FL'%format_, real=True)

        self.Fs = sm.Function('UF_%s_Fs'%name)#, commutative=True)
        self.Fd = sm.Function('UF_%s_Fd'%name, real=True)#, commutative=True)
        self.Fa = sm.Function('UF_%s_Fa'%name)#, commutative=True)
        
        self.Ts = sm.Function('UF_%s_Ts'%name)#, commutative=True)
        self.Td = sm.Function('UF_%s_Td'%name)#, commutative=True)
        self.Ta = sm.Function('UF_%s_Ta'%name)#, commutative=True)
        
        self._construct_force_vector()
                
    @property
    def Qi(self):
        return self._Qi
    
    @property
    def Qj(self):
        return self._Qj
    
    @property
    def arguments_symbols(self):
        configuration_args = self.joint.arguments_symbols[1:3]
        forces_args = [self.Fs, self.Fd, self.LF]
        return configuration_args + forces_args
    @property
    def constants_symbolic_expr(self):
        return self.joint.constants_symbolic_expr[2:4]
    @property
    def constants_numeric_expr(self):
        eq1 = sm.Eq(self.Ti, zero_matrix(3, 1), evaluate=False)
        eq2 = sm.Eq(self.Tj, zero_matrix(3, 1), evaluate=False)
        return [eq1, eq2]
    
    
    def _construct_force_vector(self):
        
        dij = self.joint.dij
        distance    = sm.sqrt(dij.T*dij)
        unit_vector = dij/distance
        
        defflection = self.LF - distance[0,0]
        velocity    = (unit_vector.T*self.joint.dijd)
        
        total_force = self.Fs(defflection) - self.Fd(velocity)

        self.Fi = total_force * unit_vector
        Ti_e = 2*G(self.Pi).T * (self.Ti + Skew(self.ui).T*self.Fi)
        
        self._Qi = sm.BlockMatrix([[self.Fi], [Ti_e]])
        
        self.Fj = -self.Fi
        Tj_e = 2*G(self.Pj).T * (self.Tj + Skew(self.uj).T*self.Fj)
        self._Qj = sm.BlockMatrix([[self.Fj], [Tj_e]])
        
###############################################################################
###############################################################################

       


        