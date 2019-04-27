# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:43:54 2019

@author: khaled.ghobashy
"""

import itertools
import sympy as sm

from .abstract_matrices import vector, G, Skew, zero_matrix, matrix_symbol
from .helper_funcs import body_setter, name_setter
from .joints import dummy_cylinderical


class generic_force(object):
    
    n   = 0
    nc  = 0
    nve = 0
    
    def __init__(self,name,body_i=None,body_j=None):
        name_setter(self,name)
        if body_i : self.body_i = body_i 
        if body_j : self.body_j = body_j
        
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
    def body_i(self,body_i):
        body_setter(self,body_i,'i')
        self.Gi  = body_i.G
        self._construct_force_i()
    
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self,body_j):
        body_setter(self,body_j,'j')
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
        config_args = self.joint.arguments_symbols
        forces_inputs = [self.Fi,self.Ti,self.Fj,self.Tj]
        args = config_args + forces_inputs
        return args
    @property
    def runtime_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return self.joint.constants_symbolic_expr
    @property
    def constants_numeric_expr(self):
        return []
    @property
    def constants_symbols(self):
        constants_expr = itertools.chain(self.constants_symbolic_expr,
                                         self.constants_numeric_expr)
        return [expr.lhs for expr in constants_expr]


        
    def _construct_force_i(self):
        bname = self.body_i.id_name
        F_format = (self.prefix,'F',bname,self.id_name)
        T_format = (self.prefix,'T',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        T_raw_name, T_frm_name = self._formatter(*T_format)
        self.Fi   = vector(F_raw_name,format_as=F_frm_name)
        self.Ti   = vector(T_raw_name,format_as=T_frm_name)
        self.Ti_e = 2*G(self.Pi).T*(self.Ti + Skew(self.ui).T*self.Fi)
    
    def _construct_force_j(self):
        bname = self.body_j.id_name
        F_format = (self.prefix,'F',bname,self.id_name)
        T_format = (self.prefix,'T',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        T_raw_name, T_frm_name = self._formatter(*T_format)
        self.Fj   = vector(F_raw_name,format_as=F_frm_name)
        self.Tj   = vector(T_raw_name,format_as=T_frm_name)
        self.Tj_e = 2*G(self.Pj).T*(self.Tj + Skew(self.uj).T*self.Fj)
    
    @staticmethod
    def _formatter(*args):
        raw_name = '%s%s_%s_%s'%(*args,)
        frm_name = r'{%s{%s}^{%s}_{%s}}'%(*args,)
        return (raw_name,frm_name)

###############################################################################
###############################################################################

class gravity_force(generic_force):
    
    def __init__(self,name,body,*args):
        name = 'gravity'
        super().__init__(name,body,*args)
    
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [zero_matrix(4,1)]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3,1)], [zero_matrix(4,1)]])
    
    @property
    def arguments_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return []
    @property
    def constants_numeric_expr(self):
        gravity = sm.Eq(self.Fi,self.body_i.m*sm.Matrix([0,0,-9.81e3]))
        return [gravity]

###############################################################################
###############################################################################

class centrifugal_force(generic_force):
    
    def __init__(self,name,body,*args):
        name = 'centrifugal'
        super().__init__(name,body,*args)
    
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

class internal_force(generic_force):
    
    def __init__(self,name,body_i=None,body_j=None):
        super().__init__(name,body_i,body_j)
        self.joint = dummy_cylinderical(name,body_i,body_j)
        format_ = (self.prefix,self.id_name)
        self.LF = matrix_symbol('%s%s_FL'%format_,1,1)

        self.Fs = sm.Function('Fs_%s'%name)#('dx')
        self.Fd = sm.Function('Fd_%s'%name)#('dv')
        self.Fa = sm.Function('Fa_%s'%name)#('dv')
        
        self.Ts = sm.Function('Ts_%s'%name)
        self.Td = sm.Function('Td_%s'%name)
        self.Ta = sm.Function('Ta_%s'%name)
                
    @property
    def Qi(self):
        dij = self.joint.dij
        distance    = sm.sqrt(dij.T*dij)
        unit_vector = dij/distance
        
        defflection = self.LF - distance
        velocity    = unit_vector.T*self.joint.dijd
             
        self.Fi = unit_vector*(self.Fs(defflection) - self.Fd(velocity))
        Ti_e = 2*G(self.Pi).T*(self.Ti + Skew(self.ui).T*self.Fi)
        
        force_vector = sm.BlockMatrix([[self.Fi], [Ti_e]])
        return force_vector
    
    @property
    def Qj(self):
        self.Fj = -self.Fi
        Tj_e = 2*G(self.Pj).T*(self.Tj + Skew(self.uj).T*self.Fj)
        force_vector = sm.BlockMatrix([[self.Fj], [Tj_e]])
        return force_vector
    
    @property
    def arguments_symbols(self):
        configuration_args = self.joint.arguments_symbols[1:3]
        forces_args = [self.Fs,self.Fd,self.LF,self.Ti,self.Tj]
        return configuration_args + forces_args
    @property
    def constants_symbolic_expr(self):
        return self.joint.constants_symbolic_expr[2:4]

###############################################################################
###############################################################################
    
        