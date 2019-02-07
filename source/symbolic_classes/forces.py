# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:43:54 2019

@author: khaled.ghobashy
"""

import sympy as sm
from source.symbolic_classes.abstract_matrices import (vector, G, Skew)


class generic_force(object):
    
    def __init__(self,name,body_i=None,body_j=None):
        splited_name = name.split('.')
        self._id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self.prefix  = (self.prefix+'.' if self.prefix!='' else self.prefix)
        self._name   = name
                
        if body_i and body_j:
            self.body_i = body_i
            self.body_j = body_j
            self.construct()
            
    @property
    def name(self):
        return self._name
    @property
    def id_name(self):
        splited_name = self.name.split('.')
        return ''.join(splited_name[-1])
    
    def construct(self):
        self._construct_equations()
    
    @property
    def body_i(self):
        return self._body_i
    @body_i.setter
    def body_i(self,body_i):
        self._body_i = body_i
        bname = body_i.id_name
        self.Ri  = body_i.R
        self.Rdi = body_i.Rd
        self.Pi  = body_i.P
        self.Pdi = body_i.Pd
        self.Ai  = body_i.A
        self.Gi  = body_i.G 
        
        fromat_   = (self.prefix,bname,self.id_name)
        u_raw_name = '%subar_%s_%s'%fromat_
        u_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromat_
        
        self.ui_bar = vector(u_raw_name,body_i,u_frm_name)        
        self.ui     = self.ui_bar.express()
        
        F_format_i = (self.prefix,'F',bname,self.id_name)
        M_format_i = (self.prefix,'M',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format_i)
        M_raw_name, M_frm_name = self._formatter(*M_format_i)
        self.Fi = vector(F_raw_name,format_as=F_frm_name)
        self.Mi = vector(M_raw_name,format_as=M_frm_name)
    
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self,body_j):
        self._body_j = body_j
        self.Rj  = body_j.R
        self.Rdj = body_j.Rd
        self.Pj  = body_j.P
        self.Pdj = body_j.Pd
        self.Aj  = body_j.A
        
        fromat_   = (self.prefix,body_j.id_name,self.id_name)
        u_raw_name = '%subar_%s_%s'%fromat_
        u_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromat_
        
        self.uj_bar = vector(u_raw_name,body_j,u_frm_name)        
        self.uj     = self.uj_bar.express()
        
        body_j_name = self.body_j.id_name
        F_format_j = (self.prefix,'F',body_j_name,self.id_name)
        M_format_j = (self.prefix,'M',body_j_name,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format_j)
        M_raw_name, M_frm_name = self._formatter(*M_format_j)
        self.Fj = vector(F_raw_name,format_as=F_frm_name)
        self.Mj = vector(M_raw_name,format_as=M_frm_name)
    
    
    @property
    def Qi(self):
        return sm.BlockMatrix(self._equations[0])
    @property
    def Qj(self):
        return sm.BlockMatrix(self._equations[1])
    
    @property
    def arguments(self):
        return [self.Fi,self.Mi,self.Fj,self.Mj]
    
    @staticmethod
    def _formatter(*args):
        raw_name = '%s%s_%s_%s'%(*args,)
        frm_name = r'{%s{%s}^{%s}_{%s}}'%(*args,)
        return (raw_name,frm_name)
    
#    def _create_force_arguments(self):
#        
#        body_i_name = self.body_i.id_name
#        F_format_i = (self.prefix,'F',body_i_name,self.id_name)
#        M_format_i = (self.prefix,'M',body_i_name,self.id_name)
#        F_raw_name, F_frm_name = self._formatter(*F_format_i)
#        M_raw_name, M_frm_name = self._formatter(*M_format_i)
#        self.Fi = vector(F_raw_name,format_as=F_frm_name)
#        self.Mi = vector(M_raw_name,format_as=M_frm_name)
#
#        body_j_name = self.body_j.id_name
#        F_format_j = (self.prefix,'F',body_j_name,self.id_name)
#        M_format_j = (self.prefix,'M',body_j_name,self.id_name)
#        F_raw_name, F_frm_name = self._formatter(*F_format_j)
#        M_raw_name, M_frm_name = self._formatter(*M_format_j)
#        self.Fj = vector(F_raw_name,format_as=F_frm_name)
#        self.Mj = vector(M_raw_name,format_as=M_frm_name)
#    
    def _construct_equations(self):
        
        Mi_e = 2*G(self.Pi).T*(self.Mi + Skew(self.ui).T*self.Fi)
        Mj_e = 2*G(self.Pj).T*(self.Mj + Skew(self.uj).T*self.Fj)
        
        self._equations = [[[self.Fi], [Mi_e]], [[self.Fj], [Mj_e]]]
        
        
class gravity_force(generic_force):
    
    def __init__(self,body):
        pass
    
    
    
        