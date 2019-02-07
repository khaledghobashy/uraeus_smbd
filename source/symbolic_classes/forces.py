# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:43:54 2019

@author: khaled.ghobashy
"""

import sympy as sm
from source.symbolic_classes.abstract_matrices import (reference_frame,
                                                       global_frame,vector, G, 
                                                       Skew, 
                                                       zero_matrix, matrix_symbol)
from source.symbolic_classes.spatial_joints import cylinderical


class generic_force(object):
    
    def __init__(self,name,body_i=None,body_j=None):
        splited_name = name.split('.')
        self._id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self.prefix  = (self.prefix+'.' if self.prefix!='' else self.prefix)
        self._name   = name
                
        if body_i : self.body_i = body_i 
        if body_j : self.body_j = body_j
            
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
        self.Ri  = body_i.R
        self.Rdi = body_i.Rd
        self.Pi  = body_i.P
        self.Pdi = body_i.Pd
        self.Ai  = body_i.A
        self.Gi  = body_i.G
        
        bname = body_i.id_name
        
        fromat_   = (self.prefix,bname,self.id_name)
        u_raw_name = '%subar_%s_%s'%fromat_
        u_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromat_
        
        self.ui_bar = vector(u_raw_name,body_i,u_frm_name)        
        self.ui     = self.ui_bar.express()
        
        F_format = (self.prefix,'F',bname,self.id_name)
        M_format = (self.prefix,'M',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        M_raw_name, M_frm_name = self._formatter(*M_format)
        
        self.Fi   = vector(F_raw_name,format_as=F_frm_name)
        self.Mi   = vector(M_raw_name,format_as=M_frm_name)
        self.Mi_e = 2*G(self.Pi).T*(self.Mi + Skew(self.ui).T*self.Fi)
    
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
        
        bname = body_j.id_name

        fromat_   = (self.prefix,bname,self.id_name)
        u_raw_name = '%subar_%s_%s'%fromat_
        u_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromat_
        
        self.uj_bar = vector(u_raw_name,body_j,u_frm_name)        
        self.uj     = self.uj_bar.express()
        
        F_format = (self.prefix,'F',bname,self.id_name)
        M_format = (self.prefix,'M',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        M_raw_name, M_frm_name = self._formatter(*M_format)
        
        self.Fj   = vector(F_raw_name,format_as=F_frm_name)
        self.Mj   = vector(M_raw_name,format_as=M_frm_name)
        self.Mj_e = 2*G(self.Pj).T*(self.Mj + Skew(self.uj).T*self.Fj)
    
    
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [self.Mi_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[self.Fj], [self.Mj_e]])
    
    @property
    def arguments(self):
        return [self.Fi,self.Mi,self.Fj,self.Mj]
    
    @staticmethod
    def _formatter(*args):
        raw_name = '%s%s_%s_%s'%(*args,)
        frm_name = r'{%s{%s}^{%s}_{%s}}'%(*args,)
        return (raw_name,frm_name)
    
    def _construct_equations(self):
        
        Mi_e = 2*G(self.Pi).T*(self.Mi + Skew(self.ui).T*self.Fi)
        Mj_e = 2*G(self.Pj).T*(self.Mj + Skew(self.uj).T*self.Fj)
        
        self._equations = [[[self.Fi], [Mi_e]], [[self.Fj], [Mj_e]]]
        
        
class gravity_force(generic_force):
    
    def __init__(self,body):
        name = 'G'
        super().__init__(name,body)
    
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [zero_matrix(4,1)]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3,1)], [zero_matrix(4,1)]])


class centrifugal_force(generic_force):
    
    def __init__(self,body):
        name = 'C'
        super().__init__(name,body)
    
    @property
    def Qi(self):
        Mi_e = 8*G(self.Pdi).T*self.body_i.Jbar*G(self.Pdi)*self.Pi
        return sm.BlockMatrix([[zero_matrix(3,1)], [Mi_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3,1)], [zero_matrix(4,1)]])
        

class internal_force(generic_force):
    
    def __init__(self,name,body_i=None,body_j=None):
        super().__init__(name,body_i,body_j)
        
        reference_frame.set_global_frame(body_i.global_frame)
        
        virtual_joint = cylinderical(name,body_i,body_j)
        self.joint = virtual_joint
        
        format_ = (self.prefix,self.id_name)
        self.LF = matrix_symbol('%s%s_FL'%format_,1,1)
        
        self.Kt = sm.symbols('%s%s_Kt'%format_)
        self.Ct = sm.symbols('%s%s_Ct'%format_)
        self.Kr = sm.symbols('%s%s_Kr'%format_)
        self.Cr = sm.symbols('%s%s_Cr'%format_)
        
    @property
    def Qi(self):
        distance    = sm.sqrt(self.joint.dij.T*self.joint.dij)
        unit_vector = self.joint.dij/distance
        
        defflection = self.LF - distance
        velocity    = unit_vector.T*self.joint.dijd
             
        self.Fi = unit_vector*(self.Kt*defflection - self.Ct*velocity)
        Mi_e = 2*G(self.Pi).T*(self.Mi + Skew(self.ui).T*self.Fi)
        return sm.BlockMatrix([[self.Fi], [Mi_e]])
    @property
    def Qj(self):
        self.Fj = -self.Fi
        Mj_e = 2*G(self.Pj).T*(self.Mj + Skew(self.uj).T*self.Fj)
        return sm.BlockMatrix([[self.Fj], [Mj_e]])

    
    
        