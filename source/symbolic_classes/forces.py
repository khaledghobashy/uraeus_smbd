# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 08:43:54 2019

@author: khaled.ghobashy
"""

import sympy as sm
from source.symbolic_classes.abstract_matrices import (reference_frame,
                                                       global_frame,vector, G, 
                                                       Skew, Force, Moment,
                                                       zero_matrix, matrix_symbol)
from source.symbolic_classes.spatial_joints import dummy_cylinderical


class generic_force(object):
    
    def __init__(self,name,body_i=None,body_j=None):
        splited_name = name.split('.')
        self._id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self.prefix  = (self.prefix+'.' if self.prefix!='' else self.prefix)
        self._name   = name
        self.joint = dummy_cylinderical(name,body_i,body_j)
                
        if body_i : self.body_i = body_i 
        if body_j : self.body_j = body_j
        
    def __getattr__(self,attr):
        try:
            object.__getattribute__(self,attr)
        except AttributeError:
            j = object.__getattribute__(self,'joint')
            return getattr(j,attr)
                
    @property
    def body_i(self):
        return self._body_i
    @body_i.setter
    def body_i(self,body_i):
        self.joint.body_i = body_i
        self.Gi  = body_i.G
        self._construct_force_i()
    
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self,body_j):
        self.joint.body_j = body_j
        self.Gj  = body_j.G
        self._construct_force_j()
        
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [self.Mi_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[self.Fj], [self.Mj_e]])
    
    @property
    def arguments(self):
        config_args = self.joint.arguments
        forces_inputs = [self.Fi,self.Mi,self.Fj,self.Mj]
        args = config_args + forces_inputs
        return args
    
    @property
    def constants(self):
        return self.joint.constants
        
    
    def _construct_force_i(self):
        bname = self.body_i.id_name
        F_format = (self.prefix,'F',bname,self.id_name)
        M_format = (self.prefix,'M',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        M_raw_name, M_frm_name = self._formatter(*M_format)
        self.Fi   = vector(F_raw_name,format_as=F_frm_name)
        self.Mi   = vector(M_raw_name,format_as=M_frm_name)
        self.Mi_e = 2*G(self.Pi).T*(self.Mi + Skew(self.ui).T*self.Fi)
    
    def _construct_force_j(self):
        bname = self.body_j.id_name
        F_format = (self.prefix,'F',bname,self.id_name)
        M_format = (self.prefix,'M',bname,self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        M_raw_name, M_frm_name = self._formatter(*M_format)
        self.Fj   = vector(F_raw_name,format_as=F_frm_name)
        self.Mj   = vector(M_raw_name,format_as=M_frm_name)
        self.Mj_e = 2*G(self.Pj).T*(self.Mj + Skew(self.uj).T*self.Fj)
    
    @staticmethod
    def _formatter(*args):
        raw_name = '%s%s_%s_%s'%(*args,)
        frm_name = r'{%s{%s}^{%s}_{%s}}'%(*args,)
        return (raw_name,frm_name)
        
class gravity_force(generic_force):
    
    def __init__(self,name,body,*args):
#        name = 'G'
        super().__init__(name,body)
    
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [zero_matrix(4,1)]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3,1)], [zero_matrix(4,1)]])


class centrifugal_force(generic_force):
    
    def __init__(self,name,body,*args):
#        name = 'C'
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
        
        virtual_joint = dummy_cylinderical(name,body_i,body_j)
        self.joint = virtual_joint
        
        format_ = (self.prefix,self.id_name)
        self.LF = matrix_symbol('%s%s_FL'%format_,1,1)

        self.Fs = sm.Function('Fs_%s'%name)#('dx')
        self.Fd = sm.Function('Fd_%s'%name)#('dv')
        self.Fa = sm.Function('Fa_%s'%name)#('dv')
        
        self.Ms = sm.Function('Ms_%s'%name)#('dx')
        self.Md = sm.Function('Md_%s'%name)#('dv')
        self.Ma = sm.Function('Ma_%s'%name)#('dv')
                
    @property
    def Qi(self):
        distance    = sm.sqrt(self.joint.dij.T*self.joint.dij)
        unit_vector = self.joint.dij/distance
        
        defflection = self.LF - distance
        velocity    = unit_vector.T*self.joint.dijd
             
        self.Fi = unit_vector*(self.Fs(defflection) - self.Fd(velocity))
        Mi_e = 2*G(self.Pi).T*(self.Mi + Skew(self.ui).T*self.Fi)
        
        force_vector = sm.BlockMatrix([[self.Fi], [Mi_e]])
        return force_vector
    
    @property
    def Qj(self):
        self.Fj = -self.Fi
        Mj_e = 2*G(self.Pj).T*(self.Mj + Skew(self.uj).T*self.Fj)
        force_vector = sm.BlockMatrix([[self.Fj], [Mj_e]])
        return force_vector

    
    
        