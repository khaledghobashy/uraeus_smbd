# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:57:47 2019

@author: khale
"""

import sympy as sm

from source.symbolic_classes.abstract_matrices import (reference_frame, abstract_mbs, 
                               vector, quatrenion, zero_matrix, A, mbs_string)


class body(reference_frame,abstract_mbs):
    
    n   = 7
    nc  = 1
    nve = 1
    
    def __init__(self,sname):
        super().__init__(sname)
        #print('Inside body()')
        self._key = sname
        
        prefix, id_, name = sname
        
        format_ = (prefix,id_+name)
        self.R  = vector('%sR_%s'%format_, format_as='{%sR_{%s}}'%format_)
        self.Rd = vector('%sRd_%s'%format_, format_as='{%s\dot{R}_{%s}}'%format_)
        
        self.P  = quatrenion('%sP_%s'%format_, format_as='{%sP_{%s}}'%format_)
        self.Pd = quatrenion('%sPd_%s'%format_, format_as='{%s\dot{P}_{%s}}'%format_)
        
        #print('Generating DCM')
        self.A = A(self.P)
                
        #print('Generating Normalized Equations')
        self.normalized_pos_equation = sm.sqrt(self.P.T*self.P)-sm.Identity(1)
        self.normalized_vel_equation = zero_matrix(1,1)
        self.normalized_acc_equation = 2*sm.sqrt(self.Pd.T*self.Pd)
        self.normalized_jacobian = [zero_matrix(1,3), 2*self.P.T]
        
        #print('Exiting Body \n')
    
    @property
    def name(self):
        return self._key.id_ + self._key.name
        
    
#    def rename(self,name,prefix=''):
#        super().rename(prefix+name)
#        R_fromated_name  = '{%s}{R_{%s}}'%(prefix,name)
#        Rd_fromated_name = '{%s}{Rd_{%s}}'%(prefix,name)
#        self.R.rename(name,R_fromated_name)
#        self.Rd.rename(name,Rd_fromated_name)
#        P_fromated_name  = '{%s}{P_{%s}}'%(prefix,name)
#        Pd_fromated_name = '{%s}{Pd_{%s}}'%(prefix,name)
#        self.P.rename(name,P_fromated_name)
#        self.Pd.rename(name,Pd_fromated_name)


class ground(body):
    
    n   = 7
    nc  = 7
    nve = 2
    
    def __new__(cls,*args):
        name = 'ground'
        return super().__new__(cls,name)
    def __init__(self,*args):
        super().__init__('ground')
        self._key = 'ground'
        self.P_ground = quatrenion('Pg_%s'%self.name,format_as='{Pg_{%s}}'%self.name)
        
        self.normalized_pos_equation = sm.BlockMatrix([[self.R], [self.P-self.P_ground]])
        self.normalized_vel_equation = sm.BlockMatrix([[zero_matrix(3,1)],[zero_matrix(4,1)]])
        self.normalized_acc_equation = sm.BlockMatrix([[zero_matrix(3,1)],[zero_matrix(4,1)]])
        self.normalized_jacobian = sm.BlockMatrix([[sm.Identity(3),zero_matrix(3,4)],
                                                   [zero_matrix(4,3),sm.Identity(4)]])
    
    def numerical_arguments(self):
        eq = sm.Eq(self.P_ground,sm.Matrix([1,0,0,0]))
        return super().numerical_arguments() + [eq]


class virtual_body(body):
    
    n   = 7
    nc  = 7
    nve = 2
    
    def __init__(self,name):
        self._key = 'vb_%s'%name
    
    def numerical_arguments(self):
        return []

