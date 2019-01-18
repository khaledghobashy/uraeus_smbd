# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:57:47 2019

@author: khale
"""

import sympy as sm

from source.symbolic_classes.abstract_matrices import (reference_frame, vector, 
                                                       quatrenion, zero_matrix, A)


class body(reference_frame):
    
    n   = 7
    nc  = 1
    nve = 1
    
    def __init__(self,name):
        self._name = name
        super().__init__(name)
        #print('Inside body()')
        
        splited_name = name.split('.')
        self.id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self.prefix  = (self.prefix+'.' if self.prefix!='' else self.prefix)
        
        format_ = (self.prefix,self.id_name)
        
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
        return self._name
    
    @property
    def arguments(self):
        R  = sm.Eq(self.R,sm.MutableDenseMatrix([0,0,0]))
        P  = sm.Eq(self.P,sm.MutableDenseMatrix([1,0,0,0]))
        Rd = sm.Eq(self.Rd,sm.MutableDenseMatrix([0,0,0]))
        Pd = sm.Eq(self.Pd,sm.MutableDenseMatrix([0,0,0,0]))
        return [R,P,Rd,Pd]
    @property
    def constants(self):
        return []
        


class ground(body):
    
    n   = 7
    nc  = 7
    nve = 2
    
    def __new__(cls,*args):
        name = 'ground'
        return super().__new__(cls,name)
    def __init__(self,*args):
        name = 'ground'
        super().__init__(name)
        self.P_ground = quatrenion('Pg_%s'%self.name,format_as='{Pg_{%s}}'%self.name)
        
        self.normalized_pos_equation = sm.BlockMatrix([[self.R], [self.P-self.P_ground]])
        self.normalized_vel_equation = sm.BlockMatrix([[zero_matrix(3,1)],[zero_matrix(4,1)]])
        self.normalized_acc_equation = sm.BlockMatrix([[zero_matrix(3,1)],[zero_matrix(4,1)]])
        self.normalized_jacobian = sm.BlockMatrix([[sm.Identity(3),zero_matrix(3,4)],
                                                   [zero_matrix(4,3),sm.Identity(4)]])
    
    @property
    def arguments(self):
        eq = sm.Eq(self.P_ground,sm.Matrix([1,0,0,0]))
        return super().arguments + [eq]


class virtual_body(body):
    
    n   = 0
    nc  = 0
    nve = 0
    
    def __init__(self,name):
        self._key = 'vb_%s'%name
    
    @property
    def arguments(self):
        return []

