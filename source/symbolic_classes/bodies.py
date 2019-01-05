# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:57:47 2019

@author: khale
"""

import sympy as sm

from source.symbolic_classes.abstract_matrices import (reference_frame, abstract_mbs, 
                               vector, quatrenion, zero_matrix, A)


class body(reference_frame,abstract_mbs):
    
    n   = 7
    nc  = 1
    nve = 1
    
    def __init__(self,name):
        super().__init__(name)
        #print('Inside body()')
        self._key = name
        
        self.R  = vector('R_%s'%name, format_as='{R_{%s}}'%name)
        self.Rd = vector('Rd_%s'%name, format_as='{Rd_{%s}}'%name)
        
        self.P  = quatrenion('P_%s'%name, format_as='{P_{%s}}'%name)
        self.Pd = quatrenion('Pd_%s'%name, format_as='{Pd_{%s}}'%name)
        
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
        return self._key

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
    

