#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:55:14 2019

@author: khaledghobashy
"""

import numpy as np

from .spatial_alg import skew_matrix

def dcm2ep(dcm):
    ''' 
    extracting euler parameters from a transformation matrix
    Note: this is not fully developed.
    ===========================================================================
    inputs  : 
        A   : The transofrmation matrix
    ===========================================================================
    outputs : 
        p   : set containing the four parameters
    ===========================================================================
    '''
    trA = dcm.trace()
    
    e0s = (trA+1)/4
    e1s = (2*dcm[0,0]-trA+1)/4
    e2s = (2*dcm[1,1]-trA+1)/4
    e3s = (2*dcm[2,2]-trA+1)/4
    
    emax = max([e0s,e1s,e2s,e2s,e3s])
    
    # Case 1 : e0 != zero
    if e0s == emax:
        e0 = abs(np.sqrt(e0s))
        e1 = (dcm[2,1]-dcm[1,2])/(4*e0)
        e2 = (dcm[0,2]-dcm[2,0])/(4*e0)
        e3 = (dcm[0,1]-dcm[1,0])/(4*e0)
        
    elif e1s == emax:
        e1 = abs(np.sqrt(e1s))
        e0 = (dcm[2,1]-dcm[1,2])/(4*e1)
        e2 = (dcm[0,1]+dcm[1,0])/(4*e1)
        e3 = (dcm[0,2]+dcm[2,0])/(4*e1)
    
    elif e2s == emax:
        e2 = abs(np.sqrt(e2s))
        e0 = (dcm[0,2]-dcm[2,0])/(4*e2)
        e1 = (dcm[0,1]+dcm[1,0])/(4*e2)
        e3 = (dcm[2,1]+dcm[1,2])/(4*e2)
    
    elif e3s == emax:
        e3 = abs(np.sqrt(e3s))
        e0 = (dcm[0,1]-dcm[1,0])/(4*e3)
        e1 = (dcm[0,2]+dcm[2,0])/(4*e3)
        e2 = (dcm[2,1]+dcm[1,2])/(4*e3)
    
    p = np.array([[e0],[e1],[e2],[e3]])
    return p


def E(p):
    ''' 
    A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the global coordinate system.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
    ===========================================================================
    outputs : 
        E   : 3x4 ndarray
    ===========================================================================
    '''
    e0,e1,e2,e3 = [float(i) for i in p]
    m = np.array([[-e1, e0,-e3, e2],
                [-e2, e3, e0,-e1],
                [-e3,-e2, e1, e0]])
    return m

def G(p):
    # Note: This is half the G_bar given in shabana's book
    ''' 
    A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the body coordinate system.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
    ===========================================================================
    outputs : 
        G   : 3x4 ndarray
    ===========================================================================
    '''
    e0,e1,e2,e3 = [float(i) for i in p]
    m = np.array([[-e1, e0, e3,-e2],
                [-e2,-e3, e0, e1],
                [-e3, e2,-e1, e0]])
    return m

def A(p):
    '''
    evaluating the transformation matrix as a function of euler parameters
    Note: The matrix is defined as a prodcut of the two special matrices 
    of euler parameters, the E and G matrices. This function is faster.
    ===========================================================================
    inputs  : 
        p   : set "any list-like object" containing the four parameters
    ===========================================================================
    outputs : 
        A   : The transofrmation matrix 
    ===========================================================================
    '''
    return E(p).dot(G(p).T)



def B(p,a):
    ''' 
    This matrix represents the variation of the body orientation with respect
    to the change in euler parameters. This can be thought as the jacobian of
    the A.dot(a), where A is the transformation matrix in terms of euler 
    parameters.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
        a   : vector 
    ===========================================================================
    outputs : 
        B   : 3x4 ndarray representing the jacobian of A.dot(a)
    ===========================================================================
    '''
    e0,e1,e2,e3 = [float(i) for i in p]
    e = np.array([[e1],[e2],[e3]])
    a = np.array(a).reshape((3,1))
    a_s = skew_matrix(a)
    I = np.eye(3)
    e_s = skew_matrix(e)
    
    m = 2*np.bmat([[(e0*I+e_s).dot(a), e.dot(a.T)-(e0*I+e_s).dot(a_s)]])
    
    return m
