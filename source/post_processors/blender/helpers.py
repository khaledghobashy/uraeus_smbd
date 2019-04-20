# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:51:59 2019

@author: khale
"""

import numpy as np

###############################################################################
###############################################################################

def skew_matrix(v):
    vs = np.array([[0,-v[2,0],v[1,0]],
                 [v[2,0],0,-v[0,0]],
                 [-v[1,0],v[0,0],0]])
    return vs

def orthogonal_vector(v):
    dummy = np.ones((3,1))
    i = np.argmax(np.abs(v))
    dummy[i] = 0
    m = np.linalg.multi_dot([skew_matrix(v),dummy])
    return m

def triad(v1,v2=None):
    k = v1/np.linalg.norm(v1)
    
    if v2 is not None:
        i = v2/np.linalg.norm(v2)
    else:
        i = orthogonal_vector(k)
        i = i/np.linalg.norm(i)
    
    j = np.linalg.multi_dot([skew_matrix(k),i])
    j = j/np.linalg.norm(j)
    
    m = np.concatenate([i,j,k],axis=1)
    return m

###############################################################################
###############################################################################

def mirrored(v):
    if v.shape != (3,1):
        return v
    else:
        m = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=np.float64)
        return m.dot(v)

def centered(*args):
    p = np.sum(args,0)
    p = p/len(args)
    return p

def oriented(*args):
    if len(args) == 2:
        v = args[1] - args[0]
    elif len(args) == 3:
        a1 = args[1] - args[0]
        a2 = args[2] - args[0]
        v = np.cross(a2,a1,axisa=0,axisb=0)
        v = np.reshape(v,(3,1))
    v = v/np.linalg.norm(v)
    return v

###############################################################################
###############################################################################
