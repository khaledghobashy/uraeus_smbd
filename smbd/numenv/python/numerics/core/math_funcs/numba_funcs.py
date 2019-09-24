#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:31:42 2019

@author: khaledghobashy
"""

import numba
import numpy as np


@numba.njit(cache=True, nogil=True)
def sparse_assembler(blocks, b_rows, b_cols, n_rows):
    
    m = 0
    row_counter = 0 
    nnz_counter = 0
    prev_rows_size = 0
    prev_cols_size = 0
    
    nnz = len(b_rows)
    
    e_data = np.empty((14*n_rows, ))
    e_rows = np.empty((14*n_rows, ), dtype=np.int64)
    e_cols = np.empty((14*n_rows, ), dtype=np.int64)
    
    
    for v in range(nnz):
        vi = b_rows[v]
        vj = b_cols[v]
        
        if vi != row_counter:
            row_counter +=1
            prev_rows_size += m
            prev_cols_size  = 0
        
        arr = blocks[v]
        if not np.any(arr):
            continue
        
        m, n = arr.shape
        
        if n == 3:
            prev_cols_size = 7*(vj//2)
        elif n == 4:
            prev_cols_size = 7*(vj//2) + 3
            
        nnz_counter = update_arrays(e_data, e_rows, e_cols, m, n, 
                                    prev_rows_size, prev_cols_size, arr, 
                                    nnz_counter)
    
    e_data = e_data[:nnz_counter]
    e_rows = e_rows[:nnz_counter]
    e_cols = e_cols[:nnz_counter]
    
    return e_data, e_rows, e_cols
    
#    return mat


@numba.njit(cache=True, nogil=True)
def update_arrays(e_data, e_rows, e_cols, m, n, r, c, arr, nnz_counter):
    
    for i in range(m):
        for j in range(n):
            value = arr[i,j]
            if abs(value)> 1e-5:
#                mat[r+i, c+j] = value
                e_rows[nnz_counter] = (r + i)
                e_cols[nnz_counter] = (c + j)
                e_data[nnz_counter] = (value)
                
                nnz_counter += 1
    
    return nnz_counter


def matrix_assembler(data, rows, cols, shape):
    
    n_rows = shape[0]

    e_data, e_rows, e_cols = sparse_assembler(data, rows, cols, n_rows)
    
    mat = np.zeros(shape)
    mat[e_rows, e_cols] = e_data
    
    return mat

@numba.njit(cache=True, nogil=True)
def matrix_assembler_temp(data, rows, cols, shape):    
    mat = np.zeros(shape)
    sparse_assembler(data, rows, cols, mat)
    return mat





@numba.njit(cache=True, nogil=True)
def skew_matrix(v):
    x, y, z = v.flat
    vs = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return vs

@numba.njit(cache=True, nogil=True)
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
    
    e0s = (trA + 1) / 4
    e1s = (2*dcm[0,0] - trA + 1) / 4
    e2s = (2*dcm[1,1] - trA + 1) / 4
    e3s = (2*dcm[2,2] - trA + 1) / 4
    
    emax = max([e0s, e1s, e2s, e2s, e3s])
    
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

@numba.njit(cache=True, nogil=True)
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
    e0,e1,e2,e3 = p.flat
    m = np.array([[-e1, e0,-e3, e2],
                  [-e2, e3, e0,-e1],
                  [-e3,-e2, e1, e0]])
    return m

@numba.njit(cache=True, nogil=True)
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
    e0,e1,e2,e3 = p.flat
    m = np.array([[-e1, e0, e3,-e2],
                [-e2,-e3, e0, e1],
                [-e3, e2,-e1, e0]])
    return m

@numba.njit(cache=True, nogil=True)
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


@numba.njit(cache=True, nogil=True)
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
    e0,e1,e2,e3 = p.flat
    e = np.array([[e1],[e2],[e3]])
    a = np.array(a).reshape((3,1))
    a_s = skew_matrix(a)
    I = np.eye(3)
    e_s = skew_matrix(e)
    
    m = 2*np.bmat([[(e0*I+e_s).dot(a), e.dot(a.T)-(e0*I+e_s).dot(a_s)]])
    
    return m

