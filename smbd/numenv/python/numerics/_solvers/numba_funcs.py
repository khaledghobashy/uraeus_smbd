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


