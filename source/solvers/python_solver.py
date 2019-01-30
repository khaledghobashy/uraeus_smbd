# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:21:35 2019

@author: khale
"""

import numpy as np
import scipy as sc
from scipy.sparse.linalg import spsolve
import pandas as pd


def scipy_matrix_assembler(data,rows,cols,n,m):
    mat = sc.empty((m+1,n+1),dtype=np.object)
    mat[rows,cols] = data
    return sc.sparse.bmat(mat)

class solver(object):
    
    def __init__(self,model):
        self.model = model
        self.nrows = model.nrows
        self.ncols = model.ncols
        
        model.set_initial_states()
                
        self.pos_history = {0:self.model.config.q_initial}
        self.vel_history = {}
        self.acc_history = {}
    
    def _matrix_assembler(self,data,rows,cols):
        mat = scipy_matrix_assembler(data,rows,cols)
        return mat
    
    def eval_pos_eq(self):
        self.model.eval_pos_eq()
        data, rows, cols = self.model.pos_eq_blocks,self.model.pos_rows,self.model.pos_cols
        mat = self._matrix_assembler(data,rows,cols)
        return mat
        
    def eval_vel_eq(self):
        self.model.eval_vel_eq()
        data, rows, cols = self.model.vel_eq_blocks,self.model.vel_rows,self.model.vel_cols
        mat = self._matrix_assembler(data,rows,cols)
        return mat
        
    def eval_acc_eq(self):
        self.model.eval_acc_eq()
        data, rows, cols = self.model.acc_eq_blocks,self.model.acc_rows,self.model.acc_cols
        mat = self._matrix_assembler(data,rows,cols)
        return mat
        
    def eval_jac_eq(self):
        self.model.eval_jac_eq()
        data, rows, cols = self.model.jac_eq_blocks,self.model.jac_rows,self.model.jac_cols
        mat = self._matrix_assembler(data,rows,cols)
        return mat
    
        
    def newton_raphson(self,guess=None):
        
        guess = (self.model.config.q_initial if guess is None else guess)
        self.model.set_q(guess)
        
        A = self.eval_jac_eq().A
        b = self.eval_pos_eq().A
        delta_q = np.linalg.solve(A,-b)
        
        itr=0
        while np.linalg.norm(delta_q)>1e-5:
            print(np.linalg.norm(delta_q))
            guess=guess+delta_q
            
            self.model.set_q(guess)
            b = self.eval_pos_eq().A
            delta_q = np.linalg.solve(A,-b)
            
            if itr%5==0 and itr!=0:
                A = self.eval_jac_eq().A
                delta_q = np.linalg.solve(A,-b)
            if itr>200:
                print("Iterations exceded \n")
                break
            itr+=1
        self.pos = guess
    
    
    def solve_kds(self,time_array):
        dt = time_array[1]-time_array[0]
        
        A = self.eval_jac_eq().A
        
        vel_rhs = self.eval_vel_eq().A
        v0 = np.linalg.solve(A,-vel_rhs)
        self.model.set_qd(v0)
        self.vel_history[0] = v0
        
        acc_rhs = self.eval_acc_eq().A
        self.acc_history[0] = np.linalg.solve(A,-acc_rhs)
        
        print('\nRunning System Kinematic Analysis:')
        for i,step in enumerate(time_array[1:]):
            self.model.t = step

            g = self.pos_history[i] + self.vel_history[i]*dt  + 0.5*self.acc_history[i]*(dt**2)
            
            self.newton_raphson(g)
            self.pos_history[i+1] = self.pos
            A = self.eval_jac_eq().A
            
            vel_rhs = self.eval_vel_eq().A
            vi = np.linalg.solve(A,-vel_rhs)
            self.model.set_qd(vi)
            self.vel_history[i+1] = vi

            acc_rhs = self.eval_acc_eq().A
            self.acc_history[i+1] = np.linalg.solve(A,-acc_rhs)
            
            i+=1