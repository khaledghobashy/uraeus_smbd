# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:21:35 2019

@author: khale
"""

import numpy as np
import scipy as sc
from scipy.sparse.linalg import spsolve
import pandas as pd


def scipy_matrix_assembler(data,rows,cols,shape):
    mat = sc.empty(shape,dtype=np.object)
    mat[rows,cols] = data
    return sc.sparse.bmat(mat)

class solver(object):
    
    def __init__(self,model):
        self.model = model
        
        self.nrows = model.nrows
        self.ncols = model.ncols
        self.jac_shape = (self.nrows,self.ncols)
        
        model.set_initial_states()
                
        self.pos_history = {0:self.model.q0}
        self.vel_history = {}
        self.acc_history = {}
        
    def assemble_equations(self,data):
        mat = np.concatenate(data)
        return mat
    
    def set_time(self,t):
        self.model.t = t
    
    def set_gen_coordinates(self,q):
        self.model.set_gen_coordinates(q)
    
    def set_gen_velocities(self,qd):
        self.model.set_gen_velocities(qd)
    
    def eval_pos_eq(self):
        self.model.eval_pos_eq()
        data = self.model.pos_eq_blocks
        mat = self.assemble_equations(data)
        return mat
        
    def eval_vel_eq(self):
        self.model.eval_vel_eq()
        data = self.model.vel_eq_blocks
        mat = self.assemble_equations(data)
        return mat
        
    def eval_acc_eq(self):
        self.model.eval_acc_eq()
        data = self.model.acc_eq_blocks
        mat = self.assemble_equations(data)
        return mat
        
    def eval_jac_eq(self):
        self.model.eval_jac_eq()
        rows = self.model.jac_rows
        cols = self.model.jac_cols
        data = self.model.jac_eq_blocks
        mat = scipy_matrix_assembler(data,rows,cols,self.jac_shape)
        return mat
    
        
    def newton_raphson(self,guess):
        self.set_gen_coordinates(guess)
        
        A = self.eval_jac_eq().A
        b = self.eval_pos_eq()
        delta_q = np.linalg.solve(A,-b)
        
        itr=0
        while np.linalg.norm(delta_q)>1e-5:
            print(np.linalg.norm(delta_q))
            guess = guess + delta_q
            
            self.set_gen_coordinates(guess)
            b = self.eval_pos_eq()
            delta_q = np.linalg.solve(A,-b)
            
            if itr%5==0 and itr!=0:
                print("Updating Jacobian \n")
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
        
        vel_rhs = self.eval_vel_eq()
        v0 = np.linalg.solve(A,-vel_rhs)
        self.set_gen_velocities(v0)
        self.vel_history[0] = v0
        
        acc_rhs = self.eval_acc_eq()
        self.acc_history[0] = np.linalg.solve(A,-acc_rhs)
        
        print('\nRunning System Kinematic Analysis:')
        for i,t in enumerate(time_array):
            self.set_time(t)

            g = self.pos_history[i] + self.vel_history[i]*dt  + 0.5*self.acc_history[i]*(dt**2)
            
            self.newton_raphson(g)
            self.pos_history[i+1] = self.pos
            A = self.eval_jac_eq().A
            
            vel_rhs = self.eval_vel_eq()
            vi = np.linalg.solve(A,-vel_rhs)
            self.set_gen_velocities(vi)
            self.vel_history[i+1] = vi

            acc_rhs = self.eval_acc_eq()
            self.acc_history[i+1] = np.linalg.solve(A,-acc_rhs)
            
            i+=1
        
        