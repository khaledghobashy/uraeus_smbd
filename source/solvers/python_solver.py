# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:21:35 2019

@author: khale
"""

import sys
import numpy as np
import scipy as sc
from scipy.sparse.linalg import spsolve
import pandas as pd

def solve(A,b):
    x = np.linalg.solve(A.A,b)
    x = np.reshape(x,(x.shape[0],1))
    return x

def progress_bar(steps,i):
    sys.stdout.write('\r')
    length=(100*(1+i)//(4*steps))
    percentage=100*(1+i)//steps
    sys.stdout.write("Progress: ")
    sys.stdout.write("[%-25s] %d%% of %s steps." % ('='*length,percentage, steps+1))
    sys.stdout.flush()

def scipy_matrix_assembler(data,rows,cols,shape):
    mat = sc.empty(shape,dtype=np.object)
    mat[rows,cols] = data
    return sc.sparse.bmat(mat,format='csc')

class solver(object):
    
    def __init__(self,model):
        self.model = model
        
        self.nrows = model.nrows
        self.ncols = model.ncols
        self.jac_shape = (self.nrows,self.ncols)
        
        model.set_initial_states()
                
        self._pos_history = {0:self.model.q0}
        self._vel_history = {}
        self._acc_history = {}
        
        sorted_coordinates = {v:k for k,v in self.model.indicies_map.items()}
        self.coordinates_indicies = []
        for name in sorted_coordinates.values():
            self.coordinates_indicies += ['%s.%s'%(name,i) 
            for i in ['x','y','z','e0','e1','e2','e3']]
    
    def creat_results_dataframes(self):
        self.pos_dataframe = pd.DataFrame(
                data = np.concatenate(list(self._pos_history.values()),1).T,
                columns = self.coordinates_indicies)
        self.vel_dataframe = pd.DataFrame(
                data = np.concatenate(list(self._vel_history.values()),1).T,
                columns = self.coordinates_indicies)
        self.acc_dataframe = pd.DataFrame(
                data = np.concatenate(list(self._acc_history.values()),1).T,
                columns = self.coordinates_indicies)
    
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
        
        A = self.eval_jac_eq()
        b = self.eval_pos_eq()
        delta_q = solve(A,-b)
        
        itr=0
        while np.linalg.norm(delta_q)>1e-5:
            print(np.linalg.norm(delta_q))
            guess = guess + delta_q
            
            self.set_gen_coordinates(guess)
            b = self.eval_pos_eq()
            delta_q = solve(A,-b)
            
            if itr%5==0 and itr!=0:
#                print("Updating Jacobian \n")
                A = self.eval_jac_eq()
                delta_q = solve(A,-b)
            if itr>50:
                print("Iterations exceded \n")
                break
            itr+=1
        self.pos = guess
    
    
    def solve_kds(self,time_array,run_id='',save=False):
        dt = time_array[1]-time_array[0]
        
        A = self.eval_jac_eq()
        
        vel_rhs = self.eval_vel_eq()
        v0 = solve(A,-vel_rhs)
        self.set_gen_velocities(v0)
        self._vel_history[0] = v0
        
        acc_rhs = self.eval_acc_eq()
        self._acc_history[0] = solve(A,-acc_rhs)
        
        print('\nRunning System Kinematic Analysis:')
        for i,t in enumerate(time_array[1:]):
            progress_bar(len(time_array)-1,i)
            self.set_time(t)

            g = self._pos_history[i] + self._vel_history[i]*dt  + 0.5*self._acc_history[i]*(dt**2)
            
            self.newton_raphson(g)
            self._pos_history[i+1] = self.pos
            A = self.eval_jac_eq()
            
            vel_rhs = self.eval_vel_eq()
            vi = solve(A,-vel_rhs)
            self.set_gen_velocities(vi)
            self._vel_history[i+1] = vi

            acc_rhs = self.eval_acc_eq()
            self._acc_history[i+1] = solve(A,-acc_rhs)
            
            i+=1
        
        self.creat_results_dataframes()
    
    def save_data(self,data,filename):
        pass

        


