# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:21:35 2019

@author: khale
"""

import numpy as np
from scipy.sparse.linalg import spsolve
import pandas as pd

class solver(object):
    
    def __init__(self,model):
        self.model = model
        self.model.set_initial_configuration()
        self.model.eval_constants()
        
        '''self.pos_history = pd.DataFrame(self.model.q_initial.T)
        self.vel_history = pd.DataFrame(0*self.model.q_initial.T)
        self.acc_history = pd.DataFrame(0*self.model.q_initial.T)'''
        
        self.pos_history = {0:self.model.q_initial}
        self.vel_history = {}
        self.acc_history = {}
        
    def newton_raphson(self,guess=None):
        
        guess = (self.model.q_initial if guess is None else guess)
        self.model.set_coordinates(guess)
        self.model.eval_jacobian()
        self.model.eval_pos_eq()
        
        A = self.model.jacobian.A
        b = self.model.pos_rhs.A
        
        delta_q = np.linalg.solve(A,-b)
        
        itr=0
        while np.linalg.norm(delta_q)>1e-5:
            print(np.linalg.norm(delta_q))
            guess=guess+delta_q
            
            self.model.set_coordinates(guess)
            self.model.eval_pos_eq()
            b = self.model.pos_rhs.A
            delta_q = np.linalg.solve(A,-b)
            
            if itr%5==0 and itr!=0:
                self.model.eval_jacobian()
                A = self.model.jacobian.A
                delta_q = np.linalg.solve(A,-b)
            
            if itr>200:
                print("Iterations exceded \n")
                break
            
            itr+=1

        self.pos = guess
        self.model.eval_jacobian()
    
    def solve_kds(self,time_array):
        dt = time_array[1]-time_array[0]
        
        self.model.eval_jacobian()
        A = self.model.jacobian.A
        
        self.model.eval_vel_eq()
        vel_rhs = self.model.vel_rhs.A
        v0 = np.linalg.solve(A,-vel_rhs)
        self.model.set_velocities(v0)
        self.vel_history[0] = v0
        
        self.model.eval_acc_eq()
        acc_rhs = self.model.acc_rhs.A
        self.acc_history[0] = np.linalg.solve(A,-acc_rhs)
        
        print('\nRunning System Kinematic Analysis:')
        for i,step in enumerate(time_array[1:]):
            self.model.t = step

            g = self.pos_history[i] + self.vel_history[i]*dt  + 0.5*self.acc_history[i]*(dt**2)
            #print('det = '+str(np.linalg.det(A))+'\n')
            self.newton_raphson(g)
            self.pos_history[i+1] = self.pos
            A = self.model.jacobian.A
            
            self.model.eval_vel_eq()
            vel_rhs = self.model.vel_rhs.A
            vi = np.linalg.solve(A,-vel_rhs)
            self.model.set_velocities(vi)
            self.vel_history[i+1] = vi

            self.model.eval_acc_eq()
            acc_rhs = self.model.acc_rhs.A
            self.acc_history[i+1] = np.linalg.solve(A,-acc_rhs)
            
            i+=1