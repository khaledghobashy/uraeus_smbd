# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 13:21:35 2019

@author: khale
"""

import sys
import numpy as np
import scipy as sc
import scipy.integrate
from scipy.sparse.linalg import spsolve
import pandas as pd

import numba

from smbd.numenv.python.numerics.matrix_funcs import sparse_assembler


def solve(A, b):
    x = spsolve(A, b)
    shape = (x.size, 1)
    x = np.reshape(x, shape)
    return x

def progress_bar(steps, i):
    sys.stdout.write('\r')
    length=(100*(1+i)//(4*steps))
    percentage=100*(1+i)//steps
    sys.stdout.write("[%-25s] %d%%, (%s/%s) steps." % ('='*length,percentage,i+1, steps))
    sys.stdout.flush()


def sparse_assembler_py(blocks, b_rows, b_cols): 
    
    e_data = []
    e_rows = []
    e_cols = []
    
    row_counter = 0 
    prev_rows_size = 0
    prev_cols_size = 0
    nnz = len(b_rows)
    
    m = 0
    
    for v in range(nnz):
        vi = b_rows[v]
        vj = b_cols[v]
        
        if vi != row_counter:
            row_counter +=1
            prev_rows_size += m
            prev_cols_size  = 0
        
        arr = blocks[v]
        m, n = arr.shape
        
        if n==3:
            prev_cols_size = 7*(vj//2)
        elif n==4:
            prev_cols_size = 7*(vj//2)+3
        
        for i in range(m):
            for j in range(n):
                value = arr[i,j]
                if abs(value)> 1e-5:
                    e_rows.append(prev_rows_size + i)
                    e_cols.append(prev_cols_size + j)
                    e_data.append(value)
    
    return e_data, e_rows, e_cols


def scipy_matrix_assembler(data, rows, cols, shape):
    e_data, e_rows, e_cols = sparse_assembler_py(data, rows, cols)
    mat = sc.sparse.coo_matrix((e_data, (e_rows, e_cols)), shape=shape)
    return mat

def scipy_matrix_assembler2(data, rows, cols, shape):
    e_data = []
    e_rows = []
    e_cols = []
    
    sparse_assembler(data, rows, cols, e_data, e_rows, e_cols)
    
    mat = sc.sparse.coo_matrix((e_data, (e_rows, e_cols)), shape=shape)
    return mat

###############################################################################
###############################################################################

class abstract_solver(object):
    
    def __init__(self, model):
        self.model = model
        self._initialize_model()
        
        self._nrows = model.nrows
        self._ncols = model.ncols
        self._jac_shape = (self._nrows,self._ncols)
        
        self._pos_history = {0 : model.q0}
        self._vel_history = {0 : 0*model.q0}
        self._acc_history = {}
        self._lgr_history = {}
        
        self._create_indicies()
        
    
    def set_initial_states(self, coordinates, velocities):
        self._q0 = coordinates
        self._qd0 = velocities
        self.model.set_gen_coordinates(self._q0)
        self.model.set_gen_velocities(self._qd0)
        self._pos_history = {0 : self._q0}
        self._vel_history = {0 : self._qd0}

            
    def set_time_array(self, duration, spacing):
        
        if duration > spacing:
            time_array = np.arange(0, duration, spacing)
            step_size  = spacing
        elif duration < spacing:
            time_array, step_size = np.linspace(0, duration, spacing, retstep=True)
        else:
            raise ValueError('Time array is not properly sampled.')
        self.time_array = time_array
        self.step_size  = step_size
        
    def eval_reactions(self):
        self._reactions = {}
        time_array = self.time_array
        bar_length = len(time_array)
        print('\nEvaluating System Constraints Reactions:')
        for i, t in enumerate(time_array):
            progress_bar(bar_length, i)
            self._set_time(t)
            lamda = self._eval_lagrange_multipliers(i)
            self._eval_reactions_eq(lamda)
            self._reactions[i] = self.model.reactions
        
        self.values = {i:np.concatenate(list(v.values())) for i,v in self._reactions.items()}
        
        self.reactions_dataframe = pd.DataFrame(
                data = np.concatenate(list(self.values.values()),1).T,
                columns = self._reactions_indicies)
        self.reactions_dataframe['time'] = time_array
    
    
    def _initialize_model(self):
        model = self.model
        model.initialize()
        q0 = model.q0
        model.set_gen_coordinates(q0)

    def _create_indicies(self):
        model = self.model
        sorted_coordinates = {v:k for k,v in model.indicies_map.items()}
        self._coordinates_indicies = []
        for name in sorted_coordinates.values():
            self._coordinates_indicies += ['%s.%s'%(name, i) 
            for i in ['x', 'y', 'z', 'e0', 'e1', 'e2', 'e3']]
            
        self._reactions_indicies = []
        for name in model.reactions_indicies:
            self._reactions_indicies += ['%s.%s'%(name, i) 
            for i in ['x','y','z']]

    def _creat_results_dataframes(self):
        columns = self._coordinates_indicies
        
        self.pos_dataframe = pd.DataFrame(
                data = np.concatenate(list(self._pos_history.values()),1).T,
                columns = columns)
        self.vel_dataframe = pd.DataFrame(
                data = np.concatenate(list(self._vel_history.values()),1).T,
                columns = columns)
        self.acc_dataframe = pd.DataFrame(
                data = np.concatenate(list(self._acc_history.values()),1).T,
                columns = columns)
        
        time_array = self.time_array
        self.pos_dataframe['time'] = time_array
        self.vel_dataframe['time'] = time_array
        self.acc_dataframe['time'] = time_array
    
    def _assemble_equations(self, data):
        mat = np.concatenate(data)
        return mat
    
    def _set_time(self, t):
        self.model.t = t
    
    def _set_gen_coordinates(self, q):
        self.model.set_gen_coordinates(q)
    
    def _set_gen_velocities(self, qd):
        self.model.set_gen_velocities(qd)
    
    def _set_gen_accelerations(self, qdd):
        self.model.set_gen_accelerations(qdd)
    
    def _eval_pos_eq(self):
        self.model.eval_pos_eq()
        data = self.model.pos_eq_blocks
        mat = self._assemble_equations(data)
        return mat
        
    def _eval_vel_eq(self):
        self.model.eval_vel_eq()
        data = self.model.vel_eq_blocks
        mat = self._assemble_equations(data)
        return mat
        
    def _eval_acc_eq(self):
        self.model.eval_acc_eq()
        data = self.model.acc_eq_blocks
        mat = self._assemble_equations(data)
        return mat
            
    def _eval_jac_eq(self):
        self.model.eval_jac_eq()
        self._jac_rows = rows = self.model.jac_rows
        self._jac_cols = cols = self.model.jac_cols
        self._jac_data = data = self.model.jac_eq_blocks
        shape = (self.model.nc, self.model.n)
        mat = scipy_matrix_assembler(data, rows, cols, shape)
        return mat.tocsr()
    
    def _eval_mass_eq(self):
        self.model.eval_mass_eq()
        self._mass_data = data = self.model.mass_eq_blocks
        n = self.model.n #self.model.ncols
        self._mass_rows = rows = cols = np.arange(self.model.ncols, dtype=np.intc)
        mat = scipy_matrix_assembler(data, rows, cols, (n,n))
        return mat.tocsr()
    
    def _eval_frc_eq(self):
        self.model.eval_frc_eq()
        data = self.model.frc_eq_blocks
        mat = self._assemble_equations(data)
        return mat
        
    def _eval_reactions_eq(self, lamda):
        self.model.set_lagrange_multipliers(lamda)
        self.model.eval_reactions_eq()
    
    def _newton_raphson(self, guess):
        self._set_gen_coordinates(guess)
        
        A = self._eval_jac_eq()
        b = self._eval_pos_eq()
        delta_q = solve(A, -b)
        
        itr=0
        while np.linalg.norm(delta_q)>1e-5:
#            print(np.linalg.norm(delta_q))
            guess = guess + delta_q
            
            self._set_gen_coordinates(guess)
            b = self._eval_pos_eq()
            delta_q = solve(A, -b)
            
            if itr%5==0 and itr!=0:
                A = self._eval_jac_eq()
                delta_q = solve(A, -b)
            if itr>50:
                print("Iterations exceded \n")
                raise ValueError("Iterations exceded \n")
                break
            itr+=1
        self._pos = guess
        self._jac = self._eval_jac_eq()

###############################################################################
###############################################################################

class kds_solver(abstract_solver):
    
    def __init__(self, model):
        super().__init__(model)
        if self.model.n != self.model.nc:
            raise ValueError('Model is not fully constrained.')
    
    def solve(self, run_id):        
        time_array = self.time_array
        dt = self.step_size
        
        A = self._eval_jac_eq()
        vel_rhs = self._eval_vel_eq()
        v0 = solve(A, -vel_rhs)
        self._set_gen_velocities(v0)
        self._vel_history[0] = v0
        
        acc_rhs = self._eval_acc_eq()
        self._acc_history[0] = solve(A, -acc_rhs)
        
        print('\nRunning System Kinematic Analysis:')
        bar_length = len(time_array)-1
        for i,t in enumerate(time_array[1:]):
            progress_bar(bar_length, i)
            self._set_time(t)

            g =   self._pos_history[i] \
                + self._vel_history[i]*dt \
                + 0.5*self._acc_history[i]*(dt**2)
            
            self._newton_raphson(g)
            self._pos_history[i+1] = self._pos
            A = self._eval_jac_eq()
            
            vel_rhs = self._eval_vel_eq()
            vi = solve(A,-vel_rhs)
            self._set_gen_velocities(vi)
            self._vel_history[i+1] = vi

            acc_rhs = self._eval_acc_eq()
            self._acc_history[i+1] = solve(A,-acc_rhs)
        
        print('\n')
        self._creat_results_dataframes()    
    
    def _eval_lagrange_multipliers(self, i):
        self._set_gen_coordinates(self._pos_history[i])
        self._set_gen_velocities(self._vel_history[i])
        self._set_gen_accelerations(self._acc_history[i])
        
        applied_forces = self._eval_frc_eq()
        mass_matrix = self._eval_mass_eq()
        jac = self._eval_jac_eq()
        qdd = self._acc_history[i]
        inertia_forces = mass_matrix.dot(qdd)
        rhs = applied_forces - inertia_forces
        lamda = solve(jac.T, rhs)
        
        return lamda

###############################################################################
###############################################################################

class dds_solver(abstract_solver):
    
    def __init__(self, model):
        super().__init__(model)
        if self.model.n == self.model.nc:
            raise ValueError('Model is fully constrained.')
    
    def solve(self, run_id):
        time_array = self.time_array
        dt = self.step_size
        bar_length = len(time_array)-1
        print('\nStarting System Dynamic Analysis:')
        
        self._extract_independent_coordinates()
        print('Estimated DOF : %s'%(len(self.independent_cord),))
        print('Estimated Independent Coordinates : %s'%(self.independent_cord,))
        
        pos_t0 = self._pos_history[0]
        vel_t0 = self._vel_history[0]
        
        self._newton_raphson(pos_t0)

        M, J, Qt, Qd = self._eval_augmented_matricies(pos_t0, vel_t0)
        acc_t0, lamda_t0 = self._solve_augmented_system(M, J, Qt, Qd)        
        self._acc_history[0] = acc_t0
        self._lgr_history[0] = lamda_t0

        self.integrator = self._runge_kutta
        
        print('\nRunning System Dynamic Analysis:')
        i = 0
        while i != bar_length:
            progress_bar(bar_length,i)
            t = time_array[i+1]
            self._extract_independent_coordinates(self._jac[:-self.dof,:])
            self._set_time(t)
            self._solve_time_step(t, i, dt)
            i += 1            
        print('\n')
        self._creat_results_dataframes()

    def _solve_time_step(self, t, i, dt):
        
        q  = self._pos_history[i]
        qd = self._vel_history[i]
        
        q_v  = self.get_indpenednt_q(q)
        qd_v = self.get_indpenednt_q(qd)
        state_vector = np.concatenate([q_v, qd_v])
        
        soln = self.integrator(self.SSODE, state_vector, t, dt, i)
        y1 = soln[:self.dof, 0]
        y2 = soln[self.dof:, 0]

        guess = self._pos_history[i] \
              + self._vel_history[i]*dt \
              + 0.5*self._acc_history[i]*(dt**2)
        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = y1[c]
        
        self._newton_raphson(guess)
        A  = self._jac
        qi = self._pos
        
        vel_rhs = self._eval_vel_eq(y2)
        vi = solve(A, -vel_rhs)
        
        self._set_gen_coordinates(qi)
        self._set_gen_velocities(vi)
        
        J  = A[:-self.dof,:]
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        acc_ti, lamda_ti = self._solve_augmented_system(M, J, Qt, Qd)
        
        self._pos_history[i+1] = qi
        self._vel_history[i+1] = vi
        self._acc_history[i+1] = acc_ti
        self._lgr_history[i+1] = lamda_ti
        
        
    def _extract_independent_coordinates(self, jacobian=None):
        A = super()._eval_jac_eq() if jacobian is None else jacobian
        rows, cols = A.shape
        permutaion_mat = sc.linalg.lu(A.A.T)[0]
        independent_cols = permutaion_mat[:, rows:]
        self.dof = dof = independent_cols.shape[1]
        independent_cord = [self._coordinates_indicies[np.argmax(independent_cols[:,i])] for i in range(dof) ]
        self.permutaion_mat  = permutaion_mat.T
        self.independent_cols = independent_cols
        self.independent_cord = independent_cord
    
    
    def _eval_augmented_matricies(self, q , qd):
        self._set_gen_coordinates(q)
        self._set_gen_velocities(qd)
        J  = super()._eval_jac_eq()
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        return M, J, Qt, Qd
    
        
    def _solve_augmented_system(self, M, J, Qt, Qd):
        
        J = J.A
        z = np.zeros((self.model.nc, self.model.nc))

        u = np.concatenate([M.A, J.T], axis=1)
        l = np.concatenate([J, z], axis=1)
        
        A = np.concatenate([u, l], axis=0)
        A = sc.sparse.coo_matrix(A).tocsr()
        
#        A = sc.sparse.bmat([[M, J.T], [J, None]], format='csc')
        b = np.concatenate([Qt, -Qd])
        x = solve(A, b)
        n = len(self._coordinates_indicies)
        accelerations = x[:n]
        lamda = x[n:]
        return accelerations, lamda
    
    def _eval_pos_eq(self):
        A = super()._eval_pos_eq()
        Z = np.zeros((self.dof,1))
        A = np.concatenate([A,Z])
        return A

    def _eval_vel_eq(self,ind_vel_i):
        A = super()._eval_vel_eq()
        V = np.array(ind_vel_i)
        V = np.reshape(V,(self.dof,1))
        A = np.concatenate([A,-V])
        return A
    
    def _eval_jac_eq(self):
        A = np.concatenate([super()._eval_jac_eq().A, self.independent_cols.T])
        A = scipy.sparse.coo_matrix(A)
        return A.tocsr()
    
            
    def get_indpenednt_q(self, q):
        dof = self.dof
        P = self.permutaion_mat
        qp = P@q
        qv = qp[-dof:,:]
        return qv
    
    
    def SSODE(self, state_vector, t, i):
          
        self._set_time(t)
        
        y1 = state_vector[:self.dof]
        y2 = state_vector[self.dof:]
        
        dt = self.step_size
        
        guess = self._pos_history[i] \
              + self._vel_history[i]*dt \
              + 0.5*self._acc_history[i]*(dt**2)

        for c in range(self.dof): 
            guess[np.argmax(self.independent_cols[:, c]), 0] = y1[c]
                        
        self._newton_raphson(guess)
        self._set_gen_coordinates(self._pos)
                
        vel_rhs = self._eval_vel_eq(y2)
        vi = solve(self._jac, -vel_rhs)
        self._set_gen_velocities(vi)

        J  = self._jac[:-self.dof,:]
        M  = self._eval_mass_eq()
        Qt = self._eval_frc_eq()
        Qd = self._eval_acc_eq()
        acc_ti, lamda_ti = self._solve_augmented_system(M, J, Qt, Qd)
        
        y3 = self.get_indpenednt_q(acc_ti)
        
        rhs_vector = np.concatenate([y2, y3])
        
        return rhs_vector
        
    
    @staticmethod
    def _forward_euler(func, state_vector, t, h, i):
        f1 = func(state_vector, t, i)
        yn = state_vector + h*f1
        return yn
    
    @staticmethod
    def _runge_kutta(func, state_vector, t, h, i):
        
        f1 = h*func(state_vector, t, i)
        f2 = h*func(state_vector + 0.5*f1, t + 0.5*h, i)
        f3 = h*func(state_vector + 0.5*f2, t + 0.5*h, i)
        f4 = h*func(state_vector + f3, t + h, i)
        
        yn = state_vector + (1/6) * (f1 + 2*f2 + 2*f3 + f4)
        
        return yn
        


