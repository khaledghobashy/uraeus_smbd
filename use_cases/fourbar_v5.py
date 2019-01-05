
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, sparse_assembler, triad as Triad
#from source.solvers.py_numerical_functions import A, B
from scipy.misc import derivative
from numpy import cos, sin
from source.solvers.python_solver import solver

import scipy.sparse as sp



class inputs(object):

    def __init__(self):
        self.C = np.array([[1, 0, 0]],dtype=np.float64)
        self.F = lambda t : 0
        
        self.pt_jcs_a = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_a = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.pt_jcs_d = np.array([[100], [0], [0]],dtype=np.float64)
        self.ax_jcs_d = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.pt_jcs_b = np.array([[0], [50], [0]],dtype=np.float64)
        self.ax_jcs_b = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.pt_jcs_c = np.array([[100], [50], [0]],dtype=np.float64)
        self.ax1_jcs_c = np.array([[1], [0], [0]],dtype=np.float64)
        self.ax2_jcs_c = np.array([[-1], [0], [0]],dtype=np.float64)
        
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[0], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_l1 = np.array([[0], [25], [0]],dtype=np.float64)
        self.P_sub_rbs_l1 = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_l1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_l1 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_l2 = np.array([[50], [50], [0]],dtype=np.float64)
        self.P_sub_rbs_l2 = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_l2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_l2 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_l3 = np.array([[100], [25], [0]],dtype=np.float64)
        self.P_sub_rbs_l3 = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_l3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_l3 = np.array([[0], [0], [0], [0]],dtype=np.float64)


class numerical_assembly(object):

    def __init__(self,config):
        self.F = config.F
        self.t = 0

        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        self.pos_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        self.vel_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        self.acc_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jacobian_rows_blocks = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,11,11,12,13,14])
        self.jacobian_cols_blocks = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,6,7,0,1,6,7,0,1,6,7,2,3,4,5,4,5,6,7,4,5,6,7,0,1,0,1,3,5,7])

    def eval_constants(self):
        config = self.config

        c0 = A(config.P_ground).T
        c1 = config.pt_jcs_a
        c2 = -1.0*multi_dot([c0,config.R_ground])
        c3 = A(config.P_sub_rbs_l1).T
        c4 = -1.0*multi_dot([c3,config.R_sub_rbs_l1])
        c5 = Triad(config.ax_jcs_a)
        c6 = config.pt_jcs_d
        c7 = A(config.P_sub_rbs_l3).T
        c8 = -1.0*multi_dot([c7,config.R_sub_rbs_l3])
        c9 = Triad(config.ax_jcs_d)
        c10 = config.pt_jcs_b
        c11 = A(config.P_sub_rbs_l2).T
        c12 = -1.0*multi_dot([c11,config.R_sub_rbs_l2])
        c13 = Triad(config.ax_jcs_b)
        c14 = config.pt_jcs_c
        c15 = Triad(config.ax1_jcs_c)

        self.ubar_ground_jcs_a = (multi_dot([c0,c1]) + c2)
        self.ubar_sub_rbs_l1_jcs_a = (multi_dot([c3,c1]) + c4)
        self.Mbar_ground_jcs_a = multi_dot([c0,c5])
        self.Mbar_sub_rbs_l1_jcs_a = multi_dot([c3,c5])
        self.ubar_ground_jcs_d = (multi_dot([c0,c6]) + c2)
        self.ubar_sub_rbs_l3_jcs_d = (multi_dot([c7,c6]) + c8)
        self.Mbar_ground_jcs_d = multi_dot([c0,c9])
        self.Mbar_sub_rbs_l3_jcs_d = multi_dot([c7,c9])
        self.ubar_sub_rbs_l1_jcs_b = (multi_dot([c3,c10]) + c4)
        self.ubar_sub_rbs_l2_jcs_b = (multi_dot([c11,c10]) + c12)
        self.Mbar_sub_rbs_l1_jcs_b = multi_dot([c3,c13])
        self.Mbar_sub_rbs_l2_jcs_b = multi_dot([c11,c13])
        self.ubar_sub_rbs_l2_jcs_c = (multi_dot([c11,c14]) + c12)
        self.ubar_sub_rbs_l3_jcs_c = (multi_dot([c7,c14]) + c8)
        self.Mbar_sub_rbs_l2_jcs_c = multi_dot([c11,c15])
        self.Mbar_sub_rbs_l3_jcs_c = multi_dot([c7,Triad(config.ax2_jcs_c, c15[0:3,1:2])])

    
    def set_coordinates(self,q):
        self.R_ground = q[0:3,0:1]
        self.P_ground = q[3:7,0:1]
        self.R_sub_rbs_l1 = q[7:10,0:1]
        self.P_sub_rbs_l1 = q[10:14,0:1]
        self.R_sub_rbs_l2 = q[14:17,0:1]
        self.P_sub_rbs_l2 = q[17:21,0:1]
        self.R_sub_rbs_l3 = q[21:24,0:1]
        self.P_sub_rbs_l3 = q[24:28,0:1]
    
    
        
    def set_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        self.Rd_sub_rbs_l1 = qd[7:10,0:1]
        self.Pd_sub_rbs_l1 = qd[10:14,0:1]
        self.Rd_sub_rbs_l2 = qd[14:17,0:1]
        self.Pd_sub_rbs_l2 = qd[17:21,0:1]
        self.Rd_sub_rbs_l3 = qd[21:24,0:1]
        self.Pd_sub_rbs_l3 = qd[24:28,0:1]
    
    
        
    def set_initial_configuration(self):
        config = self.config
    
        q = np.concatenate([config.R_ground,
        config.P_ground,
        config.R_sub_rbs_l1,
        config.P_sub_rbs_l1,
        config.R_sub_rbs_l2,
        config.P_sub_rbs_l2,
        config.R_sub_rbs_l3,
        config.P_sub_rbs_l3])
    
        qd = np.concatenate([config.Rd_ground,
        config.Pd_ground,
        config.Rd_sub_rbs_l1,
        config.Pd_sub_rbs_l1,
        config.Rd_sub_rbs_l2,
        config.Pd_sub_rbs_l2,
        config.Rd_sub_rbs_l3,
        config.Pd_sub_rbs_l3])
    
        self.set_coordinates(q)
        self.set_velocities(qd)
        self.q_initial = q.copy()
    
    
    
        
    def eval_pos_eq(self):
        F = self.F
        t = self.t
    
        x0 = self.R_ground
        x1 = self.R_sub_rbs_l1
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_sub_rbs_l1
        x5 = A(x4)
        x6 = x3.T
        x7 = self.Mbar_sub_rbs_l1_jcs_a[:,2:3]
        x8 = F(t)
        x9 = self.Mbar_sub_rbs_l1_jcs_a[:,0:1]
        x10 = -1.0*self.R_sub_rbs_l3
        x11 = self.P_sub_rbs_l3
        x12 = A(x11)
        x13 = self.Mbar_sub_rbs_l3_jcs_d[:,2:3]
        x14 = self.R_sub_rbs_l2
        x15 = self.P_sub_rbs_l2
        x16 = A(x15)
        x17 = -1.0*np.eye(1,dtype=np.float64)
    
        self.pos_level_data_blocks = [(x0 + -1.0*x1 + multi_dot([x3,self.ubar_ground_jcs_a]) + -1.0*multi_dot([x5,self.ubar_sub_rbs_l1_jcs_a])),multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x7]),multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x7]),(-sin(x8)*multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x9]) - cos(x8)*-1.0*multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x9])),(x0 + x10 + multi_dot([x3,self.ubar_ground_jcs_d]) + -1.0*multi_dot([x12,self.ubar_sub_rbs_l3_jcs_d])),multi_dot([self.Mbar_ground_jcs_d[:,0:1].T,x6,x12,x13]),multi_dot([self.Mbar_ground_jcs_d[:,1:2].T,x6,x12,x13]),(x1 + -1.0*x14 + multi_dot([x5,self.ubar_sub_rbs_l1_jcs_b]) + -1.0*multi_dot([x16,self.ubar_sub_rbs_l2_jcs_b])),(x14 + x10 + multi_dot([x16,self.ubar_sub_rbs_l2_jcs_c]) + -1.0*multi_dot([x12,self.ubar_sub_rbs_l3_jcs_c])),multi_dot([self.Mbar_sub_rbs_l2_jcs_c[:,0:1].T,x16.T,x12,self.Mbar_sub_rbs_l3_jcs_c[:,0:1]]),x0,(x2 + -1.0*self.Pg_ground),(x17 + (multi_dot([x4.T,x4]))**(1.0/2.0)),(x17 + (multi_dot([x15.T,x15]))**(1.0/2.0)),(x17 + (multi_dot([x11.T,x11]))**(1.0/2.0))]
    
        self.pos_level_rows_explicit = []
        self.pos_level_cols_explicit = []
        self.pos_level_data_explicit = []
    
        sparse_assembler(self.pos_level_data_blocks, self.pos_level_rows_blocks, self.pos_level_cols_blocks,
                         self.pos_level_data_explicit, self.pos_level_rows_explicit, self.pos_level_cols_explicit)
    
        self.pos_rhs = sc.sparse.coo_matrix(
        (self.pos_level_data_explicit,
        (self.pos_level_rows_explicit,self.pos_level_cols_explicit)),
        (28,1))
    
    
    
        
    def eval_vel_eq(self):
        F = self.F
        t = self.t
    
        x0 = np.zeros((3,1),dtype=np.float64)
        x1 = np.zeros((1,1),dtype=np.float64)
        x2 = F(t)
    
        self.vel_level_data_blocks = [x0,x1,x1,(x1 + cos(x2)*derivative(F,t,0.1,1)*-1.0*np.eye(1,dtype=np.float64)),x0,x1,x1,x0,x0,x1,x0,np.zeros((4,1),dtype=np.float64),x1,x1,x1]
    
        self.vel_level_rows_explicit = []
        self.vel_level_cols_explicit = []
        self.vel_level_data_explicit = []
    
        sparse_assembler(self.vel_level_data_blocks, self.vel_level_rows_blocks, self.vel_level_cols_blocks,
                         self.vel_level_data_explicit, self.vel_level_rows_explicit, self.vel_level_cols_explicit)
    
        self.vel_rhs = sc.sparse.coo_matrix(
        (self.vel_level_data_explicit,
        (self.vel_level_rows_explicit,self.vel_level_cols_explicit)),
        (28,1))
    
    
    
        
    def eval_acc_eq(self):
        F = self.F
        t = self.t
    
        a0 = self.Pd_ground
        a1 = self.Pd_sub_rbs_l1
        a2 = self.Mbar_ground_jcs_a[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_sub_rbs_l1_jcs_a[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_sub_rbs_l1
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_a[:,1:2]
        a13 = self.Mbar_sub_rbs_l1_jcs_a[:,0:1]
        a14 = F(t)
        a15 = cos(a14)
        a16 = self.Mbar_ground_jcs_a[:,1:2]
        a17 = sin(a14)
        a18 = self.Mbar_ground_jcs_a[:,0:1]
        a19 = self.Pd_sub_rbs_l3
        a20 = self.Mbar_ground_jcs_d[:,0:1]
        a21 = self.Mbar_sub_rbs_l3_jcs_d[:,2:3]
        a22 = B(a19,a21)
        a23 = a21.T
        a24 = self.P_sub_rbs_l3
        a25 = A(a24).T
        a26 = B(a24,a21)
        a27 = self.Mbar_ground_jcs_d[:,1:2]
        a28 = self.Pd_sub_rbs_l2
        a29 = self.Mbar_sub_rbs_l3_jcs_c[:,0:1]
        a30 = self.Mbar_sub_rbs_l2_jcs_c[:,0:1]
        a31 = self.P_sub_rbs_l2
        a32 = a28.T
    
        self.acc_level_data_blocks = [(multi_dot([B(a0,self.ubar_ground_jcs_a),a0]) + -1.0*multi_dot([B(a1,self.ubar_sub_rbs_l1_jcs_a),a1])),(multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + 2.0*multi_dot([a10,B(a3,a2).T,a11,a1])),(multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + 2.0*multi_dot([a10,B(a3,a12).T,a11,a1])),(multi_dot([a13.T,a9,(a15*B(a0,a16) + a17*-1.0*B(a0,a18)),a0]) + multi_dot([(a15*multi_dot([a16.T,a4]) + a17*-1.0*multi_dot([a18.T,a4])),B(a1,a13),a1]) + -1*a15*derivative(F,t,0.1,2) + a17*pow(derivative(F,t,0.1,1), 2)*np.eye(1,dtype=np.float64) + 2.0*multi_dot([((a15*multi_dot([B(a3,a16),a0])).T + a17.T*-1.0*multi_dot([a10,B(a3,a18).T])),B(a8,a13),a1])),(multi_dot([B(a0,self.ubar_ground_jcs_d),a0]) + -1.0*multi_dot([B(a19,self.ubar_sub_rbs_l3_jcs_d),a19])),(multi_dot([a20.T,a4,a22,a19]) + multi_dot([a23,a25,B(a0,a20),a0]) + 2.0*multi_dot([a10,B(a3,a20).T,a26,a19])),(multi_dot([a27.T,a4,a22,a19]) + multi_dot([a23,a25,B(a0,a27),a0]) + 2.0*multi_dot([a10,B(a3,a27).T,a26,a19])),(multi_dot([B(a1,self.ubar_sub_rbs_l1_jcs_b),a1]) + -1.0*multi_dot([B(a28,self.ubar_sub_rbs_l2_jcs_b),a28])),(multi_dot([B(a28,self.ubar_sub_rbs_l2_jcs_c),a28]) + -1.0*multi_dot([B(a19,self.ubar_sub_rbs_l3_jcs_c),a19])),(multi_dot([a29.T,a25,B(a28,a30),a28]) + multi_dot([a30.T,A(a31).T,B(a19,a29),a19]) + 2.0*multi_dot([a32,B(a31,a30).T,B(a24,a29),a19])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a32,a28]))**(1.0/2.0),2.0*(multi_dot([a19.T,a19]))**(1.0/2.0)]
    
        self.acc_level_rows_explicit = []
        self.acc_level_cols_explicit = []
        self.acc_level_data_explicit = []
    
        sparse_assembler(self.acc_level_data_blocks, self.acc_level_rows_blocks, self.acc_level_cols_blocks,
                         self.acc_level_data_explicit, self.acc_level_rows_explicit, self.acc_level_cols_explicit)
    
        self.acc_rhs = sc.sparse.coo_matrix(
        (self.acc_level_data_explicit,
        (self.acc_level_rows_explicit,self.acc_level_cols_explicit)),
        (28,1))
    
    
    
        
    def eval_jacobian(self):
        F = self.F
        t = self.t
    
        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_sub_rbs_l1_jcs_a[:,2:3]
        j4 = j3.T
        j5 = self.P_sub_rbs_l1
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_a[:,0:1]
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = -1.0*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_sub_rbs_l1_jcs_a[:,0:1]
        j13 = F(t)
        j14 = cos(j13)
        j15 = self.Mbar_ground_jcs_a[:,1:2]
        j16 = self.Mbar_ground_jcs_a[:,0:1]
        j17 = self.Mbar_sub_rbs_l3_jcs_d[:,2:3]
        j18 = j17.T
        j19 = self.P_sub_rbs_l3
        j20 = A(j19).T
        j21 = self.Mbar_ground_jcs_d[:,0:1]
        j22 = self.Mbar_ground_jcs_d[:,1:2]
        j23 = B(j19,j17)
        j24 = self.P_sub_rbs_l2
        j25 = self.Mbar_sub_rbs_l3_jcs_c[:,0:1]
        j26 = self.Mbar_sub_rbs_l2_jcs_c[:,0:1]
    
        self.jacobian_data_blocks = [j0,B(j1,self.ubar_ground_jcs_a),j9,-1.0*B(j5,self.ubar_sub_rbs_l1_jcs_a),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j2,multi_dot([j12.T,j6,(j14*B(j1,j15) + sin(j13)*-1.0*B(j1,j16))]),j2,multi_dot([(j14*multi_dot([j15.T,j10]) + sin(j13)*-1.0*multi_dot([j16.T,j10])),B(j5,j12)]),j0,B(j1,self.ubar_ground_jcs_d),j9,-1.0*B(j19,self.ubar_sub_rbs_l3_jcs_d),j2,multi_dot([j18,j20,B(j1,j21)]),j2,multi_dot([j21.T,j10,j23]),j2,multi_dot([j18,j20,B(j1,j22)]),j2,multi_dot([j22.T,j10,j23]),j0,B(j5,self.ubar_sub_rbs_l1_jcs_b),j9,-1.0*B(j24,self.ubar_sub_rbs_l2_jcs_b),j0,B(j24,self.ubar_sub_rbs_l2_jcs_c),j9,-1.0*B(j19,self.ubar_sub_rbs_l3_jcs_c),j2,multi_dot([j25.T,j20,B(j24,j26)]),j2,multi_dot([j26.T,A(j24).T,B(j19,j25)]),j0,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j5.T,2.0*j24.T,2.0*j19.T]
    
        self.jacobian_rows_explicit = []
        self.jacobian_cols_explicit = []
        self.jacobian_data_explicit = []
    
        sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
                         self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)
    
        self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))
    
    

def solve_system(time_array):
    
    try:
        config = inputs()
        config.F = lambda t : np.deg2rad(360)*t
        assembled = numerical_assembly(config)
        assembled.eval_constants()
        assembled.set_initial_configuration()
        soln = solver(assembled)
        soln.solve_kds(time_array)
        return soln
    except np.linalg.LinAlgError:
        return soln

time_array = np.arange(0,2.5,0.01)
soln = solve_system(time_array)

import pandas as pd
import matplotlib.pyplot as plt

pos_history = pd.DataFrame(np.concatenate(list(soln.pos_history.values()),1).T,index=soln.pos_history.keys(),columns=range(28))
shape = pos_history.shape
time_array_mod = time_array[:shape[0]-1]

plt.figure(figsize=(10,6))
plt.plot(time_array_mod,pos_history[7][:-1])
plt.plot(time_array_mod,pos_history[8][:-1])
plt.grid()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(time_array_mod,pos_history[14][:-1])
plt.plot(time_array_mod,pos_history[15][:-1])
plt.grid()
plt.show()


plt.figure(figsize=(10,6))
plt.plot(time_array_mod,pos_history[21][:-1])
plt.plot(time_array_mod,pos_history[22][:-1])
plt.grid()
plt.show()
