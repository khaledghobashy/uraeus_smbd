
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
        
        self.F = lambda t : 0
        self.C = np.array([[0, 0, 1]],dtype=np.float64)
        
        self.pt_jcs_uca_chassis = np.array([[0], [294], [180]],dtype=np.float64)
        self.ax_jcs_uca_chassis = np.array([[1], [0], [0]],dtype=np.float64)
        
        self.pt_jcs_lca_chassis = np.array([[0], [245], [-106]],dtype=np.float64)
        self.ax_jcs_lca_chassis = np.array([[1], [0], [0]],dtype=np.float64)
        
        self.pt_jcs_strut_chassis = np.array([[-165], [534], [639]],dtype=np.float64)
        self.ax1_jcs_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcs_strut_chassis = np.array([[0], [0], [-1]],dtype=np.float64)
        
        self.pt_jcs_tie_steering = np.array([[402], [267], [108]],dtype=np.float64)
        self.ax1_jcs_tie_steering = np.array([[0], [1], [0]],dtype=np.float64)
        self.ax2_jcs_tie_steering = np.array([[0], [-1], [0]],dtype=np.float64)
        
        self.pt_jcs_uca_upright = np.array([[0], [722], [187]],dtype=np.float64)
        self.ax_jcs_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
       
        self.pt_jcs_lca_upright = np.array([[0], [776], [-181]],dtype=np.float64)
        self.ax_jcs_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
       
        self.pt_jcs_strut_lca = np.array([[-165], [534], [-79]],dtype=np.float64)
        self.ax1_jcs_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcs_strut_lca = np.array([[0], [0], [-1]],dtype=np.float64)
        
        self.pt_jcs_tie_upright = np.array([[399], [720], [110]],dtype=np.float64)
        self.ax_jcs_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
       
        self.pt_jcs_strut = np.array([[-165], [534], [300]],dtype=np.float64)
        self.ax_jcs_strut = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_uca = np.array([[0], [428], [182]],dtype=np.float64)
        self.P_sub_rbs_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_lca = np.array([[0], [422], [-131]],dtype=np.float64)
        self.P_sub_rbs_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_upright = np.array([[133], [739], [38]],dtype=np.float64)
        self.P_sub_rbs_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_upper_strut = np.array([[-165], [534], [459]],dtype=np.float64)
        self.P_sub_rbs_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_lower_strut = np.array([[-165], [534], [100]],dtype=np.float64)
        self.P_sub_rbs_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_sub_rbs_tie_rod = np.array([[400], [493], [109]],dtype=np.float64)
        self.P_sub_rbs_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)


class numerical_assembly(object):

    def __init__(self,config):
        self.F = config.F
        self.t = 0
        self.C = config.C

        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
        self.pos_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
        self.vel_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27])
        self.acc_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jacobian_rows_blocks = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,21,21,22,23,24,25,26,27])
        self.jacobian_cols_blocks = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,4,5,0,1,4,5,0,1,4,5,0,1,6,7,0,1,8,9,0,1,8,9,0,1,12,13,0,1,12,13,2,3,6,7,4,5,6,7,4,5,10,11,4,5,10,11,6,7,12,13,8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11,0,1,0,1,3,5,7,9,11,13])

    def eval_constants(self):
        config = self.config

        c0 = A(config.P_ground).T
        c1 = config.pt_jcs_uca_chassis
        c2 = -1.0*multi_dot([c0,config.R_ground])
        c3 = A(config.P_sub_rbs_uca).T
        c4 = -1.0*multi_dot([c3,config.R_sub_rbs_uca])
        c5 = Triad(config.ax_jcs_uca_chassis)
        c6 = config.pt_jcs_lca_chassis
        c7 = A(config.P_sub_rbs_lca).T
        c8 = -1.0*multi_dot([c7,config.R_sub_rbs_lca])
        c9 = Triad(config.ax_jcs_lca_chassis)
        c10 = config.pt_jcs_strut_chassis
        c11 = A(config.P_sub_rbs_upper_strut).T
        c12 = -1.0*multi_dot([c11,config.R_sub_rbs_upper_strut])
        c13 = Triad(config.ax1_jcs_strut_chassis)
        c14 = config.pt_jcs_tie_steering
        c15 = A(config.P_sub_rbs_tie_rod).T
        c16 = -1.0*multi_dot([c15,config.R_sub_rbs_tie_rod])
        c17 = Triad(config.ax1_jcs_tie_steering)
        c18 = config.pt_jcs_uca_upright
        c19 = A(config.P_sub_rbs_upright).T
        c20 = -1.0*multi_dot([c19,config.R_sub_rbs_upright])
        c21 = Triad(config.ax_jcs_uca_upright)
        c22 = config.pt_jcs_lca_upright
        c23 = Triad(config.ax_jcs_lca_upright)
        c24 = config.pt_jcs_strut_lca
        c25 = A(config.P_sub_rbs_lower_strut).T
        c26 = -1.0*multi_dot([c25,config.R_sub_rbs_lower_strut])
        c27 = Triad(config.ax1_jcs_strut_lca)
        c28 = config.pt_jcs_tie_upright
        c29 = Triad(config.ax_jcs_tie_upright)
        c30 = config.pt_jcs_strut
        c31 = Triad(config.ax_jcs_strut)

        self.ubar_ground_jcs_uca_chassis = (multi_dot([c0,c1]) + c2)
        self.ubar_sub_rbs_uca_jcs_uca_chassis = (multi_dot([c3,c1]) + c4)
        self.Mbar_ground_jcs_uca_chassis = multi_dot([c0,c5])
        self.Mbar_sub_rbs_uca_jcs_uca_chassis = multi_dot([c3,c5])
        self.ubar_ground_jcs_lca_chassis = (multi_dot([c0,c6]) + c2)
        self.ubar_sub_rbs_lca_jcs_lca_chassis = (multi_dot([c7,c6]) + c8)
        self.Mbar_ground_jcs_lca_chassis = multi_dot([c0,c9])
        self.Mbar_sub_rbs_lca_jcs_lca_chassis = multi_dot([c7,c9])
        self.ubar_ground_jcs_strut_chassis = (multi_dot([c0,c10]) + c2)
        self.ubar_sub_rbs_upper_strut_jcs_strut_chassis = (multi_dot([c11,c10]) + c12)
        self.Mbar_ground_jcs_strut_chassis = multi_dot([c0,c13])
        self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis = multi_dot([c11,Triad(config.ax2_jcs_strut_chassis, c13[0:3,1:2])])
        self.ubar_ground_jcs_tie_steering = (multi_dot([c0,c14]) + c2)
        self.ubar_sub_rbs_tie_rod_jcs_tie_steering = (multi_dot([c15,c14]) + c16)
        self.Mbar_ground_jcs_tie_steering = multi_dot([c0,c17])
        self.Mbar_sub_rbs_tie_rod_jcs_tie_steering = multi_dot([c15,Triad(config.ax2_jcs_tie_steering, c17[0:3,1:2])])
        self.ubar_sub_rbs_uca_jcs_uca_upright = (multi_dot([c3,c18]) + c4)
        self.ubar_sub_rbs_upright_jcs_uca_upright = (multi_dot([c19,c18]) + c20)
        self.Mbar_sub_rbs_uca_jcs_uca_upright = multi_dot([c3,c21])
        self.Mbar_sub_rbs_upright_jcs_uca_upright = multi_dot([c19,c21])
        self.ubar_sub_rbs_lca_jcs_lca_upright = (multi_dot([c7,c22]) + c8)
        self.ubar_sub_rbs_upright_jcs_lca_upright = (multi_dot([c19,c22]) + c20)
        self.Mbar_sub_rbs_lca_jcs_lca_upright = multi_dot([c7,c23])
        self.Mbar_sub_rbs_upright_jcs_lca_upright = multi_dot([c19,c23])
        self.ubar_sub_rbs_lca_jcs_strut_lca = (multi_dot([c7,c24]) + c8)
        self.ubar_sub_rbs_lower_strut_jcs_strut_lca = (multi_dot([c25,c24]) + c26)
        self.Mbar_sub_rbs_lca_jcs_strut_lca = multi_dot([c7,c27])
        self.Mbar_sub_rbs_lower_strut_jcs_strut_lca = multi_dot([c25,Triad(config.ax2_jcs_strut_lca, c27[0:3,1:2])])
        self.ubar_sub_rbs_upright_jcs_tie_upright = (multi_dot([c19,c28]) + c20)
        self.ubar_sub_rbs_tie_rod_jcs_tie_upright = (multi_dot([c15,c28]) + c16)
        self.Mbar_sub_rbs_upright_jcs_tie_upright = multi_dot([c19,c29])
        self.Mbar_sub_rbs_tie_rod_jcs_tie_upright = multi_dot([c15,c29])
        self.ubar_sub_rbs_upper_strut_jcs_strut = (multi_dot([c11,c30]) + c12)
        self.ubar_sub_rbs_lower_strut_jcs_strut = (multi_dot([c25,c30]) + c26)
        self.Mbar_sub_rbs_upper_strut_jcs_strut = multi_dot([c11,c31])
        self.Mbar_sub_rbs_lower_strut_jcs_strut = multi_dot([c25,c31])

    
    def set_coordinates(self,q):
        self.R_ground = q[0:3,0:1]
        self.P_ground = q[3:7,0:1]
        self.R_sub_rbs_uca = q[7:10,0:1]
        self.P_sub_rbs_uca = q[10:14,0:1]
        self.R_sub_rbs_lca = q[14:17,0:1]
        self.P_sub_rbs_lca = q[17:21,0:1]
        self.R_sub_rbs_upright = q[21:24,0:1]
        self.P_sub_rbs_upright = q[24:28,0:1]
        self.R_sub_rbs_upper_strut = q[28:31,0:1]
        self.P_sub_rbs_upper_strut = q[31:35,0:1]
        self.R_sub_rbs_lower_strut = q[35:38,0:1]
        self.P_sub_rbs_lower_strut = q[38:42,0:1]
        self.R_sub_rbs_tie_rod = q[42:45,0:1]
        self.P_sub_rbs_tie_rod = q[45:49,0:1]
    
    
        
    def set_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        self.Rd_sub_rbs_uca = qd[7:10,0:1]
        self.Pd_sub_rbs_uca = qd[10:14,0:1]
        self.Rd_sub_rbs_lca = qd[14:17,0:1]
        self.Pd_sub_rbs_lca = qd[17:21,0:1]
        self.Rd_sub_rbs_upright = qd[21:24,0:1]
        self.Pd_sub_rbs_upright = qd[24:28,0:1]
        self.Rd_sub_rbs_upper_strut = qd[28:31,0:1]
        self.Pd_sub_rbs_upper_strut = qd[31:35,0:1]
        self.Rd_sub_rbs_lower_strut = qd[35:38,0:1]
        self.Pd_sub_rbs_lower_strut = qd[38:42,0:1]
        self.Rd_sub_rbs_tie_rod = qd[42:45,0:1]
        self.Pd_sub_rbs_tie_rod = qd[45:49,0:1]
    
    
        
    def set_initial_configuration(self):
        config = self.config
    
        q = np.concatenate([config.R_ground,
        config.P_ground,
        config.R_sub_rbs_uca,
        config.P_sub_rbs_uca,
        config.R_sub_rbs_lca,
        config.P_sub_rbs_lca,
        config.R_sub_rbs_upright,
        config.P_sub_rbs_upright,
        config.R_sub_rbs_upper_strut,
        config.P_sub_rbs_upper_strut,
        config.R_sub_rbs_lower_strut,
        config.P_sub_rbs_lower_strut,
        config.R_sub_rbs_tie_rod,
        config.P_sub_rbs_tie_rod])
    
        qd = np.concatenate([config.Rd_ground,
        config.Pd_ground,
        config.Rd_sub_rbs_uca,
        config.Pd_sub_rbs_uca,
        config.Rd_sub_rbs_lca,
        config.Pd_sub_rbs_lca,
        config.Rd_sub_rbs_upright,
        config.Pd_sub_rbs_upright,
        config.Rd_sub_rbs_upper_strut,
        config.Pd_sub_rbs_upper_strut,
        config.Rd_sub_rbs_lower_strut,
        config.Pd_sub_rbs_lower_strut,
        config.Rd_sub_rbs_tie_rod,
        config.Pd_sub_rbs_tie_rod])
    
        self.set_coordinates(q)
        self.set_velocities(qd)
        self.q_initial = q.copy()
    
    
    
        
    def eval_pos_eq(self):
        F = self.F
        t = self.t
    
        x0 = self.R_ground
        x1 = self.R_sub_rbs_uca
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_sub_rbs_uca
        x5 = A(x4)
        x6 = x3.T
        x7 = self.Mbar_sub_rbs_uca_jcs_uca_chassis[:,2:3]
        x8 = self.R_sub_rbs_lca
        x9 = self.P_sub_rbs_lca
        x10 = A(x9)
        x11 = self.Mbar_sub_rbs_lca_jcs_lca_chassis[:,2:3]
        x12 = self.R_sub_rbs_upright
        x13 = self.R_sub_rbs_upper_strut
        x14 = self.P_sub_rbs_upper_strut
        x15 = A(x14)
        x16 = -1.0*self.R_sub_rbs_tie_rod
        x17 = self.P_sub_rbs_tie_rod
        x18 = A(x17)
        x19 = -1.0*x12
        x20 = self.P_sub_rbs_upright
        x21 = A(x20)
        x22 = -1.0*self.R_sub_rbs_lower_strut
        x23 = self.P_sub_rbs_lower_strut
        x24 = A(x23)
        x25 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,0:1].T
        x26 = x15.T
        x27 = self.Mbar_sub_rbs_lower_strut_jcs_strut[:,2:3]
        x28 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,1:2].T
        x29 = (x13 + x22 + multi_dot([x15,self.ubar_sub_rbs_upper_strut_jcs_strut]) + -1.0*multi_dot([x24,self.ubar_sub_rbs_lower_strut_jcs_strut]))
        x30 = -1.0*np.eye(1,dtype=np.float64)
    
        self.pos_level_data_blocks = [(x0 + -1.0*x1 + multi_dot([x3,self.ubar_ground_jcs_uca_chassis]) + -1.0*multi_dot([x5,self.ubar_sub_rbs_uca_jcs_uca_chassis])),multi_dot([self.Mbar_ground_jcs_uca_chassis[:,0:1].T,x6,x5,x7]),multi_dot([self.Mbar_ground_jcs_uca_chassis[:,1:2].T,x6,x5,x7]),(x0 + -1.0*x8 + multi_dot([x3,self.ubar_ground_jcs_lca_chassis]) + -1.0*multi_dot([x10,self.ubar_sub_rbs_lca_jcs_lca_chassis])),multi_dot([self.Mbar_ground_jcs_lca_chassis[:,0:1].T,x6,x10,x11]),multi_dot([self.Mbar_ground_jcs_lca_chassis[:,1:2].T,x6,x10,x11]),-1*np.eye(1)*F(t) + x12[2],(x0 + -1.0*x13 + multi_dot([x3,self.ubar_ground_jcs_strut_chassis]) + -1.0*multi_dot([x15,self.ubar_sub_rbs_upper_strut_jcs_strut_chassis])),multi_dot([self.Mbar_ground_jcs_strut_chassis[:,0:1].T,x6,x15,self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis[:,0:1]]),(x0 + x16 + multi_dot([x3,self.ubar_ground_jcs_tie_steering]) + -1.0*multi_dot([x18,self.ubar_sub_rbs_tie_rod_jcs_tie_steering])),multi_dot([self.Mbar_ground_jcs_tie_steering[:,0:1].T,x6,x18,self.Mbar_sub_rbs_tie_rod_jcs_tie_steering[:,0:1]]),(x1 + x19 + multi_dot([x5,self.ubar_sub_rbs_uca_jcs_uca_upright]) + -1.0*multi_dot([x21,self.ubar_sub_rbs_upright_jcs_uca_upright])),(x8 + x19 + multi_dot([x10,self.ubar_sub_rbs_lca_jcs_lca_upright]) + -1.0*multi_dot([x21,self.ubar_sub_rbs_upright_jcs_lca_upright])),(x8 + x22 + multi_dot([x10,self.ubar_sub_rbs_lca_jcs_strut_lca]) + -1.0*multi_dot([x24,self.ubar_sub_rbs_lower_strut_jcs_strut_lca])),multi_dot([self.Mbar_sub_rbs_lca_jcs_strut_lca[:,0:1].T,x10.T,x24,self.Mbar_sub_rbs_lower_strut_jcs_strut_lca[:,0:1]]),(x12 + x16 + multi_dot([x21,self.ubar_sub_rbs_upright_jcs_tie_upright]) + -1.0*multi_dot([x18,self.ubar_sub_rbs_tie_rod_jcs_tie_upright])),multi_dot([x25,x26,x24,x27]),multi_dot([x28,x26,x24,x27]),multi_dot([x25,x26,x29]),multi_dot([x28,x26,x29]),x0,(x2 + -1.0*self.Pg_ground),(x30 + (multi_dot([x4.T,x4]))**(1.0/2.0)),(x30 + (multi_dot([x9.T,x9]))**(1.0/2.0)),(x30 + (multi_dot([x20.T,x20]))**(1.0/2.0)),(x30 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x30 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x30 + (multi_dot([x17.T,x17]))**(1.0/2.0))]
    
        self.pos_level_rows_explicit = []
        self.pos_level_cols_explicit = []
        self.pos_level_data_explicit = []
    
        sparse_assembler(self.pos_level_data_blocks, self.pos_level_rows_blocks, self.pos_level_cols_blocks,
                         self.pos_level_data_explicit, self.pos_level_rows_explicit, self.pos_level_cols_explicit)
    
        self.pos_rhs = sc.sparse.coo_matrix(
        (self.pos_level_data_explicit,
        (self.pos_level_rows_explicit,self.pos_level_cols_explicit)),
        (49,1))
    
    
    
        
    def eval_vel_eq(self):
        F = self.F
        t = self.t
    
        x0 = np.zeros((3,1),dtype=np.float64)
        x1 = np.zeros((1,1),dtype=np.float64)
    
        self.vel_level_data_blocks = [x0,x1,x1,x0,x1,x1,np.eye(1)*derivative(F,t,0.1,1),x0,x1,x0,x1,x0,x0,x0,x1,x0,x1,x1,x1,x1,x0,np.zeros((4,1),dtype=np.float64),x1,x1,x1,x1,x1,x1]
    
        self.vel_level_rows_explicit = []
        self.vel_level_cols_explicit = []
        self.vel_level_data_explicit = []
    
        sparse_assembler(self.vel_level_data_blocks, self.vel_level_rows_blocks, self.vel_level_cols_blocks,
                         self.vel_level_data_explicit, self.vel_level_rows_explicit, self.vel_level_cols_explicit)
    
        self.vel_rhs = sc.sparse.coo_matrix(
        (self.vel_level_data_explicit,
        (self.vel_level_rows_explicit,self.vel_level_cols_explicit)),
        (49,1))
    
    
    
        
    def eval_acc_eq(self):
        F = self.F
        t = self.t
    
        a0 = self.Pd_ground
        a1 = self.Pd_sub_rbs_uca
        a2 = self.Mbar_sub_rbs_uca_jcs_uca_chassis[:,2:3]
        a3 = a2.T
        a4 = self.P_sub_rbs_uca
        a5 = A(a4).T
        a6 = self.Mbar_ground_jcs_uca_chassis[:,0:1]
        a7 = self.P_ground
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = self.Mbar_ground_jcs_uca_chassis[:,1:2]
        a13 = self.Pd_sub_rbs_lca
        a14 = self.Mbar_ground_jcs_lca_chassis[:,0:1]
        a15 = self.Mbar_sub_rbs_lca_jcs_lca_chassis[:,2:3]
        a16 = B(a13,a15)
        a17 = a15.T
        a18 = self.P_sub_rbs_lca
        a19 = A(a18).T
        a20 = B(a18,a15)
        a21 = self.Mbar_ground_jcs_lca_chassis[:,1:2]
        a22 = self.Pd_sub_rbs_upper_strut
        a23 = self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis[:,0:1]
        a24 = self.P_sub_rbs_upper_strut
        a25 = A(a24).T
        a26 = self.Mbar_ground_jcs_strut_chassis[:,0:1]
        a27 = self.Pd_sub_rbs_tie_rod
        a28 = self.Mbar_sub_rbs_tie_rod_jcs_tie_steering[:,0:1]
        a29 = self.P_sub_rbs_tie_rod
        a30 = self.Mbar_ground_jcs_tie_steering[:,0:1]
        a31 = self.Pd_sub_rbs_upright
        a32 = self.Pd_sub_rbs_lower_strut
        a33 = self.Mbar_sub_rbs_lca_jcs_strut_lca[:,0:1]
        a34 = self.Mbar_sub_rbs_lower_strut_jcs_strut_lca[:,0:1]
        a35 = self.P_sub_rbs_lower_strut
        a36 = A(a35).T
        a37 = a13.T
        a38 = self.Mbar_sub_rbs_lower_strut_jcs_strut[:,2:3]
        a39 = a38.T
        a40 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,0:1]
        a41 = B(a22,a40)
        a42 = a40.T
        a43 = B(a32,a38)
        a44 = a22.T
        a45 = B(a24,a40).T
        a46 = B(a35,a38)
        a47 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,1:2]
        a48 = B(a22,a47)
        a49 = a47.T
        a50 = B(a24,a47).T
        a51 = self.ubar_sub_rbs_lower_strut_jcs_strut
        a52 = self.ubar_sub_rbs_upper_strut_jcs_strut
        a53 = (multi_dot([B(a32,a51),a32]) + -1.0*multi_dot([B(a22,a52),a22]))
        a54 = (self.Rd_sub_rbs_upper_strut + -1.0*self.Rd_sub_rbs_lower_strut + multi_dot([B(a35,a51),a32]) + multi_dot([B(a24,a52),a22]))
        a55 = (self.R_sub_rbs_upper_strut.T + -1.0*self.R_sub_rbs_lower_strut.T + multi_dot([a52.T,a25]) + -1.0*multi_dot([a51.T,a36]))
    
        self.acc_level_data_blocks = [(multi_dot([B(a0,self.ubar_ground_jcs_uca_chassis),a0]) + -1.0*multi_dot([B(a1,self.ubar_sub_rbs_uca_jcs_uca_chassis),a1])),(multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2.0*multi_dot([a10,B(a7,a6).T,a11,a1])),(multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2.0*multi_dot([a10,B(a7,a12).T,a11,a1])),(multi_dot([B(a0,self.ubar_ground_jcs_lca_chassis),a0]) + -1.0*multi_dot([B(a13,self.ubar_sub_rbs_lca_jcs_lca_chassis),a13])),(multi_dot([a14.T,a8,a16,a13]) + multi_dot([a17,a19,B(a0,a14),a0]) + 2.0*multi_dot([a10,B(a7,a14).T,a20,a13])),(multi_dot([a21.T,a8,a16,a13]) + multi_dot([a17,a19,B(a0,a21),a0]) + 2.0*multi_dot([a10,B(a7,a21).T,a20,a13])),np.eye(1)*derivative(F,t,0.1,2),(multi_dot([B(a0,self.ubar_ground_jcs_strut_chassis),a0]) + -1.0*multi_dot([B(a22,self.ubar_sub_rbs_upper_strut_jcs_strut_chassis),a22])),(multi_dot([a23.T,a25,B(a0,a26),a0]) + multi_dot([a26.T,a8,B(a22,a23),a22]) + 2.0*multi_dot([a10,B(a7,a26).T,B(a24,a23),a22])),(multi_dot([B(a0,self.ubar_ground_jcs_tie_steering),a0]) + -1.0*multi_dot([B(a27,self.ubar_sub_rbs_tie_rod_jcs_tie_steering),a27])),(multi_dot([a28.T,A(a29).T,B(a0,a30),a0]) + multi_dot([a30.T,a8,B(a27,a28),a27]) + 2.0*multi_dot([a10,B(a7,a30).T,B(a29,a28),a27])),(multi_dot([B(a1,self.ubar_sub_rbs_uca_jcs_uca_upright),a1]) + -1.0*multi_dot([B(a31,self.ubar_sub_rbs_upright_jcs_uca_upright),a31])),(multi_dot([B(a13,self.ubar_sub_rbs_lca_jcs_lca_upright),a13]) + -1.0*multi_dot([B(a31,self.ubar_sub_rbs_upright_jcs_lca_upright),a31])),(multi_dot([B(a13,self.ubar_sub_rbs_lca_jcs_strut_lca),a13]) + -1.0*multi_dot([B(a32,self.ubar_sub_rbs_lower_strut_jcs_strut_lca),a32])),(multi_dot([a33.T,a19,B(a32,a34),a32]) + multi_dot([a34.T,a36,B(a13,a33),a13]) + 2.0*multi_dot([a37,B(a18,a33).T,B(a35,a34),a32])),(multi_dot([B(a31,self.ubar_sub_rbs_upright_jcs_tie_upright),a31]) + -1.0*multi_dot([B(a27,self.ubar_sub_rbs_tie_rod_jcs_tie_upright),a27])),(multi_dot([a39,a36,a41,a22]) + multi_dot([a42,a25,a43,a32]) + 2.0*multi_dot([a44,a45,a46,a32])),(multi_dot([a39,a36,a48,a22]) + multi_dot([a49,a25,a43,a32]) + 2.0*multi_dot([a44,a50,a46,a32])),(multi_dot([a42,a25,a53]) + 2.0*multi_dot([a44,a45,a54]) + multi_dot([a55,a41,a22])),(multi_dot([a49,a25,a53]) + 2.0*multi_dot([a44,a50,a54]) + multi_dot([a55,a48,a22])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a37,a13]))**(1.0/2.0),2.0*(multi_dot([a31.T,a31]))**(1.0/2.0),2.0*(multi_dot([a44,a22]))**(1.0/2.0),2.0*(multi_dot([a32.T,a32]))**(1.0/2.0),2.0*(multi_dot([a27.T,a27]))**(1.0/2.0)]
    
        self.acc_level_rows_explicit = []
        self.acc_level_cols_explicit = []
        self.acc_level_data_explicit = []
    
        sparse_assembler(self.acc_level_data_blocks, self.acc_level_rows_blocks, self.acc_level_cols_blocks,
                         self.acc_level_data_explicit, self.acc_level_rows_explicit, self.acc_level_cols_explicit)
    
        self.acc_rhs = sc.sparse.coo_matrix(
        (self.acc_level_data_explicit,
        (self.acc_level_rows_explicit,self.acc_level_cols_explicit)),
        (49,1))
    
    
    
        
    def eval_jacobian(self):
        
        C = self.C
    
        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_sub_rbs_uca_jcs_uca_chassis[:,2:3]
        j4 = j3.T
        j5 = self.P_sub_rbs_uca
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_uca_chassis[:,0:1]
        j8 = self.Mbar_ground_jcs_uca_chassis[:,1:2]
        j9 = -1.0*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_sub_rbs_lca_jcs_lca_chassis[:,2:3]
        j13 = j12.T
        j14 = self.P_sub_rbs_lca
        j15 = A(j14).T
        j16 = self.Mbar_ground_jcs_lca_chassis[:,0:1]
        j17 = self.Mbar_ground_jcs_lca_chassis[:,1:2]
        j18 = B(j14,j12)
        j19 = np.zeros((1,4),dtype=np.float64)
        j20 = self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis[:,0:1]
        j21 = self.P_sub_rbs_upper_strut
        j22 = A(j21).T
        j23 = self.Mbar_ground_jcs_strut_chassis[:,0:1]
        j24 = self.Mbar_sub_rbs_tie_rod_jcs_tie_steering[:,0:1]
        j25 = self.P_sub_rbs_tie_rod
        j26 = self.Mbar_ground_jcs_tie_steering[:,0:1]
        j27 = self.P_sub_rbs_upright
        j28 = self.Mbar_sub_rbs_lower_strut_jcs_strut_lca[:,0:1]
        j29 = self.P_sub_rbs_lower_strut
        j30 = A(j29).T
        j31 = self.Mbar_sub_rbs_lca_jcs_strut_lca[:,0:1]
        j32 = self.Mbar_sub_rbs_lower_strut_jcs_strut[:,2:3]
        j33 = j32.T
        j34 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,0:1]
        j35 = B(j21,j34)
        j36 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,1:2]
        j37 = B(j21,j36)
        j38 = j34.T
        j39 = multi_dot([j38,j22])
        j40 = self.ubar_sub_rbs_upper_strut_jcs_strut
        j41 = B(j21,j40)
        j42 = self.ubar_sub_rbs_lower_strut_jcs_strut
        j43 = (self.R_sub_rbs_upper_strut.T + -1.0*self.R_sub_rbs_lower_strut.T + multi_dot([j40.T,j22]) + -1.0*multi_dot([j42.T,j30]))
        j44 = j36.T
        j45 = multi_dot([j44,j22])
        j46 = B(j29,j32)
        j47 = B(j29,j42)
    
        self.jacobian_data_blocks = [j0,B(j1,self.ubar_ground_jcs_uca_chassis),j9,-1.0*B(j5,self.ubar_sub_rbs_uca_jcs_uca_chassis),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j0,B(j1,self.ubar_ground_jcs_lca_chassis),j9,-1.0*B(j14,self.ubar_sub_rbs_lca_jcs_lca_chassis),j2,multi_dot([j13,j15,B(j1,j16)]),j2,multi_dot([j16.T,j10,j18]),j2,multi_dot([j13,j15,B(j1,j17)]),j2,multi_dot([j17.T,j10,j18]),j2,j19,C,j19,j0,B(j1,self.ubar_ground_jcs_strut_chassis),j9,-1.0*B(j21,self.ubar_sub_rbs_upper_strut_jcs_strut_chassis),j2,multi_dot([j20.T,j22,B(j1,j23)]),j2,multi_dot([j23.T,j10,B(j21,j20)]),j0,B(j1,self.ubar_ground_jcs_tie_steering),j9,-1.0*B(j25,self.ubar_sub_rbs_tie_rod_jcs_tie_steering),j2,multi_dot([j24.T,A(j25).T,B(j1,j26)]),j2,multi_dot([j26.T,j10,B(j25,j24)]),j0,B(j5,self.ubar_sub_rbs_uca_jcs_uca_upright),j9,-1.0*B(j27,self.ubar_sub_rbs_upright_jcs_uca_upright),j0,B(j14,self.ubar_sub_rbs_lca_jcs_lca_upright),j9,-1.0*B(j27,self.ubar_sub_rbs_upright_jcs_lca_upright),j0,B(j14,self.ubar_sub_rbs_lca_jcs_strut_lca),j9,-1.0*B(j29,self.ubar_sub_rbs_lower_strut_jcs_strut_lca),j2,multi_dot([j28.T,j30,B(j14,j31)]),j2,multi_dot([j31.T,j15,B(j29,j28)]),j0,B(j27,self.ubar_sub_rbs_upright_jcs_tie_upright),j9,-1.0*B(j25,self.ubar_sub_rbs_tie_rod_jcs_tie_upright),j2,multi_dot([j33,j30,j35]),j2,multi_dot([j38,j22,j46]),j2,multi_dot([j33,j30,j37]),j2,multi_dot([j44,j22,j46]),-1.0*j39,(-1.0*multi_dot([j38,j22,j41]) + multi_dot([j43,j35])),j39,multi_dot([j38,j22,j47]),-1.0*j45,(-1.0*multi_dot([j44,j22,j41]) + multi_dot([j43,j37])),j45,multi_dot([j44,j22,j47]),j0,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j5.T,2.0*j14.T,2.0*j27.T,2.0*j21.T,2.0*j29.T,2.0*j25.T]
    
        self.jacobian_rows_explicit = []
        self.jacobian_cols_explicit = []
        self.jacobian_data_explicit = []
    
        sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
                         self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)
    
        self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))
    
    

def solve_system(time_array):
    
    try:
        config = inputs()
        config.F = lambda t : 170*sin(t) + 38
        assembled = numerical_assembly(config)
        assembled.eval_constants()
        assembled.set_initial_configuration()
        soln = solver(assembled)
        soln.solve_kds(time_array)
        return soln
    except np.linalg.LinAlgError:
        return soln

time_array = np.arange(0,2*np.pi,0.01)
soln = solve_system(time_array)

import pandas as pd
import matplotlib.pyplot as plt

pos_history = pd.DataFrame(np.concatenate(list(soln.pos_history.values()),1).T,index=soln.pos_history.keys(),columns=range(49))
shape = pos_history.shape
time_array_mod = time_array[:shape[0]-1]

plt.figure(figsize=(8,3))
plt.plot(pos_history[23][:-1],pos_history[22][:-1])
#plt.plot(time_array_mod,pos_history[22][:-1])
#plt.plot(time_array_mod,pos_history[23][:-1])
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
