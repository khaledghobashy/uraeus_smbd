
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from matrix_funcs import A, B, sparse_assembler, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.F_mcs_zact = lambda t : 0
        self.pt_jcs_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcs_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcs_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcs_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.F_jcs_hub_bearing = lambda t : 0
        self.pt_jcs_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_sub_rbs_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_sub_rbs_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_sub_rbs_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_sub_rbs_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)


class numerical_assembly(object):

    def __init__(self,config):
        self.F = config.F
        self.t = 0

        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        self.pos_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        self.vel_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        self.acc_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jacobian_rows_blocks = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,25,25,26,27,28,29,30,31,32])
        self.jacobian_cols_blocks = np.array([0,1,6,7,0,1,2,3,0,1,2,3,0,1,2,3,0,1,4,5,0,1,4,5,0,1,4,5,0,1,8,9,0,1,8,9,0,1,12,13,0,1,12,13,2,3,6,7,4,5,6,7,4,5,10,11,4,5,10,11,6,7,12,13,6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11,0,1,0,1,3,5,7,9,11,13,15])

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
        c13 = Triad(config.ax_jcs_strut_chassis)
        c14 = config.pt_jcs_tie_steering
        c15 = A(config.P_sub_rbs_tie_rod).T
        c16 = -1.0*multi_dot([c15,config.R_sub_rbs_tie_rod])
        c17 = Triad(config.ax_jcs_tie_steering)
        c18 = config.pt_jcs_uca_upright
        c19 = A(config.P_sub_rbs_upright).T
        c20 = -1.0*multi_dot([c19,config.R_sub_rbs_upright])
        c21 = Triad(config.ax_jcs_uca_upright)
        c22 = config.pt_jcs_lca_upright
        c23 = Triad(config.ax_jcs_lca_upright)
        c24 = config.pt_jcs_strut_lca
        c25 = A(config.P_sub_rbs_lower_strut).T
        c26 = -1.0*multi_dot([c25,config.R_sub_rbs_lower_strut])
        c27 = Triad(config.ax_jcs_strut_lca)
        c28 = config.pt_jcs_tie_upright
        c29 = Triad(config.ax_jcs_tie_upright)
        c30 = config.pt_jcs_hub_bearing
        c31 = A(config.P_sub_rbs_hub).T
        c32 = Triad(config.ax_jcs_hub_bearing)
        c33 = config.pt_jcs_strut
        c34 = Triad(config.ax_jcs_strut)

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
        self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis = multi_dot([c11,Triad("'ax2_jcs_strut_chassis'", 'c13[0:3,1:2]')])
        self.ubar_ground_jcs_tie_steering = (multi_dot([c0,c14]) + c2)
        self.ubar_sub_rbs_tie_rod_jcs_tie_steering = (multi_dot([c15,c14]) + c16)
        self.Mbar_ground_jcs_tie_steering = multi_dot([c0,c17])
        self.Mbar_sub_rbs_tie_rod_jcs_tie_steering = multi_dot([c15,Triad("'ax2_jcs_tie_steering'", 'c17[0:3,1:2]')])
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
        self.Mbar_sub_rbs_lower_strut_jcs_strut_lca = multi_dot([c25,Triad("'ax2_jcs_strut_lca'", 'c27[0:3,1:2]')])
        self.ubar_sub_rbs_upright_jcs_tie_upright = (multi_dot([c19,c28]) + c20)
        self.ubar_sub_rbs_tie_rod_jcs_tie_upright = (multi_dot([c15,c28]) + c16)
        self.Mbar_sub_rbs_upright_jcs_tie_upright = multi_dot([c19,c29])
        self.Mbar_sub_rbs_tie_rod_jcs_tie_upright = multi_dot([c15,c29])
        self.ubar_sub_rbs_upright_jcs_hub_bearing = (multi_dot([c19,c30]) + c20)
        self.ubar_sub_rbs_hub_jcs_hub_bearing = (multi_dot([c31,c30]) + -1.0*multi_dot([c31,'R_sub_rbs_hub']))
        self.Mbar_sub_rbs_upright_jcs_hub_bearing = multi_dot([c19,c32])
        self.Mbar_sub_rbs_hub_jcs_hub_bearing = multi_dot([c31,c32])
        self.ubar_sub_rbs_upper_strut_jcs_strut = (multi_dot([c11,c33]) + c12)
        self.ubar_sub_rbs_lower_strut_jcs_strut = (multi_dot([c25,c33]) + c26)
        self.Mbar_sub_rbs_upper_strut_jcs_strut = multi_dot([c11,c34])
        self.Mbar_sub_rbs_lower_strut_jcs_strut = multi_dot([c25,c34])

    
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
    self.R_sub_rbs_hub = q[49:52,0:1]
    self.P_sub_rbs_hub = q[52:56,0:1]


    
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
    self.Rd_sub_rbs_hub = qd[49:52,0:1]
    self.Pd_sub_rbs_hub = qd[52:56,0:1]


    
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
    config.P_sub_rbs_tie_rod,
    config.R_sub_rbs_hub,
    config.P_sub_rbs_hub])

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
    config.Pd_sub_rbs_tie_rod,
    config.Rd_sub_rbs_hub,
    config.Pd_sub_rbs_hub])

    self.set_coordinates(q)
    self.set_velocities(qd)
    self.q_initial = q.copy()



    
def eval_pos_eq(self):
    F = self.F
    t = self.t

    x0 = self.R_ground
    x1 = np.eye(1,dtype=np.float64)
    x2 = self.R_sub_rbs_uca
    x3 = self.P_ground
    x4 = A(x3)
    x5 = self.P_sub_rbs_uca
    x6 = A(x5)
    x7 = x4.T
    x8 = self.Mbar_sub_rbs_uca_jcs_uca_chassis[:,2:3]
    x9 = self.R_sub_rbs_lca
    x10 = self.P_sub_rbs_lca
    x11 = A(x10)
    x12 = self.Mbar_sub_rbs_lca_jcs_lca_chassis[:,2:3]
    x13 = self.R_sub_rbs_upper_strut
    x14 = self.P_sub_rbs_upper_strut
    x15 = A(x14)
    x16 = -1.0*self.R_sub_rbs_tie_rod
    x17 = self.P_sub_rbs_tie_rod
    x18 = A(x17)
    x19 = self.R_sub_rbs_upright
    x20 = -1.0*x19
    x21 = self.P_sub_rbs_upright
    x22 = A(x21)
    x23 = -1.0*self.R_sub_rbs_lower_strut
    x24 = self.P_sub_rbs_lower_strut
    x25 = A(x24)
    x26 = self.P_sub_rbs_hub
    x27 = A(x26)
    x28 = x22.T
    x29 = self.Mbar_sub_rbs_hub_jcs_hub_bearing[:,2:3]
    x30 = F_jcs_hub_bearing(t)
    x31 = self.Mbar_sub_rbs_hub_jcs_hub_bearing[:,0:1]
    x32 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,0:1].T
    x33 = x15.T
    x34 = self.Mbar_sub_rbs_lower_strut_jcs_strut[:,2:3]
    x35 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,1:2].T
    x36 = (x13 + x23 + multi_dot([x15,self.ubar_sub_rbs_upper_strut_jcs_strut]) + -1.0*multi_dot([x25,self.ubar_sub_rbs_lower_strut_jcs_strut]))
    x37 = -1.0*x1

    self.pos_level_data_blocks = [-1*F_mcs_zact(t) + x0[2]*x1,(x0 + -1.0*x2 + multi_dot([x4,self.ubar_ground_jcs_uca_chassis]) + -1.0*multi_dot([x6,self.ubar_sub_rbs_uca_jcs_uca_chassis])),multi_dot([self.Mbar_ground_jcs_uca_chassis[:,0:1].T,x7,x6,x8]),multi_dot([self.Mbar_ground_jcs_uca_chassis[:,1:2].T,x7,x6,x8]),(x0 + -1.0*x9 + multi_dot([x4,self.ubar_ground_jcs_lca_chassis]) + -1.0*multi_dot([x11,self.ubar_sub_rbs_lca_jcs_lca_chassis])),multi_dot([self.Mbar_ground_jcs_lca_chassis[:,0:1].T,x7,x11,x12]),multi_dot([self.Mbar_ground_jcs_lca_chassis[:,1:2].T,x7,x11,x12]),(x0 + -1.0*x13 + multi_dot([x4,self.ubar_ground_jcs_strut_chassis]) + -1.0*multi_dot([x15,self.ubar_sub_rbs_upper_strut_jcs_strut_chassis])),multi_dot([self.Mbar_ground_jcs_strut_chassis[:,0:1].T,x7,x15,self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis[:,0:1]]),(x0 + x16 + multi_dot([x4,self.ubar_ground_jcs_tie_steering]) + -1.0*multi_dot([x18,self.ubar_sub_rbs_tie_rod_jcs_tie_steering])),multi_dot([self.Mbar_ground_jcs_tie_steering[:,0:1].T,x7,x18,self.Mbar_sub_rbs_tie_rod_jcs_tie_steering[:,0:1]]),(x2 + x20 + multi_dot([x6,self.ubar_sub_rbs_uca_jcs_uca_upright]) + -1.0*multi_dot([x22,self.ubar_sub_rbs_upright_jcs_uca_upright])),(x9 + x20 + multi_dot([x11,self.ubar_sub_rbs_lca_jcs_lca_upright]) + -1.0*multi_dot([x22,self.ubar_sub_rbs_upright_jcs_lca_upright])),(x9 + x23 + multi_dot([x11,self.ubar_sub_rbs_lca_jcs_strut_lca]) + -1.0*multi_dot([x25,self.ubar_sub_rbs_lower_strut_jcs_strut_lca])),multi_dot([self.Mbar_sub_rbs_lca_jcs_strut_lca[:,0:1].T,x11.T,x25,self.Mbar_sub_rbs_lower_strut_jcs_strut_lca[:,0:1]]),(x19 + x16 + multi_dot([x22,self.ubar_sub_rbs_upright_jcs_tie_upright]) + -1.0*multi_dot([x18,self.ubar_sub_rbs_tie_rod_jcs_tie_upright])),(x19 + -1.0*self.R_sub_rbs_hub + multi_dot([x22,self.ubar_sub_rbs_upright_jcs_hub_bearing]) + -1.0*multi_dot([x27,self.ubar_sub_rbs_hub_jcs_hub_bearing])),multi_dot([self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,0:1].T,x28,x27,x29]),multi_dot([self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,1:2].T,x28,x27,x29]),(cos(x30)*multi_dot([self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,1:2].T,x28,x27,x31]) + sin(x30)*-1.0*multi_dot([self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,0:1].T,x28,x27,x31])),multi_dot([x32,x33,x25,x34]),multi_dot([x35,x33,x25,x34]),multi_dot([x32,x33,x36]),multi_dot([x35,x33,x36]),x0,(x3 + -1.0*'Pg_ground'),(x37 + (multi_dot([x5.T,x5]))**(1.0/2.0)),(x37 + (multi_dot([x10.T,x10]))**(1.0/2.0)),(x37 + (multi_dot([x21.T,x21]))**(1.0/2.0)),(x37 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x37 + (multi_dot([x24.T,x24]))**(1.0/2.0)),(x37 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x37 + (multi_dot([x26.T,x26]))**(1.0/2.0))]

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

    x0 = np.zeros((1,1),dtype=np.float64)
    x1 = np.eye(1,dtype=np.float64)
    x2 = np.zeros((3,1),dtype=np.float64)

    self.vel_level_data_blocks = [(x0 + derivative(F,t,0.1,1)*-1.0*x1),x2,x0,x0,x2,x0,x0,x2,x0,x2,x0,x2,x2,x2,x0,x2,x2,x0,x0,(x0 + derivative(F,t,0.1,1)*-1.0*x1),x0,x0,x0,x0,x2,np.zeros((4,1),dtype=np.float64),x0,x0,x0,x0,x0,x0,x0]

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

    a0 = np.eye(1,dtype=np.float64)
    a1 = self.Pd_ground
    a2 = self.Pd_sub_rbs_uca
    a3 = self.Mbar_ground_jcs_uca_chassis[:,0:1]
    a4 = self.P_ground
    a5 = A(a4).T
    a6 = self.Mbar_sub_rbs_uca_jcs_uca_chassis[:,2:3]
    a7 = B(a2,a6)
    a8 = a6.T
    a9 = self.P_sub_rbs_uca
    a10 = A(a9).T
    a11 = a1.T
    a12 = B(a9,a6)
    a13 = self.Mbar_ground_jcs_uca_chassis[:,1:2]
    a14 = self.Pd_sub_rbs_lca
    a15 = self.Mbar_sub_rbs_lca_jcs_lca_chassis[:,2:3]
    a16 = a15.T
    a17 = self.P_sub_rbs_lca
    a18 = A(a17).T
    a19 = self.Mbar_ground_jcs_lca_chassis[:,0:1]
    a20 = B(a14,a15)
    a21 = B(a17,a15)
    a22 = self.Mbar_ground_jcs_lca_chassis[:,1:2]
    a23 = self.Pd_sub_rbs_upper_strut
    a24 = self.Mbar_ground_jcs_strut_chassis[:,0:1]
    a25 = self.Mbar_sub_rbs_upper_strut_jcs_strut_chassis[:,0:1]
    a26 = self.P_sub_rbs_upper_strut
    a27 = A(a26).T
    a28 = self.Pd_sub_rbs_tie_rod
    a29 = self.Mbar_ground_jcs_tie_steering[:,0:1]
    a30 = self.Mbar_sub_rbs_tie_rod_jcs_tie_steering[:,0:1]
    a31 = self.P_sub_rbs_tie_rod
    a32 = self.Pd_sub_rbs_upright
    a33 = self.Pd_sub_rbs_lower_strut
    a34 = self.Mbar_sub_rbs_lca_jcs_strut_lca[:,0:1]
    a35 = self.Mbar_sub_rbs_lower_strut_jcs_strut_lca[:,0:1]
    a36 = self.P_sub_rbs_lower_strut
    a37 = A(a36).T
    a38 = a14.T
    a39 = self.Pd_sub_rbs_hub
    a40 = self.Mbar_sub_rbs_hub_jcs_hub_bearing[:,2:3]
    a41 = a40.T
    a42 = self.P_sub_rbs_hub
    a43 = A(a42).T
    a44 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,0:1]
    a45 = self.P_sub_rbs_upright
    a46 = A(a45).T
    a47 = B(a39,a40)
    a48 = a32.T
    a49 = B(a42,a40)
    a50 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,1:2]
    a51 = F_jcs_hub_bearing(t)
    a52 = self.Mbar_sub_rbs_hub_jcs_hub_bearing[:,0:1]
    a53 = cos(a51)
    a54 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,1:2]
    a55 = sin(a51)
    a56 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,0:1]
    a57 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,0:1]
    a58 = a57.T
    a59 = self.Mbar_sub_rbs_lower_strut_jcs_strut[:,2:3]
    a60 = B(a33,a59)
    a61 = a59.T
    a62 = B(a23,a57)
    a63 = a23.T
    a64 = B(a26,a57).T
    a65 = B(a36,a59)
    a66 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,1:2]
    a67 = a66.T
    a68 = B(a23,a66)
    a69 = B(a26,a66).T
    a70 = self.ubar_sub_rbs_lower_strut_jcs_strut
    a71 = self.ubar_sub_rbs_upper_strut_jcs_strut
    a72 = (multi_dot([B(a33,a70),a33]) + -1.0*multi_dot([B(a23,a71),a23]))
    a73 = (self.Rd_sub_rbs_upper_strut + -1.0*self.Rd_sub_rbs_lower_strut + multi_dot([B(a36,a70),a33]) + multi_dot([B(a26,a71),a23]))
    a74 = (self.R_sub_rbs_upper_strut.T + -1.0*self.R_sub_rbs_lower_strut.T + multi_dot([a71.T,a27]) + -1.0*multi_dot([a70.T,a37]))

    self.acc_level_data_blocks = [(np.zeros((1,1),dtype=np.float64) + derivative(F,t,0.1,2)*-1.0*a0),(multi_dot([B(a1,self.ubar_ground_jcs_uca_chassis),a1]) + -1.0*multi_dot([B(a2,self.ubar_sub_rbs_uca_jcs_uca_chassis),a2])),(multi_dot([a3.T,a5,a7,a2]) + multi_dot([a8,a10,B(a1,a3),a1]) + 2.0*multi_dot([a11,B(a4,a3).T,a12,a2])),(multi_dot([a13.T,a5,a7,a2]) + multi_dot([a8,a10,B(a1,a13),a1]) + 2.0*multi_dot([a11,B(a4,a13).T,a12,a2])),(multi_dot([B(a1,self.ubar_ground_jcs_lca_chassis),a1]) + -1.0*multi_dot([B(a14,self.ubar_sub_rbs_lca_jcs_lca_chassis),a14])),(multi_dot([a16,a18,B(a1,a19),a1]) + multi_dot([a19.T,a5,a20,a14]) + 2.0*multi_dot([a11,B(a4,a19).T,a21,a14])),(multi_dot([a16,a18,B(a1,a22),a1]) + multi_dot([a22.T,a5,a20,a14]) + 2.0*multi_dot([a11,B(a4,a22).T,a21,a14])),(multi_dot([B(a1,self.ubar_ground_jcs_strut_chassis),a1]) + -1.0*multi_dot([B(a23,self.ubar_sub_rbs_upper_strut_jcs_strut_chassis),a23])),(multi_dot([a24.T,a5,B(a23,a25),a23]) + multi_dot([a25.T,a27,B(a1,a24),a1]) + 2.0*multi_dot([a11,B(a4,a24).T,B(a26,a25),a23])),(multi_dot([B(a1,self.ubar_ground_jcs_tie_steering),a1]) + -1.0*multi_dot([B(a28,self.ubar_sub_rbs_tie_rod_jcs_tie_steering),a28])),(multi_dot([a29.T,a5,B(a28,a30),a28]) + multi_dot([a30.T,A(a31).T,B(a1,a29),a1]) + 2.0*multi_dot([a11,B(a4,a29).T,B(a31,a30),a28])),(multi_dot([B(a2,self.ubar_sub_rbs_uca_jcs_uca_upright),a2]) + -1.0*multi_dot([B(a32,self.ubar_sub_rbs_upright_jcs_uca_upright),a32])),(multi_dot([B(a14,self.ubar_sub_rbs_lca_jcs_lca_upright),a14]) + -1.0*multi_dot([B(a32,self.ubar_sub_rbs_upright_jcs_lca_upright),a32])),(multi_dot([B(a14,self.ubar_sub_rbs_lca_jcs_strut_lca),a14]) + -1.0*multi_dot([B(a33,self.ubar_sub_rbs_lower_strut_jcs_strut_lca),a33])),(multi_dot([a34.T,a18,B(a33,a35),a33]) + multi_dot([a35.T,a37,B(a14,a34),a14]) + 2.0*multi_dot([a38,B(a17,a34).T,B(a36,a35),a33])),(multi_dot([B(a32,self.ubar_sub_rbs_upright_jcs_tie_upright),a32]) + -1.0*multi_dot([B(a28,self.ubar_sub_rbs_tie_rod_jcs_tie_upright),a28])),(multi_dot([B(a32,self.ubar_sub_rbs_upright_jcs_hub_bearing),a32]) + -1.0*multi_dot([B(a39,self.ubar_sub_rbs_hub_jcs_hub_bearing),a39])),(multi_dot([a41,a43,B(a32,a44),a32]) + multi_dot([a44.T,a46,a47,a39]) + 2.0*multi_dot([a48,B(a45,a44).T,a49,a39])),(multi_dot([a41,a43,B(a32,a50),a32]) + multi_dot([a50.T,a46,a47,a39]) + 2.0*multi_dot([a48,B(a45,a50).T,a49,a39])),(derivative(F,t,0.1,2)*-1.0*a0 + multi_dot([a52.T,a43,(a53*B(a32,a54) + a55*-1.0*B(a32,a56)),a32]) + multi_dot([(a53*multi_dot([a54.T,a46]) + a55*-1.0*multi_dot([a56.T,a46])),B(a39,a52),a39]) + 2.0*multi_dot([((a53*multi_dot([B(a45,a54),a32])).T + transpose(a55)*-1.0*multi_dot([a48,B(a45,a56).T])),B(a42,a52),a39])),(multi_dot([a58,a27,a60,a33]) + multi_dot([a61,a37,a62,a23]) + 2.0*multi_dot([a63,a64,a65,a33])),(multi_dot([a67,a27,a60,a33]) + multi_dot([a61,a37,a68,a23]) + 2.0*multi_dot([a63,a69,a65,a33])),(multi_dot([a58,a27,a72]) + 2.0*multi_dot([a63,a64,a73]) + multi_dot([a74,a62,a23])),(multi_dot([a67,a27,a72]) + 2.0*multi_dot([a63,a69,a73]) + multi_dot([a74,a68,a23])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a2.T,a2]))**(1.0/2.0),2.0*(multi_dot([a38,a14]))**(1.0/2.0),2.0*(multi_dot([a48,a32]))**(1.0/2.0),2.0*(multi_dot([a63,a23]))**(1.0/2.0),2.0*(multi_dot([a33.T,a33]))**(1.0/2.0),2.0*(multi_dot([a28.T,a28]))**(1.0/2.0),2.0*(multi_dot([a39.T,a39]))**(1.0/2.0)]

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

    j0 = np.zeros((1,4),dtype=np.float64)
    j1 = np.zeros((1,3),dtype=np.float64)
    j2 = np.eye(3,dtype=np.float64)
    j3 = self.P_ground
    j4 = self.Mbar_sub_rbs_uca_jcs_uca_chassis[:,2:3]
    j5 = j4.T
    j6 = self.P_sub_rbs_uca
    j7 = A(j6).T
    j8 = self.Mbar_ground_jcs_uca_chassis[:,0:1]
    j9 = self.Mbar_ground_jcs_uca_chassis[:,1:2]
    j10 = -1.0*j2
    j11 = A(j3).T
    j12 = B(j6,j4)
    j13 = self.Mbar_sub_rbs_lca_jcs_lca_chassis[:,2:3]
    j14 = j13.T
    j15 = self.P_sub_rbs_lca
    j16 = A(j15).T
    j17 = self.Mbar_ground_jcs_lca_chassis[:,0:1]
    j18 = self.Mbar_ground_jcs_lca_chassis[:,1:2]
    j19 = B(j15,j13)
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
    j32 = self.Mbar_sub_rbs_hub_jcs_hub_bearing[:,2:3]
    j33 = j32.T
    j34 = self.P_sub_rbs_hub
    j35 = A(j34).T
    j36 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,0:1]
    j37 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,1:2]
    j38 = A(j27).T
    j39 = B(j34,j32)
    j40 = self.Mbar_sub_rbs_hub_jcs_hub_bearing[:,0:1]
    j41 = F_jcs_hub_bearing(t)
    j42 = cos(j41)
    j43 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,1:2]
    j44 = self.Mbar_sub_rbs_upright_jcs_hub_bearing[:,0:1]
    j45 = self.Mbar_sub_rbs_lower_strut_jcs_strut[:,2:3]
    j46 = j45.T
    j47 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,0:1]
    j48 = B(j21,j47)
    j49 = self.Mbar_sub_rbs_upper_strut_jcs_strut[:,1:2]
    j50 = B(j21,j49)
    j51 = j47.T
    j52 = multi_dot([j51,j22])
    j53 = self.ubar_sub_rbs_upper_strut_jcs_strut
    j54 = B(j21,j53)
    j55 = self.ubar_sub_rbs_lower_strut_jcs_strut
    j56 = (self.R_sub_rbs_upper_strut.T + -1.0*self.R_sub_rbs_lower_strut.T + multi_dot([j53.T,j22]) + -1.0*multi_dot([j55.T,j30]))
    j57 = j49.T
    j58 = multi_dot([j57,j22])
    j59 = B(j29,j45)
    j60 = B(j29,j55)

    self.jacobian_data_blocks = [J_mcs_zact,j0,j1,j0,j2,B(j3,self.ubar_ground_jcs_uca_chassis),j10,-1.0*B(j6,self.ubar_sub_rbs_uca_jcs_uca_chassis),j1,multi_dot([j5,j7,B(j3,j8)]),j1,multi_dot([j8.T,j11,j12]),j1,multi_dot([j5,j7,B(j3,j9)]),j1,multi_dot([j9.T,j11,j12]),j2,B(j3,self.ubar_ground_jcs_lca_chassis),j10,-1.0*B(j15,self.ubar_sub_rbs_lca_jcs_lca_chassis),j1,multi_dot([j14,j16,B(j3,j17)]),j1,multi_dot([j17.T,j11,j19]),j1,multi_dot([j14,j16,B(j3,j18)]),j1,multi_dot([j18.T,j11,j19]),j2,B(j3,self.ubar_ground_jcs_strut_chassis),j10,-1.0*B(j21,self.ubar_sub_rbs_upper_strut_jcs_strut_chassis),j1,multi_dot([j20.T,j22,B(j3,j23)]),j1,multi_dot([j23.T,j11,B(j21,j20)]),j2,B(j3,self.ubar_ground_jcs_tie_steering),j10,-1.0*B(j25,self.ubar_sub_rbs_tie_rod_jcs_tie_steering),j1,multi_dot([j24.T,A(j25).T,B(j3,j26)]),j1,multi_dot([j26.T,j11,B(j25,j24)]),j2,B(j6,self.ubar_sub_rbs_uca_jcs_uca_upright),j10,-1.0*B(j27,self.ubar_sub_rbs_upright_jcs_uca_upright),j2,B(j15,self.ubar_sub_rbs_lca_jcs_lca_upright),j10,-1.0*B(j27,self.ubar_sub_rbs_upright_jcs_lca_upright),j2,B(j15,self.ubar_sub_rbs_lca_jcs_strut_lca),j10,-1.0*B(j29,self.ubar_sub_rbs_lower_strut_jcs_strut_lca),j1,multi_dot([j28.T,j30,B(j15,j31)]),j1,multi_dot([j31.T,j16,B(j29,j28)]),j2,B(j27,self.ubar_sub_rbs_upright_jcs_tie_upright),j10,-1.0*B(j25,self.ubar_sub_rbs_tie_rod_jcs_tie_upright),j2,B(j27,self.ubar_sub_rbs_upright_jcs_hub_bearing),j10,-1.0*B(j34,self.ubar_sub_rbs_hub_jcs_hub_bearing),j1,multi_dot([j33,j35,B(j27,j36)]),j1,multi_dot([j36.T,j38,j39]),j1,multi_dot([j33,j35,B(j27,j37)]),j1,multi_dot([j37.T,j38,j39]),j1,multi_dot([j40.T,j35,(j42*B(j27,j43) + sin(j41)*-1.0*B(j27,j44))]),j1,multi_dot([(j42*multi_dot([j43.T,j38]) + sin(j41)*-1.0*multi_dot([j44.T,j38])),B(j34,j40)]),j1,multi_dot([j46,j30,j48]),j1,multi_dot([j51,j22,j59]),j1,multi_dot([j46,j30,j50]),j1,multi_dot([j57,j22,j59]),-1.0*j52,(-1.0*multi_dot([j51,j22,j54]) + multi_dot([j56,j48])),j52,multi_dot([j51,j22,j60]),-1.0*j58,(-1.0*multi_dot([j57,j22,j54]) + multi_dot([j56,j50])),j58,multi_dot([j57,j22,j60]),j2,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j6.T,2.0*j15.T,2.0*j27.T,2.0*j21.T,2.0*j29.T,2.0*j25.T,2.0*j34.T]

    self.jacobian_rows_explicit = []
    self.jacobian_cols_explicit = []
    self.jacobian_data_explicit = []

    sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
                     self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)

    self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))



