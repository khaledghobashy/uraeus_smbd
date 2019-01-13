
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin
from source.solvers.python_solver import solver


class inputs(object):

    def __init__(self):
        self.pt_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_rc_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_rc_cyl = np.array([[0], [0], [1]],dtype=np.float64)
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.P_ST_rbs_coupler).T
        c1 = self.pt_jcs_rc_sph
        c2 = -1.0*multi_dot([c0,self.R_ST_rbs_coupler])
        c3 = A(self.P_ST_rbr_rocker).T
        c4 = Triad(self.ax_jcs_rc_sph,)
        c5 = self.pt_jcs_rc_cyl
        c6 = A(self.P_ST_rbl_rocker).T
        c7 = Triad(self.ax_jcs_rc_cyl,)

        self.ubar_ST_rbs_coupler_jcs_rc_sph = (multi_dot([c0,c1]) + c2)
        self.ubar_ST_rbr_rocker_jcs_rc_sph = (multi_dot([c3,c1]) + -1.0*multi_dot([c3,self.R_ST_rbr_rocker]))
        self.Mbar_ST_rbs_coupler_jcs_rc_sph = multi_dot([c0,c4])
        self.Mbar_ST_rbr_rocker_jcs_rc_sph = multi_dot([c3,c4])
        self.ubar_ST_rbs_coupler_jcs_rc_cyl = (multi_dot([c0,c5]) + c2)
        self.ubar_ST_rbl_rocker_jcs_rc_cyl = (multi_dot([c6,c5]) + -1.0*multi_dot([c6,self.R_ST_rbl_rocker]))
        self.Mbar_ST_rbs_coupler_jcs_rc_cyl = multi_dot([c0,c7])
        self.Mbar_ST_rbl_rocker_jcs_rc_cyl = multi_dot([c6,c7])

    @property
    def q_initial(self):
        q = np.concatenate([self.R_ground,self.P_ground,self.R_ST_rbs_coupler,self.P_ST_rbs_coupler,self.R_ST_rbr_rocker,self.P_ST_rbr_rocker,self.R_ST_rbl_rocker,self.P_ST_rbl_rocker])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.Rd_ground,self.Pd_ground,self.Rd_ST_rbs_coupler,self.Pd_ST_rbs_coupler,self.Rd_ST_rbr_rocker,self.Pd_ST_rbr_rocker,self.Rd_ST_rbl_rocker,self.Pd_ST_rbl_rocker])
        return qd



class numerical_assembly(object):

    def __init__(self,config):
        self.t = 0.0
        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.nve = 19

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,6,6,7,8,9])
        self.jac_cols = np.array([2,3,4,5,2,3,6,7,2,3,6,7,2,3,6,7,2,3,6,7,0,1,0,1,3,5,7])

    
    def set_q(self,q):
        self.R_ground = q[0:3,0:1]
        self.P_ground = q[3:7,0:1]
        self.R_ST_rbs_coupler = q[7:10,0:1]
        self.P_ST_rbs_coupler = q[10:14,0:1]
        self.R_ST_rbr_rocker = q[14:17,0:1]
        self.P_ST_rbr_rocker = q[17:21,0:1]
        self.R_ST_rbl_rocker = q[21:24,0:1]
        self.P_ST_rbl_rocker = q[24:28,0:1]

    
    def set_qd(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        self.Rd_ST_rbs_coupler = qd[7:10,0:1]
        self.Pd_ST_rbs_coupler = qd[10:14,0:1]
        self.Rd_ST_rbr_rocker = qd[14:17,0:1]
        self.Pd_ST_rbr_rocker = qd[17:21,0:1]
        self.Rd_ST_rbl_rocker = qd[21:24,0:1]
        self.Pd_ST_rbl_rocker = qd[24:28,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ST_rbs_coupler
        x1 = self.P_ST_rbs_coupler
        x2 = A(x1)
        x3 = self.P_ST_rbr_rocker
        x4 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1].T
        x5 = x2.T
        x6 = self.P_ST_rbl_rocker
        x7 = A(x6)
        x8 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        x9 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2].T
        x10 = (x0 + -1.0*self.R_ST_rbl_rocker + multi_dot([x2,config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3]]) + multi_dot([x2,config.ubar_ST_rbs_coupler_jcs_rc_cyl]) + -1.0*multi_dot([x7,config.ubar_ST_rbl_rocker_jcs_rc_cyl]))
        x11 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1.0*self.R_ST_rbr_rocker + multi_dot([x2,config.ubar_ST_rbs_coupler_jcs_rc_sph]) + -1.0*multi_dot([A(x3),config.ubar_ST_rbr_rocker_jcs_rc_sph])),multi_dot([x4,x5,x7,x8]),multi_dot([x9,x5,x7,x8]),multi_dot([x4,x5,x10]),multi_dot([x9,x5,x10]),self.R_ground,(self.P_ground + -1.0*config.Pg_ground),(x11 + (multi_dot([x1.T,x1]))**(1.0/2.0)),(x11 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x11 + (multi_dot([x6.T,x6]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v1,v1,v0,np.zeros((4,1),dtype=np.float64),v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_ST_rbs_coupler
        a1 = self.Pd_ST_rbr_rocker
        a2 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1]
        a3 = a2.T
        a4 = self.P_ST_rbs_coupler
        a5 = A(a4).T
        a6 = self.Pd_ST_rbl_rocker
        a7 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        a8 = B(a6,a7)
        a9 = a7.T
        a10 = self.P_ST_rbl_rocker
        a11 = A(a10).T
        a12 = B(a0,a2)
        a13 = a0.T
        a14 = B(a4,a2).T
        a15 = B(a10,a7)
        a16 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2]
        a17 = a16.T
        a18 = B(a0,a16)
        a19 = B(a4,a16).T
        a20 = config.ubar_ST_rbs_coupler_jcs_rc_cyl
        a21 = config.ubar_ST_rbl_rocker_jcs_rc_cyl
        a22 = (multi_dot([B(a0,a20),a0]) + -1.0*multi_dot([B(a6,a21),a6]))
        a23 = (self.Rd_ST_rbs_coupler + -1.0*self.Rd_ST_rbl_rocker + multi_dot([B(a10,a21),a6]) + multi_dot([B(a4,a20),a0]))
        a24 = (self.R_ST_rbs_coupler.T + -1.0*self.R_ST_rbl_rocker.T + multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3].T,a5]) + multi_dot([a20.T,a5]) + -1.0*multi_dot([a21.T,a11]))

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_ST_rbs_coupler_jcs_rc_sph),a0]) + -1.0*multi_dot([B(a1,config.ubar_ST_rbr_rocker_jcs_rc_sph),a1])),(multi_dot([a3,a5,a8,a6]) + multi_dot([a9,a11,a12,a0]) + 2.0*multi_dot([a13,a14,a15,a6])),(multi_dot([a17,a5,a8,a6]) + multi_dot([a9,a11,a18,a0]) + 2.0*multi_dot([a13,a19,a15,a6])),(multi_dot([a3,a5,a22]) + 2.0*multi_dot([a13,a14,a23]) + multi_dot([a24,a12,a0])),(multi_dot([a17,a5,a22]) + 2.0*multi_dot([a13,a19,a23]) + multi_dot([a24,a18,a0])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a13,a0]))**(1.0/2.0),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a6.T,a6]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_ST_rbs_coupler
        j2 = self.P_ST_rbr_rocker
        j3 = np.zeros((1,3),dtype=np.float64)
        j4 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        j5 = j4.T
        j6 = self.P_ST_rbl_rocker
        j7 = A(j6).T
        j8 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1]
        j9 = B(j1,j8)
        j10 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2]
        j11 = B(j1,j10)
        j12 = j8.T
        j13 = A(j1).T
        j14 = multi_dot([j12,j13])
        j15 = config.ubar_ST_rbs_coupler_jcs_rc_cyl
        j16 = B(j1,j15)
        j17 = config.ubar_ST_rbl_rocker_jcs_rc_cyl
        j18 = (self.R_ST_rbs_coupler.T + -1.0*self.R_ST_rbl_rocker.T + multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3].T,j13]) + multi_dot([j15.T,j13]) + -1.0*multi_dot([j17.T,j7]))
        j19 = j10.T
        j20 = multi_dot([j19,j13])
        j21 = B(j6,j4)
        j22 = B(j6,j17)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_ST_rbs_coupler_jcs_rc_sph),-1.0*j0,-1.0*B(j2,config.ubar_ST_rbr_rocker_jcs_rc_sph),j3,multi_dot([j5,j7,j9]),j3,multi_dot([j12,j13,j21]),j3,multi_dot([j5,j7,j11]),j3,multi_dot([j19,j13,j21]),j14,(multi_dot([j12,j13,j16]) + multi_dot([j18,j9])),-1.0*j14,-1.0*multi_dot([j12,j13,j22]),j20,(multi_dot([j19,j13,j16]) + multi_dot([j18,j11])),-1.0*j20,-1.0*multi_dot([j19,j13,j22]),j0,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j1.T,2.0*j2.T,2.0*j6.T]
  
