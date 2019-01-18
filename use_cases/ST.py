
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rc_cyl = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcs_rc_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.R_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.P_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.ST.Rd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.Pd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ST.R_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.P_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.ST.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ST.R_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.P_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.ST.Rd_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.Pd_rbl_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.ST.P_rbr_rocker).T
        c1 = self.pt1_jcs_rc_sph
        c2 = A(self.ST.P_rbs_coupler).T
        c3 = -1.0*multi_dot([c2,self.ST.R_rbs_coupler])
        c4 = Triad(self.ax1_jcs_rc_sph,)
        c5 = self.pt1_jcs_rc_cyl
        c6 = A(self.ST.P_rbl_rocker).T
        c7 = Triad(self.ax1_jcs_rc_cyl,)

        self.ubar_rbr_rocker_jcs_rc_sph = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,self.ST.R_rbr_rocker]))
        self.ubar_rbs_coupler_jcs_rc_sph = (multi_dot([c2,c1]) + c3)
        self.Mbar_rbr_rocker_jcs_rc_sph = multi_dot([c0,c4])
        self.Mbar_rbs_coupler_jcs_rc_sph = multi_dot([c2,c4])
        self.ubar_rbs_coupler_jcs_rc_cyl = (multi_dot([c2,c5]) + c3)
        self.ubar_rbl_rocker_jcs_rc_cyl = (multi_dot([c6,c5]) + -1.0*multi_dot([c6,self.ST.R_rbl_rocker]))
        self.Mbar_rbs_coupler_jcs_rc_cyl = multi_dot([c2,c7])
        self.Mbar_rbl_rocker_jcs_rc_cyl = multi_dot([c6,c7])

    @property
    def q_initial(self):
        q = np.concatenate([self.ST.R_rbr_rocker,self.ST.P_rbr_rocker,self.ST.R_rbs_coupler,self.ST.P_rbs_coupler,self.ST.R_rbl_rocker,self.ST.P_rbl_rocker])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.ST.Rd_rbr_rocker,self.ST.Pd_rbr_rocker,self.ST.Rd_rbs_coupler,self.ST.Pd_rbs_coupler,self.ST.Rd_rbl_rocker,self.ST.Pd_rbl_rocker])
        return qd



class numerical_assembly(object):

    def __init__(self,config):
        self.t = 0.0
        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_rows = np.array([0,1,2,3,4,5,6,7])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,6,7])
        self.jac_cols = np.array([4,5,6,7,6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,5,7,9])

        self.nrows = max(self.pos_rows)
        self.ncols = max(self.jac_cols)

    
    def set_q(self,q):
        self.ST.R_rbr_rocker = q[0:3,0:1]
        self.ST.P_rbr_rocker = q[3:7,0:1]
        self.ST.R_rbs_coupler = q[7:10,0:1]
        self.ST.P_rbs_coupler = q[10:14,0:1]
        self.ST.R_rbl_rocker = q[14:17,0:1]
        self.ST.P_rbl_rocker = q[17:21,0:1]

    
    def set_qd(self,qd):
        self.ST.Rd_rbr_rocker = qd[0:3,0:1]
        self.ST.Pd_rbr_rocker = qd[3:7,0:1]
        self.ST.Rd_rbs_coupler = qd[7:10,0:1]
        self.ST.Pd_rbs_coupler = qd[10:14,0:1]
        self.ST.Rd_rbl_rocker = qd[14:17,0:1]
        self.ST.Pd_rbl_rocker = qd[17:21,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.ST.R_rbs_coupler
        x1 = self.ST.P_rbr_rocker
        x2 = self.ST.P_rbs_coupler
        x3 = A(x2)
        x4 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,0:1].T
        x5 = x3.T
        x6 = self.ST.P_rbl_rocker
        x7 = A(x6)
        x8 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]
        x9 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,1:2].T
        x10 = (x0 + -1.0*self.ST.R_rbl_rocker + multi_dot([x3,config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]]) + multi_dot([x3,config.ubar_rbs_coupler_jcs_rc_cyl]) + -1.0*multi_dot([x7,config.ubar_rbl_rocker_jcs_rc_cyl]))
        x11 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(self.ST.R_rbr_rocker + -1.0*x0 + multi_dot([A(x1),config.ubar_rbr_rocker_jcs_rc_sph]) + -1.0*multi_dot([x3,config.ubar_rbs_coupler_jcs_rc_sph])),multi_dot([x4,x5,x7,x8]),multi_dot([x9,x5,x7,x8]),multi_dot([x4,x5,x10]),multi_dot([x9,x5,x10]),(x11 + (multi_dot([x1.T,x1]))**(1.0/2.0)),(x11 + (multi_dot([x2.T,x2]))**(1.0/2.0)),(x11 + (multi_dot([x6.T,x6]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [np.zeros((3,1),dtype=np.float64),v0,v0,v0,v0,v0,v0,v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.ST.Pd_rbr_rocker
        a1 = self.ST.Pd_rbs_coupler
        a2 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,0:1]
        a3 = a2.T
        a4 = self.ST.P_rbs_coupler
        a5 = A(a4).T
        a6 = self.ST.Pd_rbl_rocker
        a7 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]
        a8 = B(a6,a7)
        a9 = a7.T
        a10 = self.ST.P_rbl_rocker
        a11 = A(a10).T
        a12 = B(a1,a2)
        a13 = a1.T
        a14 = B(a4,a2).T
        a15 = B(a10,a7)
        a16 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,1:2]
        a17 = a16.T
        a18 = B(a1,a16)
        a19 = B(a4,a16).T
        a20 = config.ubar_rbs_coupler_jcs_rc_cyl
        a21 = config.ubar_rbl_rocker_jcs_rc_cyl
        a22 = (multi_dot([B(a1,a20),a1]) + -1.0*multi_dot([B(a6,a21),a6]))
        a23 = (self.ST.Rd_rbs_coupler + -1.0*self.ST.Rd_rbl_rocker + multi_dot([B(a10,a21),a6]) + multi_dot([B(a4,a20),a1]))
        a24 = (self.ST.R_rbs_coupler.T + -1.0*self.ST.R_rbl_rocker.T + multi_dot([config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3].T,a5]) + multi_dot([a20.T,a5]) + -1.0*multi_dot([a21.T,a11]))

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_rbr_rocker_jcs_rc_sph),a0]) + -1.0*multi_dot([B(a1,config.ubar_rbs_coupler_jcs_rc_sph),a1])),(multi_dot([a3,a5,a8,a6]) + multi_dot([a9,a11,a12,a1]) + 2.0*multi_dot([a13,a14,a15,a6])),(multi_dot([a17,a5,a8,a6]) + multi_dot([a9,a11,a18,a1]) + 2.0*multi_dot([a13,a19,a15,a6])),(multi_dot([a3,a5,a22]) + 2.0*multi_dot([a13,a14,a23]) + multi_dot([a24,a12,a1])),(multi_dot([a17,a5,a22]) + 2.0*multi_dot([a13,a19,a23]) + multi_dot([a24,a18,a1])),2.0*(multi_dot([a0.T,a0]))**(1.0/2.0),2.0*(multi_dot([a13,a1]))**(1.0/2.0),2.0*(multi_dot([a6.T,a6]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.ST.P_rbr_rocker
        j2 = self.ST.P_rbs_coupler
        j3 = np.zeros((1,3),dtype=np.float64)
        j4 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]
        j5 = j4.T
        j6 = self.ST.P_rbl_rocker
        j7 = A(j6).T
        j8 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,0:1]
        j9 = B(j2,j8)
        j10 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,1:2]
        j11 = B(j2,j10)
        j12 = j8.T
        j13 = A(j2).T
        j14 = multi_dot([j12,j13])
        j15 = config.ubar_rbs_coupler_jcs_rc_cyl
        j16 = B(j2,j15)
        j17 = config.ubar_rbl_rocker_jcs_rc_cyl
        j18 = (self.ST.R_rbs_coupler.T + -1.0*self.ST.R_rbl_rocker.T + multi_dot([config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3].T,j13]) + multi_dot([j15.T,j13]) + -1.0*multi_dot([j17.T,j7]))
        j19 = j10.T
        j20 = multi_dot([j19,j13])
        j21 = B(j6,j4)
        j22 = B(j6,j17)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_rbr_rocker_jcs_rc_sph),-1.0*j0,-1.0*B(j2,config.ubar_rbs_coupler_jcs_rc_sph),j3,multi_dot([j5,j7,j9]),j3,multi_dot([j12,j13,j21]),j3,multi_dot([j5,j7,j11]),j3,multi_dot([j19,j13,j21]),j14,(multi_dot([j12,j13,j16]) + multi_dot([j18,j9])),-1.0*j14,-1.0*multi_dot([j12,j13,j22]),j20,(multi_dot([j19,j13,j16]) + multi_dot([j18,j11])),-1.0*j20,-1.0*multi_dot([j19,j13,j22]),2.0*j1.T,2.0*j2.T,2.0*j6.T]
  
