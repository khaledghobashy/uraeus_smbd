
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcs_rc_cyl = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcs_rc_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.R_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.P_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.ST.Rd_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.Pd_rbl_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ST.R_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.P_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.ST.Rd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.Pd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ST.R_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.P_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.ST.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.ST.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.ST.P_rbl_rocker).T
        c1 = self.pt1_jcs_rc_cyl
        c2 = A(self.ST.P_rbs_coupler).T
        c3 = -1.0*multi_dot([c2,self.ST.R_rbs_coupler])
        c4 = Triad(self.ax1_jcs_rc_cyl,)
        c5 = A(self.ST.P_rbr_rocker).T
        c6 = self.pt1_jcs_rc_sph
        c7 = Triad(self.ax1_jcs_rc_sph,)

        self.ubar_rbl_rocker_jcs_rc_cyl = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,self.ST.R_rbl_rocker]))
        self.ubar_rbs_coupler_jcs_rc_cyl = (multi_dot([c2,c1]) + c3)
        self.Mbar_rbl_rocker_jcs_rc_cyl = multi_dot([c0,c4])
        self.Mbar_rbs_coupler_jcs_rc_cyl = multi_dot([c2,c4])
        self.ubar_rbr_rocker_jcs_rc_sph = (multi_dot([c5,c6]) + -1.0*multi_dot([c5,self.ST.R_rbr_rocker]))
        self.ubar_rbs_coupler_jcs_rc_sph = (multi_dot([c2,c6]) + c3)
        self.Mbar_rbr_rocker_jcs_rc_sph = multi_dot([c5,c7])
        self.Mbar_rbs_coupler_jcs_rc_sph = multi_dot([c2,c7])

    @property
    def q_initial(self):
        q = np.concatenate([self.ST.R_rbl_rocker,self.ST.P_rbl_rocker,self.ST.R_rbr_rocker,self.ST.P_rbr_rocker,self.ST.R_rbs_coupler,self.ST.P_rbs_coupler])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.ST.Rd_rbl_rocker,self.ST.Pd_rbl_rocker,self.ST.Rd_rbr_rocker,self.ST.Pd_rbr_rocker,self.ST.Rd_rbs_coupler,self.ST.Pd_rbs_coupler])
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
        self.jac_cols = np.array([2,3,6,7,2,3,6,7,2,3,6,7,2,3,6,7,4,5,6,7,3,5,7])

        self.nrows = max(self.pos_rows)
        self.ncols = max(self.jac_cols)

    
    def set_q(self,q):
        self.ST.R_rbl_rocker = q[0:3,0:1]
        self.ST.P_rbl_rocker = q[3:7,0:1]
        self.ST.R_rbr_rocker = q[7:10,0:1]
        self.ST.P_rbr_rocker = q[10:14,0:1]
        self.ST.R_rbs_coupler = q[14:17,0:1]
        self.ST.P_rbs_coupler = q[17:21,0:1]

    
    def set_qd(self,qd):
        self.ST.Rd_rbl_rocker = qd[0:3,0:1]
        self.ST.Pd_rbl_rocker = qd[3:7,0:1]
        self.ST.Rd_rbr_rocker = qd[7:10,0:1]
        self.ST.Pd_rbr_rocker = qd[10:14,0:1]
        self.ST.Rd_rbs_coupler = qd[14:17,0:1]
        self.ST.Pd_rbs_coupler = qd[17:21,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1].T
        x1 = self.ST.P_rbl_rocker
        x2 = A(x1)
        x3 = x2.T
        x4 = self.ST.P_rbs_coupler
        x5 = A(x4)
        x6 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]
        x7 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2].T
        x8 = -1.0*self.ST.R_rbs_coupler
        x9 = (self.ST.R_rbl_rocker + x8 + multi_dot([x2,config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]]) + multi_dot([x2,config.ubar_rbl_rocker_jcs_rc_cyl]) + -1.0*multi_dot([x5,config.ubar_rbs_coupler_jcs_rc_cyl]))
        x10 = self.ST.P_rbr_rocker
        x11 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [multi_dot([x0,x3,x5,x6]),multi_dot([x7,x3,x5,x6]),multi_dot([x0,x3,x9]),multi_dot([x7,x3,x9]),(self.ST.R_rbr_rocker + x8 + multi_dot([A(x10),config.ubar_rbr_rocker_jcs_rc_sph]) + -1.0*multi_dot([x5,config.ubar_rbs_coupler_jcs_rc_sph])),(x11 + (multi_dot([x1.T,x1]))**(1.0/2.0)),(x11 + (multi_dot([x10.T,x10]))**(1.0/2.0)),(x11 + (multi_dot([x4.T,x4]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v0,v0,v0,np.zeros((3,1),dtype=np.float64),v0,v0,v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]
        a1 = a0.T
        a2 = self.ST.P_rbs_coupler
        a3 = A(a2).T
        a4 = self.ST.Pd_rbl_rocker
        a5 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.ST.P_rbl_rocker
        a9 = A(a8).T
        a10 = self.ST.Pd_rbs_coupler
        a11 = B(a10,a0)
        a12 = a4.T
        a13 = B(a8,a5).T
        a14 = B(a2,a0)
        a15 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]
        a16 = B(a4,a15)
        a17 = a15.T
        a18 = B(a8,a15).T
        a19 = config.ubar_rbl_rocker_jcs_rc_cyl
        a20 = config.ubar_rbs_coupler_jcs_rc_cyl
        a21 = (multi_dot([B(a4,a19),a4]) + -1.0*multi_dot([B(a10,a20),a10]))
        a22 = (self.ST.Rd_rbl_rocker + -1.0*self.ST.Rd_rbs_coupler + multi_dot([B(a8,a19),a4]) + multi_dot([B(a2,a20),a10]))
        a23 = (self.ST.R_rbl_rocker.T + -1.0*self.ST.R_rbs_coupler.T + multi_dot([config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3].T,a9]) + multi_dot([a19.T,a9]) + -1.0*multi_dot([a20.T,a3]))
        a24 = self.ST.Pd_rbr_rocker

        self.acc_eq_blocks = [(multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + 2.0*multi_dot([a12,a13,a14,a10])),(multi_dot([a1,a3,a16,a4]) + multi_dot([a17,a9,a11,a10]) + 2.0*multi_dot([a12,a18,a14,a10])),(multi_dot([a7,a9,a21]) + 2.0*multi_dot([a12,a13,a22]) + multi_dot([a23,a6,a4])),(multi_dot([a17,a9,a21]) + 2.0*multi_dot([a12,a18,a22]) + multi_dot([a23,a16,a4])),(multi_dot([B(a24,config.ubar_rbr_rocker_jcs_rc_sph),a24]) + -1.0*multi_dot([B(a10,config.ubar_rbs_coupler_jcs_rc_sph),a10])),2.0*(multi_dot([a12,a4]))**(1.0/2.0),2.0*(multi_dot([a24.T,a24]))**(1.0/2.0),2.0*(multi_dot([a10.T,a10]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]
        j2 = j1.T
        j3 = self.ST.P_rbs_coupler
        j4 = A(j3).T
        j5 = self.ST.P_rbl_rocker
        j6 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]
        j7 = B(j5,j6)
        j8 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = config.ubar_rbl_rocker_jcs_rc_cyl
        j14 = B(j5,j13)
        j15 = config.ubar_rbs_coupler_jcs_rc_cyl
        j16 = (self.ST.R_rbl_rocker.T + -1.0*self.ST.R_rbs_coupler.T + multi_dot([config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3].T,j11]) + multi_dot([j13.T,j11]) + -1.0*multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = B(j3,j1)
        j20 = B(j3,j15)
        j21 = np.eye(3,dtype=np.float64)
        j22 = self.ST.P_rbr_rocker

        self.jac_eq_blocks = [j0,multi_dot([j2,j4,j7]),j0,multi_dot([j10,j11,j19]),j0,multi_dot([j2,j4,j9]),j0,multi_dot([j17,j11,j19]),j12,(multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),-1.0*j12,-1.0*multi_dot([j10,j11,j20]),j18,(multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),-1.0*j18,-1.0*multi_dot([j17,j11,j20]),j21,B(j22,config.ubar_rbr_rocker_jcs_rc_sph),-1.0*j21,-1.0*B(j3,config.ubar_rbs_coupler_jcs_rc_sph),2.0*j5.T,2.0*j22.T,2.0*j3.T]
  
