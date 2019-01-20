
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcr_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_rocker_ch = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_rocker_ch = np.array([[0], [0], [0]],dtype=np.float64)
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

        c0 = A('ST.P_vbs_chassis').T
        c1 = self.pt1_jcr_rocker_ch
        c2 = -1.0*multi_dot([c0,'ST.R_vbs_chassis'])
        c3 = A(self.ST.P_rbr_rocker).T
        c4 = -1.0*multi_dot([c3,self.ST.R_rbr_rocker])
        c5 = Triad(self.ax1_jcr_rocker_ch,)
        c6 = self.pt1_jcl_rocker_ch
        c7 = A(self.ST.P_rbl_rocker).T
        c8 = -1.0*multi_dot([c7,self.ST.R_rbl_rocker])
        c9 = Triad(self.ax1_jcl_rocker_ch,)
        c10 = self.pt1_jcs_rc_sph
        c11 = A(self.ST.P_rbs_coupler).T
        c12 = -1.0*multi_dot([c11,self.ST.R_rbs_coupler])
        c13 = Triad(self.ax1_jcs_rc_sph,)
        c14 = self.pt1_jcs_rc_cyl
        c15 = Triad(self.ax1_jcs_rc_cyl,)

        self.ubar_vbs_chassis_jcr_rocker_ch = (multi_dot([c0,c1]) + c2)
        self.ubar_rbr_rocker_jcr_rocker_ch = (multi_dot([c3,c1]) + c4)
        self.Mbar_vbs_chassis_jcr_rocker_ch = multi_dot([c0,c5])
        self.Mbar_rbr_rocker_jcr_rocker_ch = multi_dot([c3,c5])
        self.ubar_vbs_chassis_jcl_rocker_ch = (multi_dot([c0,c6]) + c2)
        self.ubar_rbl_rocker_jcl_rocker_ch = (multi_dot([c7,c6]) + c8)
        self.Mbar_vbs_chassis_jcl_rocker_ch = multi_dot([c0,c9])
        self.Mbar_rbl_rocker_jcl_rocker_ch = multi_dot([c7,c9])
        self.ubar_rbr_rocker_jcs_rc_sph = (multi_dot([c3,c10]) + c4)
        self.ubar_rbs_coupler_jcs_rc_sph = (multi_dot([c11,c10]) + c12)
        self.Mbar_rbr_rocker_jcs_rc_sph = multi_dot([c3,c13])
        self.Mbar_rbs_coupler_jcs_rc_sph = multi_dot([c11,c13])
        self.ubar_rbs_coupler_jcs_rc_cyl = (multi_dot([c11,c14]) + c12)
        self.ubar_rbl_rocker_jcs_rc_cyl = (multi_dot([c7,c14]) + c8)
        self.Mbar_rbs_coupler_jcs_rc_cyl = multi_dot([c11,c15])
        self.Mbar_rbl_rocker_jcs_rc_cyl = multi_dot([c7,c15])

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

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,12,13])
        self.jac_cols = np.array([2,3,4,5,2,3,4,5,2,3,4,5,2,3,8,9,2,3,8,9,2,3,8,9,4,5,6,7,6,7,8,9,6,7,8,9,6,7,8,9,6,7,8,9,5,7,9])

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

        x0 = 'ST.R_vbs_chassis'
        x1 = self.ST.R_rbr_rocker
        x2 = A('ST.P_vbs_chassis')
        x3 = self.ST.P_rbr_rocker
        x4 = A(x3)
        x5 = x2.T
        x6 = config.Mbar_rbr_rocker_jcr_rocker_ch[:,2:3]
        x7 = -1.0*self.ST.R_rbl_rocker
        x8 = self.ST.P_rbl_rocker
        x9 = A(x8)
        x10 = config.Mbar_rbl_rocker_jcl_rocker_ch[:,2:3]
        x11 = self.ST.R_rbs_coupler
        x12 = self.ST.P_rbs_coupler
        x13 = A(x12)
        x14 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,0:1].T
        x15 = x13.T
        x16 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]
        x17 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,1:2].T
        x18 = (x11 + x7 + multi_dot([x13,config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]]) + multi_dot([x13,config.ubar_rbs_coupler_jcs_rc_cyl]) + -1.0*multi_dot([x9,config.ubar_rbl_rocker_jcs_rc_cyl]))
        x19 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1.0*x1 + multi_dot([x2,config.ubar_vbs_chassis_jcr_rocker_ch]) + -1.0*multi_dot([x4,config.ubar_rbr_rocker_jcr_rocker_ch])),multi_dot([config.Mbar_vbs_chassis_jcr_rocker_ch[:,0:1].T,x5,x4,x6]),multi_dot([config.Mbar_vbs_chassis_jcr_rocker_ch[:,1:2].T,x5,x4,x6]),(x0 + x7 + multi_dot([x2,config.ubar_vbs_chassis_jcl_rocker_ch]) + -1.0*multi_dot([x9,config.ubar_rbl_rocker_jcl_rocker_ch])),multi_dot([config.Mbar_vbs_chassis_jcl_rocker_ch[:,0:1].T,x5,x9,x10]),multi_dot([config.Mbar_vbs_chassis_jcl_rocker_ch[:,1:2].T,x5,x9,x10]),(x1 + -1.0*x11 + multi_dot([x4,config.ubar_rbr_rocker_jcs_rc_sph]) + -1.0*multi_dot([x13,config.ubar_rbs_coupler_jcs_rc_sph])),multi_dot([x14,x15,x9,x16]),multi_dot([x17,x15,x9,x16]),multi_dot([x14,x15,x18]),multi_dot([x17,x15,x18]),(x19 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x19 + (multi_dot([x12.T,x12]))**(1.0/2.0)),(x19 + (multi_dot([x8.T,x8]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v0,v1,v1,v0,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = 'ST.Pd_vbs_chassis'
        a1 = self.ST.Pd_rbr_rocker
        a2 = config.Mbar_rbr_rocker_jcr_rocker_ch[:,2:3]
        a3 = a2.T
        a4 = self.ST.P_rbr_rocker
        a5 = A(a4).T
        a6 = config.Mbar_vbs_chassis_jcr_rocker_ch[:,0:1]
        a7 = 'ST.P_vbs_chassis'
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = config.Mbar_vbs_chassis_jcr_rocker_ch[:,1:2]
        a13 = self.ST.Pd_rbl_rocker
        a14 = config.Mbar_vbs_chassis_jcl_rocker_ch[:,0:1]
        a15 = config.Mbar_rbl_rocker_jcl_rocker_ch[:,2:3]
        a16 = B(a13,a15)
        a17 = a15.T
        a18 = self.ST.P_rbl_rocker
        a19 = A(a18).T
        a20 = B(a18,a15)
        a21 = config.Mbar_vbs_chassis_jcl_rocker_ch[:,1:2]
        a22 = self.ST.Pd_rbs_coupler
        a23 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,0:1]
        a24 = a23.T
        a25 = self.ST.P_rbs_coupler
        a26 = A(a25).T
        a27 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]
        a28 = B(a13,a27)
        a29 = a27.T
        a30 = B(a22,a23)
        a31 = a22.T
        a32 = B(a25,a23).T
        a33 = B(a18,a27)
        a34 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,1:2]
        a35 = a34.T
        a36 = B(a22,a34)
        a37 = B(a25,a34).T
        a38 = config.ubar_rbs_coupler_jcs_rc_cyl
        a39 = config.ubar_rbl_rocker_jcs_rc_cyl
        a40 = (multi_dot([B(a22,a38),a22]) + -1.0*multi_dot([B(a13,a39),a13]))
        a41 = (self.ST.Rd_rbs_coupler + -1.0*self.ST.Rd_rbl_rocker + multi_dot([B(a18,a39),a13]) + multi_dot([B(a25,a38),a22]))
        a42 = (self.ST.R_rbs_coupler.T + -1.0*self.ST.R_rbl_rocker.T + multi_dot([config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3].T,a26]) + multi_dot([a38.T,a26]) + -1.0*multi_dot([a39.T,a19]))

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_vbs_chassis_jcr_rocker_ch),a0]) + -1.0*multi_dot([B(a1,config.ubar_rbr_rocker_jcr_rocker_ch),a1])),(multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2.0*multi_dot([a10,B(a7,a6).T,a11,a1])),(multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2.0*multi_dot([a10,B(a7,a12).T,a11,a1])),(multi_dot([B(a0,config.ubar_vbs_chassis_jcl_rocker_ch),a0]) + -1.0*multi_dot([B(a13,config.ubar_rbl_rocker_jcl_rocker_ch),a13])),(multi_dot([a14.T,a8,a16,a13]) + multi_dot([a17,a19,B(a0,a14),a0]) + 2.0*multi_dot([a10,B(a7,a14).T,a20,a13])),(multi_dot([a21.T,a8,a16,a13]) + multi_dot([a17,a19,B(a0,a21),a0]) + 2.0*multi_dot([a10,B(a7,a21).T,a20,a13])),(multi_dot([B(a1,config.ubar_rbr_rocker_jcs_rc_sph),a1]) + -1.0*multi_dot([B(a22,config.ubar_rbs_coupler_jcs_rc_sph),a22])),(multi_dot([a24,a26,a28,a13]) + multi_dot([a29,a19,a30,a22]) + 2.0*multi_dot([a31,a32,a33,a13])),(multi_dot([a35,a26,a28,a13]) + multi_dot([a29,a19,a36,a22]) + 2.0*multi_dot([a31,a37,a33,a13])),(multi_dot([a24,a26,a40]) + 2.0*multi_dot([a31,a32,a41]) + multi_dot([a42,a30,a22])),(multi_dot([a35,a26,a40]) + 2.0*multi_dot([a31,a37,a41]) + multi_dot([a42,a36,a22])),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a31,a22]))**(1.0/2.0),2.0*(multi_dot([a13.T,a13]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = 'ST.P_vbs_chassis'
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = config.Mbar_rbr_rocker_jcr_rocker_ch[:,2:3]
        j4 = j3.T
        j5 = self.ST.P_rbr_rocker
        j6 = A(j5).T
        j7 = config.Mbar_vbs_chassis_jcr_rocker_ch[:,0:1]
        j8 = config.Mbar_vbs_chassis_jcr_rocker_ch[:,1:2]
        j9 = -1.0*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = config.Mbar_rbl_rocker_jcl_rocker_ch[:,2:3]
        j13 = j12.T
        j14 = self.ST.P_rbl_rocker
        j15 = A(j14).T
        j16 = config.Mbar_vbs_chassis_jcl_rocker_ch[:,0:1]
        j17 = config.Mbar_vbs_chassis_jcl_rocker_ch[:,1:2]
        j18 = B(j14,j12)
        j19 = self.ST.P_rbs_coupler
        j20 = config.Mbar_rbl_rocker_jcs_rc_cyl[:,2:3]
        j21 = j20.T
        j22 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,0:1]
        j23 = B(j19,j22)
        j24 = config.Mbar_rbs_coupler_jcs_rc_cyl[:,1:2]
        j25 = B(j19,j24)
        j26 = j22.T
        j27 = A(j19).T
        j28 = multi_dot([j26,j27])
        j29 = config.ubar_rbs_coupler_jcs_rc_cyl
        j30 = B(j19,j29)
        j31 = config.ubar_rbl_rocker_jcs_rc_cyl
        j32 = (self.ST.R_rbs_coupler.T + -1.0*self.ST.R_rbl_rocker.T + multi_dot([config.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3].T,j27]) + multi_dot([j29.T,j27]) + -1.0*multi_dot([j31.T,j15]))
        j33 = j24.T
        j34 = multi_dot([j33,j27])
        j35 = B(j14,j20)
        j36 = B(j14,j31)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_vbs_chassis_jcr_rocker_ch),j9,-1.0*B(j5,config.ubar_rbr_rocker_jcr_rocker_ch),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j0,B(j1,config.ubar_vbs_chassis_jcl_rocker_ch),j9,-1.0*B(j14,config.ubar_rbl_rocker_jcl_rocker_ch),j2,multi_dot([j13,j15,B(j1,j16)]),j2,multi_dot([j16.T,j10,j18]),j2,multi_dot([j13,j15,B(j1,j17)]),j2,multi_dot([j17.T,j10,j18]),j0,B(j5,config.ubar_rbr_rocker_jcs_rc_sph),j9,-1.0*B(j19,config.ubar_rbs_coupler_jcs_rc_sph),j2,multi_dot([j21,j15,j23]),j2,multi_dot([j26,j27,j35]),j2,multi_dot([j21,j15,j25]),j2,multi_dot([j33,j27,j35]),j28,(multi_dot([j26,j27,j30]) + multi_dot([j32,j23])),-1.0*j28,-1.0*multi_dot([j26,j27,j36]),j34,(multi_dot([j33,j27,j30]) + multi_dot([j32,j25])),-1.0*j34,-1.0*multi_dot([j33,j27,j36]),2.0*j5.T,2.0*j19.T,2.0*j14.T]
  
