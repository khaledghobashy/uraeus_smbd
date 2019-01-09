
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin
from source.solvers.python_solver import solver


class inputs(object):

    def __init__(self):
        self.pt_jcr_rocker_ch = np.array([[0], [90], [0]],dtype=np.float64)
        self.ax_jcr_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.F_jcr_rocker_ch = lambda t : 0
        
        self.pt_jcl_rocker_ch = np.array([[0], [-90], [0]],dtype=np.float64)
        self.ax_jcl_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.pt_jcs_rc_sph = np.array([[90], [-90], [0]],dtype=np.float64)
        self.ax_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.pt_jcs_rc_cyl = np.array([[90], [90], [0]],dtype=np.float64)
        self.ax_jcs_rc_cyl = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_ST_rbs_coupler = np.array([[90], [0], [0]],dtype=np.float64)
        self.P_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_ST_rbr_rocker = np.array([[45], [90], [0]],dtype=np.float64)
        self.P_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_ST_rbl_rocker = np.array([[45], [-90], [0]],dtype=np.float64)
        self.P_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.P_ground).T
        c1 = self.pt_jcr_rocker_ch
        c2 = -1.0*multi_dot([c0,self.R_ground])
        c3 = A(self.P_ST_rbr_rocker).T
        c4 = -1.0*multi_dot([c3,self.R_ST_rbr_rocker])
        c5 = Triad(self.ax_jcr_rocker_ch,)
        c6 = self.pt_jcl_rocker_ch
        c7 = A(self.P_ST_rbl_rocker).T
        c8 = -1.0*multi_dot([c7,self.R_ST_rbl_rocker])
        c9 = Triad(self.ax_jcl_rocker_ch,)
        c10 = A(self.P_ST_rbs_coupler).T
        c11 = self.pt_jcs_rc_sph
        c12 = -1.0*multi_dot([c10,self.R_ST_rbs_coupler])
        c13 = Triad(self.ax_jcs_rc_sph,)
        c14 = self.pt_jcs_rc_cyl
        c15 = Triad(self.ax_jcs_rc_cyl,)

        self.ubar_ground_jcr_rocker_ch = (multi_dot([c0,c1]) + c2)
        self.ubar_ST_rbr_rocker_jcr_rocker_ch = (multi_dot([c3,c1]) + c4)
        self.Mbar_ground_jcr_rocker_ch = multi_dot([c0,c5])
        self.Mbar_ST_rbr_rocker_jcr_rocker_ch = multi_dot([c3,c5])
        self.ubar_ground_jcl_rocker_ch = (multi_dot([c0,c6]) + c2)
        self.ubar_ST_rbl_rocker_jcl_rocker_ch = (multi_dot([c7,c6]) + c8)
        self.Mbar_ground_jcl_rocker_ch = multi_dot([c0,c9])
        self.Mbar_ST_rbl_rocker_jcl_rocker_ch = multi_dot([c7,c9])
        self.ubar_ST_rbs_coupler_jcs_rc_sph = (multi_dot([c10,c11]) + c12)
        self.ubar_ST_rbr_rocker_jcs_rc_sph = (multi_dot([c3,c11]) + c4)
        self.Mbar_ST_rbs_coupler_jcs_rc_sph = multi_dot([c10,c13])
        self.Mbar_ST_rbr_rocker_jcs_rc_sph = multi_dot([c3,c13])
        self.ubar_ST_rbs_coupler_jcs_rc_cyl = (multi_dot([c10,c14]) + c12)
        self.ubar_ST_rbl_rocker_jcs_rc_cyl = (multi_dot([c7,c14]) + c8)
        self.Mbar_ST_rbs_coupler_jcs_rc_cyl = multi_dot([c10,c15])
        self.Mbar_ST_rbl_rocker_jcs_rc_cyl = multi_dot([c7,c15])

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

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,13,13,14,15,16])
        self.jac_cols = np.array([0,1,4,5,0,1,4,5,0,1,4,5,0,1,4,5,0,1,6,7,0,1,6,7,0,1,6,7,2,3,4,5,2,3,6,7,2,3,6,7,2,3,6,7,2,3,6,7,0,1,0,1,3,5,7])

    
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

        x0 = self.R_ground
        x1 = -1.0*self.R_ST_rbr_rocker
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_ST_rbr_rocker
        x5 = A(x4)
        x6 = x3.T
        x7 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        x8 = config.F_jcr_rocker_ch(t,)
        x9 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,0:1]
        x10 = -1.0*self.R_ST_rbl_rocker
        x11 = self.P_ST_rbl_rocker
        x12 = A(x11)
        x13 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        x14 = self.R_ST_rbs_coupler
        x15 = self.P_ST_rbs_coupler
        x16 = A(x15)
        x17 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1].T
        x18 = x16.T
        x19 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        x20 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2].T
        x21 = (x14 + x10 + multi_dot([x16,config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3]]) + multi_dot([x16,config.ubar_ST_rbs_coupler_jcs_rc_cyl]) + -1.0*multi_dot([x12,config.ubar_ST_rbl_rocker_jcs_rc_cyl]))
        x22 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x1 + multi_dot([x3,config.ubar_ground_jcr_rocker_ch]) + -1.0*multi_dot([x5,config.ubar_ST_rbr_rocker_jcr_rocker_ch])),multi_dot([config.Mbar_ground_jcr_rocker_ch[:,0:1].T,x6,x5,x7]),multi_dot([config.Mbar_ground_jcr_rocker_ch[:,1:2].T,x6,x5,x7]),(cos(x8)*multi_dot([config.Mbar_ground_jcr_rocker_ch[:,1:2].T,x6,x5,x9]) + sin(x8)*-1.0*multi_dot([config.Mbar_ground_jcr_rocker_ch[:,0:1].T,x6,x5,x9])),(x0 + x10 + multi_dot([x3,config.ubar_ground_jcl_rocker_ch]) + -1.0*multi_dot([x12,config.ubar_ST_rbl_rocker_jcl_rocker_ch])),multi_dot([config.Mbar_ground_jcl_rocker_ch[:,0:1].T,x6,x12,x13]),multi_dot([config.Mbar_ground_jcl_rocker_ch[:,1:2].T,x6,x12,x13]),(x14 + x1 + multi_dot([x16,config.ubar_ST_rbs_coupler_jcs_rc_sph]) + -1.0*multi_dot([x5,config.ubar_ST_rbr_rocker_jcs_rc_sph])),multi_dot([x17,x18,x12,x19]),multi_dot([x20,x18,x12,x19]),multi_dot([x17,x18,x21]),multi_dot([x20,x18,x21]),x0,(x2 + -1.0*config.Pg_ground),(x22 + (multi_dot([x15.T,x15]))**(1.0/2.0)),(x22 + (multi_dot([x4.T,x4]))**(1.0/2.0)),(x22 + (multi_dot([x11.T,x11]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,(v1 + -1*derivative(config.F_jcr_rocker_ch,t,0.1,1)*np.eye(1,dtype=np.float64)),v0,v1,v1,v0,v1,v1,v1,v1,v0,np.zeros((4,1),dtype=np.float64),v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_ground
        a1 = self.Pd_ST_rbr_rocker
        a2 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        a3 = a2.T
        a4 = self.P_ST_rbr_rocker
        a5 = A(a4).T
        a6 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        a7 = self.P_ground
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        a13 = config.F_jcr_rocker_ch(t,)
        a14 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,0:1]
        a15 = cos(a13)
        a16 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        a17 = sin(a13)
        a18 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        a19 = self.Pd_ST_rbl_rocker
        a20 = config.Mbar_ground_jcl_rocker_ch[:,0:1]
        a21 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        a22 = B(a19,a21)
        a23 = a21.T
        a24 = self.P_ST_rbl_rocker
        a25 = A(a24).T
        a26 = B(a24,a21)
        a27 = config.Mbar_ground_jcl_rocker_ch[:,1:2]
        a28 = self.Pd_ST_rbs_coupler
        a29 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1]
        a30 = a29.T
        a31 = self.P_ST_rbs_coupler
        a32 = A(a31).T
        a33 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        a34 = B(a19,a33)
        a35 = a33.T
        a36 = B(a28,a29)
        a37 = a28.T
        a38 = B(a31,a29).T
        a39 = B(a24,a33)
        a40 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2]
        a41 = a40.T
        a42 = B(a28,a40)
        a43 = B(a31,a40).T
        a44 = config.ubar_ST_rbs_coupler_jcs_rc_cyl
        a45 = config.ubar_ST_rbl_rocker_jcs_rc_cyl
        a46 = (multi_dot([B(a28,a44),a28]) + -1.0*multi_dot([B(a19,a45),a19]))
        a47 = (self.Rd_ST_rbs_coupler + -1.0*self.Rd_ST_rbl_rocker + multi_dot([B(a24,a45),a19]) + multi_dot([B(a31,a44),a28]))
        a48 = (self.R_ST_rbs_coupler.T + -1.0*self.R_ST_rbl_rocker.T + multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3].T,a32]) + multi_dot([a44.T,a32]) + -1.0*multi_dot([a45.T,a25]))

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_ground_jcr_rocker_ch),a0]) + -1.0*multi_dot([B(a1,config.ubar_ST_rbr_rocker_jcr_rocker_ch),a1])),(multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2.0*multi_dot([a10,B(a7,a6).T,a11,a1])),(multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2.0*multi_dot([a10,B(a7,a12).T,a11,a1])),(derivative(config.F_jcr_rocker_ch,t,0.1,2)*-1.0*np.eye(1,dtype=np.float64) + multi_dot([a14.T,a5,(a15*B(a0,a16) + a17*-1.0*B(a0,a18)),a0]) + multi_dot([(a15*multi_dot([a16.T,a8]) + a17*-1.0*multi_dot([a18.T,a8])),B(a1,a14),a1]) + 2.0*multi_dot([((a15*multi_dot([B(a7,a16),a0])).T + a17*-1.0*multi_dot([a10,B(a7,a18).T])),B(a4,a14),a1])),(multi_dot([B(a0,config.ubar_ground_jcl_rocker_ch),a0]) + -1.0*multi_dot([B(a19,config.ubar_ST_rbl_rocker_jcl_rocker_ch),a19])),(multi_dot([a20.T,a8,a22,a19]) + multi_dot([a23,a25,B(a0,a20),a0]) + 2.0*multi_dot([a10,B(a7,a20).T,a26,a19])),(multi_dot([a27.T,a8,a22,a19]) + multi_dot([a23,a25,B(a0,a27),a0]) + 2.0*multi_dot([a10,B(a7,a27).T,a26,a19])),(multi_dot([B(a28,config.ubar_ST_rbs_coupler_jcs_rc_sph),a28]) + -1.0*multi_dot([B(a1,config.ubar_ST_rbr_rocker_jcs_rc_sph),a1])),(multi_dot([a30,a32,a34,a19]) + multi_dot([a35,a25,a36,a28]) + 2.0*multi_dot([a37,a38,a39,a19])),(multi_dot([a41,a32,a34,a19]) + multi_dot([a35,a25,a42,a28]) + 2.0*multi_dot([a37,a43,a39,a19])),(multi_dot([a30,a32,a46]) + 2.0*multi_dot([a37,a38,a47]) + multi_dot([a48,a36,a28])),(multi_dot([a41,a32,a46]) + 2.0*multi_dot([a37,a43,a47]) + multi_dot([a48,a42,a28])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a37,a28]))**(1.0/2.0),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a19.T,a19]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        j4 = j3.T
        j5 = self.P_ST_rbr_rocker
        j6 = A(j5).T
        j7 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        j8 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        j9 = -1.0*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,0:1]
        j13 = config.F_jcr_rocker_ch(t,)
        j14 = cos(j13)
        j15 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        j16 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        j17 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        j18 = j17.T
        j19 = self.P_ST_rbl_rocker
        j20 = A(j19).T
        j21 = config.Mbar_ground_jcl_rocker_ch[:,0:1]
        j22 = config.Mbar_ground_jcl_rocker_ch[:,1:2]
        j23 = B(j19,j17)
        j24 = self.P_ST_rbs_coupler
        j25 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        j26 = j25.T
        j27 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1]
        j28 = B(j24,j27)
        j29 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2]
        j30 = B(j24,j29)
        j31 = j27.T
        j32 = A(j24).T
        j33 = multi_dot([j31,j32])
        j34 = config.ubar_ST_rbs_coupler_jcs_rc_cyl
        j35 = B(j24,j34)
        j36 = config.ubar_ST_rbl_rocker_jcs_rc_cyl
        j37 = (self.R_ST_rbs_coupler.T + -1.0*self.R_ST_rbl_rocker.T + multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3].T,j32]) + multi_dot([j34.T,j32]) + -1.0*multi_dot([j36.T,j20]))
        j38 = j29.T
        j39 = multi_dot([j38,j32])
        j40 = B(j19,j25)
        j41 = B(j19,j36)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_ground_jcr_rocker_ch),j9,-1.0*B(j5,config.ubar_ST_rbr_rocker_jcr_rocker_ch),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j2,multi_dot([j12.T,j6,(j14*B(j1,j15) + sin(j13)*-1.0*B(j1,j16))]),j2,multi_dot([(j14*multi_dot([j15.T,j10]) + sin(j13)*-1.0*multi_dot([j16.T,j10])),B(j5,j12)]),j0,B(j1,config.ubar_ground_jcl_rocker_ch),j9,-1.0*B(j19,config.ubar_ST_rbl_rocker_jcl_rocker_ch),j2,multi_dot([j18,j20,B(j1,j21)]),j2,multi_dot([j21.T,j10,j23]),j2,multi_dot([j18,j20,B(j1,j22)]),j2,multi_dot([j22.T,j10,j23]),j0,B(j24,config.ubar_ST_rbs_coupler_jcs_rc_sph),j9,-1.0*B(j5,config.ubar_ST_rbr_rocker_jcs_rc_sph),j2,multi_dot([j26,j20,j28]),j2,multi_dot([j31,j32,j40]),j2,multi_dot([j26,j20,j30]),j2,multi_dot([j38,j32,j40]),j33,(multi_dot([j31,j32,j35]) + multi_dot([j37,j28])),-1.0*j33,-1.0*multi_dot([j31,j32,j41]),j39,(multi_dot([j38,j32,j35]) + multi_dot([j37,j30])),-1.0*j39,-1.0*multi_dot([j38,j32,j41]),j0,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j24.T,2.0*j5.T,2.0*j19.T]
  
