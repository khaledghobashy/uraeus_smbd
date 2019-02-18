
import os
import numpy as np
import pandas as pd
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad                



path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.R_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker = 1
        self.Jbar_rbs_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rod = 1
        self.Jbar_rbs_rod = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_slider = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_slider = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_slider = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_slider = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_slider = 1
        self.Jbar_rbs_slider = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcs_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_uni = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_uni = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_uni = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_trans = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_trans = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_rocker,self.P_rbs_rocker,self.R_rbs_rod,self.P_rbs_rod,self.R_rbs_slider,self.P_rbs_slider])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_rocker,self.Pd_rbs_rocker,self.Rd_rbs_rod,self.Pd_rbs_rod,self.Rd_rbs_slider,self.Pd_rbs_slider])
        return qd

    def load_from_csv(self,csv_file):
        file_path = os.path.join(path,csv_file)
        dataframe = pd.read_csv(file_path,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def _set_arguments(self):
    
        pass




class topology(object):

    def __init__(self,prefix='',cfg=None):
        self.t = 0.0
        self.config = (configuration() if cfg is None else cfg)
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 21
        self.nrows = 16
        self.ncols = 2*4
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,14,15])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
        self.rbs_rocker = indicies_map[p+'rbs_rocker']
        self.rbs_rod = indicies_map[p+'rbs_rod']
        self.rbs_slider = indicies_map[p+'rbs_slider']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_rod*2,self.rbs_rod*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_rod*2,self.rbs_rod*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_rod*2,self.rbs_rod*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_rod*2,self.rbs_rod*2+1,self.rbs_rod*2,self.rbs_rod*2+1,self.rbs_slider*2,self.rbs_slider*2+1,self.rbs_rod*2,self.rbs_rod*2+1,self.rbs_slider*2,self.rbs_slider*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_slider*2,self.rbs_slider*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_slider*2,self.rbs_slider*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_slider*2,self.rbs_slider*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_slider*2,self.rbs_slider*2+1,self.rbs_rocker*2+1,self.rbs_rod*2+1,self.rbs_slider*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_rocker_gravity = np.array([[0], [0], [9810.0*config.m_rbs_rocker]],dtype=np.float64)
        self.F_rbs_rod_gravity = np.array([[0], [0], [9810.0*config.m_rbs_rod]],dtype=np.float64)
        self.F_rbs_slider_gravity = np.array([[0], [0], [9810.0*config.m_rbs_slider]],dtype=np.float64)

        c0 = A(config.P_vbs_ground).T
        c1 = triad(config.ax1_jcs_rev)
        c2 = A(config.P_rbs_rocker).T
        c3 = config.pt1_jcs_rev
        c4 = -1*multi_dot([c0,config.R_vbs_ground])
        c5 = -1*multi_dot([c2,config.R_rbs_rocker])
        c6 = triad(config.ax1_jcs_cyl)
        c7 = A(config.P_rbs_rod).T
        c8 = config.pt1_jcs_cyl
        c9 = -1*multi_dot([c7,config.R_rbs_rod])
        c10 = triad(config.ax1_jcs_uni)
        c11 = A(config.P_rbs_slider).T
        c12 = config.pt1_jcs_uni
        c13 = -1*multi_dot([c11,config.R_rbs_slider])
        c14 = triad(config.ax1_jcs_trans)
        c15 = config.pt1_jcs_trans

        self.Mbar_vbs_ground_jcs_rev = multi_dot([c0,c1])
        self.Mbar_rbs_rocker_jcs_rev = multi_dot([c2,c1])
        self.ubar_vbs_ground_jcs_rev = (multi_dot([c0,c3]) + c4)
        self.ubar_rbs_rocker_jcs_rev = (multi_dot([c2,c3]) + c5)
        self.Mbar_rbs_rocker_jcs_cyl = multi_dot([c2,c6])
        self.Mbar_rbs_rod_jcs_cyl = multi_dot([c7,c6])
        self.ubar_rbs_rocker_jcs_cyl = (multi_dot([c2,c8]) + c5)
        self.ubar_rbs_rod_jcs_cyl = (multi_dot([c7,c8]) + c9)
        self.Mbar_rbs_rod_jcs_uni = multi_dot([c7,c10])
        self.Mbar_rbs_slider_jcs_uni = multi_dot([c11,triad(config.ax2_jcs_uni,c10[0:3,1:2])])
        self.ubar_rbs_rod_jcs_uni = (multi_dot([c7,c12]) + c9)
        self.ubar_rbs_slider_jcs_uni = (multi_dot([c11,c12]) + c13)
        self.Mbar_rbs_slider_jcs_trans = multi_dot([c11,c14])
        self.Mbar_vbs_ground_jcs_trans = multi_dot([c0,c14])
        self.ubar_rbs_slider_jcs_trans = (multi_dot([c11,c15]) + c13)
        self.ubar_vbs_ground_jcs_trans = (multi_dot([c0,c15]) + c4)

    
    def set_gen_coordinates(self,q):
        self.R_rbs_rocker = q[0:3,0:1]
        self.P_rbs_rocker = q[3:7,0:1]
        self.R_rbs_rod = q[7:10,0:1]
        self.P_rbs_rod = q[10:14,0:1]
        self.R_rbs_slider = q[14:17,0:1]
        self.P_rbs_slider = q[17:21,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_rocker = qd[0:3,0:1]
        self.Pd_rbs_rocker = qd[3:7,0:1]
        self.Rd_rbs_rod = qd[7:10,0:1]
        self.Pd_rbs_rod = qd[10:14,0:1]
        self.Rd_rbs_slider = qd[14:17,0:1]
        self.Pd_rbs_slider = qd[17:21,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_vbs_ground
        x1 = self.R_rbs_rocker
        x2 = A(self.P_vbs_ground)
        x3 = self.P_rbs_rocker
        x4 = A(x3)
        x5 = x2.T
        x6 = self.Mbar_rbs_rocker_jcs_rev[:,2:3]
        x7 = self.Mbar_rbs_rocker_jcs_cyl[:,0:1].T
        x8 = x4.T
        x9 = self.P_rbs_rod
        x10 = A(x9)
        x11 = self.Mbar_rbs_rod_jcs_cyl[:,2:3]
        x12 = self.Mbar_rbs_rocker_jcs_cyl[:,1:2].T
        x13 = self.R_rbs_rod
        x14 = (x1 + -1*x13 + multi_dot([x4,self.ubar_rbs_rocker_jcs_cyl]) + -1*multi_dot([x10,self.ubar_rbs_rod_jcs_cyl]))
        x15 = self.R_rbs_slider
        x16 = self.P_rbs_slider
        x17 = A(x16)
        x18 = self.Mbar_rbs_slider_jcs_trans[:,0:1].T
        x19 = x17.T
        x20 = self.Mbar_vbs_ground_jcs_trans[:,2:3]
        x21 = self.Mbar_rbs_slider_jcs_trans[:,1:2].T
        x22 = (x15 + -1*x0 + multi_dot([x17,self.ubar_rbs_slider_jcs_trans]) + -1*multi_dot([x2,self.ubar_vbs_ground_jcs_trans]))
        x23 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1*x1 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev]) + -1*multi_dot([x4,self.ubar_rbs_rocker_jcs_rev])),multi_dot([self.Mbar_vbs_ground_jcs_rev[:,0:1].T,x5,x4,x6]),multi_dot([self.Mbar_vbs_ground_jcs_rev[:,1:2].T,x5,x4,x6]),multi_dot([x7,x8,x10,x11]),multi_dot([x12,x8,x10,x11]),multi_dot([x7,x8,x14]),multi_dot([x12,x8,x14]),(x13 + -1*x15 + multi_dot([x10,self.ubar_rbs_rod_jcs_uni]) + -1*multi_dot([x17,self.ubar_rbs_slider_jcs_uni])),multi_dot([self.Mbar_rbs_rod_jcs_uni[:,0:1].T,x10.T,x17,self.Mbar_rbs_slider_jcs_uni[:,0:1]]),multi_dot([x18,x19,x2,x20]),multi_dot([x21,x19,x2,x20]),multi_dot([x18,x19,x22]),multi_dot([x21,x19,x22]),(x23 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x23 + (multi_dot([x9.T,x9]))**(1.0/2.0)),(x23 + (multi_dot([x16.T,x16]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v1,v1,v1,v1,v0,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_vbs_ground
        a1 = self.Pd_rbs_rocker
        a2 = self.Mbar_rbs_rocker_jcs_rev[:,2:3]
        a3 = a2.T
        a4 = self.P_rbs_rocker
        a5 = A(a4).T
        a6 = self.Mbar_vbs_ground_jcs_rev[:,0:1]
        a7 = self.P_vbs_ground
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = self.Mbar_vbs_ground_jcs_rev[:,1:2]
        a13 = self.Mbar_rbs_rod_jcs_cyl[:,2:3]
        a14 = a13.T
        a15 = self.P_rbs_rod
        a16 = A(a15).T
        a17 = self.Mbar_rbs_rocker_jcs_cyl[:,0:1]
        a18 = B(a1,a17)
        a19 = a17.T
        a20 = self.Pd_rbs_rod
        a21 = B(a20,a13)
        a22 = a1.T
        a23 = B(a4,a17).T
        a24 = B(a15,a13)
        a25 = self.Mbar_rbs_rocker_jcs_cyl[:,1:2]
        a26 = B(a1,a25)
        a27 = a25.T
        a28 = B(a4,a25).T
        a29 = self.ubar_rbs_rocker_jcs_cyl
        a30 = self.ubar_rbs_rod_jcs_cyl
        a31 = (multi_dot([B(a1,a29),a1]) + -1*multi_dot([B(a20,a30),a20]))
        a32 = (self.Rd_rbs_rocker + -1*self.Rd_rbs_rod + multi_dot([B(a4,a29),a1]) + multi_dot([B(a15,a30),a20]))
        a33 = (self.R_rbs_rocker.T + -1*self.R_rbs_rod.T + multi_dot([a29.T,a5]) + -1*multi_dot([a30.T,a16]))
        a34 = self.Pd_rbs_slider
        a35 = self.Mbar_rbs_rod_jcs_uni[:,0:1]
        a36 = self.Mbar_rbs_slider_jcs_uni[:,0:1]
        a37 = self.P_rbs_slider
        a38 = A(a37).T
        a39 = a20.T
        a40 = self.Mbar_rbs_slider_jcs_trans[:,0:1]
        a41 = a40.T
        a42 = self.Mbar_vbs_ground_jcs_trans[:,2:3]
        a43 = B(a0,a42)
        a44 = a42.T
        a45 = B(a34,a40)
        a46 = a34.T
        a47 = B(a37,a40).T
        a48 = B(a7,a42)
        a49 = self.Mbar_rbs_slider_jcs_trans[:,1:2]
        a50 = a49.T
        a51 = B(a34,a49)
        a52 = B(a37,a49).T
        a53 = self.ubar_rbs_slider_jcs_trans
        a54 = self.ubar_vbs_ground_jcs_trans
        a55 = (multi_dot([B(a34,a53),a34]) + -1*multi_dot([B(a0,a54),a0]))
        a56 = (self.Rd_rbs_slider + -1*self.Rd_vbs_ground + multi_dot([B(a37,a53),a34]) + multi_dot([B(a7,a54),a0]))
        a57 = (self.R_rbs_slider.T + -1*self.R_vbs_ground.T + multi_dot([a53.T,a38]) + -1*multi_dot([a54.T,a8]))

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev),a0]) + -1*multi_dot([B(a1,self.ubar_rbs_rocker_jcs_rev),a1])),(multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a6).T,a11,a1])),(multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a12).T,a11,a1])),(multi_dot([a14,a16,a18,a1]) + multi_dot([a19,a5,a21,a20]) + 2*multi_dot([a22,a23,a24,a20])),(multi_dot([a14,a16,a26,a1]) + multi_dot([a27,a5,a21,a20]) + 2*multi_dot([a22,a28,a24,a20])),(multi_dot([a19,a5,a31]) + 2*multi_dot([a22,a23,a32]) + multi_dot([a33,a18,a1])),(multi_dot([a27,a5,a31]) + 2*multi_dot([a22,a28,a32]) + multi_dot([a33,a26,a1])),(multi_dot([B(a20,self.ubar_rbs_rod_jcs_uni),a20]) + -1*multi_dot([B(a34,self.ubar_rbs_slider_jcs_uni),a34])),(multi_dot([a35.T,a16,B(a34,a36),a34]) + multi_dot([a36.T,a38,B(a20,a35),a20]) + 2*multi_dot([a39,B(a15,a35).T,B(a37,a36),a34])),(multi_dot([a41,a38,a43,a0]) + multi_dot([a44,a8,a45,a34]) + 2*multi_dot([a46,a47,a48,a0])),(multi_dot([a50,a38,a43,a0]) + multi_dot([a44,a8,a51,a34]) + 2*multi_dot([a46,a52,a48,a0])),(multi_dot([a41,a38,a55]) + 2*multi_dot([a46,a47,a56]) + multi_dot([a57,a45,a34])),(multi_dot([a50,a38,a55]) + 2*multi_dot([a46,a52,a56]) + multi_dot([a57,a51,a34])),2*(multi_dot([a22,a1]))**(1.0/2.0),2*(multi_dot([a39,a20]))**(1.0/2.0),2*(multi_dot([a46,a34]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_vbs_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_rbs_rocker_jcs_rev[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_rocker
        j6 = A(j5).T
        j7 = self.Mbar_vbs_ground_jcs_rev[:,0:1]
        j8 = self.Mbar_vbs_ground_jcs_rev[:,1:2]
        j9 = -1*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbs_rod_jcs_cyl[:,2:3]
        j13 = j12.T
        j14 = self.P_rbs_rod
        j15 = A(j14).T
        j16 = self.Mbar_rbs_rocker_jcs_cyl[:,0:1]
        j17 = B(j5,j16)
        j18 = self.Mbar_rbs_rocker_jcs_cyl[:,1:2]
        j19 = B(j5,j18)
        j20 = j16.T
        j21 = multi_dot([j20,j6])
        j22 = self.ubar_rbs_rocker_jcs_cyl
        j23 = B(j5,j22)
        j24 = self.ubar_rbs_rod_jcs_cyl
        j25 = (self.R_rbs_rocker.T + -1*self.R_rbs_rod.T + multi_dot([j22.T,j6]) + -1*multi_dot([j24.T,j15]))
        j26 = j18.T
        j27 = multi_dot([j26,j6])
        j28 = B(j14,j12)
        j29 = B(j14,j24)
        j30 = self.Mbar_rbs_slider_jcs_uni[:,0:1]
        j31 = self.P_rbs_slider
        j32 = A(j31).T
        j33 = self.Mbar_rbs_rod_jcs_uni[:,0:1]
        j34 = self.Mbar_vbs_ground_jcs_trans[:,2:3]
        j35 = j34.T
        j36 = self.Mbar_rbs_slider_jcs_trans[:,0:1]
        j37 = B(j31,j36)
        j38 = self.Mbar_rbs_slider_jcs_trans[:,1:2]
        j39 = B(j31,j38)
        j40 = j36.T
        j41 = multi_dot([j40,j32])
        j42 = self.ubar_rbs_slider_jcs_trans
        j43 = B(j31,j42)
        j44 = self.ubar_vbs_ground_jcs_trans
        j45 = (self.R_rbs_slider.T + -1*self.R_vbs_ground.T + multi_dot([j42.T,j32]) + -1*multi_dot([j44.T,j10]))
        j46 = j38.T
        j47 = multi_dot([j46,j32])
        j48 = B(j1,j34)
        j49 = B(j1,j44)

        self.jac_eq_blocks = [j0,B(j1,self.ubar_vbs_ground_jcs_rev),j9,-1*B(j5,self.ubar_rbs_rocker_jcs_rev),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j2,multi_dot([j13,j15,j17]),j2,multi_dot([j20,j6,j28]),j2,multi_dot([j13,j15,j19]),j2,multi_dot([j26,j6,j28]),j21,(multi_dot([j20,j6,j23]) + multi_dot([j25,j17])),-1*j21,-1*multi_dot([j20,j6,j29]),j27,(multi_dot([j26,j6,j23]) + multi_dot([j25,j19])),-1*j27,-1*multi_dot([j26,j6,j29]),j0,B(j14,self.ubar_rbs_rod_jcs_uni),j9,-1*B(j31,self.ubar_rbs_slider_jcs_uni),j2,multi_dot([j30.T,j32,B(j14,j33)]),j2,multi_dot([j33.T,j15,B(j31,j30)]),j2,multi_dot([j40,j32,j48]),j2,multi_dot([j35,j10,j37]),j2,multi_dot([j46,j32,j48]),j2,multi_dot([j35,j10,j39]),-1*j41,-1*multi_dot([j40,j32,j49]),j41,(multi_dot([j40,j32,j43]) + multi_dot([j45,j37])),-1*j47,-1*multi_dot([j46,j32,j49]),j47,(multi_dot([j46,j32,j43]) + multi_dot([j45,j39])),2*j5.T,2*j14.T,2*j31.T]
  
    
