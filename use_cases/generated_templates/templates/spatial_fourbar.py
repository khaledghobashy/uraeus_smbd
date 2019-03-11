
import os
import numpy as np
import pandas as pd
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, G, E, triad, skew_matrix as skew
from source.solvers.py_numerical_functions import mirrored



path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.R_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_crank = 1
        self.Jbar_rbs_crank = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker = 1
        self.Jbar_rbs_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_coupler = 1
        self.Jbar_rbs_coupler = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcs_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_crank = lambda t : 0
        self.ax1_jcs_rev_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_sph_coupler_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_sph_coupler_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_uni_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_uni_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_uni_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_crank,self.P_rbs_crank,self.R_rbs_rocker,self.P_rbs_rocker,self.R_rbs_coupler,self.P_rbs_coupler])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_crank,self.Pd_rbs_crank,self.Rd_rbs_rocker,self.Pd_rbs_rocker,self.Rd_rbs_coupler,self.Pd_rbs_coupler])
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

        self.n  = 21
        self.nc = 21
        self.nrows = 13
        self.ncols = 2*4
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,11,12])
        self.joints_reactions_indicies = ['F_rbs_crank_jcs_rev_crank','T_rbs_crank_jcs_rev_crank','F_rbs_crank_jcs_rev_crank','T_rbs_crank_jcs_rev_crank','F_rbs_rocker_jcs_rev_rocker','T_rbs_rocker_jcs_rev_rocker','F_rbs_coupler_jcs_sph_coupler_crank','T_rbs_coupler_jcs_sph_coupler_crank','F_rbs_coupler_jcs_uni_coupler_rocker','T_rbs_coupler_jcs_uni_coupler_rocker']

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
        self.rbs_crank = indicies_map[p+'rbs_crank']
        self.rbs_rocker = indicies_map[p+'rbs_rocker']
        self.rbs_coupler = indicies_map[p+'rbs_coupler']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_crank*2,self.rbs_crank*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_crank*2,self.rbs_crank*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_crank*2,self.rbs_crank*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_crank*2,self.rbs_crank*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_crank*2,self.rbs_crank*2+1,self.rbs_coupler*2,self.rbs_coupler*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_coupler*2,self.rbs_coupler*2+1,self.rbs_rocker*2,self.rbs_rocker*2+1,self.rbs_coupler*2,self.rbs_coupler*2+1,self.rbs_crank*2+1,self.rbs_rocker*2+1,self.rbs_coupler*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_crank_gravity = np.array([[0], [0], [9810.0*config.m_rbs_crank]],dtype=np.float64)
        self.F_rbs_rocker_gravity = np.array([[0], [0], [9810.0*config.m_rbs_rocker]],dtype=np.float64)
        self.F_rbs_coupler_gravity = np.array([[0], [0], [9810.0*config.m_rbs_coupler]],dtype=np.float64)

        c0 = A(config.P_rbs_crank).T
        c1 = triad(config.ax1_jcs_rev_crank)
        c2 = self.Mbar_rbs_crank_jcs_rev_crank = multi_dot([c0,c1])
        c3 = A(config.P_vbs_ground).T
        c4 = self.Mbar_vbs_ground_jcs_rev_crank = multi_dot([c3,c1])
        c5 = config.pt1_jcs_rev_crank
        c6 = -1*multi_dot([c0,config.R_rbs_crank])
        c7 = -1*multi_dot([c3,config.R_vbs_ground])
        c8 = A(config.P_rbs_rocker).T
        c9 = triad(config.ax1_jcs_rev_rocker)
        c10 = config.pt1_jcs_rev_rocker
        c11 = -1*multi_dot([c8,config.R_rbs_rocker])
        c12 = A(config.P_rbs_coupler).T
        c13 = triad(config.ax1_jcs_sph_coupler_crank)
        c14 = config.pt1_jcs_sph_coupler_crank
        c15 = -1*multi_dot([c12,config.R_rbs_coupler])
        c16 = triad(config.ax1_jcs_uni_coupler_rocker)
        c17 = config.pt1_jcs_uni_coupler_rocker

        c2
        c4
        self.ubar_rbs_crank_jcs_rev_crank = (multi_dot([c0,c5]) + c6)
        self.ubar_vbs_ground_jcs_rev_crank = (multi_dot([c3,c5]) + c7)
        c2
        c4
        self.Mbar_rbs_rocker_jcs_rev_rocker = multi_dot([c8,c9])
        self.Mbar_vbs_ground_jcs_rev_rocker = multi_dot([c3,c9])
        self.ubar_rbs_rocker_jcs_rev_rocker = (multi_dot([c8,c10]) + c11)
        self.ubar_vbs_ground_jcs_rev_rocker = (multi_dot([c3,c10]) + c7)
        self.Mbar_rbs_coupler_jcs_sph_coupler_crank = multi_dot([c12,c13])
        self.Mbar_rbs_crank_jcs_sph_coupler_crank = multi_dot([c0,c13])
        self.ubar_rbs_coupler_jcs_sph_coupler_crank = (multi_dot([c12,c14]) + c15)
        self.ubar_rbs_crank_jcs_sph_coupler_crank = (multi_dot([c0,c14]) + c6)
        self.Mbar_rbs_coupler_jcs_uni_coupler_rocker = multi_dot([c12,c16])
        self.Mbar_rbs_rocker_jcs_uni_coupler_rocker = multi_dot([c8,triad(config.ax2_jcs_uni_coupler_rocker,c16[0:3,1:2])])
        self.ubar_rbs_coupler_jcs_uni_coupler_rocker = (multi_dot([c12,c17]) + c15)
        self.ubar_rbs_rocker_jcs_uni_coupler_rocker = (multi_dot([c8,c17]) + c11)

    
    def set_gen_coordinates(self,q):
        self.R_rbs_crank = q[0:3,0:1]
        self.P_rbs_crank = q[3:7,0:1]
        self.R_rbs_rocker = q[7:10,0:1]
        self.P_rbs_rocker = q[10:14,0:1]
        self.R_rbs_coupler = q[14:17,0:1]
        self.P_rbs_coupler = q[17:21,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_crank = qd[0:3,0:1]
        self.Pd_rbs_crank = qd[3:7,0:1]
        self.Rd_rbs_rocker = qd[7:10,0:1]
        self.Pd_rbs_rocker = qd[10:14,0:1]
        self.Rd_rbs_coupler = qd[14:17,0:1]
        self.Pd_rbs_coupler = qd[17:21,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_crank = qdd[0:3,0:1]
        self.Pdd_rbs_crank = qdd[3:7,0:1]
        self.Rdd_rbs_rocker = qdd[7:10,0:1]
        self.Pdd_rbs_rocker = qdd[10:14,0:1]
        self.Rdd_rbs_coupler = qdd[14:17,0:1]
        self.Pdd_rbs_coupler = qdd[17:21,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_rev_crank = Lambda[0:5,0:1]
        self.L_jcs_rev_crank = Lambda[5:6,0:1]
        self.L_jcs_rev_rocker = Lambda[6:11,0:1]
        self.L_jcs_sph_coupler_crank = Lambda[11:14,0:1]
        self.L_jcs_uni_coupler_rocker = Lambda[14:18,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbs_crank
        x1 = -1*self.R_vbs_ground
        x2 = self.P_rbs_crank
        x3 = A(x2)
        x4 = A(self.P_vbs_ground)
        x5 = x3.T
        x6 = self.Mbar_vbs_ground_jcs_rev_crank[:,2:3]
        x7 = self.Mbar_vbs_ground_jcs_rev_crank[:,0:1]
        x8 = self.R_rbs_rocker
        x9 = self.P_rbs_rocker
        x10 = A(x9)
        x11 = x10.T
        x12 = self.Mbar_vbs_ground_jcs_rev_rocker[:,2:3]
        x13 = self.R_rbs_coupler
        x14 = self.P_rbs_coupler
        x15 = A(x14)
        x16 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x1 + multi_dot([x3,self.ubar_rbs_crank_jcs_rev_crank]) + -1*multi_dot([x4,self.ubar_vbs_ground_jcs_rev_crank])),multi_dot([self.Mbar_rbs_crank_jcs_rev_crank[:,0:1].T,x5,x4,x6]),multi_dot([self.Mbar_rbs_crank_jcs_rev_crank[:,1:2].T,x5,x4,x6]),(cos(config.AF_jcs_rev_crank(t))*multi_dot([self.Mbar_rbs_crank_jcs_rev_crank[:,1:2].T,x5,x4,x7]) + sin(config.AF_jcs_rev_crank(t))*-1*multi_dot([self.Mbar_rbs_crank_jcs_rev_crank[:,0:1].T,x5,x4,x7])),(x8 + x1 + multi_dot([x10,self.ubar_rbs_rocker_jcs_rev_rocker]) + -1*multi_dot([x4,self.ubar_vbs_ground_jcs_rev_rocker])),multi_dot([self.Mbar_rbs_rocker_jcs_rev_rocker[:,0:1].T,x11,x4,x12]),multi_dot([self.Mbar_rbs_rocker_jcs_rev_rocker[:,1:2].T,x11,x4,x12]),(x13 + -1*x0 + multi_dot([x15,self.ubar_rbs_coupler_jcs_sph_coupler_crank]) + -1*multi_dot([x3,self.ubar_rbs_crank_jcs_sph_coupler_crank])),(x13 + -1*x8 + multi_dot([x15,self.ubar_rbs_coupler_jcs_uni_coupler_rocker]) + -1*multi_dot([x10,self.ubar_rbs_rocker_jcs_uni_coupler_rocker])),multi_dot([self.Mbar_rbs_coupler_jcs_uni_coupler_rocker[:,0:1].T,x15.T,x10,self.Mbar_rbs_rocker_jcs_uni_coupler_rocker[:,0:1]]),(x16 + (multi_dot([x2.T,x2]))**(1.0/2.0)),(x16 + (multi_dot([x9.T,x9]))**(1.0/2.0)),(x16 + (multi_dot([x14.T,x14]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,-1*derivative(config.AF_jcs_rev_crank,t,0.1,1)*np.eye(1,dtype=np.float64),v0,v1,v1,v0,v0,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbs_crank
        a1 = self.Pd_vbs_ground
        a2 = self.Mbar_rbs_crank_jcs_rev_crank[:,0:1]
        a3 = self.P_rbs_crank
        a4 = A(a3).T
        a5 = self.Mbar_vbs_ground_jcs_rev_crank[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_vbs_ground
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_rbs_crank_jcs_rev_crank[:,1:2]
        a13 = self.Mbar_vbs_ground_jcs_rev_crank[:,0:1]
        a14 = self.Mbar_rbs_crank_jcs_rev_crank[:,1:2]
        a15 = self.Mbar_rbs_crank_jcs_rev_crank[:,0:1]
        a16 = self.Pd_rbs_rocker
        a17 = self.Mbar_vbs_ground_jcs_rev_rocker[:,2:3]
        a18 = a17.T
        a19 = self.Mbar_rbs_rocker_jcs_rev_rocker[:,0:1]
        a20 = self.P_rbs_rocker
        a21 = A(a20).T
        a22 = B(a1,a17)
        a23 = a16.T
        a24 = B(a8,a17)
        a25 = self.Mbar_rbs_rocker_jcs_rev_rocker[:,1:2]
        a26 = self.Pd_rbs_coupler
        a27 = self.Mbar_rbs_coupler_jcs_uni_coupler_rocker[:,0:1]
        a28 = self.P_rbs_coupler
        a29 = self.Mbar_rbs_rocker_jcs_uni_coupler_rocker[:,0:1]
        a30 = a26.T

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_rbs_crank_jcs_rev_crank),a0]) + -1*multi_dot([B(a1,self.ubar_vbs_ground_jcs_rev_crank),a1])),(multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + 2*multi_dot([a10,B(a3,a2).T,a11,a1])),(multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + 2*multi_dot([a10,B(a3,a12).T,a11,a1])),(-1*derivative(config.AF_jcs_rev_crank,t,0.1,2)*np.eye(1,dtype=np.float64) + multi_dot([a13.T,a9,(cos(config.AF_jcs_rev_crank(t))*B(a0,a14) + sin(config.AF_jcs_rev_crank(t))*-1*B(a0,a15)),a0]) + multi_dot([(cos(config.AF_jcs_rev_crank(t))*multi_dot([a14.T,a4]) + sin(config.AF_jcs_rev_crank(t))*-1*multi_dot([a15.T,a4])),B(a1,a13),a1]) + 2*multi_dot([((cos(config.AF_jcs_rev_crank(t))*multi_dot([B(a3,a14),a0])).T + sin(config.AF_jcs_rev_crank(t))*-1*multi_dot([a10,B(a3,a15).T])),B(a8,a13),a1])),(multi_dot([B(a16,self.ubar_rbs_rocker_jcs_rev_rocker),a16]) + -1*multi_dot([B(a1,self.ubar_vbs_ground_jcs_rev_rocker),a1])),(multi_dot([a18,a9,B(a16,a19),a16]) + multi_dot([a19.T,a21,a22,a1]) + 2*multi_dot([a23,B(a20,a19).T,a24,a1])),(multi_dot([a18,a9,B(a16,a25),a16]) + multi_dot([a25.T,a21,a22,a1]) + 2*multi_dot([a23,B(a20,a25).T,a24,a1])),(multi_dot([B(a26,self.ubar_rbs_coupler_jcs_sph_coupler_crank),a26]) + -1*multi_dot([B(a0,self.ubar_rbs_crank_jcs_sph_coupler_crank),a0])),(multi_dot([B(a26,self.ubar_rbs_coupler_jcs_uni_coupler_rocker),a26]) + -1*multi_dot([B(a16,self.ubar_rbs_rocker_jcs_uni_coupler_rocker),a16])),(multi_dot([a27.T,A(a28).T,B(a16,a29),a16]) + multi_dot([a29.T,a21,B(a26,a27),a26]) + 2*multi_dot([a30,B(a28,a27).T,B(a20,a29),a16])),2*(multi_dot([a10,a0]))**(1.0/2.0),2*(multi_dot([a23,a16]))**(1.0/2.0),2*(multi_dot([a30,a26]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbs_crank
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_vbs_ground_jcs_rev_crank[:,2:3]
        j4 = j3.T
        j5 = self.P_vbs_ground
        j6 = A(j5).T
        j7 = self.Mbar_rbs_crank_jcs_rev_crank[:,0:1]
        j8 = self.Mbar_rbs_crank_jcs_rev_crank[:,1:2]
        j9 = -1*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_vbs_ground_jcs_rev_crank[:,0:1]
        j13 = self.Mbar_rbs_crank_jcs_rev_crank[:,1:2]
        j14 = self.Mbar_rbs_crank_jcs_rev_crank[:,0:1]
        j15 = self.P_rbs_rocker
        j16 = self.Mbar_vbs_ground_jcs_rev_rocker[:,2:3]
        j17 = j16.T
        j18 = self.Mbar_rbs_rocker_jcs_rev_rocker[:,0:1]
        j19 = self.Mbar_rbs_rocker_jcs_rev_rocker[:,1:2]
        j20 = A(j15).T
        j21 = B(j5,j16)
        j22 = self.P_rbs_coupler
        j23 = self.Mbar_rbs_rocker_jcs_uni_coupler_rocker[:,0:1]
        j24 = self.Mbar_rbs_coupler_jcs_uni_coupler_rocker[:,0:1]

        self.jac_eq_blocks = [j9,-1*B(j5,self.ubar_vbs_ground_jcs_rev_crank),j0,B(j1,self.ubar_rbs_crank_jcs_rev_crank),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j8.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([(cos(config.AF_jcs_rev_crank(t))*multi_dot([j13.T,j10]) + sin(config.AF_jcs_rev_crank(t))*-1*multi_dot([j14.T,j10])),B(j5,j12)]),j2,multi_dot([j12.T,j6,(cos(config.AF_jcs_rev_crank(t))*B(j1,j13) + sin(config.AF_jcs_rev_crank(t))*-1*B(j1,j14))]),j9,-1*B(j5,self.ubar_vbs_ground_jcs_rev_rocker),j0,B(j15,self.ubar_rbs_rocker_jcs_rev_rocker),j2,multi_dot([j18.T,j20,j21]),j2,multi_dot([j17,j6,B(j15,j18)]),j2,multi_dot([j19.T,j20,j21]),j2,multi_dot([j17,j6,B(j15,j19)]),j9,-1*B(j1,self.ubar_rbs_crank_jcs_sph_coupler_crank),j0,B(j22,self.ubar_rbs_coupler_jcs_sph_coupler_crank),j9,-1*B(j15,self.ubar_rbs_rocker_jcs_uni_coupler_rocker),j0,B(j22,self.ubar_rbs_coupler_jcs_uni_coupler_rocker),j2,multi_dot([j24.T,A(j22).T,B(j15,j23)]),j2,multi_dot([j23.T,j20,B(j22,j24)]),2*j1.T,2*j15.T,2*j22.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3,dtype=np.float64)
        m1 = G(self.P_rbs_crank)
        m2 = G(self.P_rbs_rocker)
        m3 = G(self.P_rbs_coupler)

        self.mass_eq_blocks = [config.m_rbs_crank*m0,4*multi_dot([m1.T,config.Jbar_rbs_crank,m1]),config.m_rbs_rocker*m0,4*multi_dot([m2.T,config.Jbar_rbs_rocker,m2]),config.m_rbs_coupler*m0,4*multi_dot([m3.T,config.Jbar_rbs_coupler,m3])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_crank)
        f1 = G(self.Pd_rbs_rocker)
        f2 = G(self.Pd_rbs_coupler)

        self.frc_eq_blocks = [self.F_rbs_crank_gravity,8*multi_dot([f0.T,config.Jbar_rbs_crank,f0,self.P_rbs_crank]),self.F_rbs_rocker_gravity,8*multi_dot([f1.T,config.Jbar_rbs_rocker,f1,self.P_rbs_rocker]),self.F_rbs_coupler_gravity,8*multi_dot([f2.T,config.Jbar_rbs_coupler,f2,self.P_rbs_coupler])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_crank_jcs_rev_crank = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_crank,self.ubar_rbs_crank_jcs_rev_crank).T,multi_dot([B(self.P_rbs_crank,self.Mbar_rbs_crank_jcs_rev_crank[:,0:1]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_crank[:,2:3]]),multi_dot([B(self.P_rbs_crank,self.Mbar_rbs_crank_jcs_rev_crank[:,1:2]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_crank[:,2:3]])]]),self.L_jcs_rev_crank])
        self.F_rbs_crank_jcs_rev_crank = Q_rbs_crank_jcs_rev_crank[0:3,0:1]
        Te_rbs_crank_jcs_rev_crank = Q_rbs_crank_jcs_rev_crank[3:7,0:1]
        self.T_rbs_crank_jcs_rev_crank = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_crank),self.ubar_rbs_crank_jcs_rev_crank])),self.F_rbs_crank_jcs_rev_crank]) + 0.5*multi_dot([E(self.P_rbs_crank),Te_rbs_crank_jcs_rev_crank]))
        Q_rbs_crank_jcs_rev_crank = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcs_rev_crank(t))*B(self.P_rbs_crank,self.Mbar_rbs_crank_jcs_rev_crank[:,0:1]).T + (cos(config.AF_jcs_rev_crank(t))*B(self.P_rbs_crank,self.Mbar_rbs_crank_jcs_rev_crank[:,1:2])).T),A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_crank[:,0:1]])]]),self.L_jcs_rev_crank])
        self.F_rbs_crank_jcs_rev_crank = Q_rbs_crank_jcs_rev_crank[0:3,0:1]
        Te_rbs_crank_jcs_rev_crank = Q_rbs_crank_jcs_rev_crank[3:7,0:1]
        self.T_rbs_crank_jcs_rev_crank = 0.5*multi_dot([E(self.P_rbs_crank),Te_rbs_crank_jcs_rev_crank])
        Q_rbs_rocker_jcs_rev_rocker = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_rocker,self.ubar_rbs_rocker_jcs_rev_rocker).T,multi_dot([B(self.P_rbs_rocker,self.Mbar_rbs_rocker_jcs_rev_rocker[:,0:1]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_rocker[:,2:3]]),multi_dot([B(self.P_rbs_rocker,self.Mbar_rbs_rocker_jcs_rev_rocker[:,1:2]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_rocker[:,2:3]])]]),self.L_jcs_rev_rocker])
        self.F_rbs_rocker_jcs_rev_rocker = Q_rbs_rocker_jcs_rev_rocker[0:3,0:1]
        Te_rbs_rocker_jcs_rev_rocker = Q_rbs_rocker_jcs_rev_rocker[3:7,0:1]
        self.T_rbs_rocker_jcs_rev_rocker = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_rocker),self.ubar_rbs_rocker_jcs_rev_rocker])),self.F_rbs_rocker_jcs_rev_rocker]) + 0.5*multi_dot([E(self.P_rbs_rocker),Te_rbs_rocker_jcs_rev_rocker]))
        Q_rbs_coupler_jcs_sph_coupler_crank = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbs_coupler,self.ubar_rbs_coupler_jcs_sph_coupler_crank).T]]),self.L_jcs_sph_coupler_crank])
        self.F_rbs_coupler_jcs_sph_coupler_crank = Q_rbs_coupler_jcs_sph_coupler_crank[0:3,0:1]
        Te_rbs_coupler_jcs_sph_coupler_crank = Q_rbs_coupler_jcs_sph_coupler_crank[3:7,0:1]
        self.T_rbs_coupler_jcs_sph_coupler_crank = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_coupler),self.ubar_rbs_coupler_jcs_sph_coupler_crank])),self.F_rbs_coupler_jcs_sph_coupler_crank]) + 0.5*multi_dot([E(self.P_rbs_coupler),Te_rbs_coupler_jcs_sph_coupler_crank]))
        Q_rbs_coupler_jcs_uni_coupler_rocker = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_coupler,self.ubar_rbs_coupler_jcs_uni_coupler_rocker).T,multi_dot([B(self.P_rbs_coupler,self.Mbar_rbs_coupler_jcs_uni_coupler_rocker[:,0:1]).T,A(self.P_rbs_rocker),self.Mbar_rbs_rocker_jcs_uni_coupler_rocker[:,0:1]])]]),self.L_jcs_uni_coupler_rocker])
        self.F_rbs_coupler_jcs_uni_coupler_rocker = Q_rbs_coupler_jcs_uni_coupler_rocker[0:3,0:1]
        Te_rbs_coupler_jcs_uni_coupler_rocker = Q_rbs_coupler_jcs_uni_coupler_rocker[3:7,0:1]
        self.T_rbs_coupler_jcs_uni_coupler_rocker = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_coupler),self.ubar_rbs_coupler_jcs_uni_coupler_rocker])),self.F_rbs_coupler_jcs_uni_coupler_rocker]) + 0.5*multi_dot([E(self.P_rbs_coupler),Te_rbs_coupler_jcs_uni_coupler_rocker]))

        self.reactions = {'F_rbs_crank_jcs_rev_crank':self.F_rbs_crank_jcs_rev_crank,'T_rbs_crank_jcs_rev_crank':self.T_rbs_crank_jcs_rev_crank,'F_rbs_crank_jcs_rev_crank':self.F_rbs_crank_jcs_rev_crank,'T_rbs_crank_jcs_rev_crank':self.T_rbs_crank_jcs_rev_crank,'F_rbs_rocker_jcs_rev_rocker':self.F_rbs_rocker_jcs_rev_rocker,'T_rbs_rocker_jcs_rev_rocker':self.T_rbs_rocker_jcs_rev_rocker,'F_rbs_coupler_jcs_sph_coupler_crank':self.F_rbs_coupler_jcs_sph_coupler_crank,'T_rbs_coupler_jcs_sph_coupler_crank':self.T_rbs_coupler_jcs_sph_coupler_crank,'F_rbs_coupler_jcs_uni_coupler_rocker':self.F_rbs_coupler_jcs_uni_coupler_rocker,'T_rbs_coupler_jcs_uni_coupler_rocker':self.T_rbs_coupler_jcs_uni_coupler_rocker}

