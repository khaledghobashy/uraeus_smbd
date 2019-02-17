
import os
import numpy as np
import scipy as sc
import pandas as pd
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad
from source.solvers.py_numerical_functions import mirrored, centered, oriented




path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.R_rbs_table = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_table = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_table = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_table = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_table = 1
        self.Jbar_rbs_table = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_link_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_link_1 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_link_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_link_1 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_link_1 = 1
        self.Jbar_rbs_link_1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_link_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_link_2 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_link_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_link_2 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_link_2 = 1
        self.Jbar_rbs_link_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_link_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_link_3 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_link_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_link_3 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_link_3 = 1
        self.Jbar_rbs_link_3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_rocker_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rocker_1 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker_1 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker_1 = 1
        self.Jbar_rbs_rocker_1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_rocker_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rocker_2 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker_2 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker_2 = 1
        self.Jbar_rbs_rocker_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_rocker_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rocker_3 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker_3 = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker_3 = 1
        self.Jbar_rbs_rocker_3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_bottom_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_bottom_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_bottom_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_table,self.P_rbs_table,self.R_rbs_link_1,self.P_rbs_link_1,self.R_rbs_link_2,self.P_rbs_link_2,self.R_rbs_link_3,self.P_rbs_link_3,self.R_rbs_rocker_1,self.P_rbs_rocker_1,self.R_rbs_rocker_2,self.P_rbs_rocker_2,self.R_rbs_rocker_3,self.P_rbs_rocker_3])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_table,self.Pd_rbs_table,self.Rd_rbs_link_1,self.Pd_rbs_link_1,self.Rd_rbs_link_2,self.Pd_rbs_link_2,self.Rd_rbs_link_3,self.Pd_rbs_link_3,self.Rd_rbs_rocker_1,self.Pd_rbs_rocker_1,self.Rd_rbs_rocker_2,self.Pd_rbs_rocker_2,self.Rd_rbs_rocker_3,self.Pd_rbs_rocker_3])
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
    





class topology(object):

    def __init__(self,prefix='',cfg=None):
        self.t = 0.0
        self.config = (configuration() if cfg is None else cfg)
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 49
        self.nrows = 28
        self.ncols = 2*8
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,22,23,24,25,26,27])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
        self.rbs_table = indicies_map[p+'rbs_table']
        self.rbs_link_1 = indicies_map[p+'rbs_link_1']
        self.rbs_link_2 = indicies_map[p+'rbs_link_2']
        self.rbs_link_3 = indicies_map[p+'rbs_link_3']
        self.rbs_rocker_1 = indicies_map[p+'rbs_rocker_1']
        self.rbs_rocker_2 = indicies_map[p+'rbs_rocker_2']
        self.rbs_rocker_3 = indicies_map[p+'rbs_rocker_3']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_table*2+1,self.rbs_link_1*2+1,self.rbs_link_2*2+1,self.rbs_link_3*2+1,self.rbs_rocker_1*2+1,self.rbs_rocker_2*2+1,self.rbs_rocker_3*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        c0 = A(config.P_rbs_link_1).T
        c1 = triad(config.ax1_jcs_upper_uni_1)
        c2 = A(config.P_rbs_table).T
        c3 = config.pt1_jcs_upper_uni_1
        c4 = -1*multi_dot([c0,config.R_rbs_link_1])
        c5 = -1*multi_dot([c2,config.R_rbs_table])
        c6 = A(config.P_rbs_link_2).T
        c7 = triad(config.ax1_jcs_upper_uni_2)
        c8 = config.pt1_jcs_upper_uni_2
        c9 = -1*multi_dot([c6,config.R_rbs_link_2])
        c10 = A(config.P_rbs_link_3).T
        c11 = triad(config.ax1_jcs_upper_uni_3)
        c12 = config.pt1_jcs_upper_uni_3
        c13 = -1*multi_dot([c10,config.R_rbs_link_3])
        c14 = A(config.P_rbs_rocker_1).T
        c15 = triad(config.ax1_jcs_rev_1)
        c16 = A(config.P_vbs_ground).T
        c17 = config.pt1_jcs_rev_1
        c18 = -1*multi_dot([c14,config.R_rbs_rocker_1])
        c19 = -1*multi_dot([c16,config.R_vbs_ground])
        c20 = triad(config.ax1_jcs_bottom_uni_1)
        c21 = config.pt1_jcs_bottom_uni_1
        c22 = A(config.P_rbs_rocker_2).T
        c23 = triad(config.ax1_jcs_rev_2)
        c24 = config.pt1_jcs_rev_2
        c25 = -1*multi_dot([c22,config.R_rbs_rocker_2])
        c26 = triad(config.ax1_jcs_bottom_uni_2)
        c27 = config.pt1_jcs_bottom_uni_2
        c28 = A(config.P_rbs_rocker_3).T
        c29 = triad(config.ax1_jcs_rev_3)
        c30 = config.pt1_jcs_rev_3
        c31 = -1*multi_dot([c28,config.R_rbs_rocker_3])
        c32 = triad(config.ax1_jcs_bottom_uni_3)
        c33 = config.pt1_jcs_bottom_uni_3

        self.Mbar_rbs_link_1_jcs_upper_uni_1 = multi_dot([c0,c1])
        self.Mbar_rbs_table_jcs_upper_uni_1 = multi_dot([c2,triad(config.ax2_jcs_upper_uni_1,c1[0:3,1:2])])
        self.ubar_rbs_link_1_jcs_upper_uni_1 = (multi_dot([c0,c3]) + c4)
        self.ubar_rbs_table_jcs_upper_uni_1 = (multi_dot([c2,c3]) + c5)
        self.Mbar_rbs_link_2_jcs_upper_uni_2 = multi_dot([c6,c7])
        self.Mbar_rbs_table_jcs_upper_uni_2 = multi_dot([c2,triad(config.ax2_jcs_upper_uni_2,c7[0:3,1:2])])
        self.ubar_rbs_link_2_jcs_upper_uni_2 = (multi_dot([c6,c8]) + c9)
        self.ubar_rbs_table_jcs_upper_uni_2 = (multi_dot([c2,c8]) + c5)
        self.Mbar_rbs_link_3_jcs_upper_uni_3 = multi_dot([c10,c11])
        self.Mbar_rbs_table_jcs_upper_uni_3 = multi_dot([c2,triad(config.ax2_jcs_upper_uni_3,c11[0:3,1:2])])
        self.ubar_rbs_link_3_jcs_upper_uni_3 = (multi_dot([c10,c12]) + c13)
        self.ubar_rbs_table_jcs_upper_uni_3 = (multi_dot([c2,c12]) + c5)
        self.Mbar_rbs_rocker_1_jcs_rev_1 = multi_dot([c14,c15])
        self.Mbar_vbs_ground_jcs_rev_1 = multi_dot([c16,c15])
        self.ubar_rbs_rocker_1_jcs_rev_1 = (multi_dot([c14,c17]) + c18)
        self.ubar_vbs_ground_jcs_rev_1 = (multi_dot([c16,c17]) + c19)
        self.Mbar_rbs_rocker_1_jcs_bottom_uni_1 = multi_dot([c14,c20])
        self.Mbar_rbs_link_1_jcs_bottom_uni_1 = multi_dot([c0,triad(config.ax2_jcs_bottom_uni_1,c20[0:3,1:2])])
        self.ubar_rbs_rocker_1_jcs_bottom_uni_1 = (multi_dot([c14,c21]) + c18)
        self.ubar_rbs_link_1_jcs_bottom_uni_1 = (multi_dot([c0,c21]) + c4)
        self.Mbar_rbs_rocker_2_jcs_rev_2 = multi_dot([c22,c23])
        self.Mbar_vbs_ground_jcs_rev_2 = multi_dot([c16,c23])
        self.ubar_rbs_rocker_2_jcs_rev_2 = (multi_dot([c22,c24]) + c25)
        self.ubar_vbs_ground_jcs_rev_2 = (multi_dot([c16,c24]) + c19)
        self.Mbar_rbs_rocker_2_jcs_bottom_uni_2 = multi_dot([c22,c26])
        self.Mbar_rbs_link_2_jcs_bottom_uni_2 = multi_dot([c6,triad(config.ax2_jcs_bottom_uni_2,c26[0:3,1:2])])
        self.ubar_rbs_rocker_2_jcs_bottom_uni_2 = (multi_dot([c22,c27]) + c25)
        self.ubar_rbs_link_2_jcs_bottom_uni_2 = (multi_dot([c6,c27]) + c9)
        self.Mbar_rbs_rocker_3_jcs_rev_3 = multi_dot([c28,c29])
        self.Mbar_vbs_ground_jcs_rev_3 = multi_dot([c16,c29])
        self.ubar_rbs_rocker_3_jcs_rev_3 = (multi_dot([c28,c30]) + c31)
        self.ubar_vbs_ground_jcs_rev_3 = (multi_dot([c16,c30]) + c19)
        self.Mbar_rbs_rocker_3_jcs_bottom_uni_3 = multi_dot([c28,c32])
        self.Mbar_rbs_link_3_jcs_bottom_uni_3 = multi_dot([c10,triad(config.ax2_jcs_bottom_uni_3,c32[0:3,1:2])])
        self.ubar_rbs_rocker_3_jcs_bottom_uni_3 = (multi_dot([c28,c33]) + c31)
        self.ubar_rbs_link_3_jcs_bottom_uni_3 = (multi_dot([c10,c33]) + c13)

    
    def set_gen_coordinates(self,q):
        self.R_rbs_table = q[0:3,0:1]
        self.P_rbs_table = q[3:7,0:1]
        self.R_rbs_link_1 = q[7:10,0:1]
        self.P_rbs_link_1 = q[10:14,0:1]
        self.R_rbs_link_2 = q[14:17,0:1]
        self.P_rbs_link_2 = q[17:21,0:1]
        self.R_rbs_link_3 = q[21:24,0:1]
        self.P_rbs_link_3 = q[24:28,0:1]
        self.R_rbs_rocker_1 = q[28:31,0:1]
        self.P_rbs_rocker_1 = q[31:35,0:1]
        self.R_rbs_rocker_2 = q[35:38,0:1]
        self.P_rbs_rocker_2 = q[38:42,0:1]
        self.R_rbs_rocker_3 = q[42:45,0:1]
        self.P_rbs_rocker_3 = q[45:49,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_table = qd[0:3,0:1]
        self.Pd_rbs_table = qd[3:7,0:1]
        self.Rd_rbs_link_1 = qd[7:10,0:1]
        self.Pd_rbs_link_1 = qd[10:14,0:1]
        self.Rd_rbs_link_2 = qd[14:17,0:1]
        self.Pd_rbs_link_2 = qd[17:21,0:1]
        self.Rd_rbs_link_3 = qd[21:24,0:1]
        self.Pd_rbs_link_3 = qd[24:28,0:1]
        self.Rd_rbs_rocker_1 = qd[28:31,0:1]
        self.Pd_rbs_rocker_1 = qd[31:35,0:1]
        self.Rd_rbs_rocker_2 = qd[35:38,0:1]
        self.Pd_rbs_rocker_2 = qd[38:42,0:1]
        self.Rd_rbs_rocker_3 = qd[42:45,0:1]
        self.Pd_rbs_rocker_3 = qd[45:49,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbs_link_1
        x1 = -1*self.R_rbs_table
        x2 = self.P_rbs_link_1
        x3 = A(x2)
        x4 = self.P_rbs_table
        x5 = A(x4)
        x6 = self.R_rbs_link_2
        x7 = self.P_rbs_link_2
        x8 = A(x7)
        x9 = self.R_rbs_link_3
        x10 = self.P_rbs_link_3
        x11 = A(x10)
        x12 = self.R_rbs_rocker_1
        x13 = -1*self.R_vbs_ground
        x14 = self.P_rbs_rocker_1
        x15 = A(x14)
        x16 = A(self.P_vbs_ground)
        x17 = x15.T
        x18 = self.Mbar_vbs_ground_jcs_rev_1[:,2:3]
        x19 = self.R_rbs_rocker_2
        x20 = self.P_rbs_rocker_2
        x21 = A(x20)
        x22 = x21.T
        x23 = self.Mbar_vbs_ground_jcs_rev_2[:,2:3]
        x24 = self.R_rbs_rocker_3
        x25 = self.P_rbs_rocker_3
        x26 = A(x25)
        x27 = x26.T
        x28 = self.Mbar_vbs_ground_jcs_rev_3[:,2:3]
        x29 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x1 + multi_dot([x3,self.ubar_rbs_link_1_jcs_upper_uni_1]) + -1*multi_dot([x5,self.ubar_rbs_table_jcs_upper_uni_1])),multi_dot([self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1].T,x3.T,x5,self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]]),(x6 + x1 + multi_dot([x8,self.ubar_rbs_link_2_jcs_upper_uni_2]) + -1*multi_dot([x5,self.ubar_rbs_table_jcs_upper_uni_2])),multi_dot([self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1].T,x8.T,x5,self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]]),(x9 + x1 + multi_dot([x11,self.ubar_rbs_link_3_jcs_upper_uni_3]) + -1*multi_dot([x5,self.ubar_rbs_table_jcs_upper_uni_3])),multi_dot([self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1].T,x11.T,x5,self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]]),(x12 + x13 + multi_dot([x15,self.ubar_rbs_rocker_1_jcs_rev_1]) + -1*multi_dot([x16,self.ubar_vbs_ground_jcs_rev_1])),multi_dot([self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1].T,x17,x16,x18]),multi_dot([self.Mbar_rbs_rocker_1_jcs_rev_1[:,1:2].T,x17,x16,x18]),(x12 + -1*x0 + multi_dot([x15,self.ubar_rbs_rocker_1_jcs_bottom_uni_1]) + -1*multi_dot([x3,self.ubar_rbs_link_1_jcs_bottom_uni_1])),multi_dot([self.Mbar_rbs_rocker_1_jcs_bottom_uni_1[:,0:1].T,x17,x3,self.Mbar_rbs_link_1_jcs_bottom_uni_1[:,0:1]]),(x19 + x13 + multi_dot([x21,self.ubar_rbs_rocker_2_jcs_rev_2]) + -1*multi_dot([x16,self.ubar_vbs_ground_jcs_rev_2])),multi_dot([self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1].T,x22,x16,x23]),multi_dot([self.Mbar_rbs_rocker_2_jcs_rev_2[:,1:2].T,x22,x16,x23]),(x19 + -1*x6 + multi_dot([x21,self.ubar_rbs_rocker_2_jcs_bottom_uni_2]) + -1*multi_dot([x8,self.ubar_rbs_link_2_jcs_bottom_uni_2])),multi_dot([self.Mbar_rbs_rocker_2_jcs_bottom_uni_2[:,0:1].T,x22,x8,self.Mbar_rbs_link_2_jcs_bottom_uni_2[:,0:1]]),(x24 + x13 + multi_dot([x26,self.ubar_rbs_rocker_3_jcs_rev_3]) + -1*multi_dot([x16,self.ubar_vbs_ground_jcs_rev_3])),multi_dot([self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1].T,x27,x16,x28]),multi_dot([self.Mbar_rbs_rocker_3_jcs_rev_3[:,1:2].T,x27,x16,x28]),(x24 + -1*x9 + multi_dot([x26,self.ubar_rbs_rocker_3_jcs_bottom_uni_3]) + -1*multi_dot([x11,self.ubar_rbs_link_3_jcs_bottom_uni_3])),multi_dot([self.Mbar_rbs_rocker_3_jcs_bottom_uni_3[:,0:1].T,x27,x11,self.Mbar_rbs_link_3_jcs_bottom_uni_3[:,0:1]]),(x29 + (multi_dot([x4.T,x4]))**(1.0/2.0)),(x29 + (multi_dot([x2.T,x2]))**(1.0/2.0)),(x29 + (multi_dot([x7.T,x7]))**(1.0/2.0)),(x29 + (multi_dot([x10.T,x10]))**(1.0/2.0)),(x29 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x29 + (multi_dot([x20.T,x20]))**(1.0/2.0)),(x29 + (multi_dot([x25.T,x25]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v0,v1,v0,v1,v0,v1,v1,v0,v1,v0,v1,v1,v0,v1,v0,v1,v1,v0,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbs_link_1
        a1 = self.Pd_rbs_table
        a2 = self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]
        a3 = self.P_rbs_table
        a4 = A(a3).T
        a5 = self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]
        a6 = self.P_rbs_link_1
        a7 = A(a6).T
        a8 = a0.T
        a9 = self.Pd_rbs_link_2
        a10 = self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]
        a11 = self.P_rbs_link_2
        a12 = A(a11).T
        a13 = self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]
        a14 = a9.T
        a15 = self.Pd_rbs_link_3
        a16 = self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]
        a17 = self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]
        a18 = self.P_rbs_link_3
        a19 = A(a18).T
        a20 = a15.T
        a21 = self.Pd_rbs_rocker_1
        a22 = self.Pd_vbs_ground
        a23 = self.Mbar_vbs_ground_jcs_rev_1[:,2:3]
        a24 = a23.T
        a25 = self.P_vbs_ground
        a26 = A(a25).T
        a27 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1]
        a28 = self.P_rbs_rocker_1
        a29 = A(a28).T
        a30 = B(a22,a23)
        a31 = a21.T
        a32 = B(a25,a23)
        a33 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,1:2]
        a34 = self.Mbar_rbs_link_1_jcs_bottom_uni_1[:,0:1]
        a35 = self.Mbar_rbs_rocker_1_jcs_bottom_uni_1[:,0:1]
        a36 = self.Pd_rbs_rocker_2
        a37 = self.Mbar_vbs_ground_jcs_rev_2[:,2:3]
        a38 = a37.T
        a39 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1]
        a40 = self.P_rbs_rocker_2
        a41 = A(a40).T
        a42 = B(a22,a37)
        a43 = a36.T
        a44 = B(a25,a37)
        a45 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,1:2]
        a46 = self.Mbar_rbs_link_2_jcs_bottom_uni_2[:,0:1]
        a47 = self.Mbar_rbs_rocker_2_jcs_bottom_uni_2[:,0:1]
        a48 = self.Pd_rbs_rocker_3
        a49 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1]
        a50 = self.P_rbs_rocker_3
        a51 = A(a50).T
        a52 = self.Mbar_vbs_ground_jcs_rev_3[:,2:3]
        a53 = B(a22,a52)
        a54 = a52.T
        a55 = a48.T
        a56 = B(a25,a52)
        a57 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,1:2]
        a58 = self.Mbar_rbs_rocker_3_jcs_bottom_uni_3[:,0:1]
        a59 = self.Mbar_rbs_link_3_jcs_bottom_uni_3[:,0:1]

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_rbs_link_1_jcs_upper_uni_1),a0]) + -1*multi_dot([B(a1,self.ubar_rbs_table_jcs_upper_uni_1),a1])),(multi_dot([a2.T,a4,B(a0,a5),a0]) + multi_dot([a5.T,a7,B(a1,a2),a1]) + 2*multi_dot([a8,B(a6,a5).T,B(a3,a2),a1])),(multi_dot([B(a9,self.ubar_rbs_link_2_jcs_upper_uni_2),a9]) + -1*multi_dot([B(a1,self.ubar_rbs_table_jcs_upper_uni_2),a1])),(multi_dot([a10.T,a12,B(a1,a13),a1]) + multi_dot([a13.T,a4,B(a9,a10),a9]) + 2*multi_dot([a14,B(a11,a10).T,B(a3,a13),a1])),(multi_dot([B(a15,self.ubar_rbs_link_3_jcs_upper_uni_3),a15]) + -1*multi_dot([B(a1,self.ubar_rbs_table_jcs_upper_uni_3),a1])),(multi_dot([a16.T,a4,B(a15,a17),a15]) + multi_dot([a17.T,a19,B(a1,a16),a1]) + 2*multi_dot([a20,B(a18,a17).T,B(a3,a16),a1])),(multi_dot([B(a21,self.ubar_rbs_rocker_1_jcs_rev_1),a21]) + -1*multi_dot([B(a22,self.ubar_vbs_ground_jcs_rev_1),a22])),(multi_dot([a24,a26,B(a21,a27),a21]) + multi_dot([a27.T,a29,a30,a22]) + 2*multi_dot([a31,B(a28,a27).T,a32,a22])),(multi_dot([a24,a26,B(a21,a33),a21]) + multi_dot([a33.T,a29,a30,a22]) + 2*multi_dot([a31,B(a28,a33).T,a32,a22])),(multi_dot([B(a21,self.ubar_rbs_rocker_1_jcs_bottom_uni_1),a21]) + -1*multi_dot([B(a0,self.ubar_rbs_link_1_jcs_bottom_uni_1),a0])),(multi_dot([a34.T,a7,B(a21,a35),a21]) + multi_dot([a35.T,a29,B(a0,a34),a0]) + 2*multi_dot([a31,B(a28,a35).T,B(a6,a34),a0])),(multi_dot([B(a36,self.ubar_rbs_rocker_2_jcs_rev_2),a36]) + -1*multi_dot([B(a22,self.ubar_vbs_ground_jcs_rev_2),a22])),(multi_dot([a38,a26,B(a36,a39),a36]) + multi_dot([a39.T,a41,a42,a22]) + 2*multi_dot([a43,B(a40,a39).T,a44,a22])),(multi_dot([a38,a26,B(a36,a45),a36]) + multi_dot([a45.T,a41,a42,a22]) + 2*multi_dot([a43,B(a40,a45).T,a44,a22])),(multi_dot([B(a36,self.ubar_rbs_rocker_2_jcs_bottom_uni_2),a36]) + -1*multi_dot([B(a9,self.ubar_rbs_link_2_jcs_bottom_uni_2),a9])),(multi_dot([a46.T,a12,B(a36,a47),a36]) + multi_dot([a47.T,a41,B(a9,a46),a9]) + 2*multi_dot([a43,B(a40,a47).T,B(a11,a46),a9])),(multi_dot([B(a48,self.ubar_rbs_rocker_3_jcs_rev_3),a48]) + -1*multi_dot([B(a22,self.ubar_vbs_ground_jcs_rev_3),a22])),(multi_dot([a49.T,a51,a53,a22]) + multi_dot([a54,a26,B(a48,a49),a48]) + 2*multi_dot([a55,B(a50,a49).T,a56,a22])),(multi_dot([a57.T,a51,a53,a22]) + multi_dot([a54,a26,B(a48,a57),a48]) + 2*multi_dot([a55,B(a50,a57).T,a56,a22])),(multi_dot([B(a48,self.ubar_rbs_rocker_3_jcs_bottom_uni_3),a48]) + -1*multi_dot([B(a15,self.ubar_rbs_link_3_jcs_bottom_uni_3),a15])),(multi_dot([a58.T,a51,B(a15,a59),a15]) + multi_dot([a59.T,a19,B(a48,a58),a48]) + 2*multi_dot([a55,B(a50,a58).T,B(a18,a59),a15])),2*(multi_dot([a1.T,a1]))**(1.0/2.0),2*(multi_dot([a8,a0]))**(1.0/2.0),2*(multi_dot([a14,a9]))**(1.0/2.0),2*(multi_dot([a20,a15]))**(1.0/2.0),2*(multi_dot([a31,a21]))**(1.0/2.0),2*(multi_dot([a43,a36]))**(1.0/2.0),2*(multi_dot([a55,a48]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbs_link_1
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]
        j4 = self.P_rbs_table
        j5 = A(j4).T
        j6 = self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]
        j7 = -1*j0
        j8 = A(j1).T
        j9 = self.P_rbs_link_2
        j10 = self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]
        j11 = self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]
        j12 = A(j9).T
        j13 = self.P_rbs_link_3
        j14 = self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]
        j15 = self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]
        j16 = A(j13).T
        j17 = self.P_rbs_rocker_1
        j18 = self.Mbar_vbs_ground_jcs_rev_1[:,2:3]
        j19 = j18.T
        j20 = self.P_vbs_ground
        j21 = A(j20).T
        j22 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,0:1]
        j23 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,1:2]
        j24 = A(j17).T
        j25 = B(j20,j18)
        j26 = self.Mbar_rbs_link_1_jcs_bottom_uni_1[:,0:1]
        j27 = self.Mbar_rbs_rocker_1_jcs_bottom_uni_1[:,0:1]
        j28 = self.P_rbs_rocker_2
        j29 = self.Mbar_vbs_ground_jcs_rev_2[:,2:3]
        j30 = j29.T
        j31 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,0:1]
        j32 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,1:2]
        j33 = A(j28).T
        j34 = B(j20,j29)
        j35 = self.Mbar_rbs_link_2_jcs_bottom_uni_2[:,0:1]
        j36 = self.Mbar_rbs_rocker_2_jcs_bottom_uni_2[:,0:1]
        j37 = self.P_rbs_rocker_3
        j38 = self.Mbar_vbs_ground_jcs_rev_3[:,2:3]
        j39 = j38.T
        j40 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,0:1]
        j41 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,1:2]
        j42 = A(j37).T
        j43 = B(j20,j38)
        j44 = self.Mbar_rbs_link_3_jcs_bottom_uni_3[:,0:1]
        j45 = self.Mbar_rbs_rocker_3_jcs_bottom_uni_3[:,0:1]

        self.jac_eq_blocks = [j7,-1*B(j4,self.ubar_rbs_table_jcs_upper_uni_1),j0,B(j1,self.ubar_rbs_link_1_jcs_upper_uni_1),j2,multi_dot([j6.T,j8,B(j4,j3)]),j2,multi_dot([j3.T,j5,B(j1,j6)]),j7,-1*B(j4,self.ubar_rbs_table_jcs_upper_uni_2),j0,B(j9,self.ubar_rbs_link_2_jcs_upper_uni_2),j2,multi_dot([j11.T,j12,B(j4,j10)]),j2,multi_dot([j10.T,j5,B(j9,j11)]),j7,-1*B(j4,self.ubar_rbs_table_jcs_upper_uni_3),j0,B(j13,self.ubar_rbs_link_3_jcs_upper_uni_3),j2,multi_dot([j15.T,j16,B(j4,j14)]),j2,multi_dot([j14.T,j5,B(j13,j15)]),j7,-1*B(j20,self.ubar_vbs_ground_jcs_rev_1),j0,B(j17,self.ubar_rbs_rocker_1_jcs_rev_1),j2,multi_dot([j22.T,j24,j25]),j2,multi_dot([j19,j21,B(j17,j22)]),j2,multi_dot([j23.T,j24,j25]),j2,multi_dot([j19,j21,B(j17,j23)]),j7,-1*B(j1,self.ubar_rbs_link_1_jcs_bottom_uni_1),j0,B(j17,self.ubar_rbs_rocker_1_jcs_bottom_uni_1),j2,multi_dot([j27.T,j24,B(j1,j26)]),j2,multi_dot([j26.T,j8,B(j17,j27)]),j7,-1*B(j20,self.ubar_vbs_ground_jcs_rev_2),j0,B(j28,self.ubar_rbs_rocker_2_jcs_rev_2),j2,multi_dot([j31.T,j33,j34]),j2,multi_dot([j30,j21,B(j28,j31)]),j2,multi_dot([j32.T,j33,j34]),j2,multi_dot([j30,j21,B(j28,j32)]),j7,-1*B(j9,self.ubar_rbs_link_2_jcs_bottom_uni_2),j0,B(j28,self.ubar_rbs_rocker_2_jcs_bottom_uni_2),j2,multi_dot([j36.T,j33,B(j9,j35)]),j2,multi_dot([j35.T,j12,B(j28,j36)]),j7,-1*B(j20,self.ubar_vbs_ground_jcs_rev_3),j0,B(j37,self.ubar_rbs_rocker_3_jcs_rev_3),j2,multi_dot([j40.T,j42,j43]),j2,multi_dot([j39,j21,B(j37,j40)]),j2,multi_dot([j41.T,j42,j43]),j2,multi_dot([j39,j21,B(j37,j41)]),j7,-1*B(j13,self.ubar_rbs_link_3_jcs_bottom_uni_3),j0,B(j37,self.ubar_rbs_rocker_3_jcs_bottom_uni_3),j2,multi_dot([j45.T,j42,B(j13,j44)]),j2,multi_dot([j44.T,j16,B(j37,j45)]),2*j4.T,2*j1.T,2*j9.T,2*j13.T,2*j17.T,2*j28.T,2*j37.T]
  
