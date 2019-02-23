
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
        self.ax1_jcs_rev_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_tripod = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_tripod = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_sph_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_sph_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_sph_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_sph_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_sph_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_sph_3 = np.array([[0], [0], [0]],dtype=np.float64)                       

    
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
    
        pass




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
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_table*2,self.rbs_table*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_table*2,self.rbs_table*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_table*2+1,self.rbs_link_1*2+1,self.rbs_link_2*2+1,self.rbs_link_3*2+1,self.rbs_rocker_1*2+1,self.rbs_rocker_2*2+1,self.rbs_rocker_3*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_table_gravity = np.array([[0], [0], [9810.0*config.m_rbs_table]],dtype=np.float64)
        self.F_rbs_link_1_gravity = np.array([[0], [0], [9810.0*config.m_rbs_link_1]],dtype=np.float64)
        self.F_rbs_link_2_gravity = np.array([[0], [0], [9810.0*config.m_rbs_link_2]],dtype=np.float64)
        self.F_rbs_link_3_gravity = np.array([[0], [0], [9810.0*config.m_rbs_link_3]],dtype=np.float64)
        self.F_rbs_rocker_1_gravity = np.array([[0], [0], [9810.0*config.m_rbs_rocker_1]],dtype=np.float64)
        self.F_rbs_rocker_2_gravity = np.array([[0], [0], [9810.0*config.m_rbs_rocker_2]],dtype=np.float64)
        self.F_rbs_rocker_3_gravity = np.array([[0], [0], [9810.0*config.m_rbs_rocker_3]],dtype=np.float64)

        c0 = A(config.P_vbs_ground).T
        c1 = triad(config.ax1_jcs_rev_1)
        c2 = A(config.P_rbs_rocker_1).T
        c3 = config.pt1_jcs_rev_1
        c4 = -1*multi_dot([c0,config.R_vbs_ground])
        c5 = -1*multi_dot([c2,config.R_rbs_rocker_1])
        c6 = triad(config.ax1_jcs_rev_2)
        c7 = A(config.P_rbs_rocker_2).T
        c8 = config.pt1_jcs_rev_2
        c9 = -1*multi_dot([c7,config.R_rbs_rocker_2])
        c10 = triad(config.ax1_jcs_rev_3)
        c11 = A(config.P_rbs_rocker_3).T
        c12 = config.pt1_jcs_rev_3
        c13 = -1*multi_dot([c11,config.R_rbs_rocker_3])
        c14 = A(config.P_rbs_table).T
        c15 = triad(config.ax1_jcs_tripod)
        c16 = config.pt1_jcs_tripod
        c17 = -1*multi_dot([c14,config.R_rbs_table])
        c18 = A(config.P_rbs_link_1).T
        c19 = triad(config.ax1_jcs_upper_uni_1)
        c20 = config.pt1_jcs_upper_uni_1
        c21 = -1*multi_dot([c18,config.R_rbs_link_1])
        c22 = A(config.P_rbs_link_2).T
        c23 = triad(config.ax1_jcs_upper_uni_2)
        c24 = config.pt1_jcs_upper_uni_2
        c25 = -1*multi_dot([c22,config.R_rbs_link_2])
        c26 = A(config.P_rbs_link_3).T
        c27 = triad(config.ax1_jcs_upper_uni_3)
        c28 = config.pt1_jcs_upper_uni_3
        c29 = -1*multi_dot([c26,config.R_rbs_link_3])
        c30 = triad(config.ax1_jcs_bottom_sph_1)
        c31 = config.pt1_jcs_bottom_sph_1
        c32 = triad(config.ax1_jcs_bottom_sph_2)
        c33 = config.pt1_jcs_bottom_sph_2
        c34 = triad(config.ax1_jcs_bottom_sph_3)
        c35 = config.pt1_jcs_bottom_sph_3

        self.Mbar_vbs_ground_jcs_rev_1 = multi_dot([c0,c1])
        self.Mbar_rbs_rocker_1_jcs_rev_1 = multi_dot([c2,c1])
        self.ubar_vbs_ground_jcs_rev_1 = (multi_dot([c0,c3]) + c4)
        self.ubar_rbs_rocker_1_jcs_rev_1 = (multi_dot([c2,c3]) + c5)
        self.Mbar_vbs_ground_jcs_rev_2 = multi_dot([c0,c6])
        self.Mbar_rbs_rocker_2_jcs_rev_2 = multi_dot([c7,c6])
        self.ubar_vbs_ground_jcs_rev_2 = (multi_dot([c0,c8]) + c4)
        self.ubar_rbs_rocker_2_jcs_rev_2 = (multi_dot([c7,c8]) + c9)
        self.Mbar_vbs_ground_jcs_rev_3 = multi_dot([c0,c10])
        self.Mbar_rbs_rocker_3_jcs_rev_3 = multi_dot([c11,c10])
        self.ubar_vbs_ground_jcs_rev_3 = (multi_dot([c0,c12]) + c4)
        self.ubar_rbs_rocker_3_jcs_rev_3 = (multi_dot([c11,c12]) + c13)
        self.Mbar_rbs_table_jcs_tripod = multi_dot([c14,c15])
        self.Mbar_vbs_ground_jcs_tripod = multi_dot([c0,c15])
        self.ubar_rbs_table_jcs_tripod = (multi_dot([c14,c16]) + c17)
        self.ubar_vbs_ground_jcs_tripod = (multi_dot([c0,c16]) + c4)
        self.Mbar_rbs_link_1_jcs_upper_uni_1 = multi_dot([c18,c19])
        self.Mbar_rbs_table_jcs_upper_uni_1 = multi_dot([c14,triad(config.ax2_jcs_upper_uni_1,c19[0:3,1:2])])
        self.ubar_rbs_link_1_jcs_upper_uni_1 = (multi_dot([c18,c20]) + c21)
        self.ubar_rbs_table_jcs_upper_uni_1 = (multi_dot([c14,c20]) + c17)
        self.Mbar_rbs_link_2_jcs_upper_uni_2 = multi_dot([c22,c23])
        self.Mbar_rbs_table_jcs_upper_uni_2 = multi_dot([c14,triad(config.ax2_jcs_upper_uni_2,c23[0:3,1:2])])
        self.ubar_rbs_link_2_jcs_upper_uni_2 = (multi_dot([c22,c24]) + c25)
        self.ubar_rbs_table_jcs_upper_uni_2 = (multi_dot([c14,c24]) + c17)
        self.Mbar_rbs_link_3_jcs_upper_uni_3 = multi_dot([c26,c27])
        self.Mbar_rbs_table_jcs_upper_uni_3 = multi_dot([c14,triad(config.ax2_jcs_upper_uni_3,c27[0:3,1:2])])
        self.ubar_rbs_link_3_jcs_upper_uni_3 = (multi_dot([c26,c28]) + c29)
        self.ubar_rbs_table_jcs_upper_uni_3 = (multi_dot([c14,c28]) + c17)
        self.Mbar_rbs_rocker_1_jcs_bottom_sph_1 = multi_dot([c2,c30])
        self.Mbar_rbs_link_1_jcs_bottom_sph_1 = multi_dot([c18,c30])
        self.ubar_rbs_rocker_1_jcs_bottom_sph_1 = (multi_dot([c2,c31]) + c5)
        self.ubar_rbs_link_1_jcs_bottom_sph_1 = (multi_dot([c18,c31]) + c21)
        self.Mbar_rbs_rocker_2_jcs_bottom_sph_2 = multi_dot([c7,c32])
        self.Mbar_rbs_link_2_jcs_bottom_sph_2 = multi_dot([c22,c32])
        self.ubar_rbs_rocker_2_jcs_bottom_sph_2 = (multi_dot([c7,c33]) + c9)
        self.ubar_rbs_link_2_jcs_bottom_sph_2 = (multi_dot([c22,c33]) + c25)
        self.Mbar_rbs_rocker_3_jcs_bottom_sph_3 = multi_dot([c11,c34])
        self.Mbar_rbs_link_3_jcs_bottom_sph_3 = multi_dot([c26,c34])
        self.ubar_rbs_rocker_3_jcs_bottom_sph_3 = (multi_dot([c11,c35]) + c13)
        self.ubar_rbs_link_3_jcs_bottom_sph_3 = (multi_dot([c26,c35]) + c29)

    
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

        x0 = self.R_vbs_ground
        x1 = self.R_rbs_rocker_1
        x2 = A(self.P_vbs_ground)
        x3 = self.P_rbs_rocker_1
        x4 = A(x3)
        x5 = x2.T
        x6 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        x7 = self.R_rbs_rocker_2
        x8 = self.P_rbs_rocker_2
        x9 = A(x8)
        x10 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        x11 = self.R_rbs_rocker_3
        x12 = self.P_rbs_rocker_3
        x13 = A(x12)
        x14 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        x15 = self.Mbar_rbs_table_jcs_tripod[:,0:1].T
        x16 = self.P_rbs_table
        x17 = A(x16)
        x18 = x17.T
        x19 = self.R_rbs_table
        x20 = (x19 + -1*x0 + multi_dot([x17,self.ubar_rbs_table_jcs_tripod]) + -1*multi_dot([x2,self.ubar_vbs_ground_jcs_tripod]))
        x21 = self.R_rbs_link_1
        x22 = -1*x19
        x23 = self.P_rbs_link_1
        x24 = A(x23)
        x25 = self.R_rbs_link_2
        x26 = self.P_rbs_link_2
        x27 = A(x26)
        x28 = self.R_rbs_link_3
        x29 = self.P_rbs_link_3
        x30 = A(x29)
        x31 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1*x1 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_1]) + -1*multi_dot([x4,self.ubar_rbs_rocker_1_jcs_rev_1])),multi_dot([self.Mbar_vbs_ground_jcs_rev_1[:,0:1].T,x5,x4,x6]),multi_dot([self.Mbar_vbs_ground_jcs_rev_1[:,1:2].T,x5,x4,x6]),(x0 + -1*x7 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_2]) + -1*multi_dot([x9,self.ubar_rbs_rocker_2_jcs_rev_2])),multi_dot([self.Mbar_vbs_ground_jcs_rev_2[:,0:1].T,x5,x9,x10]),multi_dot([self.Mbar_vbs_ground_jcs_rev_2[:,1:2].T,x5,x9,x10]),(x0 + -1*x11 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_3]) + -1*multi_dot([x13,self.ubar_rbs_rocker_3_jcs_rev_3])),multi_dot([self.Mbar_vbs_ground_jcs_rev_3[:,0:1].T,x5,x13,x14]),multi_dot([self.Mbar_vbs_ground_jcs_rev_3[:,1:2].T,x5,x13,x14]),multi_dot([x15,x18,x2,self.Mbar_vbs_ground_jcs_tripod[:,1:2]]),multi_dot([x15,x18,x20]),multi_dot([self.Mbar_rbs_table_jcs_tripod[:,1:2].T,x18,x20]),(x21 + x22 + multi_dot([x24,self.ubar_rbs_link_1_jcs_upper_uni_1]) + -1*multi_dot([x17,self.ubar_rbs_table_jcs_upper_uni_1])),multi_dot([self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1].T,x24.T,x17,self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]]),(x25 + x22 + multi_dot([x27,self.ubar_rbs_link_2_jcs_upper_uni_2]) + -1*multi_dot([x17,self.ubar_rbs_table_jcs_upper_uni_2])),multi_dot([self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1].T,x27.T,x17,self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]]),(x28 + x22 + multi_dot([x30,self.ubar_rbs_link_3_jcs_upper_uni_3]) + -1*multi_dot([x17,self.ubar_rbs_table_jcs_upper_uni_3])),multi_dot([self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1].T,x30.T,x17,self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]]),(x1 + -1*x21 + multi_dot([x4,self.ubar_rbs_rocker_1_jcs_bottom_sph_1]) + -1*multi_dot([x24,self.ubar_rbs_link_1_jcs_bottom_sph_1])),(x7 + -1*x25 + multi_dot([x9,self.ubar_rbs_rocker_2_jcs_bottom_sph_2]) + -1*multi_dot([x27,self.ubar_rbs_link_2_jcs_bottom_sph_2])),(x11 + -1*x28 + multi_dot([x13,self.ubar_rbs_rocker_3_jcs_bottom_sph_3]) + -1*multi_dot([x30,self.ubar_rbs_link_3_jcs_bottom_sph_3])),(x31 + (multi_dot([x16.T,x16]))**(1.0/2.0)),(x31 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x31 + (multi_dot([x26.T,x26]))**(1.0/2.0)),(x31 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x31 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x31 + (multi_dot([x8.T,x8]))**(1.0/2.0)),(x31 + (multi_dot([x12.T,x12]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v0,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v0,v1,v0,v1,v0,v0,v0,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_vbs_ground
        a1 = self.Pd_rbs_rocker_1
        a2 = self.Mbar_vbs_ground_jcs_rev_1[:,0:1]
        a3 = self.P_vbs_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_rocker_1
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_vbs_ground_jcs_rev_1[:,1:2]
        a13 = self.Pd_rbs_rocker_2
        a14 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        a15 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        a16 = B(a13,a15)
        a17 = a15.T
        a18 = self.P_rbs_rocker_2
        a19 = A(a18).T
        a20 = B(a18,a15)
        a21 = self.Mbar_vbs_ground_jcs_rev_2[:,1:2]
        a22 = self.Pd_rbs_rocker_3
        a23 = self.Mbar_vbs_ground_jcs_rev_3[:,0:1]
        a24 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        a25 = B(a22,a24)
        a26 = a24.T
        a27 = self.P_rbs_rocker_3
        a28 = A(a27).T
        a29 = B(a27,a24)
        a30 = self.Mbar_vbs_ground_jcs_rev_3[:,1:2]
        a31 = self.Mbar_rbs_table_jcs_tripod[:,0:1]
        a32 = a31.T
        a33 = self.P_rbs_table
        a34 = A(a33).T
        a35 = self.Mbar_vbs_ground_jcs_tripod[:,1:2]
        a36 = self.Pd_rbs_table
        a37 = B(a36,a31)
        a38 = a36.T
        a39 = B(a33,a31).T
        a40 = self.ubar_rbs_table_jcs_tripod
        a41 = self.ubar_vbs_ground_jcs_tripod
        a42 = (multi_dot([B(a36,a40),a36]) + -1*multi_dot([B(a0,a41),a0]))
        a43 = (self.Rd_rbs_table + -1*self.Rd_vbs_ground + multi_dot([B(a33,a40),a36]) + multi_dot([B(a3,a41),a0]))
        a44 = (self.R_rbs_table.T + -1*self.R_vbs_ground.T + multi_dot([a40.T,a34]) + -1*multi_dot([a41.T,a4]))
        a45 = self.Mbar_rbs_table_jcs_tripod[:,1:2]
        a46 = self.Pd_rbs_link_1
        a47 = self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]
        a48 = self.P_rbs_link_1
        a49 = self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]
        a50 = a46.T
        a51 = self.Pd_rbs_link_2
        a52 = self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]
        a53 = self.P_rbs_link_2
        a54 = self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]
        a55 = a51.T
        a56 = self.Pd_rbs_link_3
        a57 = self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]
        a58 = self.P_rbs_link_3
        a59 = self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]
        a60 = a56.T

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_1),a0]) + -1*multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_rev_1),a1])),(multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + 2*multi_dot([a10,B(a3,a2).T,a11,a1])),(multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + 2*multi_dot([a10,B(a3,a12).T,a11,a1])),(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_2),a0]) + -1*multi_dot([B(a13,self.ubar_rbs_rocker_2_jcs_rev_2),a13])),(multi_dot([a14.T,a4,a16,a13]) + multi_dot([a17,a19,B(a0,a14),a0]) + 2*multi_dot([a10,B(a3,a14).T,a20,a13])),(multi_dot([a21.T,a4,a16,a13]) + multi_dot([a17,a19,B(a0,a21),a0]) + 2*multi_dot([a10,B(a3,a21).T,a20,a13])),(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_3),a0]) + -1*multi_dot([B(a22,self.ubar_rbs_rocker_3_jcs_rev_3),a22])),(multi_dot([a23.T,a4,a25,a22]) + multi_dot([a26,a28,B(a0,a23),a0]) + 2*multi_dot([a10,B(a3,a23).T,a29,a22])),(multi_dot([a30.T,a4,a25,a22]) + multi_dot([a26,a28,B(a0,a30),a0]) + 2*multi_dot([a10,B(a3,a30).T,a29,a22])),(multi_dot([a32,a34,B(a0,a35),a0]) + multi_dot([a35.T,a4,a37,a36]) + 2*multi_dot([a38,a39,B(a3,a35),a0])),(multi_dot([a32,a34,a42]) + 2*multi_dot([a38,a39,a43]) + multi_dot([a44,a37,a36])),(multi_dot([a45.T,a34,a42]) + 2*multi_dot([a38,B(a33,a45).T,a43]) + multi_dot([a44,B(a36,a45),a36])),(multi_dot([B(a46,self.ubar_rbs_link_1_jcs_upper_uni_1),a46]) + -1*multi_dot([B(a36,self.ubar_rbs_table_jcs_upper_uni_1),a36])),(multi_dot([a47.T,A(a48).T,B(a36,a49),a36]) + multi_dot([a49.T,a34,B(a46,a47),a46]) + 2*multi_dot([a50,B(a48,a47).T,B(a33,a49),a36])),(multi_dot([B(a51,self.ubar_rbs_link_2_jcs_upper_uni_2),a51]) + -1*multi_dot([B(a36,self.ubar_rbs_table_jcs_upper_uni_2),a36])),(multi_dot([a52.T,A(a53).T,B(a36,a54),a36]) + multi_dot([a54.T,a34,B(a51,a52),a51]) + 2*multi_dot([a55,B(a53,a52).T,B(a33,a54),a36])),(multi_dot([B(a56,self.ubar_rbs_link_3_jcs_upper_uni_3),a56]) + -1*multi_dot([B(a36,self.ubar_rbs_table_jcs_upper_uni_3),a36])),(multi_dot([a57.T,A(a58).T,B(a36,a59),a36]) + multi_dot([a59.T,a34,B(a56,a57),a56]) + 2*multi_dot([a60,B(a58,a57).T,B(a33,a59),a36])),(multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_bottom_sph_1),a1]) + -1*multi_dot([B(a46,self.ubar_rbs_link_1_jcs_bottom_sph_1),a46])),(multi_dot([B(a13,self.ubar_rbs_rocker_2_jcs_bottom_sph_2),a13]) + -1*multi_dot([B(a51,self.ubar_rbs_link_2_jcs_bottom_sph_2),a51])),(multi_dot([B(a22,self.ubar_rbs_rocker_3_jcs_bottom_sph_3),a22]) + -1*multi_dot([B(a56,self.ubar_rbs_link_3_jcs_bottom_sph_3),a56])),2*(multi_dot([a38,a36]))**(1.0/2.0),2*(multi_dot([a50,a46]))**(1.0/2.0),2*(multi_dot([a55,a51]))**(1.0/2.0),2*(multi_dot([a60,a56]))**(1.0/2.0),2*(multi_dot([a1.T,a1]))**(1.0/2.0),2*(multi_dot([a13.T,a13]))**(1.0/2.0),2*(multi_dot([a22.T,a22]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_vbs_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_rocker_1
        j6 = A(j5).T
        j7 = self.Mbar_vbs_ground_jcs_rev_1[:,0:1]
        j8 = self.Mbar_vbs_ground_jcs_rev_1[:,1:2]
        j9 = -1*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        j13 = j12.T
        j14 = self.P_rbs_rocker_2
        j15 = A(j14).T
        j16 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        j17 = self.Mbar_vbs_ground_jcs_rev_2[:,1:2]
        j18 = B(j14,j12)
        j19 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        j20 = j19.T
        j21 = self.P_rbs_rocker_3
        j22 = A(j21).T
        j23 = self.Mbar_vbs_ground_jcs_rev_3[:,0:1]
        j24 = self.Mbar_vbs_ground_jcs_rev_3[:,1:2]
        j25 = B(j21,j19)
        j26 = self.Mbar_vbs_ground_jcs_tripod[:,1:2]
        j27 = self.P_rbs_table
        j28 = self.Mbar_rbs_table_jcs_tripod[:,0:1]
        j29 = B(j27,j28)
        j30 = j28.T
        j31 = A(j27).T
        j32 = multi_dot([j30,j31])
        j33 = self.ubar_rbs_table_jcs_tripod
        j34 = B(j27,j33)
        j35 = self.ubar_vbs_ground_jcs_tripod
        j36 = (self.R_rbs_table.T + -1*self.R_vbs_ground.T + multi_dot([j33.T,j31]) + -1*multi_dot([j35.T,j10]))
        j37 = self.Mbar_rbs_table_jcs_tripod[:,1:2]
        j38 = j37.T
        j39 = multi_dot([j38,j31])
        j40 = B(j1,j35)
        j41 = self.P_rbs_link_1
        j42 = self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]
        j43 = self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]
        j44 = self.P_rbs_link_2
        j45 = self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]
        j46 = self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]
        j47 = self.P_rbs_link_3
        j48 = self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]
        j49 = self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]

        self.jac_eq_blocks = [j0,B(j1,self.ubar_vbs_ground_jcs_rev_1),j9,-1*B(j5,self.ubar_rbs_rocker_1_jcs_rev_1),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j0,B(j1,self.ubar_vbs_ground_jcs_rev_2),j9,-1*B(j14,self.ubar_rbs_rocker_2_jcs_rev_2),j2,multi_dot([j13,j15,B(j1,j16)]),j2,multi_dot([j16.T,j10,j18]),j2,multi_dot([j13,j15,B(j1,j17)]),j2,multi_dot([j17.T,j10,j18]),j0,B(j1,self.ubar_vbs_ground_jcs_rev_3),j9,-1*B(j21,self.ubar_rbs_rocker_3_jcs_rev_3),j2,multi_dot([j20,j22,B(j1,j23)]),j2,multi_dot([j23.T,j10,j25]),j2,multi_dot([j20,j22,B(j1,j24)]),j2,multi_dot([j24.T,j10,j25]),j2,multi_dot([j30,j31,B(j1,j26)]),j2,multi_dot([j26.T,j10,j29]),-1*j32,-1*multi_dot([j30,j31,j40]),j32,(multi_dot([j30,j31,j34]) + multi_dot([j36,j29])),-1*j39,-1*multi_dot([j38,j31,j40]),j39,(multi_dot([j38,j31,j34]) + multi_dot([j36,B(j27,j37)])),j9,-1*B(j27,self.ubar_rbs_table_jcs_upper_uni_1),j0,B(j41,self.ubar_rbs_link_1_jcs_upper_uni_1),j2,multi_dot([j43.T,A(j41).T,B(j27,j42)]),j2,multi_dot([j42.T,j31,B(j41,j43)]),j9,-1*B(j27,self.ubar_rbs_table_jcs_upper_uni_2),j0,B(j44,self.ubar_rbs_link_2_jcs_upper_uni_2),j2,multi_dot([j46.T,A(j44).T,B(j27,j45)]),j2,multi_dot([j45.T,j31,B(j44,j46)]),j9,-1*B(j27,self.ubar_rbs_table_jcs_upper_uni_3),j0,B(j47,self.ubar_rbs_link_3_jcs_upper_uni_3),j2,multi_dot([j49.T,A(j47).T,B(j27,j48)]),j2,multi_dot([j48.T,j31,B(j47,j49)]),j9,-1*B(j41,self.ubar_rbs_link_1_jcs_bottom_sph_1),j0,B(j5,self.ubar_rbs_rocker_1_jcs_bottom_sph_1),j9,-1*B(j44,self.ubar_rbs_link_2_jcs_bottom_sph_2),j0,B(j14,self.ubar_rbs_rocker_2_jcs_bottom_sph_2),j9,-1*B(j47,self.ubar_rbs_link_3_jcs_bottom_sph_3),j0,B(j21,self.ubar_rbs_rocker_3_jcs_bottom_sph_3),2*j27.T,2*j41.T,2*j44.T,2*j47.T,2*j5.T,2*j14.T,2*j21.T]
  
    
