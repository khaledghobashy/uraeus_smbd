
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
        self.ax1_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_upper_uni_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_cyl_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_cyl_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_cyl_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_cyl_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_bottom_cyl_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_bottom_cyl_3 = np.array([[0], [0], [0]],dtype=np.float64)                       

    
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
        self.nrows = 34
        self.ncols = 2*8
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,28,29,30,31,32,33])                        

    
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
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_table*2,self.rbs_table*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_1*2,self.rbs_link_1*2+1,self.rbs_rocker_1*2,self.rbs_rocker_1*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_2*2,self.rbs_link_2*2+1,self.rbs_rocker_2*2,self.rbs_rocker_2*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_link_3*2,self.rbs_link_3*2+1,self.rbs_rocker_3*2,self.rbs_rocker_3*2+1,self.rbs_table*2+1,self.rbs_link_1*2+1,self.rbs_link_2*2+1,self.rbs_link_3*2+1,self.rbs_rocker_1*2+1,self.rbs_rocker_2*2+1,self.rbs_rocker_3*2+1])

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
        c14 = A(config.P_rbs_link_1).T
        c15 = triad(config.ax1_jcs_upper_uni_1)
        c16 = A(config.P_rbs_table).T
        c17 = config.pt1_jcs_upper_uni_1
        c18 = -1*multi_dot([c14,config.R_rbs_link_1])
        c19 = -1*multi_dot([c16,config.R_rbs_table])
        c20 = A(config.P_rbs_link_2).T
        c21 = triad(config.ax1_jcs_upper_uni_2)
        c22 = config.pt1_jcs_upper_uni_2
        c23 = -1*multi_dot([c20,config.R_rbs_link_2])
        c24 = A(config.P_rbs_link_3).T
        c25 = triad(config.ax1_jcs_upper_uni_3)
        c26 = config.pt1_jcs_upper_uni_3
        c27 = -1*multi_dot([c24,config.R_rbs_link_3])
        c28 = triad(config.ax1_jcs_bottom_cyl_1)
        c29 = config.pt1_jcs_bottom_cyl_1
        c30 = triad(config.ax1_jcs_bottom_cyl_2)
        c31 = config.pt1_jcs_bottom_cyl_2
        c32 = triad(config.ax1_jcs_bottom_cyl_3)
        c33 = config.pt1_jcs_bottom_cyl_3

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
        self.Mbar_rbs_link_1_jcs_upper_uni_1 = multi_dot([c14,c15])
        self.Mbar_rbs_table_jcs_upper_uni_1 = multi_dot([c16,triad(config.ax2_jcs_upper_uni_1,c15[0:3,1:2])])
        self.ubar_rbs_link_1_jcs_upper_uni_1 = (multi_dot([c14,c17]) + c18)
        self.ubar_rbs_table_jcs_upper_uni_1 = (multi_dot([c16,c17]) + c19)
        self.Mbar_rbs_link_2_jcs_upper_uni_2 = multi_dot([c20,c21])
        self.Mbar_rbs_table_jcs_upper_uni_2 = multi_dot([c16,triad(config.ax2_jcs_upper_uni_2,c21[0:3,1:2])])
        self.ubar_rbs_link_2_jcs_upper_uni_2 = (multi_dot([c20,c22]) + c23)
        self.ubar_rbs_table_jcs_upper_uni_2 = (multi_dot([c16,c22]) + c19)
        self.Mbar_rbs_link_3_jcs_upper_uni_3 = multi_dot([c24,c25])
        self.Mbar_rbs_table_jcs_upper_uni_3 = multi_dot([c16,triad(config.ax2_jcs_upper_uni_3,c25[0:3,1:2])])
        self.ubar_rbs_link_3_jcs_upper_uni_3 = (multi_dot([c24,c26]) + c27)
        self.ubar_rbs_table_jcs_upper_uni_3 = (multi_dot([c16,c26]) + c19)
        self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1 = multi_dot([c2,c28])
        self.Mbar_rbs_link_1_jcs_bottom_cyl_1 = multi_dot([c14,c28])
        self.ubar_rbs_rocker_1_jcs_bottom_cyl_1 = (multi_dot([c2,c29]) + c5)
        self.ubar_rbs_link_1_jcs_bottom_cyl_1 = (multi_dot([c14,c29]) + c18)
        self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2 = multi_dot([c7,c30])
        self.Mbar_rbs_link_2_jcs_bottom_cyl_2 = multi_dot([c20,c30])
        self.ubar_rbs_rocker_2_jcs_bottom_cyl_2 = (multi_dot([c7,c31]) + c9)
        self.ubar_rbs_link_2_jcs_bottom_cyl_2 = (multi_dot([c20,c31]) + c23)
        self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3 = multi_dot([c11,c32])
        self.Mbar_rbs_link_3_jcs_bottom_cyl_3 = multi_dot([c24,c32])
        self.ubar_rbs_rocker_3_jcs_bottom_cyl_3 = (multi_dot([c11,c33]) + c13)
        self.ubar_rbs_link_3_jcs_bottom_cyl_3 = (multi_dot([c24,c33]) + c27)

    
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
        x15 = self.R_rbs_link_1
        x16 = -1*self.R_rbs_table
        x17 = self.P_rbs_link_1
        x18 = A(x17)
        x19 = self.P_rbs_table
        x20 = A(x19)
        x21 = self.R_rbs_link_2
        x22 = self.P_rbs_link_2
        x23 = A(x22)
        x24 = self.R_rbs_link_3
        x25 = self.P_rbs_link_3
        x26 = A(x25)
        x27 = self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1[:,0:1].T
        x28 = x4.T
        x29 = self.Mbar_rbs_link_1_jcs_bottom_cyl_1[:,2:3]
        x30 = self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1[:,1:2].T
        x31 = (x1 + -1*x15 + multi_dot([x4,self.ubar_rbs_rocker_1_jcs_bottom_cyl_1]) + -1*multi_dot([x18,self.ubar_rbs_link_1_jcs_bottom_cyl_1]))
        x32 = self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2[:,0:1].T
        x33 = x9.T
        x34 = self.Mbar_rbs_link_2_jcs_bottom_cyl_2[:,2:3]
        x35 = self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2[:,1:2].T
        x36 = (x7 + -1*x21 + multi_dot([x9,self.ubar_rbs_rocker_2_jcs_bottom_cyl_2]) + -1*multi_dot([x23,self.ubar_rbs_link_2_jcs_bottom_cyl_2]))
        x37 = self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3[:,0:1].T
        x38 = x13.T
        x39 = self.Mbar_rbs_link_3_jcs_bottom_cyl_3[:,2:3]
        x40 = self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3[:,1:2].T
        x41 = (x11 + -1*x24 + multi_dot([x13,self.ubar_rbs_rocker_3_jcs_bottom_cyl_3]) + -1*multi_dot([x26,self.ubar_rbs_link_3_jcs_bottom_cyl_3]))
        x42 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1*x1 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_1]) + -1*multi_dot([x4,self.ubar_rbs_rocker_1_jcs_rev_1])),multi_dot([self.Mbar_vbs_ground_jcs_rev_1[:,0:1].T,x5,x4,x6]),multi_dot([self.Mbar_vbs_ground_jcs_rev_1[:,1:2].T,x5,x4,x6]),(x0 + -1*x7 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_2]) + -1*multi_dot([x9,self.ubar_rbs_rocker_2_jcs_rev_2])),multi_dot([self.Mbar_vbs_ground_jcs_rev_2[:,0:1].T,x5,x9,x10]),multi_dot([self.Mbar_vbs_ground_jcs_rev_2[:,1:2].T,x5,x9,x10]),(x0 + -1*x11 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_3]) + -1*multi_dot([x13,self.ubar_rbs_rocker_3_jcs_rev_3])),multi_dot([self.Mbar_vbs_ground_jcs_rev_3[:,0:1].T,x5,x13,x14]),multi_dot([self.Mbar_vbs_ground_jcs_rev_3[:,1:2].T,x5,x13,x14]),(x15 + x16 + multi_dot([x18,self.ubar_rbs_link_1_jcs_upper_uni_1]) + -1*multi_dot([x20,self.ubar_rbs_table_jcs_upper_uni_1])),multi_dot([self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1].T,x18.T,x20,self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]]),(x21 + x16 + multi_dot([x23,self.ubar_rbs_link_2_jcs_upper_uni_2]) + -1*multi_dot([x20,self.ubar_rbs_table_jcs_upper_uni_2])),multi_dot([self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1].T,x23.T,x20,self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]]),(x24 + x16 + multi_dot([x26,self.ubar_rbs_link_3_jcs_upper_uni_3]) + -1*multi_dot([x20,self.ubar_rbs_table_jcs_upper_uni_3])),multi_dot([self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1].T,x26.T,x20,self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]]),multi_dot([x27,x28,x18,x29]),multi_dot([x30,x28,x18,x29]),multi_dot([x27,x28,x31]),multi_dot([x30,x28,x31]),multi_dot([x32,x33,x23,x34]),multi_dot([x35,x33,x23,x34]),multi_dot([x32,x33,x36]),multi_dot([x35,x33,x36]),multi_dot([x37,x38,x26,x39]),multi_dot([x40,x38,x26,x39]),multi_dot([x37,x38,x41]),multi_dot([x40,x38,x41]),(x42 + (multi_dot([x19.T,x19]))**(1.0/2.0)),(x42 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x42 + (multi_dot([x22.T,x22]))**(1.0/2.0)),(x42 + (multi_dot([x25.T,x25]))**(1.0/2.0)),(x42 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x42 + (multi_dot([x8.T,x8]))**(1.0/2.0)),(x42 + (multi_dot([x12.T,x12]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v0,v1,v1,v0,v1,v1,v0,v1,v0,v1,v0,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_vbs_ground
        a1 = self.Pd_rbs_rocker_1
        a2 = self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]
        a3 = a2.T
        a4 = self.P_rbs_rocker_1
        a5 = A(a4).T
        a6 = self.Mbar_vbs_ground_jcs_rev_1[:,0:1]
        a7 = self.P_vbs_ground
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = self.Mbar_vbs_ground_jcs_rev_1[:,1:2]
        a13 = self.Pd_rbs_rocker_2
        a14 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        a15 = a14.T
        a16 = self.P_rbs_rocker_2
        a17 = A(a16).T
        a18 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        a19 = B(a13,a14)
        a20 = B(a16,a14)
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
        a31 = self.Pd_rbs_link_1
        a32 = self.Pd_rbs_table
        a33 = self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]
        a34 = self.P_rbs_link_1
        a35 = A(a34).T
        a36 = self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]
        a37 = self.P_rbs_table
        a38 = A(a37).T
        a39 = a31.T
        a40 = self.Pd_rbs_link_2
        a41 = self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]
        a42 = self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]
        a43 = self.P_rbs_link_2
        a44 = A(a43).T
        a45 = a40.T
        a46 = self.Pd_rbs_link_3
        a47 = self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]
        a48 = self.P_rbs_link_3
        a49 = A(a48).T
        a50 = self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]
        a51 = a46.T
        a52 = self.Mbar_rbs_link_1_jcs_bottom_cyl_1[:,2:3]
        a53 = a52.T
        a54 = self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1[:,0:1]
        a55 = B(a1,a54)
        a56 = a54.T
        a57 = B(a31,a52)
        a58 = a1.T
        a59 = B(a4,a54).T
        a60 = B(a34,a52)
        a61 = self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1[:,1:2]
        a62 = B(a1,a61)
        a63 = a61.T
        a64 = B(a4,a61).T
        a65 = self.ubar_rbs_rocker_1_jcs_bottom_cyl_1
        a66 = self.ubar_rbs_link_1_jcs_bottom_cyl_1
        a67 = (multi_dot([B(a1,a65),a1]) + -1*multi_dot([B(a31,a66),a31]))
        a68 = (self.Rd_rbs_rocker_1 + -1*self.Rd_rbs_link_1 + multi_dot([B(a34,a66),a31]) + multi_dot([B(a4,a65),a1]))
        a69 = (self.R_rbs_rocker_1.T + -1*self.R_rbs_link_1.T + multi_dot([a65.T,a5]) + -1*multi_dot([a66.T,a35]))
        a70 = self.Mbar_rbs_link_2_jcs_bottom_cyl_2[:,2:3]
        a71 = a70.T
        a72 = self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2[:,0:1]
        a73 = B(a13,a72)
        a74 = a72.T
        a75 = B(a40,a70)
        a76 = a13.T
        a77 = B(a16,a72).T
        a78 = B(a43,a70)
        a79 = self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2[:,1:2]
        a80 = B(a13,a79)
        a81 = a79.T
        a82 = B(a16,a79).T
        a83 = self.ubar_rbs_rocker_2_jcs_bottom_cyl_2
        a84 = self.ubar_rbs_link_2_jcs_bottom_cyl_2
        a85 = (multi_dot([B(a13,a83),a13]) + -1*multi_dot([B(a40,a84),a40]))
        a86 = (self.Rd_rbs_rocker_2 + -1*self.Rd_rbs_link_2 + multi_dot([B(a43,a84),a40]) + multi_dot([B(a16,a83),a13]))
        a87 = (self.R_rbs_rocker_2.T + -1*self.R_rbs_link_2.T + multi_dot([a83.T,a17]) + -1*multi_dot([a84.T,a44]))
        a88 = self.Mbar_rbs_link_3_jcs_bottom_cyl_3[:,2:3]
        a89 = a88.T
        a90 = self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3[:,0:1]
        a91 = B(a22,a90)
        a92 = a90.T
        a93 = B(a46,a88)
        a94 = a22.T
        a95 = B(a27,a90).T
        a96 = B(a48,a88)
        a97 = self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3[:,1:2]
        a98 = B(a22,a97)
        a99 = a97.T
        a100 = B(a27,a97).T
        a101 = self.ubar_rbs_rocker_3_jcs_bottom_cyl_3
        a102 = self.ubar_rbs_link_3_jcs_bottom_cyl_3
        a103 = (multi_dot([B(a22,a101),a22]) + -1*multi_dot([B(a46,a102),a46]))
        a104 = (self.Rd_rbs_rocker_3 + -1*self.Rd_rbs_link_3 + multi_dot([B(a48,a102),a46]) + multi_dot([B(a27,a101),a22]))
        a105 = (self.R_rbs_rocker_3.T + -1*self.R_rbs_link_3.T + multi_dot([a101.T,a28]) + -1*multi_dot([a102.T,a49]))

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_1),a0]) + -1*multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_rev_1),a1])),(multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a6).T,a11,a1])),(multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a12).T,a11,a1])),(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_2),a0]) + -1*multi_dot([B(a13,self.ubar_rbs_rocker_2_jcs_rev_2),a13])),(multi_dot([a15,a17,B(a0,a18),a0]) + multi_dot([a18.T,a8,a19,a13]) + 2*multi_dot([a10,B(a7,a18).T,a20,a13])),(multi_dot([a15,a17,B(a0,a21),a0]) + multi_dot([a21.T,a8,a19,a13]) + 2*multi_dot([a10,B(a7,a21).T,a20,a13])),(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_3),a0]) + -1*multi_dot([B(a22,self.ubar_rbs_rocker_3_jcs_rev_3),a22])),(multi_dot([a23.T,a8,a25,a22]) + multi_dot([a26,a28,B(a0,a23),a0]) + 2*multi_dot([a10,B(a7,a23).T,a29,a22])),(multi_dot([a30.T,a8,a25,a22]) + multi_dot([a26,a28,B(a0,a30),a0]) + 2*multi_dot([a10,B(a7,a30).T,a29,a22])),(multi_dot([B(a31,self.ubar_rbs_link_1_jcs_upper_uni_1),a31]) + -1*multi_dot([B(a32,self.ubar_rbs_table_jcs_upper_uni_1),a32])),(multi_dot([a33.T,a35,B(a32,a36),a32]) + multi_dot([a36.T,a38,B(a31,a33),a31]) + 2*multi_dot([a39,B(a34,a33).T,B(a37,a36),a32])),(multi_dot([B(a40,self.ubar_rbs_link_2_jcs_upper_uni_2),a40]) + -1*multi_dot([B(a32,self.ubar_rbs_table_jcs_upper_uni_2),a32])),(multi_dot([a41.T,a38,B(a40,a42),a40]) + multi_dot([a42.T,a44,B(a32,a41),a32]) + 2*multi_dot([a45,B(a43,a42).T,B(a37,a41),a32])),(multi_dot([B(a46,self.ubar_rbs_link_3_jcs_upper_uni_3),a46]) + -1*multi_dot([B(a32,self.ubar_rbs_table_jcs_upper_uni_3),a32])),(multi_dot([a47.T,a49,B(a32,a50),a32]) + multi_dot([a50.T,a38,B(a46,a47),a46]) + 2*multi_dot([a51,B(a48,a47).T,B(a37,a50),a32])),(multi_dot([a53,a35,a55,a1]) + multi_dot([a56,a5,a57,a31]) + 2*multi_dot([a58,a59,a60,a31])),(multi_dot([a53,a35,a62,a1]) + multi_dot([a63,a5,a57,a31]) + 2*multi_dot([a58,a64,a60,a31])),(multi_dot([a56,a5,a67]) + 2*multi_dot([a58,a59,a68]) + multi_dot([a69,a55,a1])),(multi_dot([a63,a5,a67]) + 2*multi_dot([a58,a64,a68]) + multi_dot([a69,a62,a1])),(multi_dot([a71,a44,a73,a13]) + multi_dot([a74,a17,a75,a40]) + 2*multi_dot([a76,a77,a78,a40])),(multi_dot([a71,a44,a80,a13]) + multi_dot([a81,a17,a75,a40]) + 2*multi_dot([a76,a82,a78,a40])),(multi_dot([a74,a17,a85]) + 2*multi_dot([a76,a77,a86]) + multi_dot([a87,a73,a13])),(multi_dot([a81,a17,a85]) + 2*multi_dot([a76,a82,a86]) + multi_dot([a87,a80,a13])),(multi_dot([a89,a49,a91,a22]) + multi_dot([a92,a28,a93,a46]) + 2*multi_dot([a94,a95,a96,a46])),(multi_dot([a89,a49,a98,a22]) + multi_dot([a99,a28,a93,a46]) + 2*multi_dot([a94,a100,a96,a46])),(multi_dot([a92,a28,a103]) + 2*multi_dot([a94,a95,a104]) + multi_dot([a105,a91,a22])),(multi_dot([a99,a28,a103]) + 2*multi_dot([a94,a100,a104]) + multi_dot([a105,a98,a22])),2*(multi_dot([a32.T,a32]))**(1.0/2.0),2*(multi_dot([a39,a31]))**(1.0/2.0),2*(multi_dot([a45,a40]))**(1.0/2.0),2*(multi_dot([a51,a46]))**(1.0/2.0),2*(multi_dot([a58,a1]))**(1.0/2.0),2*(multi_dot([a76,a13]))**(1.0/2.0),2*(multi_dot([a94,a22]))**(1.0/2.0)]

    
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
        j26 = self.P_rbs_link_1
        j27 = self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]
        j28 = self.P_rbs_table
        j29 = A(j28).T
        j30 = self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]
        j31 = A(j26).T
        j32 = self.P_rbs_link_2
        j33 = self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]
        j34 = self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]
        j35 = A(j32).T
        j36 = self.P_rbs_link_3
        j37 = self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]
        j38 = self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]
        j39 = A(j36).T
        j40 = self.Mbar_rbs_link_1_jcs_bottom_cyl_1[:,2:3]
        j41 = j40.T
        j42 = self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1[:,0:1]
        j43 = B(j5,j42)
        j44 = self.Mbar_rbs_rocker_1_jcs_bottom_cyl_1[:,1:2]
        j45 = B(j5,j44)
        j46 = j42.T
        j47 = multi_dot([j46,j6])
        j48 = self.ubar_rbs_rocker_1_jcs_bottom_cyl_1
        j49 = B(j5,j48)
        j50 = self.ubar_rbs_link_1_jcs_bottom_cyl_1
        j51 = (self.R_rbs_rocker_1.T + -1*self.R_rbs_link_1.T + multi_dot([j48.T,j6]) + -1*multi_dot([j50.T,j31]))
        j52 = j44.T
        j53 = multi_dot([j52,j6])
        j54 = B(j26,j40)
        j55 = B(j26,j50)
        j56 = self.Mbar_rbs_link_2_jcs_bottom_cyl_2[:,2:3]
        j57 = j56.T
        j58 = self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2[:,0:1]
        j59 = B(j14,j58)
        j60 = self.Mbar_rbs_rocker_2_jcs_bottom_cyl_2[:,1:2]
        j61 = B(j14,j60)
        j62 = j58.T
        j63 = multi_dot([j62,j15])
        j64 = self.ubar_rbs_rocker_2_jcs_bottom_cyl_2
        j65 = B(j14,j64)
        j66 = self.ubar_rbs_link_2_jcs_bottom_cyl_2
        j67 = (self.R_rbs_rocker_2.T + -1*self.R_rbs_link_2.T + multi_dot([j64.T,j15]) + -1*multi_dot([j66.T,j35]))
        j68 = j60.T
        j69 = multi_dot([j68,j15])
        j70 = B(j32,j56)
        j71 = B(j32,j66)
        j72 = self.Mbar_rbs_link_3_jcs_bottom_cyl_3[:,2:3]
        j73 = j72.T
        j74 = self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3[:,0:1]
        j75 = B(j21,j74)
        j76 = self.Mbar_rbs_rocker_3_jcs_bottom_cyl_3[:,1:2]
        j77 = B(j21,j76)
        j78 = j74.T
        j79 = multi_dot([j78,j22])
        j80 = self.ubar_rbs_rocker_3_jcs_bottom_cyl_3
        j81 = B(j21,j80)
        j82 = self.ubar_rbs_link_3_jcs_bottom_cyl_3
        j83 = (self.R_rbs_rocker_3.T + -1*self.R_rbs_link_3.T + multi_dot([j80.T,j22]) + -1*multi_dot([j82.T,j39]))
        j84 = j76.T
        j85 = multi_dot([j84,j22])
        j86 = B(j36,j72)
        j87 = B(j36,j82)

        self.jac_eq_blocks = [j0,B(j1,self.ubar_vbs_ground_jcs_rev_1),j9,-1*B(j5,self.ubar_rbs_rocker_1_jcs_rev_1),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j0,B(j1,self.ubar_vbs_ground_jcs_rev_2),j9,-1*B(j14,self.ubar_rbs_rocker_2_jcs_rev_2),j2,multi_dot([j13,j15,B(j1,j16)]),j2,multi_dot([j16.T,j10,j18]),j2,multi_dot([j13,j15,B(j1,j17)]),j2,multi_dot([j17.T,j10,j18]),j0,B(j1,self.ubar_vbs_ground_jcs_rev_3),j9,-1*B(j21,self.ubar_rbs_rocker_3_jcs_rev_3),j2,multi_dot([j20,j22,B(j1,j23)]),j2,multi_dot([j23.T,j10,j25]),j2,multi_dot([j20,j22,B(j1,j24)]),j2,multi_dot([j24.T,j10,j25]),j9,-1*B(j28,self.ubar_rbs_table_jcs_upper_uni_1),j0,B(j26,self.ubar_rbs_link_1_jcs_upper_uni_1),j2,multi_dot([j30.T,j31,B(j28,j27)]),j2,multi_dot([j27.T,j29,B(j26,j30)]),j9,-1*B(j28,self.ubar_rbs_table_jcs_upper_uni_2),j0,B(j32,self.ubar_rbs_link_2_jcs_upper_uni_2),j2,multi_dot([j34.T,j35,B(j28,j33)]),j2,multi_dot([j33.T,j29,B(j32,j34)]),j9,-1*B(j28,self.ubar_rbs_table_jcs_upper_uni_3),j0,B(j36,self.ubar_rbs_link_3_jcs_upper_uni_3),j2,multi_dot([j38.T,j39,B(j28,j37)]),j2,multi_dot([j37.T,j29,B(j36,j38)]),j2,multi_dot([j46,j6,j54]),j2,multi_dot([j41,j31,j43]),j2,multi_dot([j52,j6,j54]),j2,multi_dot([j41,j31,j45]),-1*j47,-1*multi_dot([j46,j6,j55]),j47,(multi_dot([j46,j6,j49]) + multi_dot([j51,j43])),-1*j53,-1*multi_dot([j52,j6,j55]),j53,(multi_dot([j52,j6,j49]) + multi_dot([j51,j45])),j2,multi_dot([j62,j15,j70]),j2,multi_dot([j57,j35,j59]),j2,multi_dot([j68,j15,j70]),j2,multi_dot([j57,j35,j61]),-1*j63,-1*multi_dot([j62,j15,j71]),j63,(multi_dot([j62,j15,j65]) + multi_dot([j67,j59])),-1*j69,-1*multi_dot([j68,j15,j71]),j69,(multi_dot([j68,j15,j65]) + multi_dot([j67,j61])),j2,multi_dot([j78,j22,j86]),j2,multi_dot([j73,j39,j75]),j2,multi_dot([j84,j22,j86]),j2,multi_dot([j73,j39,j77]),-1*j79,-1*multi_dot([j78,j22,j87]),j79,(multi_dot([j78,j22,j81]) + multi_dot([j83,j75])),-1*j85,-1*multi_dot([j84,j22,j87]),j85,(multi_dot([j84,j22,j81]) + multi_dot([j83,j77])),2*j28.T,2*j26.T,2*j32.T,2*j36.T,2*j5.T,2*j14.T,2*j21.T]
  
    
