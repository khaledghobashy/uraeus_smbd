
import numpy as np
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, G, E, triad, skew_matrix as skew



class topology(object):

    def __init__(self,prefix=''):
        self.t = 0.0
        self.prefix = (prefix if prefix=='' else prefix+'.')
        self.config = None

        self.n  = 49
        self.nc = 46
        self.nrows = 28
        self.ncols = 2*8
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_vbs_ground_jcs_rev_1', 'T_vbs_ground_jcs_rev_1', 'F_vbs_ground_jcs_rev_2', 'T_vbs_ground_jcs_rev_2', 'F_vbs_ground_jcs_rev_3', 'T_vbs_ground_jcs_rev_3', 'F_rbs_table_jcs_tripod', 'T_rbs_table_jcs_tripod', 'F_rbs_link_1_jcs_upper_uni_1', 'T_rbs_link_1_jcs_upper_uni_1', 'F_rbs_link_2_jcs_upper_uni_2', 'T_rbs_link_2_jcs_upper_uni_2', 'F_rbs_link_3_jcs_upper_uni_3', 'T_rbs_link_3_jcs_upper_uni_3', 'F_rbs_rocker_1_jcs_bottom_sph_1', 'T_rbs_rocker_1_jcs_bottom_sph_1', 'F_rbs_rocker_2_jcs_bottom_sph_2', 'T_rbs_rocker_2_jcs_bottom_sph_2', 'F_rbs_rocker_3_jcs_bottom_sph_3', 'T_rbs_rocker_3_jcs_bottom_sph_3']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_table = indicies_map[p+'rbs_table']
        self.rbs_link_1 = indicies_map[p+'rbs_link_1']
        self.rbs_link_2 = indicies_map[p+'rbs_link_2']
        self.rbs_link_3 = indicies_map[p+'rbs_link_3']
        self.rbs_rocker_1 = indicies_map[p+'rbs_rocker_1']
        self.rbs_rocker_2 = indicies_map[p+'rbs_rocker_2']
        self.rbs_rocker_3 = indicies_map[p+'rbs_rocker_3']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 22, 23, 24, 25, 26, 27])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_table*2, self.rbs_table*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_table*2, self.rbs_table*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_table*2, self.rbs_table*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_link_1*2, self.rbs_link_1*2+1, self.rbs_rocker_1*2, self.rbs_rocker_1*2+1, self.rbs_link_2*2, self.rbs_link_2*2+1, self.rbs_rocker_2*2, self.rbs_rocker_2*2+1, self.rbs_link_3*2, self.rbs_link_3*2+1, self.rbs_rocker_3*2, self.rbs_rocker_3*2+1, self.rbs_table*2+1, self.rbs_link_1*2+1, self.rbs_link_2*2+1, self.rbs_link_3*2+1, self.rbs_rocker_1*2+1, self.rbs_rocker_2*2+1, self.rbs_rocker_3*2+1]

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

        self.Mbar_vbs_ground_jcs_rev_1 = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_1)])
        self.Mbar_rbs_rocker_1_jcs_rev_1 = multi_dot([A(config.P_rbs_rocker_1).T,triad(config.ax1_jcs_rev_1)])
        self.ubar_vbs_ground_jcs_rev_1 = (multi_dot([A(config.P_vbs_ground).T,config.pt1_jcs_rev_1]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbs_rocker_1_jcs_rev_1 = (multi_dot([A(config.P_rbs_rocker_1).T,config.pt1_jcs_rev_1]) + -1*multi_dot([A(config.P_rbs_rocker_1).T,config.R_rbs_rocker_1]))
        self.Mbar_vbs_ground_jcs_rev_2 = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_2)])
        self.Mbar_rbs_rocker_2_jcs_rev_2 = multi_dot([A(config.P_rbs_rocker_2).T,triad(config.ax1_jcs_rev_2)])
        self.ubar_vbs_ground_jcs_rev_2 = (multi_dot([A(config.P_vbs_ground).T,config.pt1_jcs_rev_2]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbs_rocker_2_jcs_rev_2 = (multi_dot([A(config.P_rbs_rocker_2).T,config.pt1_jcs_rev_2]) + -1*multi_dot([A(config.P_rbs_rocker_2).T,config.R_rbs_rocker_2]))
        self.Mbar_vbs_ground_jcs_rev_3 = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_3)])
        self.Mbar_rbs_rocker_3_jcs_rev_3 = multi_dot([A(config.P_rbs_rocker_3).T,triad(config.ax1_jcs_rev_3)])
        self.ubar_vbs_ground_jcs_rev_3 = (multi_dot([A(config.P_vbs_ground).T,config.pt1_jcs_rev_3]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_rbs_rocker_3_jcs_rev_3 = (multi_dot([A(config.P_rbs_rocker_3).T,config.pt1_jcs_rev_3]) + -1*multi_dot([A(config.P_rbs_rocker_3).T,config.R_rbs_rocker_3]))
        self.Mbar_rbs_table_jcs_tripod = multi_dot([A(config.P_rbs_table).T,triad(config.ax1_jcs_tripod)])
        self.Mbar_vbs_ground_jcs_tripod = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_tripod)])
        self.ubar_rbs_table_jcs_tripod = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_tripod]) + -1*multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.ubar_vbs_ground_jcs_tripod = (multi_dot([A(config.P_vbs_ground).T,config.pt1_jcs_tripod]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.Mbar_rbs_link_1_jcs_upper_uni_1 = multi_dot([A(config.P_rbs_link_1).T,triad(config.ax1_jcs_upper_uni_1)])
        self.Mbar_rbs_table_jcs_upper_uni_1 = multi_dot([A(config.P_rbs_table).T,triad(config.ax2_jcs_upper_uni_1,triad(config.ax1_jcs_upper_uni_1)[0:3,1:2])])
        self.ubar_rbs_link_1_jcs_upper_uni_1 = (multi_dot([A(config.P_rbs_link_1).T,config.pt1_jcs_upper_uni_1]) + -1*multi_dot([A(config.P_rbs_link_1).T,config.R_rbs_link_1]))
        self.ubar_rbs_table_jcs_upper_uni_1 = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_upper_uni_1]) + -1*multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.Mbar_rbs_link_2_jcs_upper_uni_2 = multi_dot([A(config.P_rbs_link_2).T,triad(config.ax1_jcs_upper_uni_2)])
        self.Mbar_rbs_table_jcs_upper_uni_2 = multi_dot([A(config.P_rbs_table).T,triad(config.ax2_jcs_upper_uni_2,triad(config.ax1_jcs_upper_uni_2)[0:3,1:2])])
        self.ubar_rbs_link_2_jcs_upper_uni_2 = (multi_dot([A(config.P_rbs_link_2).T,config.pt1_jcs_upper_uni_2]) + -1*multi_dot([A(config.P_rbs_link_2).T,config.R_rbs_link_2]))
        self.ubar_rbs_table_jcs_upper_uni_2 = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_upper_uni_2]) + -1*multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.Mbar_rbs_link_3_jcs_upper_uni_3 = multi_dot([A(config.P_rbs_link_3).T,triad(config.ax1_jcs_upper_uni_3)])
        self.Mbar_rbs_table_jcs_upper_uni_3 = multi_dot([A(config.P_rbs_table).T,triad(config.ax2_jcs_upper_uni_3,triad(config.ax1_jcs_upper_uni_3)[0:3,1:2])])
        self.ubar_rbs_link_3_jcs_upper_uni_3 = (multi_dot([A(config.P_rbs_link_3).T,config.pt1_jcs_upper_uni_3]) + -1*multi_dot([A(config.P_rbs_link_3).T,config.R_rbs_link_3]))
        self.ubar_rbs_table_jcs_upper_uni_3 = (multi_dot([A(config.P_rbs_table).T,config.pt1_jcs_upper_uni_3]) + -1*multi_dot([A(config.P_rbs_table).T,config.R_rbs_table]))
        self.Mbar_rbs_rocker_1_jcs_bottom_sph_1 = multi_dot([A(config.P_rbs_rocker_1).T,triad(config.ax1_jcs_bottom_sph_1)])
        self.Mbar_rbs_link_1_jcs_bottom_sph_1 = multi_dot([A(config.P_rbs_link_1).T,triad(config.ax1_jcs_bottom_sph_1)])
        self.ubar_rbs_rocker_1_jcs_bottom_sph_1 = (multi_dot([A(config.P_rbs_rocker_1).T,config.pt1_jcs_bottom_sph_1]) + -1*multi_dot([A(config.P_rbs_rocker_1).T,config.R_rbs_rocker_1]))
        self.ubar_rbs_link_1_jcs_bottom_sph_1 = (multi_dot([A(config.P_rbs_link_1).T,config.pt1_jcs_bottom_sph_1]) + -1*multi_dot([A(config.P_rbs_link_1).T,config.R_rbs_link_1]))
        self.Mbar_rbs_rocker_2_jcs_bottom_sph_2 = multi_dot([A(config.P_rbs_rocker_2).T,triad(config.ax1_jcs_bottom_sph_2)])
        self.Mbar_rbs_link_2_jcs_bottom_sph_2 = multi_dot([A(config.P_rbs_link_2).T,triad(config.ax1_jcs_bottom_sph_2)])
        self.ubar_rbs_rocker_2_jcs_bottom_sph_2 = (multi_dot([A(config.P_rbs_rocker_2).T,config.pt1_jcs_bottom_sph_2]) + -1*multi_dot([A(config.P_rbs_rocker_2).T,config.R_rbs_rocker_2]))
        self.ubar_rbs_link_2_jcs_bottom_sph_2 = (multi_dot([A(config.P_rbs_link_2).T,config.pt1_jcs_bottom_sph_2]) + -1*multi_dot([A(config.P_rbs_link_2).T,config.R_rbs_link_2]))
        self.Mbar_rbs_rocker_3_jcs_bottom_sph_3 = multi_dot([A(config.P_rbs_rocker_3).T,triad(config.ax1_jcs_bottom_sph_3)])
        self.Mbar_rbs_link_3_jcs_bottom_sph_3 = multi_dot([A(config.P_rbs_link_3).T,triad(config.ax1_jcs_bottom_sph_3)])
        self.ubar_rbs_rocker_3_jcs_bottom_sph_3 = (multi_dot([A(config.P_rbs_rocker_3).T,config.pt1_jcs_bottom_sph_3]) + -1*multi_dot([A(config.P_rbs_rocker_3).T,config.R_rbs_rocker_3]))
        self.ubar_rbs_link_3_jcs_bottom_sph_3 = (multi_dot([A(config.P_rbs_link_3).T,config.pt1_jcs_bottom_sph_3]) + -1*multi_dot([A(config.P_rbs_link_3).T,config.R_rbs_link_3]))

    
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

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_table = qdd[0:3,0:1]
        self.Pdd_rbs_table = qdd[3:7,0:1]
        self.Rdd_rbs_link_1 = qdd[7:10,0:1]
        self.Pdd_rbs_link_1 = qdd[10:14,0:1]
        self.Rdd_rbs_link_2 = qdd[14:17,0:1]
        self.Pdd_rbs_link_2 = qdd[17:21,0:1]
        self.Rdd_rbs_link_3 = qdd[21:24,0:1]
        self.Pdd_rbs_link_3 = qdd[24:28,0:1]
        self.Rdd_rbs_rocker_1 = qdd[28:31,0:1]
        self.Pdd_rbs_rocker_1 = qdd[31:35,0:1]
        self.Rdd_rbs_rocker_2 = qdd[35:38,0:1]
        self.Pdd_rbs_rocker_2 = qdd[38:42,0:1]
        self.Rdd_rbs_rocker_3 = qdd[42:45,0:1]
        self.Pdd_rbs_rocker_3 = qdd[45:49,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_rev_1 = Lambda[0:5,0:1]
        self.L_jcs_rev_2 = Lambda[5:10,0:1]
        self.L_jcs_rev_3 = Lambda[10:15,0:1]
        self.L_jcs_tripod = Lambda[15:18,0:1]
        self.L_jcs_upper_uni_1 = Lambda[18:22,0:1]
        self.L_jcs_upper_uni_2 = Lambda[22:26,0:1]
        self.L_jcs_upper_uni_3 = Lambda[26:30,0:1]
        self.L_jcs_bottom_sph_1 = Lambda[30:33,0:1]
        self.L_jcs_bottom_sph_2 = Lambda[33:36,0:1]
        self.L_jcs_bottom_sph_3 = Lambda[36:39,0:1]

    
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

        self.pos_eq_blocks = [(x0 + -1*x1 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_1]) + -1*multi_dot([x4,self.ubar_rbs_rocker_1_jcs_rev_1])),
        multi_dot([self.Mbar_vbs_ground_jcs_rev_1[:,0:1].T,x5,x4,x6]),
        multi_dot([self.Mbar_vbs_ground_jcs_rev_1[:,1:2].T,x5,x4,x6]),
        (x0 + -1*x7 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_2]) + -1*multi_dot([x9,self.ubar_rbs_rocker_2_jcs_rev_2])),
        multi_dot([self.Mbar_vbs_ground_jcs_rev_2[:,0:1].T,x5,x9,x10]),
        multi_dot([self.Mbar_vbs_ground_jcs_rev_2[:,1:2].T,x5,x9,x10]),
        (x0 + -1*x11 + multi_dot([x2,self.ubar_vbs_ground_jcs_rev_3]) + -1*multi_dot([x13,self.ubar_rbs_rocker_3_jcs_rev_3])),
        multi_dot([self.Mbar_vbs_ground_jcs_rev_3[:,0:1].T,x5,x13,x14]),
        multi_dot([self.Mbar_vbs_ground_jcs_rev_3[:,1:2].T,x5,x13,x14]),
        multi_dot([x15,x18,x2,self.Mbar_vbs_ground_jcs_tripod[:,1:2]]),
        multi_dot([x15,x18,x20]),
        multi_dot([self.Mbar_rbs_table_jcs_tripod[:,1:2].T,x18,x20]),
        (x21 + x22 + multi_dot([x24,self.ubar_rbs_link_1_jcs_upper_uni_1]) + -1*multi_dot([x17,self.ubar_rbs_table_jcs_upper_uni_1])),
        multi_dot([self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1].T,x24.T,x17,self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]]),
        (x25 + x22 + multi_dot([x27,self.ubar_rbs_link_2_jcs_upper_uni_2]) + -1*multi_dot([x17,self.ubar_rbs_table_jcs_upper_uni_2])),
        multi_dot([self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1].T,x27.T,x17,self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]]),
        (x28 + x22 + multi_dot([x30,self.ubar_rbs_link_3_jcs_upper_uni_3]) + -1*multi_dot([x17,self.ubar_rbs_table_jcs_upper_uni_3])),
        multi_dot([self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1].T,x30.T,x17,self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]]),
        (x1 + -1*x21 + multi_dot([x4,self.ubar_rbs_rocker_1_jcs_bottom_sph_1]) + -1*multi_dot([x24,self.ubar_rbs_link_1_jcs_bottom_sph_1])),
        (x7 + -1*x25 + multi_dot([x9,self.ubar_rbs_rocker_2_jcs_bottom_sph_2]) + -1*multi_dot([x27,self.ubar_rbs_link_2_jcs_bottom_sph_2])),
        (x11 + -1*x28 + multi_dot([x13,self.ubar_rbs_rocker_3_jcs_bottom_sph_3]) + -1*multi_dot([x30,self.ubar_rbs_link_3_jcs_bottom_sph_3])),
        (x31 + (multi_dot([x16.T,x16]))**(1.0/2.0)),
        (x31 + (multi_dot([x23.T,x23]))**(1.0/2.0)),
        (x31 + (multi_dot([x26.T,x26]))**(1.0/2.0)),
        (x31 + (multi_dot([x29.T,x29]))**(1.0/2.0)),
        (x31 + (multi_dot([x3.T,x3]))**(1.0/2.0)),
        (x31 + (multi_dot([x8.T,x8]))**(1.0/2.0)),
        (x31 + (multi_dot([x12.T,x12]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v0,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1]

    
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
        a14 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        a15 = self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]
        a16 = B(a13,a15)
        a17 = a15.T
        a18 = self.P_rbs_rocker_2
        a19 = A(a18).T
        a20 = B(a18,a15)
        a21 = self.Mbar_vbs_ground_jcs_rev_2[:,1:2]
        a22 = self.Pd_rbs_rocker_3
        a23 = self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]
        a24 = a23.T
        a25 = self.P_rbs_rocker_3
        a26 = A(a25).T
        a27 = self.Mbar_vbs_ground_jcs_rev_3[:,0:1]
        a28 = B(a22,a23)
        a29 = B(a25,a23)
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
        a43 = (self.Rd_rbs_table + -1*self.Rd_vbs_ground + multi_dot([B(a33,a40),a36]) + -1*multi_dot([B(a7,a41),a0]))
        a44 = (self.R_rbs_table.T + -1*self.R_vbs_ground.T + multi_dot([a40.T,a34]) + -1*multi_dot([a41.T,a8]))
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

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_1),a0]) + -1*multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_rev_1),a1])),
        (multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a6).T,a11,a1])),
        (multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a12).T,a11,a1])),
        (multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_2),a0]) + -1*multi_dot([B(a13,self.ubar_rbs_rocker_2_jcs_rev_2),a13])),
        (multi_dot([a14.T,a8,a16,a13]) + multi_dot([a17,a19,B(a0,a14),a0]) + 2*multi_dot([a10,B(a7,a14).T,a20,a13])),
        (multi_dot([a21.T,a8,a16,a13]) + multi_dot([a17,a19,B(a0,a21),a0]) + 2*multi_dot([a10,B(a7,a21).T,a20,a13])),
        (multi_dot([B(a0,self.ubar_vbs_ground_jcs_rev_3),a0]) + -1*multi_dot([B(a22,self.ubar_rbs_rocker_3_jcs_rev_3),a22])),
        (multi_dot([a24,a26,B(a0,a27),a0]) + multi_dot([a27.T,a8,a28,a22]) + 2*multi_dot([a10,B(a7,a27).T,a29,a22])),
        (multi_dot([a24,a26,B(a0,a30),a0]) + multi_dot([a30.T,a8,a28,a22]) + 2*multi_dot([a10,B(a7,a30).T,a29,a22])),
        (multi_dot([a32,a34,B(a0,a35),a0]) + multi_dot([a35.T,a8,a37,a36]) + 2*multi_dot([a38,a39,B(a7,a35),a0])),
        (multi_dot([a32,a34,a42]) + 2*multi_dot([a38,a39,a43]) + multi_dot([a44,a37,a36])),
        (multi_dot([a45.T,a34,a42]) + 2*multi_dot([a38,B(a33,a45).T,a43]) + multi_dot([a44,B(a36,a45),a36])),
        (multi_dot([B(a46,self.ubar_rbs_link_1_jcs_upper_uni_1),a46]) + -1*multi_dot([B(a36,self.ubar_rbs_table_jcs_upper_uni_1),a36])),
        (multi_dot([a47.T,A(a48).T,B(a36,a49),a36]) + multi_dot([a49.T,a34,B(a46,a47),a46]) + 2*multi_dot([a50,B(a48,a47).T,B(a33,a49),a36])),
        (multi_dot([B(a51,self.ubar_rbs_link_2_jcs_upper_uni_2),a51]) + -1*multi_dot([B(a36,self.ubar_rbs_table_jcs_upper_uni_2),a36])),
        (multi_dot([a52.T,A(a53).T,B(a36,a54),a36]) + multi_dot([a54.T,a34,B(a51,a52),a51]) + 2*multi_dot([a55,B(a53,a52).T,B(a33,a54),a36])),
        (multi_dot([B(a56,self.ubar_rbs_link_3_jcs_upper_uni_3),a56]) + -1*multi_dot([B(a36,self.ubar_rbs_table_jcs_upper_uni_3),a36])),
        (multi_dot([a57.T,A(a58).T,B(a36,a59),a36]) + multi_dot([a59.T,a34,B(a56,a57),a56]) + 2*multi_dot([a60,B(a58,a57).T,B(a33,a59),a36])),
        (multi_dot([B(a1,self.ubar_rbs_rocker_1_jcs_bottom_sph_1),a1]) + -1*multi_dot([B(a46,self.ubar_rbs_link_1_jcs_bottom_sph_1),a46])),
        (multi_dot([B(a13,self.ubar_rbs_rocker_2_jcs_bottom_sph_2),a13]) + -1*multi_dot([B(a51,self.ubar_rbs_link_2_jcs_bottom_sph_2),a51])),
        (multi_dot([B(a22,self.ubar_rbs_rocker_3_jcs_bottom_sph_3),a22]) + -1*multi_dot([B(a56,self.ubar_rbs_link_3_jcs_bottom_sph_3),a56])),
        2*(multi_dot([a38,a36]))**(1.0/2.0),
        2*(multi_dot([a50,a46]))**(1.0/2.0),
        2*(multi_dot([a55,a51]))**(1.0/2.0),
        2*(multi_dot([a60,a56]))**(1.0/2.0),
        2*(multi_dot([a1.T,a1]))**(1.0/2.0),
        2*(multi_dot([a13.T,a13]))**(1.0/2.0),
        2*(multi_dot([a22.T,a22]))**(1.0/2.0)]

    
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

        self.jac_eq_blocks = [j0,
        B(j1,self.ubar_vbs_ground_jcs_rev_1),
        j9,
        -1*B(j5,self.ubar_rbs_rocker_1_jcs_rev_1),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j0,
        B(j1,self.ubar_vbs_ground_jcs_rev_2),
        j9,
        -1*B(j14,self.ubar_rbs_rocker_2_jcs_rev_2),
        j2,
        multi_dot([j13,j15,B(j1,j16)]),
        j2,
        multi_dot([j16.T,j10,j18]),
        j2,
        multi_dot([j13,j15,B(j1,j17)]),
        j2,
        multi_dot([j17.T,j10,j18]),
        j0,
        B(j1,self.ubar_vbs_ground_jcs_rev_3),
        j9,
        -1*B(j21,self.ubar_rbs_rocker_3_jcs_rev_3),
        j2,
        multi_dot([j20,j22,B(j1,j23)]),
        j2,
        multi_dot([j23.T,j10,j25]),
        j2,
        multi_dot([j20,j22,B(j1,j24)]),
        j2,
        multi_dot([j24.T,j10,j25]),
        j2,
        multi_dot([j30,j31,B(j1,j26)]),
        j2,
        multi_dot([j26.T,j10,j29]),
        -1*j32,
        -1*multi_dot([j30,j31,j40]),
        j32,
        (multi_dot([j30,j31,j34]) + multi_dot([j36,j29])),
        -1*j39,
        -1*multi_dot([j38,j31,j40]),
        j39,
        (multi_dot([j38,j31,j34]) + multi_dot([j36,B(j27,j37)])),
        j9,
        -1*B(j27,self.ubar_rbs_table_jcs_upper_uni_1),
        j0,
        B(j41,self.ubar_rbs_link_1_jcs_upper_uni_1),
        j2,
        multi_dot([j43.T,A(j41).T,B(j27,j42)]),
        j2,
        multi_dot([j42.T,j31,B(j41,j43)]),
        j9,
        -1*B(j27,self.ubar_rbs_table_jcs_upper_uni_2),
        j0,
        B(j44,self.ubar_rbs_link_2_jcs_upper_uni_2),
        j2,
        multi_dot([j46.T,A(j44).T,B(j27,j45)]),
        j2,
        multi_dot([j45.T,j31,B(j44,j46)]),
        j9,
        -1*B(j27,self.ubar_rbs_table_jcs_upper_uni_3),
        j0,
        B(j47,self.ubar_rbs_link_3_jcs_upper_uni_3),
        j2,
        multi_dot([j49.T,A(j47).T,B(j27,j48)]),
        j2,
        multi_dot([j48.T,j31,B(j47,j49)]),
        j9,
        -1*B(j41,self.ubar_rbs_link_1_jcs_bottom_sph_1),
        j0,
        B(j5,self.ubar_rbs_rocker_1_jcs_bottom_sph_1),
        j9,
        -1*B(j44,self.ubar_rbs_link_2_jcs_bottom_sph_2),
        j0,
        B(j14,self.ubar_rbs_rocker_2_jcs_bottom_sph_2),
        j9,
        -1*B(j47,self.ubar_rbs_link_3_jcs_bottom_sph_3),
        j0,
        B(j21,self.ubar_rbs_rocker_3_jcs_bottom_sph_3),
        2*j27.T,
        2*j41.T,
        2*j44.T,
        2*j47.T,
        2*j5.T,
        2*j14.T,
        2*j21.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3,dtype=np.float64)
        m1 = G(self.P_rbs_table)
        m2 = G(self.P_rbs_link_1)
        m3 = G(self.P_rbs_link_2)
        m4 = G(self.P_rbs_link_3)
        m5 = G(self.P_rbs_rocker_1)
        m6 = G(self.P_rbs_rocker_2)
        m7 = G(self.P_rbs_rocker_3)

        self.mass_eq_blocks = [config.m_rbs_table*m0,
        4*multi_dot([m1.T,config.Jbar_rbs_table,m1]),
        config.m_rbs_link_1*m0,
        4*multi_dot([m2.T,config.Jbar_rbs_link_1,m2]),
        config.m_rbs_link_2*m0,
        4*multi_dot([m3.T,config.Jbar_rbs_link_2,m3]),
        config.m_rbs_link_3*m0,
        4*multi_dot([m4.T,config.Jbar_rbs_link_3,m4]),
        config.m_rbs_rocker_1*m0,
        4*multi_dot([m5.T,config.Jbar_rbs_rocker_1,m5]),
        config.m_rbs_rocker_2*m0,
        4*multi_dot([m6.T,config.Jbar_rbs_rocker_2,m6]),
        config.m_rbs_rocker_3*m0,
        4*multi_dot([m7.T,config.Jbar_rbs_rocker_3,m7])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_table)
        f1 = G(self.Pd_rbs_link_1)
        f2 = G(self.Pd_rbs_link_2)
        f3 = G(self.Pd_rbs_link_3)
        f4 = G(self.Pd_rbs_rocker_1)
        f5 = G(self.Pd_rbs_rocker_2)
        f6 = G(self.Pd_rbs_rocker_3)

        self.frc_eq_blocks = [self.F_rbs_table_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_table,f0,self.P_rbs_table]),
        self.F_rbs_link_1_gravity,
        8*multi_dot([f1.T,config.Jbar_rbs_link_1,f1,self.P_rbs_link_1]),
        self.F_rbs_link_2_gravity,
        8*multi_dot([f2.T,config.Jbar_rbs_link_2,f2,self.P_rbs_link_2]),
        self.F_rbs_link_3_gravity,
        8*multi_dot([f3.T,config.Jbar_rbs_link_3,f3,self.P_rbs_link_3]),
        self.F_rbs_rocker_1_gravity,
        8*multi_dot([f4.T,config.Jbar_rbs_rocker_1,f4,self.P_rbs_rocker_1]),
        self.F_rbs_rocker_2_gravity,
        8*multi_dot([f5.T,config.Jbar_rbs_rocker_2,f5,self.P_rbs_rocker_2]),
        self.F_rbs_rocker_3_gravity,
        8*multi_dot([f6.T,config.Jbar_rbs_rocker_3,f6,self.P_rbs_rocker_3])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_vbs_ground_jcs_rev_1 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_vbs_ground,self.ubar_vbs_ground_jcs_rev_1).T,multi_dot([B(self.P_vbs_ground,self.Mbar_vbs_ground_jcs_rev_1[:,0:1]).T,A(self.P_rbs_rocker_1),self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]]),multi_dot([B(self.P_vbs_ground,self.Mbar_vbs_ground_jcs_rev_1[:,1:2]).T,A(self.P_rbs_rocker_1),self.Mbar_rbs_rocker_1_jcs_rev_1[:,2:3]])]]),self.L_jcs_rev_1])
        self.F_vbs_ground_jcs_rev_1 = Q_vbs_ground_jcs_rev_1[0:3,0:1]
        Te_vbs_ground_jcs_rev_1 = Q_vbs_ground_jcs_rev_1[3:7,0:1]
        self.T_vbs_ground_jcs_rev_1 = (-1*multi_dot([skew(multi_dot([A(self.P_vbs_ground),self.ubar_vbs_ground_jcs_rev_1])),self.F_vbs_ground_jcs_rev_1]) + 0.5*multi_dot([E(self.P_vbs_ground),Te_vbs_ground_jcs_rev_1]))
        Q_vbs_ground_jcs_rev_2 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_vbs_ground,self.ubar_vbs_ground_jcs_rev_2).T,multi_dot([B(self.P_vbs_ground,self.Mbar_vbs_ground_jcs_rev_2[:,0:1]).T,A(self.P_rbs_rocker_2),self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]]),multi_dot([B(self.P_vbs_ground,self.Mbar_vbs_ground_jcs_rev_2[:,1:2]).T,A(self.P_rbs_rocker_2),self.Mbar_rbs_rocker_2_jcs_rev_2[:,2:3]])]]),self.L_jcs_rev_2])
        self.F_vbs_ground_jcs_rev_2 = Q_vbs_ground_jcs_rev_2[0:3,0:1]
        Te_vbs_ground_jcs_rev_2 = Q_vbs_ground_jcs_rev_2[3:7,0:1]
        self.T_vbs_ground_jcs_rev_2 = (-1*multi_dot([skew(multi_dot([A(self.P_vbs_ground),self.ubar_vbs_ground_jcs_rev_2])),self.F_vbs_ground_jcs_rev_2]) + 0.5*multi_dot([E(self.P_vbs_ground),Te_vbs_ground_jcs_rev_2]))
        Q_vbs_ground_jcs_rev_3 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_vbs_ground,self.ubar_vbs_ground_jcs_rev_3).T,multi_dot([B(self.P_vbs_ground,self.Mbar_vbs_ground_jcs_rev_3[:,0:1]).T,A(self.P_rbs_rocker_3),self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]]),multi_dot([B(self.P_vbs_ground,self.Mbar_vbs_ground_jcs_rev_3[:,1:2]).T,A(self.P_rbs_rocker_3),self.Mbar_rbs_rocker_3_jcs_rev_3[:,2:3]])]]),self.L_jcs_rev_3])
        self.F_vbs_ground_jcs_rev_3 = Q_vbs_ground_jcs_rev_3[0:3,0:1]
        Te_vbs_ground_jcs_rev_3 = Q_vbs_ground_jcs_rev_3[3:7,0:1]
        self.T_vbs_ground_jcs_rev_3 = (-1*multi_dot([skew(multi_dot([A(self.P_vbs_ground),self.ubar_vbs_ground_jcs_rev_3])),self.F_vbs_ground_jcs_rev_3]) + 0.5*multi_dot([E(self.P_vbs_ground),Te_vbs_ground_jcs_rev_3]))
        Q_rbs_table_jcs_tripod = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbs_table),self.Mbar_rbs_table_jcs_tripod[:,0:1]]),multi_dot([A(self.P_rbs_table),self.Mbar_rbs_table_jcs_tripod[:,1:2]])],[multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_tripod[:,0:1]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_tripod[:,1:2]]),(multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_tripod[:,0:1]).T,(-1*self.R_vbs_ground + multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_tripod]) + -1*multi_dot([A(self.P_vbs_ground),self.ubar_vbs_ground_jcs_tripod]) + self.R_rbs_table)]) + multi_dot([B(self.P_rbs_table,self.ubar_rbs_table_jcs_tripod).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_tripod[:,0:1]])),(multi_dot([B(self.P_rbs_table,self.Mbar_rbs_table_jcs_tripod[:,1:2]).T,(-1*self.R_vbs_ground + multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_tripod]) + -1*multi_dot([A(self.P_vbs_ground),self.ubar_vbs_ground_jcs_tripod]) + self.R_rbs_table)]) + multi_dot([B(self.P_rbs_table,self.ubar_rbs_table_jcs_tripod).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_tripod[:,1:2]]))]]),self.L_jcs_tripod])
        self.F_rbs_table_jcs_tripod = Q_rbs_table_jcs_tripod[0:3,0:1]
        Te_rbs_table_jcs_tripod = Q_rbs_table_jcs_tripod[3:7,0:1]
        self.T_rbs_table_jcs_tripod = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_table),self.ubar_rbs_table_jcs_tripod])),self.F_rbs_table_jcs_tripod]) + 0.5*multi_dot([E(self.P_rbs_table),Te_rbs_table_jcs_tripod]))
        Q_rbs_link_1_jcs_upper_uni_1 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_link_1,self.ubar_rbs_link_1_jcs_upper_uni_1).T,multi_dot([B(self.P_rbs_link_1,self.Mbar_rbs_link_1_jcs_upper_uni_1[:,0:1]).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_upper_uni_1[:,0:1]])]]),self.L_jcs_upper_uni_1])
        self.F_rbs_link_1_jcs_upper_uni_1 = Q_rbs_link_1_jcs_upper_uni_1[0:3,0:1]
        Te_rbs_link_1_jcs_upper_uni_1 = Q_rbs_link_1_jcs_upper_uni_1[3:7,0:1]
        self.T_rbs_link_1_jcs_upper_uni_1 = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_link_1),self.ubar_rbs_link_1_jcs_upper_uni_1])),self.F_rbs_link_1_jcs_upper_uni_1]) + 0.5*multi_dot([E(self.P_rbs_link_1),Te_rbs_link_1_jcs_upper_uni_1]))
        Q_rbs_link_2_jcs_upper_uni_2 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_link_2,self.ubar_rbs_link_2_jcs_upper_uni_2).T,multi_dot([B(self.P_rbs_link_2,self.Mbar_rbs_link_2_jcs_upper_uni_2[:,0:1]).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_upper_uni_2[:,0:1]])]]),self.L_jcs_upper_uni_2])
        self.F_rbs_link_2_jcs_upper_uni_2 = Q_rbs_link_2_jcs_upper_uni_2[0:3,0:1]
        Te_rbs_link_2_jcs_upper_uni_2 = Q_rbs_link_2_jcs_upper_uni_2[3:7,0:1]
        self.T_rbs_link_2_jcs_upper_uni_2 = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_link_2),self.ubar_rbs_link_2_jcs_upper_uni_2])),self.F_rbs_link_2_jcs_upper_uni_2]) + 0.5*multi_dot([E(self.P_rbs_link_2),Te_rbs_link_2_jcs_upper_uni_2]))
        Q_rbs_link_3_jcs_upper_uni_3 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_link_3,self.ubar_rbs_link_3_jcs_upper_uni_3).T,multi_dot([B(self.P_rbs_link_3,self.Mbar_rbs_link_3_jcs_upper_uni_3[:,0:1]).T,A(self.P_rbs_table),self.Mbar_rbs_table_jcs_upper_uni_3[:,0:1]])]]),self.L_jcs_upper_uni_3])
        self.F_rbs_link_3_jcs_upper_uni_3 = Q_rbs_link_3_jcs_upper_uni_3[0:3,0:1]
        Te_rbs_link_3_jcs_upper_uni_3 = Q_rbs_link_3_jcs_upper_uni_3[3:7,0:1]
        self.T_rbs_link_3_jcs_upper_uni_3 = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_link_3),self.ubar_rbs_link_3_jcs_upper_uni_3])),self.F_rbs_link_3_jcs_upper_uni_3]) + 0.5*multi_dot([E(self.P_rbs_link_3),Te_rbs_link_3_jcs_upper_uni_3]))
        Q_rbs_rocker_1_jcs_bottom_sph_1 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbs_rocker_1,self.ubar_rbs_rocker_1_jcs_bottom_sph_1).T]]),self.L_jcs_bottom_sph_1])
        self.F_rbs_rocker_1_jcs_bottom_sph_1 = Q_rbs_rocker_1_jcs_bottom_sph_1[0:3,0:1]
        Te_rbs_rocker_1_jcs_bottom_sph_1 = Q_rbs_rocker_1_jcs_bottom_sph_1[3:7,0:1]
        self.T_rbs_rocker_1_jcs_bottom_sph_1 = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_rocker_1),self.ubar_rbs_rocker_1_jcs_bottom_sph_1])),self.F_rbs_rocker_1_jcs_bottom_sph_1]) + 0.5*multi_dot([E(self.P_rbs_rocker_1),Te_rbs_rocker_1_jcs_bottom_sph_1]))
        Q_rbs_rocker_2_jcs_bottom_sph_2 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbs_rocker_2,self.ubar_rbs_rocker_2_jcs_bottom_sph_2).T]]),self.L_jcs_bottom_sph_2])
        self.F_rbs_rocker_2_jcs_bottom_sph_2 = Q_rbs_rocker_2_jcs_bottom_sph_2[0:3,0:1]
        Te_rbs_rocker_2_jcs_bottom_sph_2 = Q_rbs_rocker_2_jcs_bottom_sph_2[3:7,0:1]
        self.T_rbs_rocker_2_jcs_bottom_sph_2 = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_rocker_2),self.ubar_rbs_rocker_2_jcs_bottom_sph_2])),self.F_rbs_rocker_2_jcs_bottom_sph_2]) + 0.5*multi_dot([E(self.P_rbs_rocker_2),Te_rbs_rocker_2_jcs_bottom_sph_2]))
        Q_rbs_rocker_3_jcs_bottom_sph_3 = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbs_rocker_3,self.ubar_rbs_rocker_3_jcs_bottom_sph_3).T]]),self.L_jcs_bottom_sph_3])
        self.F_rbs_rocker_3_jcs_bottom_sph_3 = Q_rbs_rocker_3_jcs_bottom_sph_3[0:3,0:1]
        Te_rbs_rocker_3_jcs_bottom_sph_3 = Q_rbs_rocker_3_jcs_bottom_sph_3[3:7,0:1]
        self.T_rbs_rocker_3_jcs_bottom_sph_3 = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_rocker_3),self.ubar_rbs_rocker_3_jcs_bottom_sph_3])),self.F_rbs_rocker_3_jcs_bottom_sph_3]) + 0.5*multi_dot([E(self.P_rbs_rocker_3),Te_rbs_rocker_3_jcs_bottom_sph_3]))

        self.reactions = {'F_vbs_ground_jcs_rev_1':self.F_vbs_ground_jcs_rev_1,'T_vbs_ground_jcs_rev_1':self.T_vbs_ground_jcs_rev_1,'F_vbs_ground_jcs_rev_2':self.F_vbs_ground_jcs_rev_2,'T_vbs_ground_jcs_rev_2':self.T_vbs_ground_jcs_rev_2,'F_vbs_ground_jcs_rev_3':self.F_vbs_ground_jcs_rev_3,'T_vbs_ground_jcs_rev_3':self.T_vbs_ground_jcs_rev_3,'F_rbs_table_jcs_tripod':self.F_rbs_table_jcs_tripod,'T_rbs_table_jcs_tripod':self.T_rbs_table_jcs_tripod,'F_rbs_link_1_jcs_upper_uni_1':self.F_rbs_link_1_jcs_upper_uni_1,'T_rbs_link_1_jcs_upper_uni_1':self.T_rbs_link_1_jcs_upper_uni_1,'F_rbs_link_2_jcs_upper_uni_2':self.F_rbs_link_2_jcs_upper_uni_2,'T_rbs_link_2_jcs_upper_uni_2':self.T_rbs_link_2_jcs_upper_uni_2,'F_rbs_link_3_jcs_upper_uni_3':self.F_rbs_link_3_jcs_upper_uni_3,'T_rbs_link_3_jcs_upper_uni_3':self.T_rbs_link_3_jcs_upper_uni_3,'F_rbs_rocker_1_jcs_bottom_sph_1':self.F_rbs_rocker_1_jcs_bottom_sph_1,'T_rbs_rocker_1_jcs_bottom_sph_1':self.T_rbs_rocker_1_jcs_bottom_sph_1,'F_rbs_rocker_2_jcs_bottom_sph_2':self.F_rbs_rocker_2_jcs_bottom_sph_2,'T_rbs_rocker_2_jcs_bottom_sph_2':self.T_rbs_rocker_2_jcs_bottom_sph_2,'F_rbs_rocker_3_jcs_bottom_sph_3':self.F_rbs_rocker_3_jcs_bottom_sph_3,'T_rbs_rocker_3_jcs_bottom_sph_3':self.T_rbs_rocker_3_jcs_bottom_sph_3}

