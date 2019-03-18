
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

        self.n  = 98
        self.nc = 96
        self.nrows = 60
        self.ncols = 2*18
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbr_uca_jcr_uca_upright', 'T_rbr_uca_jcr_uca_upright', 'F_rbr_uca_jcr_uca_chassis', 'T_rbr_uca_jcr_uca_chassis', 'F_rbl_uca_jcl_uca_upright', 'T_rbl_uca_jcl_uca_upright', 'F_rbl_uca_jcl_uca_chassis', 'T_rbl_uca_jcl_uca_chassis', 'F_rbr_lca_jcr_lca_upright', 'T_rbr_lca_jcr_lca_upright', 'F_rbr_lca_jcr_lca_chassis', 'T_rbr_lca_jcr_lca_chassis', 'F_rbl_lca_jcl_lca_upright', 'T_rbl_lca_jcl_lca_upright', 'F_rbl_lca_jcl_lca_chassis', 'T_rbl_lca_jcl_lca_chassis', 'F_rbr_upright_jcr_hub_bearing', 'T_rbr_upright_jcr_hub_bearing', 'F_rbl_upright_jcl_hub_bearing', 'T_rbl_upright_jcl_hub_bearing', 'F_rbr_upper_strut_jcr_strut_chassis', 'T_rbr_upper_strut_jcr_strut_chassis', 'F_rbr_upper_strut_jcr_strut', 'T_rbr_upper_strut_jcr_strut', 'F_rbl_upper_strut_jcl_strut_chassis', 'T_rbl_upper_strut_jcl_strut_chassis', 'F_rbl_upper_strut_jcl_strut', 'T_rbl_upper_strut_jcl_strut', 'F_rbr_lower_strut_jcr_strut_lca', 'T_rbr_lower_strut_jcr_strut_lca', 'F_rbl_lower_strut_jcl_strut_lca', 'T_rbl_lower_strut_jcl_strut_lca', 'F_rbr_tie_rod_jcr_tie_upright', 'T_rbr_tie_rod_jcr_tie_upright', 'F_rbr_tie_rod_jcr_tie_steering', 'T_rbr_tie_rod_jcr_tie_steering', 'F_rbl_tie_rod_jcl_tie_upright', 'T_rbl_tie_rod_jcl_tie_upright', 'F_rbl_tie_rod_jcl_tie_steering', 'T_rbl_tie_rod_jcl_tie_steering']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbr_uca = indicies_map[p+'rbr_uca']
        self.rbl_uca = indicies_map[p+'rbl_uca']
        self.rbr_lca = indicies_map[p+'rbr_lca']
        self.rbl_lca = indicies_map[p+'rbl_lca']
        self.rbr_upright = indicies_map[p+'rbr_upright']
        self.rbl_upright = indicies_map[p+'rbl_upright']
        self.rbr_upper_strut = indicies_map[p+'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p+'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p+'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p+'rbl_lower_strut']
        self.rbr_tie_rod = indicies_map[p+'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p+'rbl_tie_rod']
        self.rbr_hub = indicies_map[p+'rbr_hub']
        self.rbl_hub = indicies_map[p+'rbl_hub']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbr_steer = indicies_map[interface_map[p+'vbr_steer']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbl_steer = indicies_map[interface_map[p+'vbl_steer']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38, 38, 38, 39, 39, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.rbr_uca*2, self.rbr_uca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_uca*2, self.rbr_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_uca*2, self.rbl_uca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_hub*2, self.rbr_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_hub*2, self.rbl_hub*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_upper_strut*2, self.rbr_upper_strut*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_upper_strut*2, self.rbl_upper_strut*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbr_lca*2, self.rbr_lca*2+1, self.rbr_lower_strut*2, self.rbr_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbl_lca*2, self.rbl_lca*2+1, self.rbl_lower_strut*2, self.rbl_lower_strut*2+1, self.rbr_upright*2, self.rbr_upright*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbr_tie_rod*2, self.rbr_tie_rod*2+1, self.vbr_steer*2, self.vbr_steer*2+1, self.rbl_upright*2, self.rbl_upright*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbl_tie_rod*2, self.rbl_tie_rod*2+1, self.vbl_steer*2, self.vbl_steer*2+1, self.rbr_uca*2+1, self.rbl_uca*2+1, self.rbr_lca*2+1, self.rbl_lca*2+1, self.rbr_upright*2+1, self.rbl_upright*2+1, self.rbr_upper_strut*2+1, self.rbl_upper_strut*2+1, self.rbr_lower_strut*2+1, self.rbl_lower_strut*2+1, self.rbr_tie_rod*2+1, self.rbl_tie_rod*2+1, self.rbr_hub*2+1, self.rbl_hub*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_uca]],dtype=np.float64)
        self.F_rbl_uca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_uca]],dtype=np.float64)
        self.F_rbr_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lca]],dtype=np.float64)
        self.F_rbl_lca_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lca]],dtype=np.float64)
        self.F_rbr_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upright]],dtype=np.float64)
        self.F_rbl_upright_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upright]],dtype=np.float64)
        self.F_rbr_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_upper_strut]],dtype=np.float64)
        self.F_rbl_upper_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_upper_strut]],dtype=np.float64)
        self.F_rbr_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_lower_strut]],dtype=np.float64)
        self.F_rbl_lower_strut_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_lower_strut]],dtype=np.float64)
        self.F_rbr_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_tie_rod]],dtype=np.float64)
        self.F_rbl_tie_rod_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_tie_rod]],dtype=np.float64)
        self.F_rbr_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_hub]],dtype=np.float64)
        self.F_rbl_hub_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_hub]],dtype=np.float64)

        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_upright)])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_uca_upright)])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_upright]) + -1*multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_uca_upright]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([A(config.P_rbr_uca).T,triad(config.ax1_jcr_uca_chassis)])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_uca_chassis)])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([A(config.P_rbr_uca).T,config.pt1_jcr_uca_chassis]) + -1*multi_dot([A(config.P_rbr_uca).T,config.R_rbr_uca]))
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_uca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_upright)])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_uca_upright)])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_upright]) + -1*multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_uca_upright]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([A(config.P_rbl_uca).T,triad(config.ax1_jcl_uca_chassis)])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_uca_chassis)])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([A(config.P_rbl_uca).T,config.pt1_jcl_uca_chassis]) + -1*multi_dot([A(config.P_rbl_uca).T,config.R_rbl_uca]))
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_uca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_upright)])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_lca_upright)])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_upright]) + -1*multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_lca_upright]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([A(config.P_rbr_lca).T,triad(config.ax1_jcr_lca_chassis)])
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_lca_chassis)])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_lca_chassis]) + -1*multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_lca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_upright)])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_lca_upright)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_upright]) + -1*multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_lca_upright]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([A(config.P_rbl_lca).T,triad(config.ax1_jcl_lca_chassis)])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_lca_chassis)])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_lca_chassis]) + -1*multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_lca_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_hub_bearing)])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([A(config.P_rbr_hub).T,triad(config.ax1_jcr_hub_bearing)])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_hub_bearing]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([A(config.P_rbr_hub).T,config.pt1_jcr_hub_bearing]) + -1*multi_dot([A(config.P_rbr_hub).T,config.R_rbr_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_hub_bearing)])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([A(config.P_rbl_hub).T,triad(config.ax1_jcl_hub_bearing)])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_hub_bearing]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([A(config.P_rbl_hub).T,config.pt1_jcl_hub_bearing]) + -1*multi_dot([A(config.P_rbl_hub).T,config.R_rbl_hub]))
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut_chassis)])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax2_jcr_strut_chassis,triad(config.ax1_jcr_strut_chassis)[0:3,1:2])])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut_chassis]) + -1*multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_strut_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([A(config.P_rbr_upper_strut).T,triad(config.ax1_jcr_strut)])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut)])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_jcr_strut]) + -1*multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut]) + -1*multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_upper_strut_far_strut = (multi_dot([A(config.P_rbr_upper_strut).T,config.pt1_far_strut]) + -1*multi_dot([A(config.P_rbr_upper_strut).T,config.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_far_strut = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt2_far_strut]) + -1*multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut_chassis)])
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax2_jcl_strut_chassis,triad(config.ax1_jcl_strut_chassis)[0:3,1:2])])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut_chassis]) + -1*multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_strut_chassis]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([A(config.P_rbl_upper_strut).T,triad(config.ax1_jcl_strut)])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut)])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_jcl_strut]) + -1*multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut]) + -1*multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_upper_strut_fal_strut = (multi_dot([A(config.P_rbl_upper_strut).T,config.pt1_fal_strut]) + -1*multi_dot([A(config.P_rbl_upper_strut).T,config.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_fal_strut = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt2_fal_strut]) + -1*multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([A(config.P_rbr_lower_strut).T,triad(config.ax1_jcr_strut_lca)])
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([A(config.P_rbr_lca).T,triad(config.ax2_jcr_strut_lca,triad(config.ax1_jcr_strut_lca)[0:3,1:2])])
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([A(config.P_rbr_lower_strut).T,config.pt1_jcr_strut_lca]) + -1*multi_dot([A(config.P_rbr_lower_strut).T,config.R_rbr_lower_strut]))
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([A(config.P_rbr_lca).T,config.pt1_jcr_strut_lca]) + -1*multi_dot([A(config.P_rbr_lca).T,config.R_rbr_lca]))
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([A(config.P_rbl_lower_strut).T,triad(config.ax1_jcl_strut_lca)])
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([A(config.P_rbl_lca).T,triad(config.ax2_jcl_strut_lca,triad(config.ax1_jcl_strut_lca)[0:3,1:2])])
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([A(config.P_rbl_lower_strut).T,config.pt1_jcl_strut_lca]) + -1*multi_dot([A(config.P_rbl_lower_strut).T,config.R_rbl_lower_strut]))
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([A(config.P_rbl_lca).T,config.pt1_jcl_strut_lca]) + -1*multi_dot([A(config.P_rbl_lca).T,config.R_rbl_lca]))
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_upright)])
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([A(config.P_rbr_upright).T,triad(config.ax1_jcr_tie_upright)])
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_upright]) + -1*multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([A(config.P_rbr_upright).T,config.pt1_jcr_tie_upright]) + -1*multi_dot([A(config.P_rbr_upright).T,config.R_rbr_upright]))
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([A(config.P_rbr_tie_rod).T,triad(config.ax1_jcr_tie_steering)])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([A(config.P_vbr_steer).T,triad(config.ax2_jcr_tie_steering,triad(config.ax1_jcr_tie_steering)[0:3,1:2])])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([A(config.P_rbr_tie_rod).T,config.pt1_jcr_tie_steering]) + -1*multi_dot([A(config.P_rbr_tie_rod).T,config.R_rbr_tie_rod]))
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([A(config.P_vbr_steer).T,config.pt1_jcr_tie_steering]) + -1*multi_dot([A(config.P_vbr_steer).T,config.R_vbr_steer]))
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_upright)])
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([A(config.P_rbl_upright).T,triad(config.ax1_jcl_tie_upright)])
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_upright]) + -1*multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([A(config.P_rbl_upright).T,config.pt1_jcl_tie_upright]) + -1*multi_dot([A(config.P_rbl_upright).T,config.R_rbl_upright]))
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([A(config.P_rbl_tie_rod).T,triad(config.ax1_jcl_tie_steering)])
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([A(config.P_vbl_steer).T,triad(config.ax2_jcl_tie_steering,triad(config.ax1_jcl_tie_steering)[0:3,1:2])])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([A(config.P_rbl_tie_rod).T,config.pt1_jcl_tie_steering]) + -1*multi_dot([A(config.P_rbl_tie_rod).T,config.R_rbl_tie_rod]))
        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([A(config.P_vbl_steer).T,config.pt1_jcl_tie_steering]) + -1*multi_dot([A(config.P_vbl_steer).T,config.R_vbl_steer]))

    
    def set_gen_coordinates(self,q):
        self.R_rbr_uca = q[0:3,0:1]
        self.P_rbr_uca = q[3:7,0:1]
        self.R_rbl_uca = q[7:10,0:1]
        self.P_rbl_uca = q[10:14,0:1]
        self.R_rbr_lca = q[14:17,0:1]
        self.P_rbr_lca = q[17:21,0:1]
        self.R_rbl_lca = q[21:24,0:1]
        self.P_rbl_lca = q[24:28,0:1]
        self.R_rbr_upright = q[28:31,0:1]
        self.P_rbr_upright = q[31:35,0:1]
        self.R_rbl_upright = q[35:38,0:1]
        self.P_rbl_upright = q[38:42,0:1]
        self.R_rbr_upper_strut = q[42:45,0:1]
        self.P_rbr_upper_strut = q[45:49,0:1]
        self.R_rbl_upper_strut = q[49:52,0:1]
        self.P_rbl_upper_strut = q[52:56,0:1]
        self.R_rbr_lower_strut = q[56:59,0:1]
        self.P_rbr_lower_strut = q[59:63,0:1]
        self.R_rbl_lower_strut = q[63:66,0:1]
        self.P_rbl_lower_strut = q[66:70,0:1]
        self.R_rbr_tie_rod = q[70:73,0:1]
        self.P_rbr_tie_rod = q[73:77,0:1]
        self.R_rbl_tie_rod = q[77:80,0:1]
        self.P_rbl_tie_rod = q[80:84,0:1]
        self.R_rbr_hub = q[84:87,0:1]
        self.P_rbr_hub = q[87:91,0:1]
        self.R_rbl_hub = q[91:94,0:1]
        self.P_rbl_hub = q[94:98,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbr_uca = qd[0:3,0:1]
        self.Pd_rbr_uca = qd[3:7,0:1]
        self.Rd_rbl_uca = qd[7:10,0:1]
        self.Pd_rbl_uca = qd[10:14,0:1]
        self.Rd_rbr_lca = qd[14:17,0:1]
        self.Pd_rbr_lca = qd[17:21,0:1]
        self.Rd_rbl_lca = qd[21:24,0:1]
        self.Pd_rbl_lca = qd[24:28,0:1]
        self.Rd_rbr_upright = qd[28:31,0:1]
        self.Pd_rbr_upright = qd[31:35,0:1]
        self.Rd_rbl_upright = qd[35:38,0:1]
        self.Pd_rbl_upright = qd[38:42,0:1]
        self.Rd_rbr_upper_strut = qd[42:45,0:1]
        self.Pd_rbr_upper_strut = qd[45:49,0:1]
        self.Rd_rbl_upper_strut = qd[49:52,0:1]
        self.Pd_rbl_upper_strut = qd[52:56,0:1]
        self.Rd_rbr_lower_strut = qd[56:59,0:1]
        self.Pd_rbr_lower_strut = qd[59:63,0:1]
        self.Rd_rbl_lower_strut = qd[63:66,0:1]
        self.Pd_rbl_lower_strut = qd[66:70,0:1]
        self.Rd_rbr_tie_rod = qd[70:73,0:1]
        self.Pd_rbr_tie_rod = qd[73:77,0:1]
        self.Rd_rbl_tie_rod = qd[77:80,0:1]
        self.Pd_rbl_tie_rod = qd[80:84,0:1]
        self.Rd_rbr_hub = qd[84:87,0:1]
        self.Pd_rbr_hub = qd[87:91,0:1]
        self.Rd_rbl_hub = qd[91:94,0:1]
        self.Pd_rbl_hub = qd[94:98,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbr_uca = qdd[0:3,0:1]
        self.Pdd_rbr_uca = qdd[3:7,0:1]
        self.Rdd_rbl_uca = qdd[7:10,0:1]
        self.Pdd_rbl_uca = qdd[10:14,0:1]
        self.Rdd_rbr_lca = qdd[14:17,0:1]
        self.Pdd_rbr_lca = qdd[17:21,0:1]
        self.Rdd_rbl_lca = qdd[21:24,0:1]
        self.Pdd_rbl_lca = qdd[24:28,0:1]
        self.Rdd_rbr_upright = qdd[28:31,0:1]
        self.Pdd_rbr_upright = qdd[31:35,0:1]
        self.Rdd_rbl_upright = qdd[35:38,0:1]
        self.Pdd_rbl_upright = qdd[38:42,0:1]
        self.Rdd_rbr_upper_strut = qdd[42:45,0:1]
        self.Pdd_rbr_upper_strut = qdd[45:49,0:1]
        self.Rdd_rbl_upper_strut = qdd[49:52,0:1]
        self.Pdd_rbl_upper_strut = qdd[52:56,0:1]
        self.Rdd_rbr_lower_strut = qdd[56:59,0:1]
        self.Pdd_rbr_lower_strut = qdd[59:63,0:1]
        self.Rdd_rbl_lower_strut = qdd[63:66,0:1]
        self.Pdd_rbl_lower_strut = qdd[66:70,0:1]
        self.Rdd_rbr_tie_rod = qdd[70:73,0:1]
        self.Pdd_rbr_tie_rod = qdd[73:77,0:1]
        self.Rdd_rbl_tie_rod = qdd[77:80,0:1]
        self.Pdd_rbl_tie_rod = qdd[80:84,0:1]
        self.Rdd_rbr_hub = qdd[84:87,0:1]
        self.Pdd_rbr_hub = qdd[87:91,0:1]
        self.Rdd_rbl_hub = qdd[91:94,0:1]
        self.Pdd_rbl_hub = qdd[94:98,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcr_uca_upright = Lambda[0:3,0:1]
        self.L_jcr_uca_chassis = Lambda[3:8,0:1]
        self.L_jcl_uca_upright = Lambda[8:11,0:1]
        self.L_jcl_uca_chassis = Lambda[11:16,0:1]
        self.L_jcr_lca_upright = Lambda[16:19,0:1]
        self.L_jcr_lca_chassis = Lambda[19:24,0:1]
        self.L_jcl_lca_upright = Lambda[24:27,0:1]
        self.L_jcl_lca_chassis = Lambda[27:32,0:1]
        self.L_jcr_hub_bearing = Lambda[32:38,0:1]
        self.L_jcl_hub_bearing = Lambda[38:44,0:1]
        self.L_jcr_strut_chassis = Lambda[44:48,0:1]
        self.L_jcr_strut = Lambda[48:52,0:1]
        self.L_jcl_strut_chassis = Lambda[52:56,0:1]
        self.L_jcl_strut = Lambda[56:60,0:1]
        self.L_jcr_strut_lca = Lambda[60:64,0:1]
        self.L_jcl_strut_lca = Lambda[64:68,0:1]
        self.L_jcr_tie_upright = Lambda[68:71,0:1]
        self.L_jcr_tie_steering = Lambda[71:75,0:1]
        self.L_jcl_tie_upright = Lambda[75:78,0:1]
        self.L_jcl_tie_steering = Lambda[78:82,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_uca
        x1 = self.R_rbr_upright
        x2 = -1*x1
        x3 = self.P_rbr_uca
        x4 = A(x3)
        x5 = self.P_rbr_upright
        x6 = A(x5)
        x7 = -1*self.R_vbs_chassis
        x8 = A(self.P_vbs_chassis)
        x9 = x4.T
        x10 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        x11 = self.R_rbl_uca
        x12 = self.R_rbl_upright
        x13 = -1*x12
        x14 = self.P_rbl_uca
        x15 = A(x14)
        x16 = self.P_rbl_upright
        x17 = A(x16)
        x18 = x15.T
        x19 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x20 = self.R_rbr_lca
        x21 = self.P_rbr_lca
        x22 = A(x21)
        x23 = x22.T
        x24 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x25 = self.R_rbl_lca
        x26 = self.P_rbl_lca
        x27 = A(x26)
        x28 = x27.T
        x29 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x30 = self.P_rbr_hub
        x31 = A(x30)
        x32 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T
        x33 = x6.T
        x34 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x35 = self.P_rbl_hub
        x36 = A(x35)
        x37 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T
        x38 = x17.T
        x39 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x40 = self.R_rbr_upper_strut
        x41 = self.P_rbr_upper_strut
        x42 = A(x41)
        x43 = x42.T
        x44 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x45 = self.P_rbr_lower_strut
        x46 = A(x45)
        x47 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x48 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x49 = self.R_rbr_lower_strut
        x50 = (x40 + -1*x49 + multi_dot([x42,self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([x46,self.ubar_rbr_lower_strut_jcr_strut]))
        x51 = self.R_rbl_upper_strut
        x52 = self.P_rbl_upper_strut
        x53 = A(x52)
        x54 = x53.T
        x55 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x56 = self.P_rbl_lower_strut
        x57 = A(x56)
        x58 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x59 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x60 = self.R_rbl_lower_strut
        x61 = (x51 + -1*x60 + multi_dot([x53,self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([x57,self.ubar_rbl_lower_strut_jcl_strut]))
        x62 = self.R_rbr_tie_rod
        x63 = self.P_rbr_tie_rod
        x64 = A(x63)
        x65 = A(self.P_vbr_steer)
        x66 = self.R_rbl_tie_rod
        x67 = self.P_rbl_tie_rod
        x68 = A(x67)
        x69 = A(self.P_vbl_steer)
        x70 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x2 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_uca_upright])),
        (x0 + x7 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_uca_chassis])),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),
        multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),
        (x11 + x13 + multi_dot([x15,self.ubar_rbl_uca_jcl_uca_upright]) + -1*multi_dot([x17,self.ubar_rbl_upright_jcl_uca_upright])),
        (x11 + x7 + multi_dot([x15,self.ubar_rbl_uca_jcl_uca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_uca_chassis])),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x18,x8,x19]),
        multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x18,x8,x19]),
        (x20 + x2 + multi_dot([x22,self.ubar_rbr_lca_jcr_lca_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_lca_upright])),
        (x20 + x7 + multi_dot([x22,self.ubar_rbr_lca_jcr_lca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_lca_chassis])),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x23,x8,x24]),
        multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x23,x8,x24]),
        (x25 + x13 + multi_dot([x27,self.ubar_rbl_lca_jcl_lca_upright]) + -1*multi_dot([x17,self.ubar_rbl_upright_jcl_lca_upright])),
        (x25 + x7 + multi_dot([x27,self.ubar_rbl_lca_jcl_lca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_lca_chassis])),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x28,x8,x29]),
        multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x28,x8,x29]),
        (x1 + -1*self.R_rbr_hub + multi_dot([x6,self.ubar_rbr_upright_jcr_hub_bearing]) + -1*multi_dot([x31,self.ubar_rbr_hub_jcr_hub_bearing])),
        multi_dot([x32,x33,x31,x34]),
        multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x33,x31,x34]),
        multi_dot([x32,x33,x31,self.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]]),
        (x12 + -1*self.R_rbl_hub + multi_dot([x17,self.ubar_rbl_upright_jcl_hub_bearing]) + -1*multi_dot([x36,self.ubar_rbl_hub_jcl_hub_bearing])),
        multi_dot([x37,x38,x36,x39]),
        multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x38,x36,x39]),
        multi_dot([x37,x38,x36,self.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]]),
        (x40 + x7 + multi_dot([x42,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_strut_chassis])),
        multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x43,x8,self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),
        multi_dot([x44,x43,x46,x47]),
        multi_dot([x48,x43,x46,x47]),
        multi_dot([x44,x43,x50]),
        multi_dot([x48,x43,x50]),
        (x51 + x7 + multi_dot([x53,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_strut_chassis])),
        multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x54,x8,self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),
        multi_dot([x55,x54,x57,x58]),
        multi_dot([x59,x54,x57,x58]),
        multi_dot([x55,x54,x61]),
        multi_dot([x59,x54,x61]),
        (x49 + -1*x20 + multi_dot([x46,self.ubar_rbr_lower_strut_jcr_strut_lca]) + -1*multi_dot([x22,self.ubar_rbr_lca_jcr_strut_lca])),
        multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x46.T,x22,self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),
        (x60 + -1*x25 + multi_dot([x57,self.ubar_rbl_lower_strut_jcl_strut_lca]) + -1*multi_dot([x27,self.ubar_rbl_lca_jcl_strut_lca])),
        multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1].T,x57.T,x27,self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]]),
        (x62 + x2 + multi_dot([x64,self.ubar_rbr_tie_rod_jcr_tie_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_tie_upright])),
        (x62 + -1*self.R_vbr_steer + multi_dot([x64,self.ubar_rbr_tie_rod_jcr_tie_steering]) + -1*multi_dot([x65,self.ubar_vbr_steer_jcr_tie_steering])),
        multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x64.T,x65,self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),
        (x66 + x13 + multi_dot([x68,self.ubar_rbl_tie_rod_jcl_tie_upright]) + -1*multi_dot([x17,self.ubar_rbl_upright_jcl_tie_upright])),
        (x66 + -1*self.R_vbl_steer + multi_dot([x68,self.ubar_rbl_tie_rod_jcl_tie_steering]) + -1*multi_dot([x69,self.ubar_vbl_steer_jcl_tie_steering])),
        multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x68.T,x69,self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),
        (x70 + (multi_dot([x3.T,x3]))**(1.0/2.0)),
        (x70 + (multi_dot([x14.T,x14]))**(1.0/2.0)),
        (x70 + (multi_dot([x21.T,x21]))**(1.0/2.0)),
        (x70 + (multi_dot([x26.T,x26]))**(1.0/2.0)),
        (x70 + (multi_dot([x5.T,x5]))**(1.0/2.0)),
        (x70 + (multi_dot([x16.T,x16]))**(1.0/2.0)),
        (x70 + (multi_dot([x41.T,x41]))**(1.0/2.0)),
        (x70 + (multi_dot([x52.T,x52]))**(1.0/2.0)),
        (x70 + (multi_dot([x45.T,x45]))**(1.0/2.0)),
        (x70 + (multi_dot([x56.T,x56]))**(1.0/2.0)),
        (x70 + (multi_dot([x63.T,x63]))**(1.0/2.0)),
        (x70 + (multi_dot([x67.T,x67]))**(1.0/2.0)),
        (x70 + (multi_dot([x30.T,x30]))**(1.0/2.0)),
        (x70 + (multi_dot([x35.T,x35]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v1,
        v0,
        v1,
        v1,
        v1,
        v0,
        v1,
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
        v1,
        v1,
        v1,
        v1,
        v0,
        v1,
        v0,
        v1,
        v0,
        v0,
        v1,
        v0,
        v0,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
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

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_rbr_upright
        a2 = self.Pd_vbs_chassis
        a3 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        a4 = a3.T
        a5 = self.P_vbs_chassis
        a6 = A(a5).T
        a7 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a8 = self.P_rbr_uca
        a9 = A(a8).T
        a10 = B(a2,a3)
        a11 = a0.T
        a12 = B(a5,a3)
        a13 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a14 = self.Pd_rbl_uca
        a15 = self.Pd_rbl_upright
        a16 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a17 = self.P_rbl_uca
        a18 = A(a17).T
        a19 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a20 = B(a2,a19)
        a21 = a19.T
        a22 = a14.T
        a23 = B(a5,a19)
        a24 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a25 = self.Pd_rbr_lca
        a26 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a27 = self.P_rbr_lca
        a28 = A(a27).T
        a29 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a30 = B(a2,a29)
        a31 = a29.T
        a32 = a25.T
        a33 = B(a5,a29)
        a34 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a35 = self.Pd_rbl_lca
        a36 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a37 = self.P_rbl_lca
        a38 = A(a37).T
        a39 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a40 = B(a2,a39)
        a41 = a39.T
        a42 = a35.T
        a43 = B(a5,a39)
        a44 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a45 = self.Pd_rbr_hub
        a46 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a47 = a46.T
        a48 = self.P_rbr_hub
        a49 = A(a48).T
        a50 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a51 = B(a1,a50)
        a52 = a50.T
        a53 = self.P_rbr_upright
        a54 = A(a53).T
        a55 = B(a45,a46)
        a56 = a1.T
        a57 = B(a53,a50).T
        a58 = B(a48,a46)
        a59 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a60 = self.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]
        a61 = self.Pd_rbl_hub
        a62 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a63 = a62.T
        a64 = self.P_rbl_upright
        a65 = A(a64).T
        a66 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a67 = B(a61,a66)
        a68 = a66.T
        a69 = self.P_rbl_hub
        a70 = A(a69).T
        a71 = B(a15,a62)
        a72 = a15.T
        a73 = B(a64,a62).T
        a74 = B(a69,a66)
        a75 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a76 = self.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        a77 = self.Pd_rbr_upper_strut
        a78 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a79 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a80 = self.P_rbr_upper_strut
        a81 = A(a80).T
        a82 = a77.T
        a83 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a84 = a83.T
        a85 = self.P_rbr_lower_strut
        a86 = A(a85).T
        a87 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a88 = B(a77,a87)
        a89 = a87.T
        a90 = self.Pd_rbr_lower_strut
        a91 = B(a90,a83)
        a92 = B(a80,a87).T
        a93 = B(a85,a83)
        a94 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a95 = B(a77,a94)
        a96 = a94.T
        a97 = B(a80,a94).T
        a98 = self.ubar_rbr_upper_strut_jcr_strut
        a99 = self.ubar_rbr_lower_strut_jcr_strut
        a100 = (multi_dot([B(a77,a98),a77]) + -1*multi_dot([B(a90,a99),a90]))
        a101 = (self.Rd_rbr_upper_strut + -1*self.Rd_rbr_lower_strut + multi_dot([B(a80,a98),a77]) + -1*multi_dot([B(a85,a99),a90]))
        a102 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([a98.T,a81]) + -1*multi_dot([a99.T,a86]))
        a103 = self.Pd_rbl_upper_strut
        a104 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a105 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a106 = self.P_rbl_upper_strut
        a107 = A(a106).T
        a108 = a103.T
        a109 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a110 = a109.T
        a111 = self.P_rbl_lower_strut
        a112 = A(a111).T
        a113 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a114 = B(a103,a113)
        a115 = a113.T
        a116 = self.Pd_rbl_lower_strut
        a117 = B(a116,a109)
        a118 = B(a106,a113).T
        a119 = B(a111,a109)
        a120 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a121 = B(a103,a120)
        a122 = a120.T
        a123 = B(a106,a120).T
        a124 = self.ubar_rbl_upper_strut_jcl_strut
        a125 = self.ubar_rbl_lower_strut_jcl_strut
        a126 = (multi_dot([B(a103,a124),a103]) + -1*multi_dot([B(a116,a125),a116]))
        a127 = (self.Rd_rbl_upper_strut + -1*self.Rd_rbl_lower_strut + multi_dot([B(a106,a124),a103]) + -1*multi_dot([B(a111,a125),a116]))
        a128 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([a124.T,a107]) + -1*multi_dot([a125.T,a112]))
        a129 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a130 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a131 = a90.T
        a132 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a133 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a134 = a116.T
        a135 = self.Pd_rbr_tie_rod
        a136 = self.Pd_vbr_steer
        a137 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a138 = self.P_vbr_steer
        a139 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a140 = self.P_rbr_tie_rod
        a141 = a135.T
        a142 = self.Pd_rbl_tie_rod
        a143 = self.Pd_vbl_steer
        a144 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a145 = self.P_rbl_tie_rod
        a146 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a147 = self.P_vbl_steer
        a148 = a142.T

        self.acc_eq_blocks = [-1*(multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_upright),a0]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_uca_upright),a1])),
        -1*(multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_chassis),a0]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_uca_chassis),a2])),
        (multi_dot([a4,a6,B(a0,a7),a0]) + multi_dot([a7.T,a9,a10,a2]) + 2*multi_dot([a11,B(a8,a7).T,a12,a2])),
        (multi_dot([a4,a6,B(a0,a13),a0]) + multi_dot([a13.T,a9,a10,a2]) + 2*multi_dot([a11,B(a8,a13).T,a12,a2])),
        -1*(multi_dot([B(a14,self.ubar_rbl_uca_jcl_uca_upright),a14]) + -1*multi_dot([B(a15,self.ubar_rbl_upright_jcl_uca_upright),a15])),
        -1*(multi_dot([B(a14,self.ubar_rbl_uca_jcl_uca_chassis),a14]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_uca_chassis),a2])),
        (multi_dot([a16.T,a18,a20,a2]) + multi_dot([a21,a6,B(a14,a16),a14]) + 2*multi_dot([a22,B(a17,a16).T,a23,a2])),
        (multi_dot([a24.T,a18,a20,a2]) + multi_dot([a21,a6,B(a14,a24),a14]) + 2*multi_dot([a22,B(a17,a24).T,a23,a2])),
        -1*(multi_dot([B(a25,self.ubar_rbr_lca_jcr_lca_upright),a25]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_lca_upright),a1])),
        -1*(multi_dot([B(a25,self.ubar_rbr_lca_jcr_lca_chassis),a25]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_lca_chassis),a2])),
        (multi_dot([a26.T,a28,a30,a2]) + multi_dot([a31,a6,B(a25,a26),a25]) + 2*multi_dot([a32,B(a27,a26).T,a33,a2])),
        (multi_dot([a34.T,a28,a30,a2]) + multi_dot([a31,a6,B(a25,a34),a25]) + 2*multi_dot([a32,B(a27,a34).T,a33,a2])),
        -1*(multi_dot([B(a35,self.ubar_rbl_lca_jcl_lca_upright),a35]) + -1*multi_dot([B(a15,self.ubar_rbl_upright_jcl_lca_upright),a15])),
        -1*(multi_dot([B(a35,self.ubar_rbl_lca_jcl_lca_chassis),a35]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_lca_chassis),a2])),
        (multi_dot([a36.T,a38,a40,a2]) + multi_dot([a41,a6,B(a35,a36),a35]) + 2*multi_dot([a42,B(a37,a36).T,a43,a2])),
        (multi_dot([a44.T,a38,a40,a2]) + multi_dot([a41,a6,B(a35,a44),a35]) + 2*multi_dot([a42,B(a37,a44).T,a43,a2])),
        -1*(multi_dot([B(a1,self.ubar_rbr_upright_jcr_hub_bearing),a1]) + -1*multi_dot([B(a45,self.ubar_rbr_hub_jcr_hub_bearing),a45])),
        (multi_dot([a47,a49,a51,a1]) + multi_dot([a52,a54,a55,a45]) + 2*multi_dot([a56,a57,a58,a45])),
        (multi_dot([a47,a49,B(a1,a59),a1]) + multi_dot([a59.T,a54,a55,a45]) + 2*multi_dot([a56,B(a53,a59).T,a58,a45])),
        (multi_dot([a60.T,a49,a51,a1]) + multi_dot([a52,a54,B(a45,a60),a45]) + 2*multi_dot([a56,a57,B(a48,a60),a45])),
        -1*(multi_dot([B(a15,self.ubar_rbl_upright_jcl_hub_bearing),a15]) + -1*multi_dot([B(a61,self.ubar_rbl_hub_jcl_hub_bearing),a61])),
        (multi_dot([a63,a65,a67,a61]) + multi_dot([a68,a70,a71,a15]) + 2*multi_dot([a72,a73,a74,a61])),
        (multi_dot([a75.T,a65,a67,a61]) + multi_dot([a68,a70,B(a15,a75),a15]) + 2*multi_dot([a72,B(a64,a75).T,a74,a61])),
        (multi_dot([a63,a65,B(a61,a76),a61]) + multi_dot([a76.T,a70,a71,a15]) + 2*multi_dot([a72,a73,B(a69,a76),a61])),
        -1*(multi_dot([B(a77,self.ubar_rbr_upper_strut_jcr_strut_chassis),a77]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_strut_chassis),a2])),
        (multi_dot([a78.T,a6,B(a77,a79),a77]) + multi_dot([a79.T,a81,B(a2,a78),a2]) + 2*multi_dot([a82,B(a80,a79).T,B(a5,a78),a2])),
        (multi_dot([a84,a86,a88,a77]) + multi_dot([a89,a81,a91,a90]) + 2*multi_dot([a82,a92,a93,a90])),
        (multi_dot([a84,a86,a95,a77]) + multi_dot([a96,a81,a91,a90]) + 2*multi_dot([a82,a97,a93,a90])),
        (multi_dot([a89,a81,a100]) + 2*multi_dot([a82,a92,a101]) + multi_dot([a102,a88,a77])),
        (multi_dot([a96,a81,a100]) + 2*multi_dot([a82,a97,a101]) + multi_dot([a102,a95,a77])),
        -1*(multi_dot([B(a103,self.ubar_rbl_upper_strut_jcl_strut_chassis),a103]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_strut_chassis),a2])),
        (multi_dot([a104.T,a6,B(a103,a105),a103]) + multi_dot([a105.T,a107,B(a2,a104),a2]) + 2*multi_dot([a108,B(a106,a105).T,B(a5,a104),a2])),
        (multi_dot([a110,a112,a114,a103]) + multi_dot([a115,a107,a117,a116]) + 2*multi_dot([a108,a118,a119,a116])),
        (multi_dot([a110,a112,a121,a103]) + multi_dot([a122,a107,a117,a116]) + 2*multi_dot([a108,a123,a119,a116])),
        (multi_dot([a115,a107,a126]) + 2*multi_dot([a108,a118,a127]) + multi_dot([a128,a114,a103])),
        (multi_dot([a122,a107,a126]) + 2*multi_dot([a108,a123,a127]) + multi_dot([a128,a121,a103])),
        -1*(multi_dot([B(a90,self.ubar_rbr_lower_strut_jcr_strut_lca),a90]) + -1*multi_dot([B(a25,self.ubar_rbr_lca_jcr_strut_lca),a25])),
        (multi_dot([a129.T,a28,B(a90,a130),a90]) + multi_dot([a130.T,a86,B(a25,a129),a25]) + 2*multi_dot([a131,B(a85,a130).T,B(a27,a129),a25])),
        -1*(multi_dot([B(a116,self.ubar_rbl_lower_strut_jcl_strut_lca),a116]) + -1*multi_dot([B(a35,self.ubar_rbl_lca_jcl_strut_lca),a35])),
        (multi_dot([a132.T,a112,B(a35,a133),a35]) + multi_dot([a133.T,a38,B(a116,a132),a116]) + 2*multi_dot([a134,B(a111,a132).T,B(a37,a133),a35])),
        -1*(multi_dot([B(a135,self.ubar_rbr_tie_rod_jcr_tie_upright),a135]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_tie_upright),a1])),
        -1*(multi_dot([B(a135,self.ubar_rbr_tie_rod_jcr_tie_steering),a135]) + -1*multi_dot([B(a136,self.ubar_vbr_steer_jcr_tie_steering),a136])),
        (multi_dot([a137.T,A(a138).T,B(a135,a139),a135]) + multi_dot([a139.T,A(a140).T,B(a136,a137),a136]) + 2*multi_dot([a141,B(a140,a139).T,B(a138,a137),a136])),
        -1*(multi_dot([B(a142,self.ubar_rbl_tie_rod_jcl_tie_upright),a142]) + -1*multi_dot([B(a15,self.ubar_rbl_upright_jcl_tie_upright),a15])),
        -1*(multi_dot([B(a142,self.ubar_rbl_tie_rod_jcl_tie_steering),a142]) + -1*multi_dot([B(a143,self.ubar_vbl_steer_jcl_tie_steering),a143])),
        (multi_dot([a144.T,A(a145).T,B(a143,a146),a143]) + multi_dot([a146.T,A(a147).T,B(a142,a144),a142]) + 2*multi_dot([a148,B(a145,a144).T,B(a147,a146),a143])),
        2*(multi_dot([a11,a0]))**(1.0/2.0),
        2*(multi_dot([a22,a14]))**(1.0/2.0),
        2*(multi_dot([a32,a25]))**(1.0/2.0),
        2*(multi_dot([a42,a35]))**(1.0/2.0),
        2*(multi_dot([a56,a1]))**(1.0/2.0),
        2*(multi_dot([a72,a15]))**(1.0/2.0),
        2*(multi_dot([a82,a77]))**(1.0/2.0),
        2*(multi_dot([a108,a103]))**(1.0/2.0),
        2*(multi_dot([a131,a90]))**(1.0/2.0),
        2*(multi_dot([a134,a116]))**(1.0/2.0),
        2*(multi_dot([a141,a135]))**(1.0/2.0),
        2*(multi_dot([a148,a142]))**(1.0/2.0),
        2*(multi_dot([a45.T,a45]))**(1.0/2.0),
        2*(multi_dot([a61.T,a61]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbr_uca
        j2 = -1*j0
        j3 = self.P_rbr_upright
        j4 = np.zeros((1,3),dtype=np.float64)
        j5 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        j6 = j5.T
        j7 = self.P_vbs_chassis
        j8 = A(j7).T
        j9 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j10 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j11 = A(j1).T
        j12 = B(j7,j5)
        j13 = self.P_rbl_uca
        j14 = self.P_rbl_upright
        j15 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j16 = j15.T
        j17 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j18 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j19 = A(j13).T
        j20 = B(j7,j15)
        j21 = self.P_rbr_lca
        j22 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j23 = j22.T
        j24 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j25 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j26 = A(j21).T
        j27 = B(j7,j22)
        j28 = self.P_rbl_lca
        j29 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j30 = j29.T
        j31 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j32 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j33 = A(j28).T
        j34 = B(j7,j29)
        j35 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j36 = j35.T
        j37 = self.P_rbr_hub
        j38 = A(j37).T
        j39 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j40 = B(j3,j39)
        j41 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j42 = self.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]
        j43 = j39.T
        j44 = A(j3).T
        j45 = B(j37,j35)
        j46 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j47 = j46.T
        j48 = self.P_rbl_hub
        j49 = A(j48).T
        j50 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j51 = B(j14,j50)
        j52 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j53 = self.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        j54 = j50.T
        j55 = A(j14).T
        j56 = B(j48,j46)
        j57 = self.P_rbr_upper_strut
        j58 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j59 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j60 = A(j57).T
        j61 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j62 = j61.T
        j63 = self.P_rbr_lower_strut
        j64 = A(j63).T
        j65 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j66 = B(j57,j65)
        j67 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j68 = B(j57,j67)
        j69 = j65.T
        j70 = multi_dot([j69,j60])
        j71 = self.ubar_rbr_upper_strut_jcr_strut
        j72 = B(j57,j71)
        j73 = self.ubar_rbr_lower_strut_jcr_strut
        j74 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([j71.T,j60]) + -1*multi_dot([j73.T,j64]))
        j75 = j67.T
        j76 = multi_dot([j75,j60])
        j77 = B(j63,j61)
        j78 = B(j63,j73)
        j79 = self.P_rbl_upper_strut
        j80 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j81 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j82 = A(j79).T
        j83 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j84 = j83.T
        j85 = self.P_rbl_lower_strut
        j86 = A(j85).T
        j87 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j88 = B(j79,j87)
        j89 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j90 = B(j79,j89)
        j91 = j87.T
        j92 = multi_dot([j91,j82])
        j93 = self.ubar_rbl_upper_strut_jcl_strut
        j94 = B(j79,j93)
        j95 = self.ubar_rbl_lower_strut_jcl_strut
        j96 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([j93.T,j82]) + -1*multi_dot([j95.T,j86]))
        j97 = j89.T
        j98 = multi_dot([j97,j82])
        j99 = B(j85,j83)
        j100 = B(j85,j95)
        j101 = self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j102 = self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j103 = self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j104 = self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j105 = self.P_rbr_tie_rod
        j106 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j107 = self.P_vbr_steer
        j108 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j109 = self.P_rbl_tie_rod
        j110 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j111 = self.P_vbl_steer
        j112 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = [j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_upright),
        j2,
        -1*B(j3,self.ubar_rbr_upright_jcr_uca_upright),
        j0,
        B(j1,self.ubar_rbr_uca_jcr_uca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_uca_chassis),
        j4,
        multi_dot([j6,j8,B(j1,j9)]),
        j4,
        multi_dot([j9.T,j11,j12]),
        j4,
        multi_dot([j6,j8,B(j1,j10)]),
        j4,
        multi_dot([j10.T,j11,j12]),
        j0,
        B(j13,self.ubar_rbl_uca_jcl_uca_upright),
        j2,
        -1*B(j14,self.ubar_rbl_upright_jcl_uca_upright),
        j0,
        B(j13,self.ubar_rbl_uca_jcl_uca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_uca_chassis),
        j4,
        multi_dot([j16,j8,B(j13,j17)]),
        j4,
        multi_dot([j17.T,j19,j20]),
        j4,
        multi_dot([j16,j8,B(j13,j18)]),
        j4,
        multi_dot([j18.T,j19,j20]),
        j0,
        B(j21,self.ubar_rbr_lca_jcr_lca_upright),
        j2,
        -1*B(j3,self.ubar_rbr_upright_jcr_lca_upright),
        j0,
        B(j21,self.ubar_rbr_lca_jcr_lca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_lca_chassis),
        j4,
        multi_dot([j23,j8,B(j21,j24)]),
        j4,
        multi_dot([j24.T,j26,j27]),
        j4,
        multi_dot([j23,j8,B(j21,j25)]),
        j4,
        multi_dot([j25.T,j26,j27]),
        j0,
        B(j28,self.ubar_rbl_lca_jcl_lca_upright),
        j2,
        -1*B(j14,self.ubar_rbl_upright_jcl_lca_upright),
        j0,
        B(j28,self.ubar_rbl_lca_jcl_lca_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_lca_chassis),
        j4,
        multi_dot([j30,j8,B(j28,j31)]),
        j4,
        multi_dot([j31.T,j33,j34]),
        j4,
        multi_dot([j30,j8,B(j28,j32)]),
        j4,
        multi_dot([j32.T,j33,j34]),
        j0,
        B(j3,self.ubar_rbr_upright_jcr_hub_bearing),
        j2,
        -1*B(j37,self.ubar_rbr_hub_jcr_hub_bearing),
        j4,
        multi_dot([j36,j38,j40]),
        j4,
        multi_dot([j43,j44,j45]),
        j4,
        multi_dot([j36,j38,B(j3,j41)]),
        j4,
        multi_dot([j41.T,j44,j45]),
        j4,
        multi_dot([j42.T,j38,j40]),
        j4,
        multi_dot([j43,j44,B(j37,j42)]),
        j0,
        B(j14,self.ubar_rbl_upright_jcl_hub_bearing),
        j2,
        -1*B(j48,self.ubar_rbl_hub_jcl_hub_bearing),
        j4,
        multi_dot([j47,j49,j51]),
        j4,
        multi_dot([j54,j55,j56]),
        j4,
        multi_dot([j47,j49,B(j14,j52)]),
        j4,
        multi_dot([j52.T,j55,j56]),
        j4,
        multi_dot([j53.T,j49,j51]),
        j4,
        multi_dot([j54,j55,B(j48,j53)]),
        j0,
        B(j57,self.ubar_rbr_upper_strut_jcr_strut_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcr_strut_chassis),
        j4,
        multi_dot([j58.T,j8,B(j57,j59)]),
        j4,
        multi_dot([j59.T,j60,B(j7,j58)]),
        j4,
        multi_dot([j62,j64,j66]),
        j4,
        multi_dot([j69,j60,j77]),
        j4,
        multi_dot([j62,j64,j68]),
        j4,
        multi_dot([j75,j60,j77]),
        j70,
        (multi_dot([j69,j60,j72]) + multi_dot([j74,j66])),
        -1*j70,
        -1*multi_dot([j69,j60,j78]),
        j76,
        (multi_dot([j75,j60,j72]) + multi_dot([j74,j68])),
        -1*j76,
        -1*multi_dot([j75,j60,j78]),
        j0,
        B(j79,self.ubar_rbl_upper_strut_jcl_strut_chassis),
        j2,
        -1*B(j7,self.ubar_vbs_chassis_jcl_strut_chassis),
        j4,
        multi_dot([j80.T,j8,B(j79,j81)]),
        j4,
        multi_dot([j81.T,j82,B(j7,j80)]),
        j4,
        multi_dot([j84,j86,j88]),
        j4,
        multi_dot([j91,j82,j99]),
        j4,
        multi_dot([j84,j86,j90]),
        j4,
        multi_dot([j97,j82,j99]),
        j92,
        (multi_dot([j91,j82,j94]) + multi_dot([j96,j88])),
        -1*j92,
        -1*multi_dot([j91,j82,j100]),
        j98,
        (multi_dot([j97,j82,j94]) + multi_dot([j96,j90])),
        -1*j98,
        -1*multi_dot([j97,j82,j100]),
        j2,
        -1*B(j21,self.ubar_rbr_lca_jcr_strut_lca),
        j0,
        B(j63,self.ubar_rbr_lower_strut_jcr_strut_lca),
        j4,
        multi_dot([j102.T,j64,B(j21,j101)]),
        j4,
        multi_dot([j101.T,j26,B(j63,j102)]),
        j2,
        -1*B(j28,self.ubar_rbl_lca_jcl_strut_lca),
        j0,
        B(j85,self.ubar_rbl_lower_strut_jcl_strut_lca),
        j4,
        multi_dot([j104.T,j86,B(j28,j103)]),
        j4,
        multi_dot([j103.T,j33,B(j85,j104)]),
        j2,
        -1*B(j3,self.ubar_rbr_upright_jcr_tie_upright),
        j0,
        B(j105,self.ubar_rbr_tie_rod_jcr_tie_upright),
        j0,
        B(j105,self.ubar_rbr_tie_rod_jcr_tie_steering),
        j2,
        -1*B(j107,self.ubar_vbr_steer_jcr_tie_steering),
        j4,
        multi_dot([j106.T,A(j107).T,B(j105,j108)]),
        j4,
        multi_dot([j108.T,A(j105).T,B(j107,j106)]),
        j2,
        -1*B(j14,self.ubar_rbl_upright_jcl_tie_upright),
        j0,
        B(j109,self.ubar_rbl_tie_rod_jcl_tie_upright),
        j0,
        B(j109,self.ubar_rbl_tie_rod_jcl_tie_steering),
        j2,
        -1*B(j111,self.ubar_vbl_steer_jcl_tie_steering),
        j4,
        multi_dot([j110.T,A(j111).T,B(j109,j112)]),
        j4,
        multi_dot([j112.T,A(j109).T,B(j111,j110)]),
        2*j1.T,
        2*j13.T,
        2*j21.T,
        2*j28.T,
        2*j3.T,
        2*j14.T,
        2*j57.T,
        2*j79.T,
        2*j63.T,
        2*j85.T,
        2*j105.T,
        2*j109.T,
        2*j37.T,
        2*j48.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3,dtype=np.float64)
        m1 = G(self.P_rbr_uca)
        m2 = G(self.P_rbl_uca)
        m3 = G(self.P_rbr_lca)
        m4 = G(self.P_rbl_lca)
        m5 = G(self.P_rbr_upright)
        m6 = G(self.P_rbl_upright)
        m7 = G(self.P_rbr_upper_strut)
        m8 = G(self.P_rbl_upper_strut)
        m9 = G(self.P_rbr_lower_strut)
        m10 = G(self.P_rbl_lower_strut)
        m11 = G(self.P_rbr_tie_rod)
        m12 = G(self.P_rbl_tie_rod)
        m13 = G(self.P_rbr_hub)
        m14 = G(self.P_rbl_hub)

        self.mass_eq_blocks = [config.m_rbr_uca*m0,
        4*multi_dot([m1.T,config.Jbar_rbr_uca,m1]),
        config.m_rbl_uca*m0,
        4*multi_dot([m2.T,config.Jbar_rbl_uca,m2]),
        config.m_rbr_lca*m0,
        4*multi_dot([m3.T,config.Jbar_rbr_lca,m3]),
        config.m_rbl_lca*m0,
        4*multi_dot([m4.T,config.Jbar_rbl_lca,m4]),
        config.m_rbr_upright*m0,
        4*multi_dot([m5.T,config.Jbar_rbr_upright,m5]),
        config.m_rbl_upright*m0,
        4*multi_dot([m6.T,config.Jbar_rbl_upright,m6]),
        config.m_rbr_upper_strut*m0,
        4*multi_dot([m7.T,config.Jbar_rbr_upper_strut,m7]),
        config.m_rbl_upper_strut*m0,
        4*multi_dot([m8.T,config.Jbar_rbl_upper_strut,m8]),
        config.m_rbr_lower_strut*m0,
        4*multi_dot([m9.T,config.Jbar_rbr_lower_strut,m9]),
        config.m_rbl_lower_strut*m0,
        4*multi_dot([m10.T,config.Jbar_rbl_lower_strut,m10]),
        config.m_rbr_tie_rod*m0,
        4*multi_dot([m11.T,config.Jbar_rbr_tie_rod,m11]),
        config.m_rbl_tie_rod*m0,
        4*multi_dot([m12.T,config.Jbar_rbl_tie_rod,m12]),
        config.m_rbr_hub*m0,
        4*multi_dot([m13.T,config.Jbar_rbr_hub,m13]),
        config.m_rbl_hub*m0,
        4*multi_dot([m14.T,config.Jbar_rbl_hub,m14])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbr_uca)
        f1 = G(self.Pd_rbl_uca)
        f2 = G(self.Pd_rbr_lca)
        f3 = G(self.Pd_rbl_lca)
        f4 = G(self.Pd_rbr_upright)
        f5 = G(self.Pd_rbl_upright)
        f6 = self.R_rbr_upper_strut
        f7 = self.R_rbr_lower_strut
        f8 = self.ubar_rbr_upper_strut_far_strut
        f9 = self.P_rbr_upper_strut
        f10 = A(f9)
        f11 = self.ubar_rbr_lower_strut_far_strut
        f12 = self.P_rbr_lower_strut
        f13 = A(f12)
        f14 = (f6.T + -1*f7.T + multi_dot([f8.T,f10.T]) + -1*multi_dot([f11.T,f13.T]))
        f15 = multi_dot([f10,f8])
        f16 = multi_dot([f13,f11])
        f17 = (f6 + -1*f7 + f15 + -1*f16)
        f18 = (multi_dot([f14,f17]))**(1.0/2.0)
        f19 = f18**(-1)
        f20 = self.Pd_rbr_upper_strut
        f21 = self.Pd_rbr_lower_strut
        f22 = config.Fd_far_strut(multi_dot([f19.T,f14,(self.Rd_rbr_upper_strut + -1*self.Rd_rbr_lower_strut + multi_dot([B(f9,f8),f20]) + -1*multi_dot([B(f12,f11),f21]))])) + config.Fs_far_strut((config.far_strut_FL + -1*f18))
        f23 = f22*multi_dot([f17,f19])
        f24 = G(f20)
        f25 = self.R_rbl_upper_strut
        f26 = self.R_rbl_lower_strut
        f27 = self.ubar_rbl_upper_strut_fal_strut
        f28 = self.P_rbl_upper_strut
        f29 = A(f28)
        f30 = self.ubar_rbl_lower_strut_fal_strut
        f31 = self.P_rbl_lower_strut
        f32 = A(f31)
        f33 = (f25.T + -1*f26.T + multi_dot([f27.T,f29.T]) + -1*multi_dot([f30.T,f32.T]))
        f34 = multi_dot([f29,f27])
        f35 = multi_dot([f32,f30])
        f36 = (f25 + -1*f26 + f34 + -1*f35)
        f37 = (multi_dot([f33,f36]))**(1.0/2.0)
        f38 = f37**(-1)
        f39 = self.Pd_rbl_upper_strut
        f40 = self.Pd_rbl_lower_strut
        f41 = config.Fd_fal_strut(multi_dot([f38.T,f33,(self.Rd_rbl_upper_strut + -1*self.Rd_rbl_lower_strut + multi_dot([B(f28,f27),f39]) + -1*multi_dot([B(f31,f30),f40]))])) + config.Fs_fal_strut((config.fal_strut_FL + -1*f37))
        f42 = f41*multi_dot([f36,f38])
        f43 = G(f39)
        f44 = np.zeros((3,1),dtype=np.float64)
        f45 = np.zeros((4,1),dtype=np.float64)
        f46 = G(f21)
        f47 = G(f40)
        f48 = G(self.Pd_rbr_tie_rod)
        f49 = G(self.Pd_rbl_tie_rod)
        f50 = G(self.Pd_rbr_hub)
        f51 = G(self.Pd_rbl_hub)

        self.frc_eq_blocks = [self.F_rbr_uca_gravity,
        8*multi_dot([f0.T,config.Jbar_rbr_uca,f0,self.P_rbr_uca]),
        self.F_rbl_uca_gravity,
        8*multi_dot([f1.T,config.Jbar_rbl_uca,f1,self.P_rbl_uca]),
        self.F_rbr_lca_gravity,
        8*multi_dot([f2.T,config.Jbar_rbr_lca,f2,self.P_rbr_lca]),
        self.F_rbl_lca_gravity,
        8*multi_dot([f3.T,config.Jbar_rbl_lca,f3,self.P_rbl_lca]),
        self.F_rbr_upright_gravity,
        8*multi_dot([f4.T,config.Jbar_rbr_upright,f4,self.P_rbr_upright]),
        self.F_rbl_upright_gravity,
        8*multi_dot([f5.T,config.Jbar_rbl_upright,f5,self.P_rbl_upright]),
        (self.F_rbr_upper_strut_gravity + f23),
        (8*multi_dot([f24.T,config.Jbar_rbr_upper_strut,f24,f9]) + 2*multi_dot([G(f9).T,(config.T_rbr_upper_strut_far_strut + f22*multi_dot([skew(f15).T,f17,f19]))])),
        (self.F_rbl_upper_strut_gravity + f42),
        (8*multi_dot([f43.T,config.Jbar_rbl_upper_strut,f43,f28]) + 2*multi_dot([G(f28).T,(config.T_rbl_upper_strut_fal_strut + f41*multi_dot([skew(f34).T,f36,f38]))])),
        (self.F_rbr_lower_strut_gravity + f44 + -1*f23),
        (f45 + 8*multi_dot([f46.T,config.Jbar_rbr_lower_strut,f46,f12]) + 2*multi_dot([G(f12).T,(config.T_rbr_lower_strut_far_strut + -1*f22*multi_dot([skew(f16).T,f17,f19]))])),
        (self.F_rbl_lower_strut_gravity + f44 + -1*f42),
        (f45 + 8*multi_dot([f47.T,config.Jbar_rbl_lower_strut,f47,f31]) + 2*multi_dot([G(f31).T,(config.T_rbl_lower_strut_fal_strut + -1*f41*multi_dot([skew(f35).T,f36,f38]))])),
        self.F_rbr_tie_rod_gravity,
        8*multi_dot([f48.T,config.Jbar_rbr_tie_rod,f48,self.P_rbr_tie_rod]),
        self.F_rbl_tie_rod_gravity,
        8*multi_dot([f49.T,config.Jbar_rbl_tie_rod,f49,self.P_rbl_tie_rod]),
        self.F_rbr_hub_gravity,
        8*multi_dot([f50.T,config.Jbar_rbr_hub,f50,self.P_rbr_hub]),
        self.F_rbl_hub_gravity,
        8*multi_dot([f51.T,config.Jbar_rbl_hub,f51,self.P_rbl_hub])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_uca_jcr_uca_upright = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_upright).T]]),self.L_jcr_uca_upright])
        self.F_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[0:3,0:1]
        Te_rbr_uca_jcr_uca_upright = Q_rbr_uca_jcr_uca_upright[3:7,0:1]
        self.T_rbr_uca_jcr_uca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_upright])),self.F_rbr_uca_jcr_uca_upright]) + 0.5*multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_upright]))
        Q_rbr_uca_jcr_uca_chassis = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_uca,self.ubar_rbr_uca_jcr_uca_chassis).T,multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_uca,self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]])]]),self.L_jcr_uca_chassis])
        self.F_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[0:3,0:1]
        Te_rbr_uca_jcr_uca_chassis = Q_rbr_uca_jcr_uca_chassis[3:7,0:1]
        self.T_rbr_uca_jcr_uca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_uca),self.ubar_rbr_uca_jcr_uca_chassis])),self.F_rbr_uca_jcr_uca_chassis]) + 0.5*multi_dot([E(self.P_rbr_uca),Te_rbr_uca_jcr_uca_chassis]))
        Q_rbl_uca_jcl_uca_upright = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_upright).T]]),self.L_jcl_uca_upright])
        self.F_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[0:3,0:1]
        Te_rbl_uca_jcl_uca_upright = Q_rbl_uca_jcl_uca_upright[3:7,0:1]
        self.T_rbl_uca_jcl_uca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_upright])),self.F_rbl_uca_jcl_uca_upright]) + 0.5*multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_upright]))
        Q_rbl_uca_jcl_uca_chassis = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_uca,self.ubar_rbl_uca_jcl_uca_chassis).T,multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_uca,self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]])]]),self.L_jcl_uca_chassis])
        self.F_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[0:3,0:1]
        Te_rbl_uca_jcl_uca_chassis = Q_rbl_uca_jcl_uca_chassis[3:7,0:1]
        self.T_rbl_uca_jcl_uca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_uca),self.ubar_rbl_uca_jcl_uca_chassis])),self.F_rbl_uca_jcl_uca_chassis]) + 0.5*multi_dot([E(self.P_rbl_uca),Te_rbl_uca_jcl_uca_chassis]))
        Q_rbr_lca_jcr_lca_upright = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_upright).T]]),self.L_jcr_lca_upright])
        self.F_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[0:3,0:1]
        Te_rbr_lca_jcr_lca_upright = Q_rbr_lca_jcr_lca_upright[3:7,0:1]
        self.T_rbr_lca_jcr_lca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_upright])),self.F_rbr_lca_jcr_lca_upright]) + 0.5*multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_upright]))
        Q_rbr_lca_jcr_lca_chassis = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_lca,self.ubar_rbr_lca_jcr_lca_chassis).T,multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbr_lca,self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]])]]),self.L_jcr_lca_chassis])
        self.F_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[0:3,0:1]
        Te_rbr_lca_jcr_lca_chassis = Q_rbr_lca_jcr_lca_chassis[3:7,0:1]
        self.T_rbr_lca_jcr_lca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_lca),self.ubar_rbr_lca_jcr_lca_chassis])),self.F_rbr_lca_jcr_lca_chassis]) + 0.5*multi_dot([E(self.P_rbr_lca),Te_rbr_lca_jcr_lca_chassis]))
        Q_rbl_lca_jcl_lca_upright = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_upright).T]]),self.L_jcl_lca_upright])
        self.F_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[0:3,0:1]
        Te_rbl_lca_jcl_lca_upright = Q_rbl_lca_jcl_lca_upright[3:7,0:1]
        self.T_rbl_lca_jcl_lca_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_upright])),self.F_rbl_lca_jcl_lca_upright]) + 0.5*multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_upright]))
        Q_rbl_lca_jcl_lca_chassis = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_lca,self.ubar_rbl_lca_jcl_lca_chassis).T,multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]]),multi_dot([B(self.P_rbl_lca,self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]])]]),self.L_jcl_lca_chassis])
        self.F_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[0:3,0:1]
        Te_rbl_lca_jcl_lca_chassis = Q_rbl_lca_jcl_lca_chassis[3:7,0:1]
        self.T_rbl_lca_jcl_lca_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_lca),self.ubar_rbl_lca_jcl_lca_chassis])),self.F_rbl_lca_jcl_lca_chassis]) + 0.5*multi_dot([E(self.P_rbl_lca),Te_rbl_lca_jcl_lca_chassis]))
        Q_rbr_upright_jcr_hub_bearing = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_upright,self.ubar_rbr_upright_jcr_hub_bearing).T,multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbr_upright,self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]).T,A(self.P_rbr_hub),self.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]])]]),self.L_jcr_hub_bearing])
        self.F_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[0:3,0:1]
        Te_rbr_upright_jcr_hub_bearing = Q_rbr_upright_jcr_hub_bearing[3:7,0:1]
        self.T_rbr_upright_jcr_hub_bearing = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_upright),self.ubar_rbr_upright_jcr_hub_bearing])),self.F_rbr_upright_jcr_hub_bearing]) + 0.5*multi_dot([E(self.P_rbr_upright),Te_rbr_upright_jcr_hub_bearing]))
        Q_rbl_upright_jcl_hub_bearing = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_upright,self.ubar_rbl_upright_jcl_hub_bearing).T,multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]]),multi_dot([B(self.P_rbl_upright,self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]).T,A(self.P_rbl_hub),self.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]])]]),self.L_jcl_hub_bearing])
        self.F_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[0:3,0:1]
        Te_rbl_upright_jcl_hub_bearing = Q_rbl_upright_jcl_hub_bearing[3:7,0:1]
        self.T_rbl_upright_jcl_hub_bearing = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_upright),self.ubar_rbl_upright_jcl_hub_bearing])),self.F_rbl_upright_jcl_hub_bearing]) + 0.5*multi_dot([E(self.P_rbl_upright),Te_rbl_upright_jcl_hub_bearing]))
        Q_rbr_upper_strut_jcr_strut_chassis = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut_chassis).T,multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]])]]),self.L_jcr_strut_chassis])
        self.F_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[0:3,0:1]
        Te_rbr_upper_strut_jcr_strut_chassis = Q_rbr_upper_strut_jcr_strut_chassis[3:7,0:1]
        self.T_rbr_upper_strut_jcr_strut_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut_chassis])),self.F_rbr_upper_strut_jcr_strut_chassis]) + 0.5*multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut_chassis]))
        Q_rbr_upper_strut_jcr_strut = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]]),multi_dot([A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]])],[multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,A(self.P_rbr_lower_strut),self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]).T,(-1*self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]])),(multi_dot([B(self.P_rbr_upper_strut,self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]).T,(-1*self.R_rbr_lower_strut + multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut]) + self.R_rbr_upper_strut)]) + multi_dot([B(self.P_rbr_upper_strut,self.ubar_rbr_upper_strut_jcr_strut).T,A(self.P_rbr_upper_strut),self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]]))]]),self.L_jcr_strut])
        self.F_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[0:3,0:1]
        Te_rbr_upper_strut_jcr_strut = Q_rbr_upper_strut_jcr_strut[3:7,0:1]
        self.T_rbr_upper_strut_jcr_strut = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_upper_strut),self.ubar_rbr_upper_strut_jcr_strut])),self.F_rbr_upper_strut_jcr_strut]) + 0.5*multi_dot([E(self.P_rbr_upper_strut),Te_rbr_upper_strut_jcr_strut]))
        Q_rbl_upper_strut_jcl_strut_chassis = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut_chassis).T,multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]])]]),self.L_jcl_strut_chassis])
        self.F_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[0:3,0:1]
        Te_rbl_upper_strut_jcl_strut_chassis = Q_rbl_upper_strut_jcl_strut_chassis[3:7,0:1]
        self.T_rbl_upper_strut_jcl_strut_chassis = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut_chassis])),self.F_rbl_upper_strut_jcl_strut_chassis]) + 0.5*multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut_chassis]))
        Q_rbl_upper_strut_jcl_strut = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]]),multi_dot([A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]])],[multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,A(self.P_rbl_lower_strut),self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]).T,(-1*self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]])),(multi_dot([B(self.P_rbl_upper_strut,self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]).T,(-1*self.R_rbl_lower_strut + multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut]) + self.R_rbl_upper_strut)]) + multi_dot([B(self.P_rbl_upper_strut,self.ubar_rbl_upper_strut_jcl_strut).T,A(self.P_rbl_upper_strut),self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]]))]]),self.L_jcl_strut])
        self.F_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[0:3,0:1]
        Te_rbl_upper_strut_jcl_strut = Q_rbl_upper_strut_jcl_strut[3:7,0:1]
        self.T_rbl_upper_strut_jcl_strut = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_upper_strut),self.ubar_rbl_upper_strut_jcl_strut])),self.F_rbl_upper_strut_jcl_strut]) + 0.5*multi_dot([E(self.P_rbl_upper_strut),Te_rbl_upper_strut_jcl_strut]))
        Q_rbr_lower_strut_jcr_strut_lca = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_lower_strut,self.ubar_rbr_lower_strut_jcr_strut_lca).T,multi_dot([B(self.P_rbr_lower_strut,self.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]).T,A(self.P_rbr_lca),self.Mbar_rbr_lca_jcr_strut_lca[:,0:1]])]]),self.L_jcr_strut_lca])
        self.F_rbr_lower_strut_jcr_strut_lca = Q_rbr_lower_strut_jcr_strut_lca[0:3,0:1]
        Te_rbr_lower_strut_jcr_strut_lca = Q_rbr_lower_strut_jcr_strut_lca[3:7,0:1]
        self.T_rbr_lower_strut_jcr_strut_lca = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_lower_strut),self.ubar_rbr_lower_strut_jcr_strut_lca])),self.F_rbr_lower_strut_jcr_strut_lca]) + 0.5*multi_dot([E(self.P_rbr_lower_strut),Te_rbr_lower_strut_jcr_strut_lca]))
        Q_rbl_lower_strut_jcl_strut_lca = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_lower_strut,self.ubar_rbl_lower_strut_jcl_strut_lca).T,multi_dot([B(self.P_rbl_lower_strut,self.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]).T,A(self.P_rbl_lca),self.Mbar_rbl_lca_jcl_strut_lca[:,0:1]])]]),self.L_jcl_strut_lca])
        self.F_rbl_lower_strut_jcl_strut_lca = Q_rbl_lower_strut_jcl_strut_lca[0:3,0:1]
        Te_rbl_lower_strut_jcl_strut_lca = Q_rbl_lower_strut_jcl_strut_lca[3:7,0:1]
        self.T_rbl_lower_strut_jcl_strut_lca = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_lower_strut),self.ubar_rbl_lower_strut_jcl_strut_lca])),self.F_rbl_lower_strut_jcl_strut_lca]) + 0.5*multi_dot([E(self.P_rbl_lower_strut),Te_rbl_lower_strut_jcl_strut_lca]))
        Q_rbr_tie_rod_jcr_tie_upright = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_upright).T]]),self.L_jcr_tie_upright])
        self.F_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[0:3,0:1]
        Te_rbr_tie_rod_jcr_tie_upright = Q_rbr_tie_rod_jcr_tie_upright[3:7,0:1]
        self.T_rbr_tie_rod_jcr_tie_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_upright])),self.F_rbr_tie_rod_jcr_tie_upright]) + 0.5*multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_upright]))
        Q_rbr_tie_rod_jcr_tie_steering = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_tie_rod,self.ubar_rbr_tie_rod_jcr_tie_steering).T,multi_dot([B(self.P_rbr_tie_rod,self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]).T,A(self.P_vbr_steer),self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]])]]),self.L_jcr_tie_steering])
        self.F_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[0:3,0:1]
        Te_rbr_tie_rod_jcr_tie_steering = Q_rbr_tie_rod_jcr_tie_steering[3:7,0:1]
        self.T_rbr_tie_rod_jcr_tie_steering = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_tie_rod),self.ubar_rbr_tie_rod_jcr_tie_steering])),self.F_rbr_tie_rod_jcr_tie_steering]) + 0.5*multi_dot([E(self.P_rbr_tie_rod),Te_rbr_tie_rod_jcr_tie_steering]))
        Q_rbl_tie_rod_jcl_tie_upright = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_upright).T]]),self.L_jcl_tie_upright])
        self.F_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[0:3,0:1]
        Te_rbl_tie_rod_jcl_tie_upright = Q_rbl_tie_rod_jcl_tie_upright[3:7,0:1]
        self.T_rbl_tie_rod_jcl_tie_upright = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_upright])),self.F_rbl_tie_rod_jcl_tie_upright]) + 0.5*multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_upright]))
        Q_rbl_tie_rod_jcl_tie_steering = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_tie_rod,self.ubar_rbl_tie_rod_jcl_tie_steering).T,multi_dot([B(self.P_rbl_tie_rod,self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]).T,A(self.P_vbl_steer),self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]])]]),self.L_jcl_tie_steering])
        self.F_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[0:3,0:1]
        Te_rbl_tie_rod_jcl_tie_steering = Q_rbl_tie_rod_jcl_tie_steering[3:7,0:1]
        self.T_rbl_tie_rod_jcl_tie_steering = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_tie_rod),self.ubar_rbl_tie_rod_jcl_tie_steering])),self.F_rbl_tie_rod_jcl_tie_steering]) + 0.5*multi_dot([E(self.P_rbl_tie_rod),Te_rbl_tie_rod_jcl_tie_steering]))

        self.reactions = {'F_rbr_uca_jcr_uca_upright':self.F_rbr_uca_jcr_uca_upright,'T_rbr_uca_jcr_uca_upright':self.T_rbr_uca_jcr_uca_upright,'F_rbr_uca_jcr_uca_chassis':self.F_rbr_uca_jcr_uca_chassis,'T_rbr_uca_jcr_uca_chassis':self.T_rbr_uca_jcr_uca_chassis,'F_rbl_uca_jcl_uca_upright':self.F_rbl_uca_jcl_uca_upright,'T_rbl_uca_jcl_uca_upright':self.T_rbl_uca_jcl_uca_upright,'F_rbl_uca_jcl_uca_chassis':self.F_rbl_uca_jcl_uca_chassis,'T_rbl_uca_jcl_uca_chassis':self.T_rbl_uca_jcl_uca_chassis,'F_rbr_lca_jcr_lca_upright':self.F_rbr_lca_jcr_lca_upright,'T_rbr_lca_jcr_lca_upright':self.T_rbr_lca_jcr_lca_upright,'F_rbr_lca_jcr_lca_chassis':self.F_rbr_lca_jcr_lca_chassis,'T_rbr_lca_jcr_lca_chassis':self.T_rbr_lca_jcr_lca_chassis,'F_rbl_lca_jcl_lca_upright':self.F_rbl_lca_jcl_lca_upright,'T_rbl_lca_jcl_lca_upright':self.T_rbl_lca_jcl_lca_upright,'F_rbl_lca_jcl_lca_chassis':self.F_rbl_lca_jcl_lca_chassis,'T_rbl_lca_jcl_lca_chassis':self.T_rbl_lca_jcl_lca_chassis,'F_rbr_upright_jcr_hub_bearing':self.F_rbr_upright_jcr_hub_bearing,'T_rbr_upright_jcr_hub_bearing':self.T_rbr_upright_jcr_hub_bearing,'F_rbl_upright_jcl_hub_bearing':self.F_rbl_upright_jcl_hub_bearing,'T_rbl_upright_jcl_hub_bearing':self.T_rbl_upright_jcl_hub_bearing,'F_rbr_upper_strut_jcr_strut_chassis':self.F_rbr_upper_strut_jcr_strut_chassis,'T_rbr_upper_strut_jcr_strut_chassis':self.T_rbr_upper_strut_jcr_strut_chassis,'F_rbr_upper_strut_jcr_strut':self.F_rbr_upper_strut_jcr_strut,'T_rbr_upper_strut_jcr_strut':self.T_rbr_upper_strut_jcr_strut,'F_rbl_upper_strut_jcl_strut_chassis':self.F_rbl_upper_strut_jcl_strut_chassis,'T_rbl_upper_strut_jcl_strut_chassis':self.T_rbl_upper_strut_jcl_strut_chassis,'F_rbl_upper_strut_jcl_strut':self.F_rbl_upper_strut_jcl_strut,'T_rbl_upper_strut_jcl_strut':self.T_rbl_upper_strut_jcl_strut,'F_rbr_lower_strut_jcr_strut_lca':self.F_rbr_lower_strut_jcr_strut_lca,'T_rbr_lower_strut_jcr_strut_lca':self.T_rbr_lower_strut_jcr_strut_lca,'F_rbl_lower_strut_jcl_strut_lca':self.F_rbl_lower_strut_jcl_strut_lca,'T_rbl_lower_strut_jcl_strut_lca':self.T_rbl_lower_strut_jcl_strut_lca,'F_rbr_tie_rod_jcr_tie_upright':self.F_rbr_tie_rod_jcr_tie_upright,'T_rbr_tie_rod_jcr_tie_upright':self.T_rbr_tie_rod_jcr_tie_upright,'F_rbr_tie_rod_jcr_tie_steering':self.F_rbr_tie_rod_jcr_tie_steering,'T_rbr_tie_rod_jcr_tie_steering':self.T_rbr_tie_rod_jcr_tie_steering,'F_rbl_tie_rod_jcl_tie_upright':self.F_rbl_tie_rod_jcl_tie_upright,'T_rbl_tie_rod_jcl_tie_upright':self.T_rbl_tie_rod_jcl_tie_upright,'F_rbl_tie_rod_jcl_tie_steering':self.F_rbl_tie_rod_jcl_tie_steering,'T_rbl_tie_rod_jcl_tie_steering':self.T_rbl_tie_rod_jcl_tie_steering}

