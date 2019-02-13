
import numpy as np
import scipy as sc
import pandas as pd
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad
from source.solvers.py_numerical_functions import mirrored, centered, oriented




class configuration(object):

    def __init__(self):
        self.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_uca = 1
        self.Jbar_rbr_uca = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_lca = 1
        self.Jbar_rbr_lca = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_upright = 1
        self.Jbar_rbr_upright = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_upper_strut = 1
        self.Jbar_rbr_upper_strut = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_lower_strut = 1
        self.Jbar_rbr_lower_strut = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_tie_rod = 1
        self.Jbar_rbr_tie_rod = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_hub = 1
        self.Jbar_rbr_hub = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt2_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Fs_far_strut = lambda t : 0
        self.Fd_far_strut = lambda t : 0
        self.ax1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    def load_from_csv(self,csv_file):
        dataframe = pd.read_csv(csv_file,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def _set_arguments(self):
        self.R_rbl_uca = mirrored(self.R_rbr_uca)
        self.P_rbl_uca = mirrored(self.P_rbr_uca)
        self.Rd_rbl_uca = mirrored(self.Rd_rbr_uca)
        self.Pd_rbl_uca = mirrored(self.Pd_rbr_uca)
        self.m_rbl_uca = self.m_rbr_uca
        self.Jbar_rbl_uca = mirrored(self.Jbar_rbr_uca)
        self.R_rbl_lca = mirrored(self.R_rbr_lca)
        self.P_rbl_lca = mirrored(self.P_rbr_lca)
        self.Rd_rbl_lca = mirrored(self.Rd_rbr_lca)
        self.Pd_rbl_lca = mirrored(self.Pd_rbr_lca)
        self.m_rbl_lca = self.m_rbr_lca
        self.Jbar_rbl_lca = mirrored(self.Jbar_rbr_lca)
        self.R_rbl_upright = mirrored(self.R_rbr_upright)
        self.P_rbl_upright = mirrored(self.P_rbr_upright)
        self.Rd_rbl_upright = mirrored(self.Rd_rbr_upright)
        self.Pd_rbl_upright = mirrored(self.Pd_rbr_upright)
        self.m_rbl_upright = self.m_rbr_upright
        self.Jbar_rbl_upright = mirrored(self.Jbar_rbr_upright)
        self.R_rbl_upper_strut = mirrored(self.R_rbr_upper_strut)
        self.P_rbl_upper_strut = mirrored(self.P_rbr_upper_strut)
        self.Rd_rbl_upper_strut = mirrored(self.Rd_rbr_upper_strut)
        self.Pd_rbl_upper_strut = mirrored(self.Pd_rbr_upper_strut)
        self.m_rbl_upper_strut = self.m_rbr_upper_strut
        self.Jbar_rbl_upper_strut = mirrored(self.Jbar_rbr_upper_strut)
        self.R_rbl_lower_strut = mirrored(self.R_rbr_lower_strut)
        self.P_rbl_lower_strut = mirrored(self.P_rbr_lower_strut)
        self.Rd_rbl_lower_strut = mirrored(self.Rd_rbr_lower_strut)
        self.Pd_rbl_lower_strut = mirrored(self.Pd_rbr_lower_strut)
        self.m_rbl_lower_strut = self.m_rbr_lower_strut
        self.Jbar_rbl_lower_strut = mirrored(self.Jbar_rbr_lower_strut)
        self.R_rbl_tie_rod = mirrored(self.R_rbr_tie_rod)
        self.P_rbl_tie_rod = mirrored(self.P_rbr_tie_rod)
        self.Rd_rbl_tie_rod = mirrored(self.Rd_rbr_tie_rod)
        self.Pd_rbl_tie_rod = mirrored(self.Pd_rbr_tie_rod)
        self.m_rbl_tie_rod = self.m_rbr_tie_rod
        self.Jbar_rbl_tie_rod = mirrored(self.Jbar_rbr_tie_rod)
        self.R_rbl_hub = mirrored(self.R_rbr_hub)
        self.P_rbl_hub = mirrored(self.P_rbr_hub)
        self.Rd_rbl_hub = mirrored(self.Rd_rbr_hub)
        self.Pd_rbl_hub = mirrored(self.Pd_rbr_hub)
        self.m_rbl_hub = self.m_rbr_hub
        self.Jbar_rbl_hub = mirrored(self.Jbar_rbr_hub)
        self.ax1_jcl_uca_upright = mirrored(self.ax1_jcr_uca_upright)
        self.pt1_jcl_uca_upright = mirrored(self.pt1_jcr_uca_upright)
        self.ax1_jcl_uca_chassis = mirrored(self.ax1_jcr_uca_chassis)
        self.pt1_jcl_uca_chassis = mirrored(self.pt1_jcr_uca_chassis)
        self.ax1_jcl_lca_upright = mirrored(self.ax1_jcr_lca_upright)
        self.pt1_jcl_lca_upright = mirrored(self.pt1_jcr_lca_upright)
        self.ax1_jcl_lca_chassis = mirrored(self.ax1_jcr_lca_chassis)
        self.pt1_jcl_lca_chassis = mirrored(self.pt1_jcr_lca_chassis)
        self.ax1_jcl_hub_bearing = mirrored(self.ax1_jcr_hub_bearing)
        self.pt1_jcl_hub_bearing = mirrored(self.pt1_jcr_hub_bearing)
        self.ax1_jcl_strut_chassis = mirrored(self.ax1_jcr_strut_chassis)
        self.ax2_jcl_strut_chassis = mirrored(self.ax2_jcr_strut_chassis)
        self.pt1_jcl_strut_chassis = mirrored(self.pt1_jcr_strut_chassis)
        self.ax1_jcl_strut = mirrored(self.ax1_jcr_strut)
        self.pt1_jcl_strut = mirrored(self.pt1_jcr_strut)
        self.pt1_fal_strut = mirrored(self.pt1_far_strut)
        self.pt2_fal_strut = mirrored(self.pt2_far_strut)
        self.Fs_fal_strut = self.Fs_far_strut
        self.Fd_fal_strut = self.Fd_far_strut
        self.ax1_jcl_strut_lca = mirrored(self.ax1_jcr_strut_lca)
        self.ax2_jcl_strut_lca = mirrored(self.ax2_jcr_strut_lca)
        self.pt1_jcl_strut_lca = mirrored(self.pt1_jcr_strut_lca)
        self.ax1_jcl_tie_upright = mirrored(self.ax1_jcr_tie_upright)
        self.pt1_jcl_tie_upright = mirrored(self.pt1_jcr_tie_upright)
        self.ax1_jcl_tie_steering = mirrored(self.ax1_jcr_tie_steering)
        self.ax2_jcl_tie_steering = mirrored(self.ax2_jcr_tie_steering)
        self.pt1_jcl_tie_steering = mirrored(self.pt1_jcr_tie_steering)





class topology(object):

    def __init__(self,config=configuration(),prefix=''):
        self.t = 0.0
        self.config = config
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 98
        self.nrows = 58
        self.ncols = 2*18
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57])                        

    
    def _set_mapping(self,indicies_map,interface_map):
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
        self.vbl_steer = indicies_map[interface_map[p+'vbl_steer']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbr_steer = indicies_map[interface_map[p+'vbr_steer']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_uca*2,self.rbr_uca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbr_uca*2+1,self.rbl_uca*2+1,self.rbr_lca*2+1,self.rbl_lca*2+1,self.rbr_upright*2+1,self.rbl_upright*2+1,self.rbr_upper_strut*2+1,self.rbl_upper_strut*2+1,self.rbr_lower_strut*2+1,self.rbl_lower_strut*2+1,self.rbr_tie_rod*2+1,self.rbl_tie_rod*2+1,self.rbr_hub*2+1,self.rbl_hub*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        c0 = A(config.P_rbr_uca).T
        c1 = triad(config.ax1_jcr_uca_upright)
        c2 = A(config.P_rbr_upright).T
        c3 = config.pt1_jcr_uca_upright
        c4 = -1*multi_dot([c0,config.R_rbr_uca])
        c5 = -1*multi_dot([c2,config.R_rbr_upright])
        c6 = triad(config.ax1_jcr_uca_chassis)
        c7 = A(config.P_vbs_chassis).T
        c8 = config.pt1_jcr_uca_chassis
        c9 = -1*multi_dot([c7,config.R_vbs_chassis])
        c10 = A(config.P_rbl_uca).T
        c11 = triad(config.ax1_jcl_uca_upright)
        c12 = A(config.P_rbl_upright).T
        c13 = config.pt1_jcl_uca_upright
        c14 = -1*multi_dot([c10,config.R_rbl_uca])
        c15 = -1*multi_dot([c12,config.R_rbl_upright])
        c16 = triad(config.ax1_jcl_uca_chassis)
        c17 = config.pt1_jcl_uca_chassis
        c18 = A(config.P_rbr_lca).T
        c19 = triad(config.ax1_jcr_lca_upright)
        c20 = config.pt1_jcr_lca_upright
        c21 = -1*multi_dot([c18,config.R_rbr_lca])
        c22 = triad(config.ax1_jcr_lca_chassis)
        c23 = config.pt1_jcr_lca_chassis
        c24 = A(config.P_rbl_lca).T
        c25 = triad(config.ax1_jcl_lca_upright)
        c26 = config.pt1_jcl_lca_upright
        c27 = -1*multi_dot([c24,config.R_rbl_lca])
        c28 = triad(config.ax1_jcl_lca_chassis)
        c29 = config.pt1_jcl_lca_chassis
        c30 = triad(config.ax1_jcr_hub_bearing)
        c31 = A(config.P_rbr_hub).T
        c32 = config.pt1_jcr_hub_bearing
        c33 = triad(config.ax1_jcl_hub_bearing)
        c34 = A(config.P_rbl_hub).T
        c35 = config.pt1_jcl_hub_bearing
        c36 = A(config.P_rbr_upper_strut).T
        c37 = triad(config.ax1_jcr_strut_chassis)
        c38 = config.pt1_jcr_strut_chassis
        c39 = -1*multi_dot([c36,config.R_rbr_upper_strut])
        c40 = triad(config.ax1_jcr_strut)
        c41 = A(config.P_rbr_lower_strut).T
        c42 = config.pt1_jcr_strut
        c43 = -1*multi_dot([c41,config.R_rbr_lower_strut])
        c44 = A(config.P_rbl_upper_strut).T
        c45 = triad(config.ax1_jcl_strut_chassis)
        c46 = config.pt1_jcl_strut_chassis
        c47 = -1*multi_dot([c44,config.R_rbl_upper_strut])
        c48 = triad(config.ax1_jcl_strut)
        c49 = A(config.P_rbl_lower_strut).T
        c50 = config.pt1_jcl_strut
        c51 = -1*multi_dot([c49,config.R_rbl_lower_strut])
        c52 = triad(config.ax1_jcr_strut_lca)
        c53 = config.pt1_jcr_strut_lca
        c54 = triad(config.ax1_jcl_strut_lca)
        c55 = config.pt1_jcl_strut_lca
        c56 = A(config.P_rbr_tie_rod).T
        c57 = triad(config.ax1_jcr_tie_upright)
        c58 = config.pt1_jcr_tie_upright
        c59 = -1*multi_dot([c56,config.R_rbr_tie_rod])
        c60 = triad(config.ax1_jcr_tie_steering)
        c61 = A(config.P_vbr_steer).T
        c62 = config.pt1_jcr_tie_steering
        c63 = A(config.P_rbl_tie_rod).T
        c64 = triad(config.ax1_jcl_tie_upright)
        c65 = config.pt1_jcl_tie_upright
        c66 = -1*multi_dot([c63,config.R_rbl_tie_rod])
        c67 = triad(config.ax1_jcl_tie_steering)
        c68 = A(config.P_vbl_steer).T
        c69 = config.pt1_jcl_tie_steering

        config.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c0,c1])
        config.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c2,c1])
        config.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c0,c3]) + c4)
        config.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c2,c3]) + c5)
        config.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([c0,c6])
        config.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([c7,c6])
        config.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([c0,c8]) + c4)
        config.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([c7,c8]) + c9)
        config.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c10,c11])
        config.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c12,c11])
        config.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c10,c13]) + c14)
        config.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c12,c13]) + c15)
        config.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([c10,c16])
        config.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([c7,c16])
        config.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([c10,c17]) + c14)
        config.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([c7,c17]) + c9)
        config.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c18,c19])
        config.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c2,c19])
        config.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c18,c20]) + c21)
        config.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c2,c20]) + c5)
        config.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([c18,c22])
        config.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([c7,c22])
        config.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([c18,c23]) + c21)
        config.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([c7,c23]) + c9)
        config.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c24,c25])
        config.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c12,c25])
        config.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c24,c26]) + c27)
        config.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c12,c26]) + c15)
        config.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([c24,c28])
        config.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([c7,c28])
        config.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([c24,c29]) + c27)
        config.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([c7,c29]) + c9)
        config.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c2,c30])
        config.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c31,c30])
        config.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c2,c32]) + c5)
        config.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c31,c32]) + -1*multi_dot([c31,config.R_rbr_hub]))
        config.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c12,c33])
        config.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c34,c33])
        config.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c12,c35]) + c15)
        config.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c34,c35]) + -1*multi_dot([c34,config.R_rbl_hub]))
        config.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([c36,c37])
        config.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([c7,triad(config.ax2_jcr_strut_chassis,c37[0:3,1:2])])
        config.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c36,c38]) + c39)
        config.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([c7,c38]) + c9)
        config.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c36,c40])
        config.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c41,c40])
        config.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c36,c42]) + c39)
        config.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c41,c42]) + c43)
        config.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([c44,c45])
        config.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([c7,triad(config.ax2_jcl_strut_chassis,c45[0:3,1:2])])
        config.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c44,c46]) + c47)
        config.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([c7,c46]) + c9)
        config.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c44,c48])
        config.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c49,c48])
        config.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c44,c50]) + c47)
        config.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c49,c50]) + c51)
        config.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c41,c52])
        config.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c18,triad(config.ax2_jcr_strut_lca,c52[0:3,1:2])])
        config.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c41,c53]) + c43)
        config.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c18,c53]) + c21)
        config.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c49,c54])
        config.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c24,triad(config.ax2_jcl_strut_lca,c54[0:3,1:2])])
        config.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c49,c55]) + c51)
        config.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c24,c55]) + c27)
        config.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c56,c57])
        config.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c2,c57])
        config.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c56,c58]) + c59)
        config.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c2,c58]) + c5)
        config.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([c56,c60])
        config.Mbar_vbr_steer_jcr_tie_steering = multi_dot([c61,triad(config.ax2_jcr_tie_steering,c60[0:3,1:2])])
        config.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([c56,c62]) + c59)
        config.ubar_vbr_steer_jcr_tie_steering = (multi_dot([c61,c62]) + -1*multi_dot([c61,config.R_vbr_steer]))
        config.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c63,c64])
        config.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c12,c64])
        config.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c63,c65]) + c66)
        config.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c12,c65]) + c15)
        config.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([c63,c67])
        config.Mbar_vbl_steer_jcl_tie_steering = multi_dot([c68,triad(config.ax2_jcl_tie_steering,c67[0:3,1:2])])
        config.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([c63,c69]) + c66)
        config.ubar_vbl_steer_jcl_tie_steering = (multi_dot([c68,c69]) + -1*multi_dot([c68,config.R_vbl_steer]))

    
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
        x10 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        x11 = self.R_rbl_uca
        x12 = self.R_rbl_upright
        x13 = -1*x12
        x14 = self.P_rbl_uca
        x15 = A(x14)
        x16 = self.P_rbl_upright
        x17 = A(x16)
        x18 = x15.T
        x19 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x20 = self.R_rbr_lca
        x21 = self.P_rbr_lca
        x22 = A(x21)
        x23 = x22.T
        x24 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x25 = self.R_rbl_lca
        x26 = self.P_rbl_lca
        x27 = A(x26)
        x28 = x27.T
        x29 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x30 = self.P_rbr_hub
        x31 = A(x30)
        x32 = x6.T
        x33 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x34 = self.P_rbl_hub
        x35 = A(x34)
        x36 = x17.T
        x37 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x38 = self.R_rbr_upper_strut
        x39 = self.P_rbr_upper_strut
        x40 = A(x39)
        x41 = x40.T
        x42 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x43 = self.P_rbr_lower_strut
        x44 = A(x43)
        x45 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x46 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x47 = self.R_rbr_lower_strut
        x48 = (x38 + -1*x47 + multi_dot([x40,config.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([x44,config.ubar_rbr_lower_strut_jcr_strut]))
        x49 = self.R_rbl_upper_strut
        x50 = self.P_rbl_upper_strut
        x51 = A(x50)
        x52 = x51.T
        x53 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x54 = self.P_rbl_lower_strut
        x55 = A(x54)
        x56 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x57 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x58 = self.R_rbl_lower_strut
        x59 = (x49 + -1*x58 + multi_dot([x51,config.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([x55,config.ubar_rbl_lower_strut_jcl_strut]))
        x60 = self.R_rbr_tie_rod
        x61 = self.P_rbr_tie_rod
        x62 = A(x61)
        x63 = A(self.P_vbr_steer)
        x64 = self.R_rbl_tie_rod
        x65 = self.P_rbl_tie_rod
        x66 = A(x65)
        x67 = A(self.P_vbl_steer)
        x68 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x2 + multi_dot([x4,config.ubar_rbr_uca_jcr_uca_upright]) + -1*multi_dot([x6,config.ubar_rbr_upright_jcr_uca_upright])),(x0 + x7 + multi_dot([x4,config.ubar_rbr_uca_jcr_uca_chassis]) + -1*multi_dot([x8,config.ubar_vbs_chassis_jcr_uca_chassis])),multi_dot([config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),multi_dot([config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),(x11 + x13 + multi_dot([x15,config.ubar_rbl_uca_jcl_uca_upright]) + -1*multi_dot([x17,config.ubar_rbl_upright_jcl_uca_upright])),(x11 + x7 + multi_dot([x15,config.ubar_rbl_uca_jcl_uca_chassis]) + -1*multi_dot([x8,config.ubar_vbs_chassis_jcl_uca_chassis])),multi_dot([config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x18,x8,x19]),multi_dot([config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x18,x8,x19]),(x20 + x2 + multi_dot([x22,config.ubar_rbr_lca_jcr_lca_upright]) + -1*multi_dot([x6,config.ubar_rbr_upright_jcr_lca_upright])),(x20 + x7 + multi_dot([x22,config.ubar_rbr_lca_jcr_lca_chassis]) + -1*multi_dot([x8,config.ubar_vbs_chassis_jcr_lca_chassis])),multi_dot([config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x23,x8,x24]),multi_dot([config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x23,x8,x24]),(x25 + x13 + multi_dot([x27,config.ubar_rbl_lca_jcl_lca_upright]) + -1*multi_dot([x17,config.ubar_rbl_upright_jcl_lca_upright])),(x25 + x7 + multi_dot([x27,config.ubar_rbl_lca_jcl_lca_chassis]) + -1*multi_dot([x8,config.ubar_vbs_chassis_jcl_lca_chassis])),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x28,x8,x29]),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x28,x8,x29]),(x1 + -1*self.R_rbr_hub + multi_dot([x6,config.ubar_rbr_upright_jcr_hub_bearing]) + -1*multi_dot([x31,config.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x32,x31,x33]),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x32,x31,x33]),(x12 + -1*self.R_rbl_hub + multi_dot([x17,config.ubar_rbl_upright_jcl_hub_bearing]) + -1*multi_dot([x35,config.ubar_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x36,x35,x37]),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x36,x35,x37]),(x38 + x7 + multi_dot([x40,config.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1*multi_dot([x8,config.ubar_vbs_chassis_jcr_strut_chassis])),multi_dot([config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x41,x8,config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),multi_dot([x42,x41,x44,x45]),multi_dot([x46,x41,x44,x45]),multi_dot([x42,x41,x48]),multi_dot([x46,x41,x48]),(x49 + x7 + multi_dot([x51,config.ubar_rbl_upper_strut_jcl_strut_chassis]) + -1*multi_dot([x8,config.ubar_vbs_chassis_jcl_strut_chassis])),multi_dot([config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x52,x8,config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),multi_dot([x53,x52,x55,x56]),multi_dot([x57,x52,x55,x56]),multi_dot([x53,x52,x59]),multi_dot([x57,x52,x59]),(x47 + -1*x20 + multi_dot([x44,config.ubar_rbr_lower_strut_jcr_strut_lca]) + -1*multi_dot([x22,config.ubar_rbr_lca_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x44.T,x22,config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),(x58 + -1*x25 + multi_dot([x55,config.ubar_rbl_lower_strut_jcl_strut_lca]) + -1*multi_dot([x27,config.ubar_rbl_lca_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1].T,x55.T,x27,config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]]),(x60 + x2 + multi_dot([x62,config.ubar_rbr_tie_rod_jcr_tie_upright]) + -1*multi_dot([x6,config.ubar_rbr_upright_jcr_tie_upright])),(x60 + -1*self.R_vbr_steer + multi_dot([x62,config.ubar_rbr_tie_rod_jcr_tie_steering]) + -1*multi_dot([x63,config.ubar_vbr_steer_jcr_tie_steering])),multi_dot([config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x62.T,x63,config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),(x64 + x13 + multi_dot([x66,config.ubar_rbl_tie_rod_jcl_tie_upright]) + -1*multi_dot([x17,config.ubar_rbl_upright_jcl_tie_upright])),(x64 + -1*self.R_vbl_steer + multi_dot([x66,config.ubar_rbl_tie_rod_jcl_tie_steering]) + -1*multi_dot([x67,config.ubar_vbl_steer_jcl_tie_steering])),multi_dot([config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x66.T,x67,config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),(x68 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x68 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x68 + (multi_dot([x21.T,x21]))**(1.0/2.0)),(x68 + (multi_dot([x26.T,x26]))**(1.0/2.0)),(x68 + (multi_dot([x5.T,x5]))**(1.0/2.0)),(x68 + (multi_dot([x16.T,x16]))**(1.0/2.0)),(x68 + (multi_dot([x39.T,x39]))**(1.0/2.0)),(x68 + (multi_dot([x50.T,x50]))**(1.0/2.0)),(x68 + (multi_dot([x43.T,x43]))**(1.0/2.0)),(x68 + (multi_dot([x54.T,x54]))**(1.0/2.0)),(x68 + (multi_dot([x61.T,x61]))**(1.0/2.0)),(x68 + (multi_dot([x65.T,x65]))**(1.0/2.0)),(x68 + (multi_dot([x30.T,x30]))**(1.0/2.0)),(x68 + (multi_dot([x34.T,x34]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v0,v1,v1,v0,v0,v1,v1,v0,v0,v1,v1,v0,v0,v1,v1,v0,v1,v1,v0,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v0,v1,v0,v0,v1,v0,v0,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_rbr_upright
        a2 = self.Pd_vbs_chassis
        a3 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        a4 = a3.T
        a5 = self.P_vbs_chassis
        a6 = A(a5).T
        a7 = config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a8 = self.P_rbr_uca
        a9 = A(a8).T
        a10 = B(a2,a3)
        a11 = a0.T
        a12 = B(a5,a3)
        a13 = config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a14 = self.Pd_rbl_uca
        a15 = self.Pd_rbl_upright
        a16 = config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a17 = self.P_rbl_uca
        a18 = A(a17).T
        a19 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a20 = B(a2,a19)
        a21 = a19.T
        a22 = a14.T
        a23 = B(a5,a19)
        a24 = config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a25 = self.Pd_rbr_lca
        a26 = config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a27 = self.P_rbr_lca
        a28 = A(a27).T
        a29 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a30 = B(a2,a29)
        a31 = a29.T
        a32 = a25.T
        a33 = B(a5,a29)
        a34 = config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a35 = self.Pd_rbl_lca
        a36 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a37 = self.P_rbl_lca
        a38 = A(a37).T
        a39 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a40 = B(a2,a39)
        a41 = a39.T
        a42 = a35.T
        a43 = B(a5,a39)
        a44 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a45 = self.Pd_rbr_hub
        a46 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a47 = self.P_rbr_upright
        a48 = A(a47).T
        a49 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a50 = B(a45,a49)
        a51 = a49.T
        a52 = self.P_rbr_hub
        a53 = A(a52).T
        a54 = a1.T
        a55 = B(a52,a49)
        a56 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a57 = self.Pd_rbl_hub
        a58 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a59 = self.P_rbl_upright
        a60 = A(a59).T
        a61 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a62 = B(a57,a61)
        a63 = a61.T
        a64 = self.P_rbl_hub
        a65 = A(a64).T
        a66 = a15.T
        a67 = B(a64,a61)
        a68 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a69 = self.Pd_rbr_upper_strut
        a70 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a71 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a72 = self.P_rbr_upper_strut
        a73 = A(a72).T
        a74 = a69.T
        a75 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a76 = a75.T
        a77 = self.Pd_rbr_lower_strut
        a78 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a79 = B(a77,a78)
        a80 = a78.T
        a81 = self.P_rbr_lower_strut
        a82 = A(a81).T
        a83 = B(a69,a75)
        a84 = B(a72,a75).T
        a85 = B(a81,a78)
        a86 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a87 = a86.T
        a88 = B(a69,a86)
        a89 = B(a72,a86).T
        a90 = config.ubar_rbr_upper_strut_jcr_strut
        a91 = config.ubar_rbr_lower_strut_jcr_strut
        a92 = (multi_dot([B(a69,a90),a69]) + -1*multi_dot([B(a77,a91),a77]))
        a93 = (self.Rd_rbr_upper_strut + -1*self.Rd_rbr_lower_strut + multi_dot([B(a81,a91),a77]) + multi_dot([B(a72,a90),a69]))
        a94 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([a90.T,a73]) + -1*multi_dot([a91.T,a82]))
        a95 = self.Pd_rbl_upper_strut
        a96 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a97 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a98 = self.P_rbl_upper_strut
        a99 = A(a98).T
        a100 = a95.T
        a101 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a102 = a101.T
        a103 = self.Pd_rbl_lower_strut
        a104 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a105 = B(a103,a104)
        a106 = a104.T
        a107 = self.P_rbl_lower_strut
        a108 = A(a107).T
        a109 = B(a95,a101)
        a110 = B(a98,a101).T
        a111 = B(a107,a104)
        a112 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a113 = a112.T
        a114 = B(a95,a112)
        a115 = B(a98,a112).T
        a116 = config.ubar_rbl_upper_strut_jcl_strut
        a117 = config.ubar_rbl_lower_strut_jcl_strut
        a118 = (multi_dot([B(a95,a116),a95]) + -1*multi_dot([B(a103,a117),a103]))
        a119 = (self.Rd_rbl_upper_strut + -1*self.Rd_rbl_lower_strut + multi_dot([B(a107,a117),a103]) + multi_dot([B(a98,a116),a95]))
        a120 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([a116.T,a99]) + -1*multi_dot([a117.T,a108]))
        a121 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a122 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a123 = a77.T
        a124 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a125 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a126 = a103.T
        a127 = self.Pd_rbr_tie_rod
        a128 = self.Pd_vbr_steer
        a129 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a130 = self.P_rbr_tie_rod
        a131 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a132 = self.P_vbr_steer
        a133 = a127.T
        a134 = self.Pd_rbl_tie_rod
        a135 = self.Pd_vbl_steer
        a136 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a137 = self.P_rbl_tie_rod
        a138 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a139 = self.P_vbl_steer
        a140 = a134.T

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_rbr_uca_jcr_uca_upright),a0]) + -1*multi_dot([B(a1,config.ubar_rbr_upright_jcr_uca_upright),a1])),(multi_dot([B(a0,config.ubar_rbr_uca_jcr_uca_chassis),a0]) + -1*multi_dot([B(a2,config.ubar_vbs_chassis_jcr_uca_chassis),a2])),(multi_dot([a4,a6,B(a0,a7),a0]) + multi_dot([a7.T,a9,a10,a2]) + 2*multi_dot([a11,B(a8,a7).T,a12,a2])),(multi_dot([a4,a6,B(a0,a13),a0]) + multi_dot([a13.T,a9,a10,a2]) + 2*multi_dot([a11,B(a8,a13).T,a12,a2])),(multi_dot([B(a14,config.ubar_rbl_uca_jcl_uca_upright),a14]) + -1*multi_dot([B(a15,config.ubar_rbl_upright_jcl_uca_upright),a15])),(multi_dot([B(a14,config.ubar_rbl_uca_jcl_uca_chassis),a14]) + -1*multi_dot([B(a2,config.ubar_vbs_chassis_jcl_uca_chassis),a2])),(multi_dot([a16.T,a18,a20,a2]) + multi_dot([a21,a6,B(a14,a16),a14]) + 2*multi_dot([a22,B(a17,a16).T,a23,a2])),(multi_dot([a24.T,a18,a20,a2]) + multi_dot([a21,a6,B(a14,a24),a14]) + 2*multi_dot([a22,B(a17,a24).T,a23,a2])),(multi_dot([B(a25,config.ubar_rbr_lca_jcr_lca_upright),a25]) + -1*multi_dot([B(a1,config.ubar_rbr_upright_jcr_lca_upright),a1])),(multi_dot([B(a25,config.ubar_rbr_lca_jcr_lca_chassis),a25]) + -1*multi_dot([B(a2,config.ubar_vbs_chassis_jcr_lca_chassis),a2])),(multi_dot([a26.T,a28,a30,a2]) + multi_dot([a31,a6,B(a25,a26),a25]) + 2*multi_dot([a32,B(a27,a26).T,a33,a2])),(multi_dot([a34.T,a28,a30,a2]) + multi_dot([a31,a6,B(a25,a34),a25]) + 2*multi_dot([a32,B(a27,a34).T,a33,a2])),(multi_dot([B(a35,config.ubar_rbl_lca_jcl_lca_upright),a35]) + -1*multi_dot([B(a15,config.ubar_rbl_upright_jcl_lca_upright),a15])),(multi_dot([B(a35,config.ubar_rbl_lca_jcl_lca_chassis),a35]) + -1*multi_dot([B(a2,config.ubar_vbs_chassis_jcl_lca_chassis),a2])),(multi_dot([a36.T,a38,a40,a2]) + multi_dot([a41,a6,B(a35,a36),a35]) + 2*multi_dot([a42,B(a37,a36).T,a43,a2])),(multi_dot([a44.T,a38,a40,a2]) + multi_dot([a41,a6,B(a35,a44),a35]) + 2*multi_dot([a42,B(a37,a44).T,a43,a2])),(multi_dot([B(a1,config.ubar_rbr_upright_jcr_hub_bearing),a1]) + -1*multi_dot([B(a45,config.ubar_rbr_hub_jcr_hub_bearing),a45])),(multi_dot([a46.T,a48,a50,a45]) + multi_dot([a51,a53,B(a1,a46),a1]) + 2*multi_dot([a54,B(a47,a46).T,a55,a45])),(multi_dot([a56.T,a48,a50,a45]) + multi_dot([a51,a53,B(a1,a56),a1]) + 2*multi_dot([a54,B(a47,a56).T,a55,a45])),(multi_dot([B(a15,config.ubar_rbl_upright_jcl_hub_bearing),a15]) + -1*multi_dot([B(a57,config.ubar_rbl_hub_jcl_hub_bearing),a57])),(multi_dot([a58.T,a60,a62,a57]) + multi_dot([a63,a65,B(a15,a58),a15]) + 2*multi_dot([a66,B(a59,a58).T,a67,a57])),(multi_dot([a68.T,a60,a62,a57]) + multi_dot([a63,a65,B(a15,a68),a15]) + 2*multi_dot([a66,B(a59,a68).T,a67,a57])),(multi_dot([B(a69,config.ubar_rbr_upper_strut_jcr_strut_chassis),a69]) + -1*multi_dot([B(a2,config.ubar_vbs_chassis_jcr_strut_chassis),a2])),(multi_dot([a70.T,a6,B(a69,a71),a69]) + multi_dot([a71.T,a73,B(a2,a70),a2]) + 2*multi_dot([a74,B(a72,a71).T,B(a5,a70),a2])),(multi_dot([a76,a73,a79,a77]) + multi_dot([a80,a82,a83,a69]) + 2*multi_dot([a74,a84,a85,a77])),(multi_dot([a87,a73,a79,a77]) + multi_dot([a80,a82,a88,a69]) + 2*multi_dot([a74,a89,a85,a77])),(multi_dot([a76,a73,a92]) + 2*multi_dot([a74,a84,a93]) + multi_dot([a94,a83,a69])),(multi_dot([a87,a73,a92]) + 2*multi_dot([a74,a89,a93]) + multi_dot([a94,a88,a69])),(multi_dot([B(a95,config.ubar_rbl_upper_strut_jcl_strut_chassis),a95]) + -1*multi_dot([B(a2,config.ubar_vbs_chassis_jcl_strut_chassis),a2])),(multi_dot([a96.T,a6,B(a95,a97),a95]) + multi_dot([a97.T,a99,B(a2,a96),a2]) + 2*multi_dot([a100,B(a98,a97).T,B(a5,a96),a2])),(multi_dot([a102,a99,a105,a103]) + multi_dot([a106,a108,a109,a95]) + 2*multi_dot([a100,a110,a111,a103])),(multi_dot([a113,a99,a105,a103]) + multi_dot([a106,a108,a114,a95]) + 2*multi_dot([a100,a115,a111,a103])),(multi_dot([a102,a99,a118]) + 2*multi_dot([a100,a110,a119]) + multi_dot([a120,a109,a95])),(multi_dot([a113,a99,a118]) + 2*multi_dot([a100,a115,a119]) + multi_dot([a120,a114,a95])),(multi_dot([B(a77,config.ubar_rbr_lower_strut_jcr_strut_lca),a77]) + -1*multi_dot([B(a25,config.ubar_rbr_lca_jcr_strut_lca),a25])),(multi_dot([a121.T,a82,B(a25,a122),a25]) + multi_dot([a122.T,a28,B(a77,a121),a77]) + 2*multi_dot([a123,B(a81,a121).T,B(a27,a122),a25])),(multi_dot([B(a103,config.ubar_rbl_lower_strut_jcl_strut_lca),a103]) + -1*multi_dot([B(a35,config.ubar_rbl_lca_jcl_strut_lca),a35])),(multi_dot([a124.T,a38,B(a103,a125),a103]) + multi_dot([a125.T,a108,B(a35,a124),a35]) + 2*multi_dot([a126,B(a107,a125).T,B(a37,a124),a35])),(multi_dot([B(a127,config.ubar_rbr_tie_rod_jcr_tie_upright),a127]) + -1*multi_dot([B(a1,config.ubar_rbr_upright_jcr_tie_upright),a1])),(multi_dot([B(a127,config.ubar_rbr_tie_rod_jcr_tie_steering),a127]) + -1*multi_dot([B(a128,config.ubar_vbr_steer_jcr_tie_steering),a128])),(multi_dot([a129.T,A(a130).T,B(a128,a131),a128]) + multi_dot([a131.T,A(a132).T,B(a127,a129),a127]) + 2*multi_dot([a133,B(a130,a129).T,B(a132,a131),a128])),(multi_dot([B(a134,config.ubar_rbl_tie_rod_jcl_tie_upright),a134]) + -1*multi_dot([B(a15,config.ubar_rbl_upright_jcl_tie_upright),a15])),(multi_dot([B(a134,config.ubar_rbl_tie_rod_jcl_tie_steering),a134]) + -1*multi_dot([B(a135,config.ubar_vbl_steer_jcl_tie_steering),a135])),(multi_dot([a136.T,A(a137).T,B(a135,a138),a135]) + multi_dot([a138.T,A(a139).T,B(a134,a136),a134]) + 2*multi_dot([a140,B(a137,a136).T,B(a139,a138),a135])),2*(multi_dot([a11,a0]))**(1.0/2.0),2*(multi_dot([a22,a14]))**(1.0/2.0),2*(multi_dot([a32,a25]))**(1.0/2.0),2*(multi_dot([a42,a35]))**(1.0/2.0),2*(multi_dot([a54,a1]))**(1.0/2.0),2*(multi_dot([a66,a15]))**(1.0/2.0),2*(multi_dot([a74,a69]))**(1.0/2.0),2*(multi_dot([a100,a95]))**(1.0/2.0),2*(multi_dot([a123,a77]))**(1.0/2.0),2*(multi_dot([a126,a103]))**(1.0/2.0),2*(multi_dot([a133,a127]))**(1.0/2.0),2*(multi_dot([a140,a134]))**(1.0/2.0),2*(multi_dot([a45.T,a45]))**(1.0/2.0),2*(multi_dot([a57.T,a57]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbr_uca
        j2 = -1*j0
        j3 = self.P_rbr_upright
        j4 = np.zeros((1,3),dtype=np.float64)
        j5 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        j6 = j5.T
        j7 = self.P_vbs_chassis
        j8 = A(j7).T
        j9 = config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j10 = config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j11 = A(j1).T
        j12 = B(j7,j5)
        j13 = self.P_rbl_uca
        j14 = self.P_rbl_upright
        j15 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j16 = j15.T
        j17 = config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j18 = config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j19 = A(j13).T
        j20 = B(j7,j15)
        j21 = self.P_rbr_lca
        j22 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j23 = j22.T
        j24 = config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j25 = config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j26 = A(j21).T
        j27 = B(j7,j22)
        j28 = self.P_rbl_lca
        j29 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j30 = j29.T
        j31 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j32 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j33 = A(j28).T
        j34 = B(j7,j29)
        j35 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j36 = j35.T
        j37 = self.P_rbr_hub
        j38 = A(j37).T
        j39 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j40 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j41 = A(j3).T
        j42 = B(j37,j35)
        j43 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j44 = j43.T
        j45 = self.P_rbl_hub
        j46 = A(j45).T
        j47 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j48 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j49 = A(j14).T
        j50 = B(j45,j43)
        j51 = self.P_rbr_upper_strut
        j52 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j53 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j54 = A(j51).T
        j55 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j56 = j55.T
        j57 = self.P_rbr_lower_strut
        j58 = A(j57).T
        j59 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j60 = B(j51,j59)
        j61 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j62 = B(j51,j61)
        j63 = j59.T
        j64 = multi_dot([j63,j54])
        j65 = config.ubar_rbr_upper_strut_jcr_strut
        j66 = B(j51,j65)
        j67 = config.ubar_rbr_lower_strut_jcr_strut
        j68 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([j65.T,j54]) + -1*multi_dot([j67.T,j58]))
        j69 = j61.T
        j70 = multi_dot([j69,j54])
        j71 = B(j57,j55)
        j72 = B(j57,j67)
        j73 = self.P_rbl_upper_strut
        j74 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j75 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j76 = A(j73).T
        j77 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j78 = j77.T
        j79 = self.P_rbl_lower_strut
        j80 = A(j79).T
        j81 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j82 = B(j73,j81)
        j83 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j84 = B(j73,j83)
        j85 = j81.T
        j86 = multi_dot([j85,j76])
        j87 = config.ubar_rbl_upper_strut_jcl_strut
        j88 = B(j73,j87)
        j89 = config.ubar_rbl_lower_strut_jcl_strut
        j90 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([j87.T,j76]) + -1*multi_dot([j89.T,j80]))
        j91 = j83.T
        j92 = multi_dot([j91,j76])
        j93 = B(j79,j77)
        j94 = B(j79,j89)
        j95 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j96 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j97 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j98 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j99 = self.P_rbr_tie_rod
        j100 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j101 = self.P_vbr_steer
        j102 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j103 = self.P_rbl_tie_rod
        j104 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j105 = self.P_vbl_steer
        j106 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = [j0,B(j1,config.ubar_rbr_uca_jcr_uca_upright),j2,-1*B(j3,config.ubar_rbr_upright_jcr_uca_upright),j0,B(j1,config.ubar_rbr_uca_jcr_uca_chassis),j2,-1*B(j7,config.ubar_vbs_chassis_jcr_uca_chassis),j4,multi_dot([j6,j8,B(j1,j9)]),j4,multi_dot([j9.T,j11,j12]),j4,multi_dot([j6,j8,B(j1,j10)]),j4,multi_dot([j10.T,j11,j12]),j0,B(j13,config.ubar_rbl_uca_jcl_uca_upright),j2,-1*B(j14,config.ubar_rbl_upright_jcl_uca_upright),j0,B(j13,config.ubar_rbl_uca_jcl_uca_chassis),j2,-1*B(j7,config.ubar_vbs_chassis_jcl_uca_chassis),j4,multi_dot([j16,j8,B(j13,j17)]),j4,multi_dot([j17.T,j19,j20]),j4,multi_dot([j16,j8,B(j13,j18)]),j4,multi_dot([j18.T,j19,j20]),j0,B(j21,config.ubar_rbr_lca_jcr_lca_upright),j2,-1*B(j3,config.ubar_rbr_upright_jcr_lca_upright),j0,B(j21,config.ubar_rbr_lca_jcr_lca_chassis),j2,-1*B(j7,config.ubar_vbs_chassis_jcr_lca_chassis),j4,multi_dot([j23,j8,B(j21,j24)]),j4,multi_dot([j24.T,j26,j27]),j4,multi_dot([j23,j8,B(j21,j25)]),j4,multi_dot([j25.T,j26,j27]),j0,B(j28,config.ubar_rbl_lca_jcl_lca_upright),j2,-1*B(j14,config.ubar_rbl_upright_jcl_lca_upright),j0,B(j28,config.ubar_rbl_lca_jcl_lca_chassis),j2,-1*B(j7,config.ubar_vbs_chassis_jcl_lca_chassis),j4,multi_dot([j30,j8,B(j28,j31)]),j4,multi_dot([j31.T,j33,j34]),j4,multi_dot([j30,j8,B(j28,j32)]),j4,multi_dot([j32.T,j33,j34]),j0,B(j3,config.ubar_rbr_upright_jcr_hub_bearing),j2,-1*B(j37,config.ubar_rbr_hub_jcr_hub_bearing),j4,multi_dot([j36,j38,B(j3,j39)]),j4,multi_dot([j39.T,j41,j42]),j4,multi_dot([j36,j38,B(j3,j40)]),j4,multi_dot([j40.T,j41,j42]),j0,B(j14,config.ubar_rbl_upright_jcl_hub_bearing),j2,-1*B(j45,config.ubar_rbl_hub_jcl_hub_bearing),j4,multi_dot([j44,j46,B(j14,j47)]),j4,multi_dot([j47.T,j49,j50]),j4,multi_dot([j44,j46,B(j14,j48)]),j4,multi_dot([j48.T,j49,j50]),j0,B(j51,config.ubar_rbr_upper_strut_jcr_strut_chassis),j2,-1*B(j7,config.ubar_vbs_chassis_jcr_strut_chassis),j4,multi_dot([j52.T,j8,B(j51,j53)]),j4,multi_dot([j53.T,j54,B(j7,j52)]),j4,multi_dot([j56,j58,j60]),j4,multi_dot([j63,j54,j71]),j4,multi_dot([j56,j58,j62]),j4,multi_dot([j69,j54,j71]),j64,(multi_dot([j63,j54,j66]) + multi_dot([j68,j60])),-1*j64,-1*multi_dot([j63,j54,j72]),j70,(multi_dot([j69,j54,j66]) + multi_dot([j68,j62])),-1*j70,-1*multi_dot([j69,j54,j72]),j0,B(j73,config.ubar_rbl_upper_strut_jcl_strut_chassis),j2,-1*B(j7,config.ubar_vbs_chassis_jcl_strut_chassis),j4,multi_dot([j74.T,j8,B(j73,j75)]),j4,multi_dot([j75.T,j76,B(j7,j74)]),j4,multi_dot([j78,j80,j82]),j4,multi_dot([j85,j76,j93]),j4,multi_dot([j78,j80,j84]),j4,multi_dot([j91,j76,j93]),j86,(multi_dot([j85,j76,j88]) + multi_dot([j90,j82])),-1*j86,-1*multi_dot([j85,j76,j94]),j92,(multi_dot([j91,j76,j88]) + multi_dot([j90,j84])),-1*j92,-1*multi_dot([j91,j76,j94]),j2,-1*B(j21,config.ubar_rbr_lca_jcr_strut_lca),j0,B(j57,config.ubar_rbr_lower_strut_jcr_strut_lca),j4,multi_dot([j96.T,j58,B(j21,j95)]),j4,multi_dot([j95.T,j26,B(j57,j96)]),j2,-1*B(j28,config.ubar_rbl_lca_jcl_strut_lca),j0,B(j79,config.ubar_rbl_lower_strut_jcl_strut_lca),j4,multi_dot([j98.T,j80,B(j28,j97)]),j4,multi_dot([j97.T,j33,B(j79,j98)]),j2,-1*B(j3,config.ubar_rbr_upright_jcr_tie_upright),j0,B(j99,config.ubar_rbr_tie_rod_jcr_tie_upright),j0,B(j99,config.ubar_rbr_tie_rod_jcr_tie_steering),j2,-1*B(j101,config.ubar_vbr_steer_jcr_tie_steering),j4,multi_dot([j100.T,A(j101).T,B(j99,j102)]),j4,multi_dot([j102.T,A(j99).T,B(j101,j100)]),j2,-1*B(j14,config.ubar_rbl_upright_jcl_tie_upright),j0,B(j103,config.ubar_rbl_tie_rod_jcl_tie_upright),j0,B(j103,config.ubar_rbl_tie_rod_jcl_tie_steering),j2,-1*B(j105,config.ubar_vbl_steer_jcl_tie_steering),j4,multi_dot([j104.T,A(j105).T,B(j103,j106)]),j4,multi_dot([j106.T,A(j103).T,B(j105,j104)]),2*j1.T,2*j13.T,2*j21.T,2*j28.T,2*j3.T,2*j14.T,2*j51.T,2*j73.T,2*j57.T,2*j79.T,2*j99.T,2*j103.T,2*j37.T,2*j45.T]
  
