
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
        self.R_rbr_pushrod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_pushrod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_pushrod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_pushrod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_pushrod = 1
        self.Jbar_rbr_pushrod = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_rocker = 1
        self.Jbar_rbr_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
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
        self.ax1_jcr_prod_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_prod_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_prod_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_prod_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_prod_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_rocker_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_rocker_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt2_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        None
        None
        self.ax1_jcr_strut_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbr_uca,self.P_rbr_uca,self.R_rbl_uca,self.P_rbl_uca,self.R_rbr_lca,self.P_rbr_lca,self.R_rbl_lca,self.P_rbl_lca,self.R_rbr_upright,self.P_rbr_upright,self.R_rbl_upright,self.P_rbl_upright,self.R_rbr_pushrod,self.P_rbr_pushrod,self.R_rbl_pushrod,self.P_rbl_pushrod,self.R_rbr_rocker,self.P_rbr_rocker,self.R_rbl_rocker,self.P_rbl_rocker,self.R_rbr_upper_strut,self.P_rbr_upper_strut,self.R_rbl_upper_strut,self.P_rbl_upper_strut,self.R_rbr_lower_strut,self.P_rbr_lower_strut,self.R_rbl_lower_strut,self.P_rbl_lower_strut,self.R_rbr_tie_rod,self.P_rbr_tie_rod,self.R_rbl_tie_rod,self.P_rbl_tie_rod,self.R_rbr_hub,self.P_rbr_hub,self.R_rbl_hub,self.P_rbl_hub])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbr_uca,self.Pd_rbr_uca,self.Rd_rbl_uca,self.Pd_rbl_uca,self.Rd_rbr_lca,self.Pd_rbr_lca,self.Rd_rbl_lca,self.Pd_rbl_lca,self.Rd_rbr_upright,self.Pd_rbr_upright,self.Rd_rbl_upright,self.Pd_rbl_upright,self.Rd_rbr_pushrod,self.Pd_rbr_pushrod,self.Rd_rbl_pushrod,self.Pd_rbl_pushrod,self.Rd_rbr_rocker,self.Pd_rbr_rocker,self.Rd_rbl_rocker,self.Pd_rbl_rocker,self.Rd_rbr_upper_strut,self.Pd_rbr_upper_strut,self.Rd_rbl_upper_strut,self.Pd_rbl_upper_strut,self.Rd_rbr_lower_strut,self.Pd_rbr_lower_strut,self.Rd_rbl_lower_strut,self.Pd_rbl_lower_strut,self.Rd_rbr_tie_rod,self.Pd_rbr_tie_rod,self.Rd_rbl_tie_rod,self.Pd_rbl_tie_rod,self.Rd_rbr_hub,self.Pd_rbr_hub,self.Rd_rbl_hub,self.Pd_rbl_hub])
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
        self.R_rbl_pushrod = mirrored(self.R_rbr_pushrod)
        self.P_rbl_pushrod = mirrored(self.P_rbr_pushrod)
        self.Rd_rbl_pushrod = mirrored(self.Rd_rbr_pushrod)
        self.Pd_rbl_pushrod = mirrored(self.Pd_rbr_pushrod)
        self.m_rbl_pushrod = self.m_rbr_pushrod
        self.Jbar_rbl_pushrod = mirrored(self.Jbar_rbr_pushrod)
        self.R_rbl_rocker = mirrored(self.R_rbr_rocker)
        self.P_rbl_rocker = mirrored(self.P_rbr_rocker)
        self.Rd_rbl_rocker = mirrored(self.Rd_rbr_rocker)
        self.Pd_rbl_rocker = mirrored(self.Pd_rbr_rocker)
        self.m_rbl_rocker = self.m_rbr_rocker
        self.Jbar_rbl_rocker = mirrored(self.Jbar_rbr_rocker)
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
        self.ax1_jcl_prod_uca = mirrored(self.ax1_jcr_prod_uca)
        self.ax2_jcl_prod_uca = mirrored(self.ax2_jcr_prod_uca)
        self.pt1_jcl_prod_uca = mirrored(self.pt1_jcr_prod_uca)
        self.ax1_jcl_lca_upright = mirrored(self.ax1_jcr_lca_upright)
        self.pt1_jcl_lca_upright = mirrored(self.pt1_jcr_lca_upright)
        self.ax1_jcl_lca_chassis = mirrored(self.ax1_jcr_lca_chassis)
        self.pt1_jcl_lca_chassis = mirrored(self.pt1_jcr_lca_chassis)
        self.ax1_jcl_hub_bearing = mirrored(self.ax1_jcr_hub_bearing)
        self.pt1_jcl_hub_bearing = mirrored(self.pt1_jcr_hub_bearing)
        self.ax1_jcl_prod_rocker = mirrored(self.ax1_jcr_prod_rocker)
        self.pt1_jcl_prod_rocker = mirrored(self.pt1_jcr_prod_rocker)
        self.ax1_jcl_rocker_chassis = mirrored(self.ax1_jcr_rocker_chassis)
        self.pt1_jcl_rocker_chassis = mirrored(self.pt1_jcr_rocker_chassis)
        self.ax1_jcl_strut_chassis = mirrored(self.ax1_jcr_strut_chassis)
        self.ax2_jcl_strut_chassis = mirrored(self.ax2_jcr_strut_chassis)
        self.pt1_jcl_strut_chassis = mirrored(self.pt1_jcr_strut_chassis)
        self.ax1_jcl_strut = mirrored(self.ax1_jcr_strut)
        self.pt1_jcl_strut = mirrored(self.pt1_jcr_strut)
        self.pt1_fal_strut = mirrored(self.pt1_far_strut)
        self.pt2_fal_strut = mirrored(self.pt2_far_strut)
        self.Fs_fal_strut = self.Fs_far_strut
        self.Fd_fal_strut = self.Fd_far_strut
        self.ax1_jcl_strut_rocker = mirrored(self.ax1_jcr_strut_rocker)
        self.ax2_jcl_strut_rocker = mirrored(self.ax2_jcr_strut_rocker)
        self.pt1_jcl_strut_rocker = mirrored(self.pt1_jcr_strut_rocker)
        self.ax1_jcl_tie_upright = mirrored(self.ax1_jcr_tie_upright)
        self.pt1_jcl_tie_upright = mirrored(self.pt1_jcr_tie_upright)
        self.ax1_jcl_tie_steering = mirrored(self.ax1_jcr_tie_steering)
        self.ax2_jcl_tie_steering = mirrored(self.ax2_jcr_tie_steering)
        self.pt1_jcl_tie_steering = mirrored(self.pt1_jcr_tie_steering)
    




class topology(object):

    def __init__(self,prefix='',cfg=None):
        self.t = 0.0
        self.config = (configuration() if cfg is None else cfg)
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 126
        self.nrows = 74
        self.ncols = 2*22
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,54,54,55,55,55,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
        self.rbr_uca = indicies_map[p+'rbr_uca']
        self.rbl_uca = indicies_map[p+'rbl_uca']
        self.rbr_lca = indicies_map[p+'rbr_lca']
        self.rbl_lca = indicies_map[p+'rbl_lca']
        self.rbr_upright = indicies_map[p+'rbr_upright']
        self.rbl_upright = indicies_map[p+'rbl_upright']
        self.rbr_pushrod = indicies_map[p+'rbr_pushrod']
        self.rbl_pushrod = indicies_map[p+'rbl_pushrod']
        self.rbr_rocker = indicies_map[p+'rbr_rocker']
        self.rbl_rocker = indicies_map[p+'rbl_rocker']
        self.rbr_upper_strut = indicies_map[p+'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p+'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p+'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p+'rbl_lower_strut']
        self.rbr_tie_rod = indicies_map[p+'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p+'rbl_tie_rod']
        self.rbr_hub = indicies_map[p+'rbr_hub']
        self.rbl_hub = indicies_map[p+'rbl_hub']
        self.vbr_steer = indicies_map[interface_map[p+'vbr_steer']]
        self.vbl_steer = indicies_map[interface_map[p+'vbl_steer']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_uca*2,self.rbr_uca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.rbr_pushrod*2,self.rbr_pushrod*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.rbr_pushrod*2,self.rbr_pushrod*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.rbl_pushrod*2,self.rbl_pushrod*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.rbl_pushrod*2,self.rbl_pushrod*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbr_pushrod*2,self.rbr_pushrod*2+1,self.rbr_rocker*2,self.rbr_rocker*2+1,self.rbr_rocker*2,self.rbr_rocker*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_rocker*2,self.rbr_rocker*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_rocker*2,self.rbr_rocker*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_pushrod*2,self.rbl_pushrod*2+1,self.rbl_rocker*2,self.rbl_rocker*2+1,self.rbl_rocker*2,self.rbl_rocker*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_rocker*2,self.rbl_rocker*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_rocker*2,self.rbl_rocker*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_rocker*2,self.rbr_rocker*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_rocker*2,self.rbr_rocker*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_rocker*2,self.rbl_rocker*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_rocker*2,self.rbl_rocker*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbr_uca*2+1,self.rbl_uca*2+1,self.rbr_lca*2+1,self.rbl_lca*2+1,self.rbr_upright*2+1,self.rbl_upright*2+1,self.rbr_pushrod*2+1,self.rbl_pushrod*2+1,self.rbr_rocker*2+1,self.rbl_rocker*2+1,self.rbr_upper_strut*2+1,self.rbl_upper_strut*2+1,self.rbr_lower_strut*2+1,self.rbl_lower_strut*2+1,self.rbr_tie_rod*2+1,self.rbl_tie_rod*2+1,self.rbr_hub*2+1,self.rbl_hub*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbr_uca_gravity = np.array([[0], [0], [9810.0*config.m_rbr_uca]],dtype=np.float64)
        self.F_rbl_uca_gravity = np.array([[0], [0], [9810.0*config.m_rbl_uca]],dtype=np.float64)
        self.F_rbr_lca_gravity = np.array([[0], [0], [9810.0*config.m_rbr_lca]],dtype=np.float64)
        self.F_rbl_lca_gravity = np.array([[0], [0], [9810.0*config.m_rbl_lca]],dtype=np.float64)
        self.F_rbr_upright_gravity = np.array([[0], [0], [9810.0*config.m_rbr_upright]],dtype=np.float64)
        self.F_rbl_upright_gravity = np.array([[0], [0], [9810.0*config.m_rbl_upright]],dtype=np.float64)
        self.F_rbr_pushrod_gravity = np.array([[0], [0], [9810.0*config.m_rbr_pushrod]],dtype=np.float64)
        self.F_rbl_pushrod_gravity = np.array([[0], [0], [9810.0*config.m_rbl_pushrod]],dtype=np.float64)
        self.F_rbr_rocker_gravity = np.array([[0], [0], [9810.0*config.m_rbr_rocker]],dtype=np.float64)
        self.F_rbl_rocker_gravity = np.array([[0], [0], [9810.0*config.m_rbl_rocker]],dtype=np.float64)
        self.F_rbr_upper_strut_gravity = np.array([[0], [0], [9810.0*config.m_rbr_upper_strut]],dtype=np.float64)
        self.F_rbl_upper_strut_gravity = np.array([[0], [0], [9810.0*config.m_rbl_upper_strut]],dtype=np.float64)
        self.F_rbr_lower_strut_gravity = np.array([[0], [0], [9810.0*config.m_rbr_lower_strut]],dtype=np.float64)
        self.F_rbl_lower_strut_gravity = np.array([[0], [0], [9810.0*config.m_rbl_lower_strut]],dtype=np.float64)
        self.F_rbr_tie_rod_gravity = np.array([[0], [0], [9810.0*config.m_rbr_tie_rod]],dtype=np.float64)
        self.F_rbl_tie_rod_gravity = np.array([[0], [0], [9810.0*config.m_rbl_tie_rod]],dtype=np.float64)
        self.F_rbr_hub_gravity = np.array([[0], [0], [9810.0*config.m_rbr_hub]],dtype=np.float64)
        self.F_rbl_hub_gravity = np.array([[0], [0], [9810.0*config.m_rbl_hub]],dtype=np.float64)
        self.F_vbr_steer_gravity = np.array([[0], [0], [9810.0*m_vbr_steer]],dtype=np.float64)
        self.F_vbl_steer_gravity = np.array([[0], [0], [9810.0*m_vbl_steer]],dtype=np.float64)
        self.F_vbs_chassis_gravity = np.array([[0], [0], [9810.0*m_vbs_chassis]],dtype=np.float64)

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
        c10 = triad(config.ax1_jcr_prod_uca)
        c11 = A(config.P_rbr_pushrod).T
        c12 = config.pt1_jcr_prod_uca
        c13 = -1*multi_dot([c11,config.R_rbr_pushrod])
        c14 = A(config.P_rbl_uca).T
        c15 = triad(config.ax1_jcl_uca_upright)
        c16 = A(config.P_rbl_upright).T
        c17 = config.pt1_jcl_uca_upright
        c18 = -1*multi_dot([c14,config.R_rbl_uca])
        c19 = -1*multi_dot([c16,config.R_rbl_upright])
        c20 = triad(config.ax1_jcl_uca_chassis)
        c21 = config.pt1_jcl_uca_chassis
        c22 = triad(config.ax1_jcl_prod_uca)
        c23 = A(config.P_rbl_pushrod).T
        c24 = config.pt1_jcl_prod_uca
        c25 = -1*multi_dot([c23,config.R_rbl_pushrod])
        c26 = A(config.P_rbr_lca).T
        c27 = triad(config.ax1_jcr_lca_upright)
        c28 = config.pt1_jcr_lca_upright
        c29 = -1*multi_dot([c26,config.R_rbr_lca])
        c30 = triad(config.ax1_jcr_lca_chassis)
        c31 = config.pt1_jcr_lca_chassis
        c32 = A(config.P_rbl_lca).T
        c33 = triad(config.ax1_jcl_lca_upright)
        c34 = config.pt1_jcl_lca_upright
        c35 = -1*multi_dot([c32,config.R_rbl_lca])
        c36 = triad(config.ax1_jcl_lca_chassis)
        c37 = config.pt1_jcl_lca_chassis
        c38 = triad(config.ax1_jcr_hub_bearing)
        c39 = A(config.P_rbr_hub).T
        c40 = config.pt1_jcr_hub_bearing
        c41 = triad(config.ax1_jcl_hub_bearing)
        c42 = A(config.P_rbl_hub).T
        c43 = config.pt1_jcl_hub_bearing
        c44 = A(config.P_rbr_rocker).T
        c45 = triad(config.ax1_jcr_prod_rocker)
        c46 = config.pt1_jcr_prod_rocker
        c47 = -1*multi_dot([c44,config.R_rbr_rocker])
        c48 = triad(config.ax1_jcr_rocker_chassis)
        c49 = config.pt1_jcr_rocker_chassis
        c50 = A(config.P_rbl_rocker).T
        c51 = triad(config.ax1_jcl_prod_rocker)
        c52 = config.pt1_jcl_prod_rocker
        c53 = -1*multi_dot([c50,config.R_rbl_rocker])
        c54 = triad(config.ax1_jcl_rocker_chassis)
        c55 = config.pt1_jcl_rocker_chassis
        c56 = A(config.P_rbr_upper_strut).T
        c57 = triad(config.ax1_jcr_strut_chassis)
        c58 = config.pt1_jcr_strut_chassis
        c59 = -1*multi_dot([c56,config.R_rbr_upper_strut])
        c60 = triad(config.ax1_jcr_strut)
        c61 = A(config.P_rbr_lower_strut).T
        c62 = config.pt1_jcr_strut
        c63 = -1*multi_dot([c61,config.R_rbr_lower_strut])
        c64 = A(config.P_rbl_upper_strut).T
        c65 = triad(config.ax1_jcl_strut_chassis)
        c66 = config.pt1_jcl_strut_chassis
        c67 = -1*multi_dot([c64,config.R_rbl_upper_strut])
        c68 = triad(config.ax1_jcl_strut)
        c69 = A(config.P_rbl_lower_strut).T
        c70 = config.pt1_jcl_strut
        c71 = -1*multi_dot([c69,config.R_rbl_lower_strut])
        c72 = triad(config.ax1_jcr_strut_rocker)
        c73 = config.pt1_jcr_strut_rocker
        c74 = triad(config.ax1_jcl_strut_rocker)
        c75 = config.pt1_jcl_strut_rocker
        c76 = A(config.P_rbr_tie_rod).T
        c77 = triad(config.ax1_jcr_tie_upright)
        c78 = config.pt1_jcr_tie_upright
        c79 = -1*multi_dot([c76,config.R_rbr_tie_rod])
        c80 = triad(config.ax1_jcr_tie_steering)
        c81 = A(config.P_vbr_steer).T
        c82 = config.pt1_jcr_tie_steering
        c83 = A(config.P_rbl_tie_rod).T
        c84 = triad(config.ax1_jcl_tie_upright)
        c85 = config.pt1_jcl_tie_upright
        c86 = -1*multi_dot([c83,config.R_rbl_tie_rod])
        c87 = triad(config.ax1_jcl_tie_steering)
        c88 = A(config.P_vbl_steer).T
        c89 = config.pt1_jcl_tie_steering

        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c0,c1])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c2,c1])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c0,c3]) + c4)
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c2,c3]) + c5)
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([c0,c6])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([c7,c6])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([c0,c8]) + c4)
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([c7,c8]) + c9)
        self.Mbar_rbr_uca_jcr_prod_uca = multi_dot([c0,c10])
        self.Mbar_rbr_pushrod_jcr_prod_uca = multi_dot([c11,triad(config.ax2_jcr_prod_uca,c10[0:3,1:2])])
        self.ubar_rbr_uca_jcr_prod_uca = (multi_dot([c0,c12]) + c4)
        self.ubar_rbr_pushrod_jcr_prod_uca = (multi_dot([c11,c12]) + c13)
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c14,c15])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c16,c15])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c14,c17]) + c18)
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c16,c17]) + c19)
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([c14,c20])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([c7,c20])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([c14,c21]) + c18)
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([c7,c21]) + c9)
        self.Mbar_rbl_uca_jcl_prod_uca = multi_dot([c14,c22])
        self.Mbar_rbl_pushrod_jcl_prod_uca = multi_dot([c23,triad(config.ax2_jcl_prod_uca,c22[0:3,1:2])])
        self.ubar_rbl_uca_jcl_prod_uca = (multi_dot([c14,c24]) + c18)
        self.ubar_rbl_pushrod_jcl_prod_uca = (multi_dot([c23,c24]) + c25)
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c26,c27])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c2,c27])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c26,c28]) + c29)
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c2,c28]) + c5)
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([c26,c30])
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([c7,c30])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([c26,c31]) + c29)
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([c7,c31]) + c9)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c32,c33])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c16,c33])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c32,c34]) + c35)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c16,c34]) + c19)
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([c32,c36])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([c7,c36])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([c32,c37]) + c35)
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([c7,c37]) + c9)
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c2,c38])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c39,c38])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c2,c40]) + c5)
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c39,c40]) + -1*multi_dot([c39,config.R_rbr_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c16,c41])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c42,c41])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c16,c43]) + c19)
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c42,c43]) + -1*multi_dot([c42,config.R_rbl_hub]))
        self.Mbar_rbr_rocker_jcr_prod_rocker = multi_dot([c44,c45])
        self.Mbar_rbr_pushrod_jcr_prod_rocker = multi_dot([c11,c45])
        self.ubar_rbr_rocker_jcr_prod_rocker = (multi_dot([c44,c46]) + c47)
        self.ubar_rbr_pushrod_jcr_prod_rocker = (multi_dot([c11,c46]) + c13)
        self.Mbar_rbr_rocker_jcr_rocker_chassis = multi_dot([c44,c48])
        self.Mbar_vbs_chassis_jcr_rocker_chassis = multi_dot([c7,c48])
        self.ubar_rbr_rocker_jcr_rocker_chassis = (multi_dot([c44,c49]) + c47)
        self.ubar_vbs_chassis_jcr_rocker_chassis = (multi_dot([c7,c49]) + c9)
        self.Mbar_rbl_rocker_jcl_prod_rocker = multi_dot([c50,c51])
        self.Mbar_rbl_pushrod_jcl_prod_rocker = multi_dot([c23,c51])
        self.ubar_rbl_rocker_jcl_prod_rocker = (multi_dot([c50,c52]) + c53)
        self.ubar_rbl_pushrod_jcl_prod_rocker = (multi_dot([c23,c52]) + c25)
        self.Mbar_rbl_rocker_jcl_rocker_chassis = multi_dot([c50,c54])
        self.Mbar_vbs_chassis_jcl_rocker_chassis = multi_dot([c7,c54])
        self.ubar_rbl_rocker_jcl_rocker_chassis = (multi_dot([c50,c55]) + c53)
        self.ubar_vbs_chassis_jcl_rocker_chassis = (multi_dot([c7,c55]) + c9)
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([c56,c57])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([c7,triad(config.ax2_jcr_strut_chassis,c57[0:3,1:2])])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c56,c58]) + c59)
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([c7,c58]) + c9)
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c56,c60])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c61,c60])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c56,c62]) + c59)
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c61,c62]) + c63)
        self.ubar_rbr_upper_strut_far_strut = (multi_dot([c56,config.pt1_far_strut]) + c59)
        self.ubar_rbr_lower_strut_far_strut = (multi_dot([c61,config.pt2_far_strut]) + c63)
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([c64,c65])
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([c7,triad(config.ax2_jcl_strut_chassis,c65[0:3,1:2])])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c64,c66]) + c67)
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([c7,c66]) + c9)
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c64,c68])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c69,c68])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c64,c70]) + c67)
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c69,c70]) + c71)
        self.ubar_rbl_upper_strut_fal_strut = (multi_dot([c64,config.pt1_fal_strut]) + c67)
        self.ubar_rbl_lower_strut_fal_strut = (multi_dot([c69,config.pt2_fal_strut]) + c71)
        self.Mbar_rbr_lower_strut_jcr_strut_rocker = multi_dot([c61,c72])
        self.Mbar_rbr_rocker_jcr_strut_rocker = multi_dot([c44,triad(config.ax2_jcr_strut_rocker,c72[0:3,1:2])])
        self.ubar_rbr_lower_strut_jcr_strut_rocker = (multi_dot([c61,c73]) + c63)
        self.ubar_rbr_rocker_jcr_strut_rocker = (multi_dot([c44,c73]) + c47)
        self.Mbar_rbl_lower_strut_jcl_strut_rocker = multi_dot([c69,c74])
        self.Mbar_rbl_rocker_jcl_strut_rocker = multi_dot([c50,triad(config.ax2_jcl_strut_rocker,c74[0:3,1:2])])
        self.ubar_rbl_lower_strut_jcl_strut_rocker = (multi_dot([c69,c75]) + c71)
        self.ubar_rbl_rocker_jcl_strut_rocker = (multi_dot([c50,c75]) + c53)
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c76,c77])
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c2,c77])
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c76,c78]) + c79)
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c2,c78]) + c5)
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([c76,c80])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([c81,triad(config.ax2_jcr_tie_steering,c80[0:3,1:2])])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([c76,c82]) + c79)
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([c81,c82]) + -1*multi_dot([c81,config.R_vbr_steer]))
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c83,c84])
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c16,c84])
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c83,c85]) + c86)
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c16,c85]) + c19)
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([c83,c87])
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([c88,triad(config.ax2_jcl_tie_steering,c87[0:3,1:2])])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([c83,c89]) + c86)
        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([c88,c89]) + -1*multi_dot([c88,config.R_vbl_steer]))

    
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
        self.R_rbr_pushrod = q[42:45,0:1]
        self.P_rbr_pushrod = q[45:49,0:1]
        self.R_rbl_pushrod = q[49:52,0:1]
        self.P_rbl_pushrod = q[52:56,0:1]
        self.R_rbr_rocker = q[56:59,0:1]
        self.P_rbr_rocker = q[59:63,0:1]
        self.R_rbl_rocker = q[63:66,0:1]
        self.P_rbl_rocker = q[66:70,0:1]
        self.R_rbr_upper_strut = q[70:73,0:1]
        self.P_rbr_upper_strut = q[73:77,0:1]
        self.R_rbl_upper_strut = q[77:80,0:1]
        self.P_rbl_upper_strut = q[80:84,0:1]
        self.R_rbr_lower_strut = q[84:87,0:1]
        self.P_rbr_lower_strut = q[87:91,0:1]
        self.R_rbl_lower_strut = q[91:94,0:1]
        self.P_rbl_lower_strut = q[94:98,0:1]
        self.R_rbr_tie_rod = q[98:101,0:1]
        self.P_rbr_tie_rod = q[101:105,0:1]
        self.R_rbl_tie_rod = q[105:108,0:1]
        self.P_rbl_tie_rod = q[108:112,0:1]
        self.R_rbr_hub = q[112:115,0:1]
        self.P_rbr_hub = q[115:119,0:1]
        self.R_rbl_hub = q[119:122,0:1]
        self.P_rbl_hub = q[122:126,0:1]

    
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
        self.Rd_rbr_pushrod = qd[42:45,0:1]
        self.Pd_rbr_pushrod = qd[45:49,0:1]
        self.Rd_rbl_pushrod = qd[49:52,0:1]
        self.Pd_rbl_pushrod = qd[52:56,0:1]
        self.Rd_rbr_rocker = qd[56:59,0:1]
        self.Pd_rbr_rocker = qd[59:63,0:1]
        self.Rd_rbl_rocker = qd[63:66,0:1]
        self.Pd_rbl_rocker = qd[66:70,0:1]
        self.Rd_rbr_upper_strut = qd[70:73,0:1]
        self.Pd_rbr_upper_strut = qd[73:77,0:1]
        self.Rd_rbl_upper_strut = qd[77:80,0:1]
        self.Pd_rbl_upper_strut = qd[80:84,0:1]
        self.Rd_rbr_lower_strut = qd[84:87,0:1]
        self.Pd_rbr_lower_strut = qd[87:91,0:1]
        self.Rd_rbl_lower_strut = qd[91:94,0:1]
        self.Pd_rbl_lower_strut = qd[94:98,0:1]
        self.Rd_rbr_tie_rod = qd[98:101,0:1]
        self.Pd_rbr_tie_rod = qd[101:105,0:1]
        self.Rd_rbl_tie_rod = qd[105:108,0:1]
        self.Pd_rbl_tie_rod = qd[108:112,0:1]
        self.Rd_rbr_hub = qd[112:115,0:1]
        self.Pd_rbr_hub = qd[115:119,0:1]
        self.Rd_rbl_hub = qd[119:122,0:1]
        self.Pd_rbl_hub = qd[122:126,0:1]

    
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
        x11 = -1*self.R_rbr_pushrod
        x12 = self.P_rbr_pushrod
        x13 = A(x12)
        x14 = self.R_rbl_uca
        x15 = self.R_rbl_upright
        x16 = -1*x15
        x17 = self.P_rbl_uca
        x18 = A(x17)
        x19 = self.P_rbl_upright
        x20 = A(x19)
        x21 = x18.T
        x22 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x23 = -1*self.R_rbl_pushrod
        x24 = self.P_rbl_pushrod
        x25 = A(x24)
        x26 = self.R_rbr_lca
        x27 = self.P_rbr_lca
        x28 = A(x27)
        x29 = x28.T
        x30 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x31 = self.R_rbl_lca
        x32 = self.P_rbl_lca
        x33 = A(x32)
        x34 = x33.T
        x35 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x36 = self.P_rbr_hub
        x37 = A(x36)
        x38 = x6.T
        x39 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x40 = self.P_rbl_hub
        x41 = A(x40)
        x42 = x20.T
        x43 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x44 = self.R_rbr_rocker
        x45 = self.P_rbr_rocker
        x46 = A(x45)
        x47 = x46.T
        x48 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        x49 = self.R_rbl_rocker
        x50 = self.P_rbl_rocker
        x51 = A(x50)
        x52 = x51.T
        x53 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        x54 = self.R_rbr_upper_strut
        x55 = self.P_rbr_upper_strut
        x56 = A(x55)
        x57 = x56.T
        x58 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x59 = self.P_rbr_lower_strut
        x60 = A(x59)
        x61 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x62 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x63 = self.R_rbr_lower_strut
        x64 = (x54 + -1*x63 + multi_dot([x56,self.ubar_rbr_upper_strut_jcr_strut]) + -1*multi_dot([x60,self.ubar_rbr_lower_strut_jcr_strut]))
        x65 = self.R_rbl_upper_strut
        x66 = self.P_rbl_upper_strut
        x67 = A(x66)
        x68 = x67.T
        x69 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x70 = self.P_rbl_lower_strut
        x71 = A(x70)
        x72 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x73 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x74 = self.R_rbl_lower_strut
        x75 = (x65 + -1*x74 + multi_dot([x67,self.ubar_rbl_upper_strut_jcl_strut]) + -1*multi_dot([x71,self.ubar_rbl_lower_strut_jcl_strut]))
        x76 = self.R_rbr_tie_rod
        x77 = self.P_rbr_tie_rod
        x78 = A(x77)
        x79 = A(self.P_vbr_steer)
        x80 = self.R_rbl_tie_rod
        x81 = self.P_rbl_tie_rod
        x82 = A(x81)
        x83 = A(self.P_vbl_steer)
        x84 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x2 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_uca_upright])),(x0 + x7 + multi_dot([x4,self.ubar_rbr_uca_jcr_uca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_uca_chassis])),multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),multi_dot([self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),(x0 + x11 + multi_dot([x4,self.ubar_rbr_uca_jcr_prod_uca]) + -1*multi_dot([x13,self.ubar_rbr_pushrod_jcr_prod_uca])),multi_dot([self.Mbar_rbr_uca_jcr_prod_uca[:,0:1].T,x9,x13,self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]]),(x14 + x16 + multi_dot([x18,self.ubar_rbl_uca_jcl_uca_upright]) + -1*multi_dot([x20,self.ubar_rbl_upright_jcl_uca_upright])),(x14 + x7 + multi_dot([x18,self.ubar_rbl_uca_jcl_uca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_uca_chassis])),multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x21,x8,x22]),multi_dot([self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x21,x8,x22]),(x14 + x23 + multi_dot([x18,self.ubar_rbl_uca_jcl_prod_uca]) + -1*multi_dot([x25,self.ubar_rbl_pushrod_jcl_prod_uca])),multi_dot([self.Mbar_rbl_uca_jcl_prod_uca[:,0:1].T,x21,x25,self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]]),(x26 + x2 + multi_dot([x28,self.ubar_rbr_lca_jcr_lca_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_lca_upright])),(x26 + x7 + multi_dot([x28,self.ubar_rbr_lca_jcr_lca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_lca_chassis])),multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x29,x8,x30]),multi_dot([self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x29,x8,x30]),(x31 + x16 + multi_dot([x33,self.ubar_rbl_lca_jcl_lca_upright]) + -1*multi_dot([x20,self.ubar_rbl_upright_jcl_lca_upright])),(x31 + x7 + multi_dot([x33,self.ubar_rbl_lca_jcl_lca_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_lca_chassis])),multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x34,x8,x35]),multi_dot([self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x34,x8,x35]),(x1 + -1*self.R_rbr_hub + multi_dot([x6,self.ubar_rbr_upright_jcr_hub_bearing]) + -1*multi_dot([x37,self.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x38,x37,x39]),multi_dot([self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x38,x37,x39]),(x15 + -1*self.R_rbl_hub + multi_dot([x20,self.ubar_rbl_upright_jcl_hub_bearing]) + -1*multi_dot([x41,self.ubar_rbl_hub_jcl_hub_bearing])),multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x42,x41,x43]),multi_dot([self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x42,x41,x43]),(x44 + x11 + multi_dot([x46,self.ubar_rbr_rocker_jcr_prod_rocker]) + -1*multi_dot([x13,self.ubar_rbr_pushrod_jcr_prod_rocker])),(x44 + x7 + multi_dot([x46,self.ubar_rbr_rocker_jcr_rocker_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_rocker_chassis])),multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1].T,x47,x8,x48]),multi_dot([self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2].T,x47,x8,x48]),(x49 + x23 + multi_dot([x51,self.ubar_rbl_rocker_jcl_prod_rocker]) + -1*multi_dot([x25,self.ubar_rbl_pushrod_jcl_prod_rocker])),(x49 + x7 + multi_dot([x51,self.ubar_rbl_rocker_jcl_rocker_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_rocker_chassis])),multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1].T,x52,x8,x53]),multi_dot([self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2].T,x52,x8,x53]),(x54 + x7 + multi_dot([x56,self.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcr_strut_chassis])),multi_dot([self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x57,x8,self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),multi_dot([x58,x57,x60,x61]),multi_dot([x62,x57,x60,x61]),multi_dot([x58,x57,x64]),multi_dot([x62,x57,x64]),(x65 + x7 + multi_dot([x67,self.ubar_rbl_upper_strut_jcl_strut_chassis]) + -1*multi_dot([x8,self.ubar_vbs_chassis_jcl_strut_chassis])),multi_dot([self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x68,x8,self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),multi_dot([x69,x68,x71,x72]),multi_dot([x73,x68,x71,x72]),multi_dot([x69,x68,x75]),multi_dot([x73,x68,x75]),(x63 + -1*x44 + multi_dot([x60,self.ubar_rbr_lower_strut_jcr_strut_rocker]) + -1*multi_dot([x46,self.ubar_rbr_rocker_jcr_strut_rocker])),multi_dot([self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1].T,x60.T,x46,self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]]),(x74 + -1*x49 + multi_dot([x71,self.ubar_rbl_lower_strut_jcl_strut_rocker]) + -1*multi_dot([x51,self.ubar_rbl_rocker_jcl_strut_rocker])),multi_dot([self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1].T,x71.T,x51,self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]]),(x76 + x2 + multi_dot([x78,self.ubar_rbr_tie_rod_jcr_tie_upright]) + -1*multi_dot([x6,self.ubar_rbr_upright_jcr_tie_upright])),(x76 + -1*self.R_vbr_steer + multi_dot([x78,self.ubar_rbr_tie_rod_jcr_tie_steering]) + -1*multi_dot([x79,self.ubar_vbr_steer_jcr_tie_steering])),multi_dot([self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x78.T,x79,self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),(x80 + x16 + multi_dot([x82,self.ubar_rbl_tie_rod_jcl_tie_upright]) + -1*multi_dot([x20,self.ubar_rbl_upright_jcl_tie_upright])),(x80 + -1*self.R_vbl_steer + multi_dot([x82,self.ubar_rbl_tie_rod_jcl_tie_steering]) + -1*multi_dot([x83,self.ubar_vbl_steer_jcl_tie_steering])),multi_dot([self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x82.T,x83,self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),(x84 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x84 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x84 + (multi_dot([x27.T,x27]))**(1.0/2.0)),(x84 + (multi_dot([x32.T,x32]))**(1.0/2.0)),(x84 + (multi_dot([x5.T,x5]))**(1.0/2.0)),(x84 + (multi_dot([x19.T,x19]))**(1.0/2.0)),(x84 + (multi_dot([x12.T,x12]))**(1.0/2.0)),(x84 + (multi_dot([x24.T,x24]))**(1.0/2.0)),(x84 + (multi_dot([x45.T,x45]))**(1.0/2.0)),(x84 + (multi_dot([x50.T,x50]))**(1.0/2.0)),(x84 + (multi_dot([x55.T,x55]))**(1.0/2.0)),(x84 + (multi_dot([x66.T,x66]))**(1.0/2.0)),(x84 + (multi_dot([x59.T,x59]))**(1.0/2.0)),(x84 + (multi_dot([x70.T,x70]))**(1.0/2.0)),(x84 + (multi_dot([x77.T,x77]))**(1.0/2.0)),(x84 + (multi_dot([x81.T,x81]))**(1.0/2.0)),(x84 + (multi_dot([x36.T,x36]))**(1.0/2.0)),(x84 + (multi_dot([x40.T,x40]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v0,v1,v1,v0,v1,v0,v0,v1,v1,v0,v1,v0,v0,v1,v1,v0,v0,v1,v1,v0,v1,v1,v0,v1,v1,v0,v0,v1,v1,v0,v0,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v0,v1,v0,v0,v1,v0,v0,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_rbr_upright
        a2 = self.Pd_vbs_chassis
        a3 = self.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a4 = self.P_rbr_uca
        a5 = A(a4).T
        a6 = self.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        a7 = B(a2,a6)
        a8 = a6.T
        a9 = self.P_vbs_chassis
        a10 = A(a9).T
        a11 = a0.T
        a12 = B(a9,a6)
        a13 = self.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a14 = self.Pd_rbr_pushrod
        a15 = self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]
        a16 = self.P_rbr_pushrod
        a17 = self.Mbar_rbr_uca_jcr_prod_uca[:,0:1]
        a18 = self.Pd_rbl_uca
        a19 = self.Pd_rbl_upright
        a20 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a21 = a20.T
        a22 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a23 = self.P_rbl_uca
        a24 = A(a23).T
        a25 = B(a2,a20)
        a26 = a18.T
        a27 = B(a9,a20)
        a28 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a29 = self.Pd_rbl_pushrod
        a30 = self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]
        a31 = self.P_rbl_pushrod
        a32 = self.Mbar_rbl_uca_jcl_prod_uca[:,0:1]
        a33 = self.Pd_rbr_lca
        a34 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a35 = self.P_rbr_lca
        a36 = A(a35).T
        a37 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a38 = B(a2,a37)
        a39 = a37.T
        a40 = a33.T
        a41 = B(a9,a37)
        a42 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a43 = self.Pd_rbl_lca
        a44 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a45 = self.P_rbl_lca
        a46 = A(a45).T
        a47 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a48 = B(a2,a47)
        a49 = a47.T
        a50 = a43.T
        a51 = B(a9,a47)
        a52 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a53 = self.Pd_rbr_hub
        a54 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a55 = a54.T
        a56 = self.P_rbr_hub
        a57 = A(a56).T
        a58 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a59 = self.P_rbr_upright
        a60 = A(a59).T
        a61 = B(a53,a54)
        a62 = a1.T
        a63 = B(a56,a54)
        a64 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a65 = self.Pd_rbl_hub
        a66 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a67 = a66.T
        a68 = self.P_rbl_hub
        a69 = A(a68).T
        a70 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a71 = self.P_rbl_upright
        a72 = A(a71).T
        a73 = B(a65,a66)
        a74 = a19.T
        a75 = B(a68,a66)
        a76 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a77 = self.Pd_rbr_rocker
        a78 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        a79 = a78.T
        a80 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        a81 = self.P_rbr_rocker
        a82 = A(a81).T
        a83 = B(a2,a78)
        a84 = a77.T
        a85 = B(a9,a78)
        a86 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        a87 = self.Pd_rbl_rocker
        a88 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        a89 = self.P_rbl_rocker
        a90 = A(a89).T
        a91 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        a92 = B(a2,a91)
        a93 = a91.T
        a94 = a87.T
        a95 = B(a9,a91)
        a96 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        a97 = self.Pd_rbr_upper_strut
        a98 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a99 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a100 = self.P_rbr_upper_strut
        a101 = A(a100).T
        a102 = a97.T
        a103 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a104 = a103.T
        a105 = self.Pd_rbr_lower_strut
        a106 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a107 = B(a105,a106)
        a108 = a106.T
        a109 = self.P_rbr_lower_strut
        a110 = A(a109).T
        a111 = B(a97,a103)
        a112 = B(a100,a103).T
        a113 = B(a109,a106)
        a114 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a115 = a114.T
        a116 = B(a97,a114)
        a117 = B(a100,a114).T
        a118 = self.ubar_rbr_upper_strut_jcr_strut
        a119 = self.ubar_rbr_lower_strut_jcr_strut
        a120 = (multi_dot([B(a97,a118),a97]) + -1*multi_dot([B(a105,a119),a105]))
        a121 = (self.Rd_rbr_upper_strut + -1*self.Rd_rbr_lower_strut + multi_dot([B(a109,a119),a105]) + multi_dot([B(a100,a118),a97]))
        a122 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([a118.T,a101]) + -1*multi_dot([a119.T,a110]))
        a123 = self.Pd_rbl_upper_strut
        a124 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a125 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a126 = self.P_rbl_upper_strut
        a127 = A(a126).T
        a128 = a123.T
        a129 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a130 = a129.T
        a131 = self.P_rbl_lower_strut
        a132 = A(a131).T
        a133 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a134 = B(a123,a133)
        a135 = a133.T
        a136 = self.Pd_rbl_lower_strut
        a137 = B(a136,a129)
        a138 = B(a126,a133).T
        a139 = B(a131,a129)
        a140 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a141 = B(a123,a140)
        a142 = a140.T
        a143 = B(a126,a140).T
        a144 = self.ubar_rbl_upper_strut_jcl_strut
        a145 = self.ubar_rbl_lower_strut_jcl_strut
        a146 = (multi_dot([B(a123,a144),a123]) + -1*multi_dot([B(a136,a145),a136]))
        a147 = (self.Rd_rbl_upper_strut + -1*self.Rd_rbl_lower_strut + multi_dot([B(a131,a145),a136]) + multi_dot([B(a126,a144),a123]))
        a148 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([a144.T,a127]) + -1*multi_dot([a145.T,a132]))
        a149 = self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]
        a150 = self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]
        a151 = a105.T
        a152 = self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]
        a153 = self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]
        a154 = a136.T
        a155 = self.Pd_rbr_tie_rod
        a156 = self.Pd_vbr_steer
        a157 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a158 = self.P_rbr_tie_rod
        a159 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a160 = self.P_vbr_steer
        a161 = a155.T
        a162 = self.Pd_rbl_tie_rod
        a163 = self.Pd_vbl_steer
        a164 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a165 = self.P_rbl_tie_rod
        a166 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a167 = self.P_vbl_steer
        a168 = a162.T

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_upright),a0]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_uca_upright),a1])),(multi_dot([B(a0,self.ubar_rbr_uca_jcr_uca_chassis),a0]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_uca_chassis),a2])),(multi_dot([a3.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a3),a0]) + 2*multi_dot([a11,B(a4,a3).T,a12,a2])),(multi_dot([a13.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a13),a0]) + 2*multi_dot([a11,B(a4,a13).T,a12,a2])),(multi_dot([B(a0,self.ubar_rbr_uca_jcr_prod_uca),a0]) + -1*multi_dot([B(a14,self.ubar_rbr_pushrod_jcr_prod_uca),a14])),(multi_dot([a15.T,A(a16).T,B(a0,a17),a0]) + multi_dot([a17.T,a5,B(a14,a15),a14]) + 2*multi_dot([a11,B(a4,a17).T,B(a16,a15),a14])),(multi_dot([B(a18,self.ubar_rbl_uca_jcl_uca_upright),a18]) + -1*multi_dot([B(a19,self.ubar_rbl_upright_jcl_uca_upright),a19])),(multi_dot([B(a18,self.ubar_rbl_uca_jcl_uca_chassis),a18]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_uca_chassis),a2])),(multi_dot([a21,a10,B(a18,a22),a18]) + multi_dot([a22.T,a24,a25,a2]) + 2*multi_dot([a26,B(a23,a22).T,a27,a2])),(multi_dot([a21,a10,B(a18,a28),a18]) + multi_dot([a28.T,a24,a25,a2]) + 2*multi_dot([a26,B(a23,a28).T,a27,a2])),(multi_dot([B(a18,self.ubar_rbl_uca_jcl_prod_uca),a18]) + -1*multi_dot([B(a29,self.ubar_rbl_pushrod_jcl_prod_uca),a29])),(multi_dot([a30.T,A(a31).T,B(a18,a32),a18]) + multi_dot([a32.T,a24,B(a29,a30),a29]) + 2*multi_dot([a26,B(a23,a32).T,B(a31,a30),a29])),(multi_dot([B(a33,self.ubar_rbr_lca_jcr_lca_upright),a33]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_lca_upright),a1])),(multi_dot([B(a33,self.ubar_rbr_lca_jcr_lca_chassis),a33]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_lca_chassis),a2])),(multi_dot([a34.T,a36,a38,a2]) + multi_dot([a39,a10,B(a33,a34),a33]) + 2*multi_dot([a40,B(a35,a34).T,a41,a2])),(multi_dot([a42.T,a36,a38,a2]) + multi_dot([a39,a10,B(a33,a42),a33]) + 2*multi_dot([a40,B(a35,a42).T,a41,a2])),(multi_dot([B(a43,self.ubar_rbl_lca_jcl_lca_upright),a43]) + -1*multi_dot([B(a19,self.ubar_rbl_upright_jcl_lca_upright),a19])),(multi_dot([B(a43,self.ubar_rbl_lca_jcl_lca_chassis),a43]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_lca_chassis),a2])),(multi_dot([a44.T,a46,a48,a2]) + multi_dot([a49,a10,B(a43,a44),a43]) + 2*multi_dot([a50,B(a45,a44).T,a51,a2])),(multi_dot([a52.T,a46,a48,a2]) + multi_dot([a49,a10,B(a43,a52),a43]) + 2*multi_dot([a50,B(a45,a52).T,a51,a2])),(multi_dot([B(a1,self.ubar_rbr_upright_jcr_hub_bearing),a1]) + -1*multi_dot([B(a53,self.ubar_rbr_hub_jcr_hub_bearing),a53])),(multi_dot([a55,a57,B(a1,a58),a1]) + multi_dot([a58.T,a60,a61,a53]) + 2*multi_dot([a62,B(a59,a58).T,a63,a53])),(multi_dot([a55,a57,B(a1,a64),a1]) + multi_dot([a64.T,a60,a61,a53]) + 2*multi_dot([a62,B(a59,a64).T,a63,a53])),(multi_dot([B(a19,self.ubar_rbl_upright_jcl_hub_bearing),a19]) + -1*multi_dot([B(a65,self.ubar_rbl_hub_jcl_hub_bearing),a65])),(multi_dot([a67,a69,B(a19,a70),a19]) + multi_dot([a70.T,a72,a73,a65]) + 2*multi_dot([a74,B(a71,a70).T,a75,a65])),(multi_dot([a67,a69,B(a19,a76),a19]) + multi_dot([a76.T,a72,a73,a65]) + 2*multi_dot([a74,B(a71,a76).T,a75,a65])),(multi_dot([B(a77,self.ubar_rbr_rocker_jcr_prod_rocker),a77]) + -1*multi_dot([B(a14,self.ubar_rbr_pushrod_jcr_prod_rocker),a14])),(multi_dot([B(a77,self.ubar_rbr_rocker_jcr_rocker_chassis),a77]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_rocker_chassis),a2])),(multi_dot([a79,a10,B(a77,a80),a77]) + multi_dot([a80.T,a82,a83,a2]) + 2*multi_dot([a84,B(a81,a80).T,a85,a2])),(multi_dot([a79,a10,B(a77,a86),a77]) + multi_dot([a86.T,a82,a83,a2]) + 2*multi_dot([a84,B(a81,a86).T,a85,a2])),(multi_dot([B(a87,self.ubar_rbl_rocker_jcl_prod_rocker),a87]) + -1*multi_dot([B(a29,self.ubar_rbl_pushrod_jcl_prod_rocker),a29])),(multi_dot([B(a87,self.ubar_rbl_rocker_jcl_rocker_chassis),a87]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_rocker_chassis),a2])),(multi_dot([a88.T,a90,a92,a2]) + multi_dot([a93,a10,B(a87,a88),a87]) + 2*multi_dot([a94,B(a89,a88).T,a95,a2])),(multi_dot([a96.T,a90,a92,a2]) + multi_dot([a93,a10,B(a87,a96),a87]) + 2*multi_dot([a94,B(a89,a96).T,a95,a2])),(multi_dot([B(a97,self.ubar_rbr_upper_strut_jcr_strut_chassis),a97]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcr_strut_chassis),a2])),(multi_dot([a98.T,a10,B(a97,a99),a97]) + multi_dot([a99.T,a101,B(a2,a98),a2]) + 2*multi_dot([a102,B(a100,a99).T,B(a9,a98),a2])),(multi_dot([a104,a101,a107,a105]) + multi_dot([a108,a110,a111,a97]) + 2*multi_dot([a102,a112,a113,a105])),(multi_dot([a115,a101,a107,a105]) + multi_dot([a108,a110,a116,a97]) + 2*multi_dot([a102,a117,a113,a105])),(multi_dot([a104,a101,a120]) + 2*multi_dot([a102,a112,a121]) + multi_dot([a122,a111,a97])),(multi_dot([a115,a101,a120]) + 2*multi_dot([a102,a117,a121]) + multi_dot([a122,a116,a97])),(multi_dot([B(a123,self.ubar_rbl_upper_strut_jcl_strut_chassis),a123]) + -1*multi_dot([B(a2,self.ubar_vbs_chassis_jcl_strut_chassis),a2])),(multi_dot([a124.T,a10,B(a123,a125),a123]) + multi_dot([a125.T,a127,B(a2,a124),a2]) + 2*multi_dot([a128,B(a126,a125).T,B(a9,a124),a2])),(multi_dot([a130,a132,a134,a123]) + multi_dot([a135,a127,a137,a136]) + 2*multi_dot([a128,a138,a139,a136])),(multi_dot([a130,a132,a141,a123]) + multi_dot([a142,a127,a137,a136]) + 2*multi_dot([a128,a143,a139,a136])),(multi_dot([a135,a127,a146]) + 2*multi_dot([a128,a138,a147]) + multi_dot([a148,a134,a123])),(multi_dot([a142,a127,a146]) + 2*multi_dot([a128,a143,a147]) + multi_dot([a148,a141,a123])),(multi_dot([B(a105,self.ubar_rbr_lower_strut_jcr_strut_rocker),a105]) + -1*multi_dot([B(a77,self.ubar_rbr_rocker_jcr_strut_rocker),a77])),(multi_dot([a149.T,a82,B(a105,a150),a105]) + multi_dot([a150.T,a110,B(a77,a149),a77]) + 2*multi_dot([a151,B(a109,a150).T,B(a81,a149),a77])),(multi_dot([B(a136,self.ubar_rbl_lower_strut_jcl_strut_rocker),a136]) + -1*multi_dot([B(a87,self.ubar_rbl_rocker_jcl_strut_rocker),a87])),(multi_dot([a152.T,a90,B(a136,a153),a136]) + multi_dot([a153.T,a132,B(a87,a152),a87]) + 2*multi_dot([a154,B(a131,a153).T,B(a89,a152),a87])),(multi_dot([B(a155,self.ubar_rbr_tie_rod_jcr_tie_upright),a155]) + -1*multi_dot([B(a1,self.ubar_rbr_upright_jcr_tie_upright),a1])),(multi_dot([B(a155,self.ubar_rbr_tie_rod_jcr_tie_steering),a155]) + -1*multi_dot([B(a156,self.ubar_vbr_steer_jcr_tie_steering),a156])),(multi_dot([a157.T,A(a158).T,B(a156,a159),a156]) + multi_dot([a159.T,A(a160).T,B(a155,a157),a155]) + 2*multi_dot([a161,B(a158,a157).T,B(a160,a159),a156])),(multi_dot([B(a162,self.ubar_rbl_tie_rod_jcl_tie_upright),a162]) + -1*multi_dot([B(a19,self.ubar_rbl_upright_jcl_tie_upright),a19])),(multi_dot([B(a162,self.ubar_rbl_tie_rod_jcl_tie_steering),a162]) + -1*multi_dot([B(a163,self.ubar_vbl_steer_jcl_tie_steering),a163])),(multi_dot([a164.T,A(a165).T,B(a163,a166),a163]) + multi_dot([a166.T,A(a167).T,B(a162,a164),a162]) + 2*multi_dot([a168,B(a165,a164).T,B(a167,a166),a163])),2*(multi_dot([a11,a0]))**(1.0/2.0),2*(multi_dot([a26,a18]))**(1.0/2.0),2*(multi_dot([a40,a33]))**(1.0/2.0),2*(multi_dot([a50,a43]))**(1.0/2.0),2*(multi_dot([a62,a1]))**(1.0/2.0),2*(multi_dot([a74,a19]))**(1.0/2.0),2*(multi_dot([a14.T,a14]))**(1.0/2.0),2*(multi_dot([a29.T,a29]))**(1.0/2.0),2*(multi_dot([a84,a77]))**(1.0/2.0),2*(multi_dot([a94,a87]))**(1.0/2.0),2*(multi_dot([a102,a97]))**(1.0/2.0),2*(multi_dot([a128,a123]))**(1.0/2.0),2*(multi_dot([a151,a105]))**(1.0/2.0),2*(multi_dot([a154,a136]))**(1.0/2.0),2*(multi_dot([a161,a155]))**(1.0/2.0),2*(multi_dot([a168,a162]))**(1.0/2.0),2*(multi_dot([a53.T,a53]))**(1.0/2.0),2*(multi_dot([a65.T,a65]))**(1.0/2.0)]

    
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
        j13 = self.Mbar_rbr_pushrod_jcr_prod_uca[:,0:1]
        j14 = self.P_rbr_pushrod
        j15 = self.Mbar_rbr_uca_jcr_prod_uca[:,0:1]
        j16 = self.P_rbl_uca
        j17 = self.P_rbl_upright
        j18 = self.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j19 = j18.T
        j20 = self.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j21 = self.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j22 = A(j16).T
        j23 = B(j7,j18)
        j24 = self.Mbar_rbl_pushrod_jcl_prod_uca[:,0:1]
        j25 = self.P_rbl_pushrod
        j26 = self.Mbar_rbl_uca_jcl_prod_uca[:,0:1]
        j27 = self.P_rbr_lca
        j28 = self.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j29 = j28.T
        j30 = self.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j31 = self.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j32 = A(j27).T
        j33 = B(j7,j28)
        j34 = self.P_rbl_lca
        j35 = self.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j36 = j35.T
        j37 = self.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j38 = self.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j39 = A(j34).T
        j40 = B(j7,j35)
        j41 = self.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j42 = j41.T
        j43 = self.P_rbr_hub
        j44 = A(j43).T
        j45 = self.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j46 = self.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j47 = A(j3).T
        j48 = B(j43,j41)
        j49 = self.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j50 = j49.T
        j51 = self.P_rbl_hub
        j52 = A(j51).T
        j53 = self.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j54 = self.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j55 = A(j17).T
        j56 = B(j51,j49)
        j57 = self.P_rbr_rocker
        j58 = self.Mbar_vbs_chassis_jcr_rocker_chassis[:,2:3]
        j59 = j58.T
        j60 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,0:1]
        j61 = self.Mbar_rbr_rocker_jcr_rocker_chassis[:,1:2]
        j62 = A(j57).T
        j63 = B(j7,j58)
        j64 = self.P_rbl_rocker
        j65 = self.Mbar_vbs_chassis_jcl_rocker_chassis[:,2:3]
        j66 = j65.T
        j67 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,0:1]
        j68 = self.Mbar_rbl_rocker_jcl_rocker_chassis[:,1:2]
        j69 = A(j64).T
        j70 = B(j7,j65)
        j71 = self.P_rbr_upper_strut
        j72 = self.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j73 = self.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j74 = A(j71).T
        j75 = self.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j76 = j75.T
        j77 = self.P_rbr_lower_strut
        j78 = A(j77).T
        j79 = self.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j80 = B(j71,j79)
        j81 = self.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j82 = B(j71,j81)
        j83 = j79.T
        j84 = multi_dot([j83,j74])
        j85 = self.ubar_rbr_upper_strut_jcr_strut
        j86 = B(j71,j85)
        j87 = self.ubar_rbr_lower_strut_jcr_strut
        j88 = (self.R_rbr_upper_strut.T + -1*self.R_rbr_lower_strut.T + multi_dot([j85.T,j74]) + -1*multi_dot([j87.T,j78]))
        j89 = j81.T
        j90 = multi_dot([j89,j74])
        j91 = B(j77,j75)
        j92 = B(j77,j87)
        j93 = self.P_rbl_upper_strut
        j94 = self.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j95 = self.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j96 = A(j93).T
        j97 = self.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j98 = j97.T
        j99 = self.P_rbl_lower_strut
        j100 = A(j99).T
        j101 = self.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j102 = B(j93,j101)
        j103 = self.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j104 = B(j93,j103)
        j105 = j101.T
        j106 = multi_dot([j105,j96])
        j107 = self.ubar_rbl_upper_strut_jcl_strut
        j108 = B(j93,j107)
        j109 = self.ubar_rbl_lower_strut_jcl_strut
        j110 = (self.R_rbl_upper_strut.T + -1*self.R_rbl_lower_strut.T + multi_dot([j107.T,j96]) + -1*multi_dot([j109.T,j100]))
        j111 = j103.T
        j112 = multi_dot([j111,j96])
        j113 = B(j99,j97)
        j114 = B(j99,j109)
        j115 = self.Mbar_rbr_rocker_jcr_strut_rocker[:,0:1]
        j116 = self.Mbar_rbr_lower_strut_jcr_strut_rocker[:,0:1]
        j117 = self.Mbar_rbl_rocker_jcl_strut_rocker[:,0:1]
        j118 = self.Mbar_rbl_lower_strut_jcl_strut_rocker[:,0:1]
        j119 = self.P_rbr_tie_rod
        j120 = self.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j121 = self.P_vbr_steer
        j122 = self.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j123 = self.P_rbl_tie_rod
        j124 = self.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j125 = self.P_vbl_steer
        j126 = self.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = [j0,B(j1,self.ubar_rbr_uca_jcr_uca_upright),j2,-1*B(j3,self.ubar_rbr_upright_jcr_uca_upright),j0,B(j1,self.ubar_rbr_uca_jcr_uca_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcr_uca_chassis),j4,multi_dot([j6,j8,B(j1,j9)]),j4,multi_dot([j9.T,j11,j12]),j4,multi_dot([j6,j8,B(j1,j10)]),j4,multi_dot([j10.T,j11,j12]),j0,B(j1,self.ubar_rbr_uca_jcr_prod_uca),j2,-1*B(j14,self.ubar_rbr_pushrod_jcr_prod_uca),j4,multi_dot([j13.T,A(j14).T,B(j1,j15)]),j4,multi_dot([j15.T,j11,B(j14,j13)]),j0,B(j16,self.ubar_rbl_uca_jcl_uca_upright),j2,-1*B(j17,self.ubar_rbl_upright_jcl_uca_upright),j0,B(j16,self.ubar_rbl_uca_jcl_uca_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcl_uca_chassis),j4,multi_dot([j19,j8,B(j16,j20)]),j4,multi_dot([j20.T,j22,j23]),j4,multi_dot([j19,j8,B(j16,j21)]),j4,multi_dot([j21.T,j22,j23]),j0,B(j16,self.ubar_rbl_uca_jcl_prod_uca),j2,-1*B(j25,self.ubar_rbl_pushrod_jcl_prod_uca),j4,multi_dot([j24.T,A(j25).T,B(j16,j26)]),j4,multi_dot([j26.T,j22,B(j25,j24)]),j0,B(j27,self.ubar_rbr_lca_jcr_lca_upright),j2,-1*B(j3,self.ubar_rbr_upright_jcr_lca_upright),j0,B(j27,self.ubar_rbr_lca_jcr_lca_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcr_lca_chassis),j4,multi_dot([j29,j8,B(j27,j30)]),j4,multi_dot([j30.T,j32,j33]),j4,multi_dot([j29,j8,B(j27,j31)]),j4,multi_dot([j31.T,j32,j33]),j0,B(j34,self.ubar_rbl_lca_jcl_lca_upright),j2,-1*B(j17,self.ubar_rbl_upright_jcl_lca_upright),j0,B(j34,self.ubar_rbl_lca_jcl_lca_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcl_lca_chassis),j4,multi_dot([j36,j8,B(j34,j37)]),j4,multi_dot([j37.T,j39,j40]),j4,multi_dot([j36,j8,B(j34,j38)]),j4,multi_dot([j38.T,j39,j40]),j0,B(j3,self.ubar_rbr_upright_jcr_hub_bearing),j2,-1*B(j43,self.ubar_rbr_hub_jcr_hub_bearing),j4,multi_dot([j42,j44,B(j3,j45)]),j4,multi_dot([j45.T,j47,j48]),j4,multi_dot([j42,j44,B(j3,j46)]),j4,multi_dot([j46.T,j47,j48]),j0,B(j17,self.ubar_rbl_upright_jcl_hub_bearing),j2,-1*B(j51,self.ubar_rbl_hub_jcl_hub_bearing),j4,multi_dot([j50,j52,B(j17,j53)]),j4,multi_dot([j53.T,j55,j56]),j4,multi_dot([j50,j52,B(j17,j54)]),j4,multi_dot([j54.T,j55,j56]),j2,-1*B(j14,self.ubar_rbr_pushrod_jcr_prod_rocker),j0,B(j57,self.ubar_rbr_rocker_jcr_prod_rocker),j0,B(j57,self.ubar_rbr_rocker_jcr_rocker_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcr_rocker_chassis),j4,multi_dot([j59,j8,B(j57,j60)]),j4,multi_dot([j60.T,j62,j63]),j4,multi_dot([j59,j8,B(j57,j61)]),j4,multi_dot([j61.T,j62,j63]),j2,-1*B(j25,self.ubar_rbl_pushrod_jcl_prod_rocker),j0,B(j64,self.ubar_rbl_rocker_jcl_prod_rocker),j0,B(j64,self.ubar_rbl_rocker_jcl_rocker_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcl_rocker_chassis),j4,multi_dot([j66,j8,B(j64,j67)]),j4,multi_dot([j67.T,j69,j70]),j4,multi_dot([j66,j8,B(j64,j68)]),j4,multi_dot([j68.T,j69,j70]),j0,B(j71,self.ubar_rbr_upper_strut_jcr_strut_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcr_strut_chassis),j4,multi_dot([j72.T,j8,B(j71,j73)]),j4,multi_dot([j73.T,j74,B(j7,j72)]),j4,multi_dot([j76,j78,j80]),j4,multi_dot([j83,j74,j91]),j4,multi_dot([j76,j78,j82]),j4,multi_dot([j89,j74,j91]),j84,(multi_dot([j83,j74,j86]) + multi_dot([j88,j80])),-1*j84,-1*multi_dot([j83,j74,j92]),j90,(multi_dot([j89,j74,j86]) + multi_dot([j88,j82])),-1*j90,-1*multi_dot([j89,j74,j92]),j0,B(j93,self.ubar_rbl_upper_strut_jcl_strut_chassis),j2,-1*B(j7,self.ubar_vbs_chassis_jcl_strut_chassis),j4,multi_dot([j94.T,j8,B(j93,j95)]),j4,multi_dot([j95.T,j96,B(j7,j94)]),j4,multi_dot([j98,j100,j102]),j4,multi_dot([j105,j96,j113]),j4,multi_dot([j98,j100,j104]),j4,multi_dot([j111,j96,j113]),j106,(multi_dot([j105,j96,j108]) + multi_dot([j110,j102])),-1*j106,-1*multi_dot([j105,j96,j114]),j112,(multi_dot([j111,j96,j108]) + multi_dot([j110,j104])),-1*j112,-1*multi_dot([j111,j96,j114]),j2,-1*B(j57,self.ubar_rbr_rocker_jcr_strut_rocker),j0,B(j77,self.ubar_rbr_lower_strut_jcr_strut_rocker),j4,multi_dot([j116.T,j78,B(j57,j115)]),j4,multi_dot([j115.T,j62,B(j77,j116)]),j2,-1*B(j64,self.ubar_rbl_rocker_jcl_strut_rocker),j0,B(j99,self.ubar_rbl_lower_strut_jcl_strut_rocker),j4,multi_dot([j118.T,j100,B(j64,j117)]),j4,multi_dot([j117.T,j69,B(j99,j118)]),j2,-1*B(j3,self.ubar_rbr_upright_jcr_tie_upright),j0,B(j119,self.ubar_rbr_tie_rod_jcr_tie_upright),j0,B(j119,self.ubar_rbr_tie_rod_jcr_tie_steering),j2,-1*B(j121,self.ubar_vbr_steer_jcr_tie_steering),j4,multi_dot([j120.T,A(j121).T,B(j119,j122)]),j4,multi_dot([j122.T,A(j119).T,B(j121,j120)]),j2,-1*B(j17,self.ubar_rbl_upright_jcl_tie_upright),j0,B(j123,self.ubar_rbl_tie_rod_jcl_tie_upright),j0,B(j123,self.ubar_rbl_tie_rod_jcl_tie_steering),j2,-1*B(j125,self.ubar_vbl_steer_jcl_tie_steering),j4,multi_dot([j124.T,A(j125).T,B(j123,j126)]),j4,multi_dot([j126.T,A(j123).T,B(j125,j124)]),2*j1.T,2*j16.T,2*j27.T,2*j34.T,2*j3.T,2*j17.T,2*j14.T,2*j25.T,2*j57.T,2*j64.T,2*j71.T,2*j93.T,2*j77.T,2*j99.T,2*j119.T,2*j123.T,2*j43.T,2*j51.T]
  
    
