
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_uca = 1
        self.Jbar_rbr_uca = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_lca = 1
        self.Jbar_rbr_lca = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_upright = 1
        self.Jbar_rbr_upright = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_upper_strut = 1
        self.Jbar_rbr_upper_strut = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_lower_strut = 1
        self.Jbar_rbr_lower_strut = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_tie_rod = 1
        self.Jbar_rbr_tie_rod = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
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
        self.far_strut_FL = np.array([[0]],dtype=np.float64)
        self.T_rbr_upper_strut_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.T_rbr_lower_strut_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbr_uca,self.P_rbr_uca,self.R_rbl_uca,self.P_rbl_uca,self.R_rbr_lca,self.P_rbr_lca,self.R_rbl_lca,self.P_rbl_lca,self.R_rbr_upright,self.P_rbr_upright,self.R_rbl_upright,self.P_rbl_upright,self.R_rbr_upper_strut,self.P_rbr_upper_strut,self.R_rbl_upper_strut,self.P_rbl_upper_strut,self.R_rbr_lower_strut,self.P_rbr_lower_strut,self.R_rbl_lower_strut,self.P_rbl_lower_strut,self.R_rbr_tie_rod,self.P_rbr_tie_rod,self.R_rbl_tie_rod,self.P_rbl_tie_rod,self.R_rbr_hub,self.P_rbr_hub,self.R_rbl_hub,self.P_rbl_hub])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbr_uca,self.Pd_rbr_uca,self.Rd_rbl_uca,self.Pd_rbl_uca,self.Rd_rbr_lca,self.Pd_rbr_lca,self.Rd_rbl_lca,self.Pd_rbl_lca,self.Rd_rbr_upright,self.Pd_rbr_upright,self.Rd_rbl_upright,self.Pd_rbl_upright,self.Rd_rbr_upper_strut,self.Pd_rbr_upper_strut,self.Rd_rbl_upper_strut,self.Pd_rbl_upper_strut,self.Rd_rbr_lower_strut,self.Pd_rbr_lower_strut,self.Rd_rbl_lower_strut,self.Pd_rbl_lower_strut,self.Rd_rbr_tie_rod,self.Pd_rbr_tie_rod,self.Rd_rbl_tie_rod,self.Pd_rbl_tie_rod,self.Rd_rbr_hub,self.Pd_rbr_hub,self.Rd_rbl_hub,self.Pd_rbl_hub])
        return qd

    def load_from_csv(self,csv_file):
        dataframe = pd.read_csv(csv_file,index_col=0)
        for ind in dataframe.index:
            value = getattr(self,ind)
            if isinstance(value, np.ndarray):
                shape = value.shape
                v = np.array(dataframe.loc[ind],dtype=np.float64)
                v = np.resize(v,shape)
                setattr(self,ind,v)
            else:
                v = dataframe.loc[ind][0]
                setattr(self,ind,v)
        self._set_arguments()

    def _set_arguments(self):
        self.R_rbl_uca = mirrored(self.R_rbr_uca)
        self.P_rbl_uca = mirrored(self.P_rbr_uca)
        self.Rd_rbl_uca = mirrored(self.Rd_rbr_uca)
        self.Pd_rbl_uca = mirrored(self.Pd_rbr_uca)
        self.Rdd_rbl_uca = mirrored(self.Rdd_rbr_uca)
        self.Pdd_rbl_uca = mirrored(self.Pdd_rbr_uca)
        self.m_rbl_uca = self.m_rbr_uca
        self.Jbar_rbl_uca = mirrored(self.Jbar_rbr_uca)
        self.R_rbl_lca = mirrored(self.R_rbr_lca)
        self.P_rbl_lca = mirrored(self.P_rbr_lca)
        self.Rd_rbl_lca = mirrored(self.Rd_rbr_lca)
        self.Pd_rbl_lca = mirrored(self.Pd_rbr_lca)
        self.Rdd_rbl_lca = mirrored(self.Rdd_rbr_lca)
        self.Pdd_rbl_lca = mirrored(self.Pdd_rbr_lca)
        self.m_rbl_lca = self.m_rbr_lca
        self.Jbar_rbl_lca = mirrored(self.Jbar_rbr_lca)
        self.R_rbl_upright = mirrored(self.R_rbr_upright)
        self.P_rbl_upright = mirrored(self.P_rbr_upright)
        self.Rd_rbl_upright = mirrored(self.Rd_rbr_upright)
        self.Pd_rbl_upright = mirrored(self.Pd_rbr_upright)
        self.Rdd_rbl_upright = mirrored(self.Rdd_rbr_upright)
        self.Pdd_rbl_upright = mirrored(self.Pdd_rbr_upright)
        self.m_rbl_upright = self.m_rbr_upright
        self.Jbar_rbl_upright = mirrored(self.Jbar_rbr_upright)
        self.R_rbl_upper_strut = mirrored(self.R_rbr_upper_strut)
        self.P_rbl_upper_strut = mirrored(self.P_rbr_upper_strut)
        self.Rd_rbl_upper_strut = mirrored(self.Rd_rbr_upper_strut)
        self.Pd_rbl_upper_strut = mirrored(self.Pd_rbr_upper_strut)
        self.Rdd_rbl_upper_strut = mirrored(self.Rdd_rbr_upper_strut)
        self.Pdd_rbl_upper_strut = mirrored(self.Pdd_rbr_upper_strut)
        self.m_rbl_upper_strut = self.m_rbr_upper_strut
        self.Jbar_rbl_upper_strut = mirrored(self.Jbar_rbr_upper_strut)
        self.R_rbl_lower_strut = mirrored(self.R_rbr_lower_strut)
        self.P_rbl_lower_strut = mirrored(self.P_rbr_lower_strut)
        self.Rd_rbl_lower_strut = mirrored(self.Rd_rbr_lower_strut)
        self.Pd_rbl_lower_strut = mirrored(self.Pd_rbr_lower_strut)
        self.Rdd_rbl_lower_strut = mirrored(self.Rdd_rbr_lower_strut)
        self.Pdd_rbl_lower_strut = mirrored(self.Pdd_rbr_lower_strut)
        self.m_rbl_lower_strut = self.m_rbr_lower_strut
        self.Jbar_rbl_lower_strut = mirrored(self.Jbar_rbr_lower_strut)
        self.R_rbl_tie_rod = mirrored(self.R_rbr_tie_rod)
        self.P_rbl_tie_rod = mirrored(self.P_rbr_tie_rod)
        self.Rd_rbl_tie_rod = mirrored(self.Rd_rbr_tie_rod)
        self.Pd_rbl_tie_rod = mirrored(self.Pd_rbr_tie_rod)
        self.Rdd_rbl_tie_rod = mirrored(self.Rdd_rbr_tie_rod)
        self.Pdd_rbl_tie_rod = mirrored(self.Pdd_rbr_tie_rod)
        self.m_rbl_tie_rod = self.m_rbr_tie_rod
        self.Jbar_rbl_tie_rod = mirrored(self.Jbar_rbr_tie_rod)
        self.R_rbl_hub = mirrored(self.R_rbr_hub)
        self.P_rbl_hub = mirrored(self.P_rbr_hub)
        self.Rd_rbl_hub = mirrored(self.Rd_rbr_hub)
        self.Pd_rbl_hub = mirrored(self.Pd_rbr_hub)
        self.Rdd_rbl_hub = mirrored(self.Rdd_rbr_hub)
        self.Pdd_rbl_hub = mirrored(self.Pdd_rbr_hub)
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
        self.fal_strut_FL = mirrored(self.far_strut_FL)
        self.T_rbl_upper_strut_fal_strut = mirrored(self.T_rbr_upper_strut_far_strut)
        self.T_rbl_lower_strut_fal_strut = mirrored(self.T_rbr_lower_strut_far_strut)
        self.ax1_jcl_strut_lca = mirrored(self.ax1_jcr_strut_lca)
        self.ax2_jcl_strut_lca = mirrored(self.ax2_jcr_strut_lca)
        self.pt1_jcl_strut_lca = mirrored(self.pt1_jcr_strut_lca)
        self.ax1_jcl_tie_upright = mirrored(self.ax1_jcr_tie_upright)
        self.pt1_jcl_tie_upright = mirrored(self.pt1_jcr_tie_upright)
        self.ax1_jcl_tie_steering = mirrored(self.ax1_jcr_tie_steering)
        self.ax2_jcl_tie_steering = mirrored(self.ax2_jcr_tie_steering)
        self.pt1_jcl_tie_steering = mirrored(self.pt1_jcr_tie_steering)
    

