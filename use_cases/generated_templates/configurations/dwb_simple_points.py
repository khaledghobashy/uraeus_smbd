
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import mirrored, centered, oriented



class configuration(object):

    def __init__(self):
        self.P_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_uca = 1
        self.Jbar_rbr_uca = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_lca = 1
        self.Jbar_rbr_lca = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_upright = 1
        self.Jbar_rbr_upright = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_upper_strut = 1
        self.Jbar_rbr_upper_strut = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_lower_strut = 1
        self.Jbar_rbr_lower_strut = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_tie_rod = 1
        self.Jbar_rbr_tie_rod = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_hub = 1
        self.Jbar_rbr_hub = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt2_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Fs_far_strut = lambda t : 0
        self.Fd_far_strut = lambda t : 0
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_ucaf = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_ucar = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_ucao = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_lcaf = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_lcar = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_lcao = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_tro = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_tri = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_wc = np.array([[0], [0], [0]],dtype=np.float64)                       

    
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
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def _set_arguments(self):
        self.hpl_ucar = mirrored(self.hpr_ucar)
        self.hpl_ucaf = mirrored(self.hpr_ucaf)
        self.hpl_ucao = mirrored(self.hpr_ucao)
        self.hpl_lcar = mirrored(self.hpr_lcar)
        self.hpl_lcaf = mirrored(self.hpr_lcaf)
        self.hpl_lcao = mirrored(self.hpr_lcao)
        self.hpr_strut_mid = centered(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.hpl_strut_lca = mirrored(self.hpr_strut_lca)
        self.hpl_strut_chassis = mirrored(self.hpr_strut_chassis)
        self.hpl_strut_mid = centered(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.hpl_tri = mirrored(self.hpr_tri)
        self.hpl_tro = mirrored(self.hpr_tro)
        self.R_rbr_upright = centered(self.hpr_ucao,self.hpr_lcao,self.hpr_wc)
        self.hpl_wc = mirrored(self.hpr_wc)
        self.R_rbl_upright = centered(self.hpl_ucao,self.hpl_lcao,self.hpl_wc)
        self.R_rbr_uca = centered(self.hpr_ucao,self.hpr_ucaf,self.hpr_ucar)
        self.R_rbl_uca = centered(self.hpl_ucao,self.hpl_ucaf,self.hpl_ucar)
        self.P_rbl_uca = mirrored(self.P_rbr_uca)
        self.Rd_rbl_uca = mirrored(self.Rd_rbr_uca)
        self.Pd_rbl_uca = mirrored(self.Pd_rbr_uca)
        self.m_rbl_uca = self.m_rbr_uca
        self.Jbar_rbl_uca = mirrored(self.Jbar_rbr_uca)
        self.R_rbr_lca = centered(self.hpr_lcao,self.hpr_lcaf,self.hpr_lcar)
        self.R_rbl_lca = centered(self.hpl_lcao,self.hpl_lcaf,self.hpl_lcar)
        self.P_rbl_lca = mirrored(self.P_rbr_lca)
        self.Rd_rbl_lca = mirrored(self.Rd_rbr_lca)
        self.Pd_rbl_lca = mirrored(self.Pd_rbr_lca)
        self.m_rbl_lca = self.m_rbr_lca
        self.Jbar_rbl_lca = mirrored(self.Jbar_rbr_lca)
        self.P_rbl_upright = mirrored(self.P_rbr_upright)
        self.Rd_rbl_upright = mirrored(self.Rd_rbr_upright)
        self.Pd_rbl_upright = mirrored(self.Pd_rbr_upright)
        self.m_rbl_upright = self.m_rbr_upright
        self.Jbar_rbl_upright = mirrored(self.Jbar_rbr_upright)
        self.R_rbr_upper_strut = centered(self.hpr_strut_chassis,self.hpr_strut_mid)
        self.R_rbl_upper_strut = centered(self.hpl_strut_chassis,self.hpl_strut_mid)
        self.P_rbl_upper_strut = mirrored(self.P_rbr_upper_strut)
        self.Rd_rbl_upper_strut = mirrored(self.Rd_rbr_upper_strut)
        self.Pd_rbl_upper_strut = mirrored(self.Pd_rbr_upper_strut)
        self.m_rbl_upper_strut = self.m_rbr_upper_strut
        self.Jbar_rbl_upper_strut = mirrored(self.Jbar_rbr_upper_strut)
        self.R_rbr_lower_strut = centered(self.hpr_strut_lca,self.hpr_strut_mid)
        self.R_rbl_lower_strut = centered(self.hpl_strut_lca,self.hpl_strut_mid)
        self.P_rbl_lower_strut = mirrored(self.P_rbr_lower_strut)
        self.Rd_rbl_lower_strut = mirrored(self.Rd_rbr_lower_strut)
        self.Pd_rbl_lower_strut = mirrored(self.Pd_rbr_lower_strut)
        self.m_rbl_lower_strut = self.m_rbr_lower_strut
        self.Jbar_rbl_lower_strut = mirrored(self.Jbar_rbr_lower_strut)
        self.R_rbr_tie_rod = centered(self.hpr_tro,self.hpr_tri)
        self.R_rbl_tie_rod = centered(self.hpl_tro,self.hpl_tri)
        self.P_rbl_tie_rod = mirrored(self.P_rbr_tie_rod)
        self.Rd_rbl_tie_rod = mirrored(self.Rd_rbr_tie_rod)
        self.Pd_rbl_tie_rod = mirrored(self.Pd_rbr_tie_rod)
        self.m_rbl_tie_rod = self.m_rbr_tie_rod
        self.Jbar_rbl_tie_rod = mirrored(self.Jbar_rbr_tie_rod)
        self.R_rbr_hub = centered(self.hpr_wc,self.R_rbr_upright)
        self.R_rbl_hub = centered(self.hpl_wc,self.R_rbl_upright)
        self.P_rbl_hub = mirrored(self.P_rbr_hub)
        self.Rd_rbl_hub = mirrored(self.Rd_rbr_hub)
        self.Pd_rbl_hub = mirrored(self.Pd_rbr_hub)
        self.m_rbl_hub = self.m_rbr_hub
        self.Jbar_rbl_hub = mirrored(self.Jbar_rbr_hub)
        self.pt1_jcr_uca_upright = self.hpr_ucao
        self.ax1_jcl_uca_upright = mirrored(self.ax1_jcr_uca_upright)
        self.pt1_jcl_uca_upright = self.hpl_ucao
        self.ax1_jcr_uca_chassis = oriented(self.hpr_ucaf,self.hpr_ucar)
        self.pt1_jcr_uca_chassis = centered(self.hpr_ucaf,self.hpr_ucar)
        self.ax1_jcl_uca_chassis = oriented(self.hpl_ucaf,self.hpl_ucar)
        self.pt1_jcl_uca_chassis = centered(self.hpl_ucaf,self.hpl_ucar)
        self.pt1_jcr_lca_upright = self.hpr_lcao
        self.ax1_jcl_lca_upright = mirrored(self.ax1_jcr_lca_upright)
        self.pt1_jcl_lca_upright = self.hpl_lcao
        self.ax1_jcr_lca_chassis = oriented(self.hpr_lcaf,self.hpr_lcar)
        self.pt1_jcr_lca_chassis = centered(self.hpr_lcaf,self.hpr_lcar)
        self.ax1_jcl_lca_chassis = oriented(self.hpl_lcaf,self.hpl_lcar)
        self.pt1_jcl_lca_chassis = centered(self.hpl_lcaf,self.hpl_lcar)
        self.pt1_jcr_hub_bearing = self.hpr_wc
        self.ax1_jcl_hub_bearing = mirrored(self.ax1_jcr_hub_bearing)
        self.pt1_jcl_hub_bearing = self.hpl_wc
        self.ax1_jcr_strut_chassis = oriented(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.ax2_jcr_strut_chassis = oriented(self.hpr_strut_lca,self.hpr_strut_chassis)
        self.pt1_jcr_strut_chassis = self.hpr_strut_chassis
        self.ax1_jcl_strut_chassis = oriented(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.ax2_jcl_strut_chassis = oriented(self.hpl_strut_lca,self.hpl_strut_chassis)
        self.pt1_jcl_strut_chassis = self.hpl_strut_chassis
        self.ax1_jcr_strut = oriented(self.hpr_strut_lca,self.hpr_strut_chassis)
        self.pt1_jcr_strut = self.hpr_strut_mid
        self.ax1_jcl_strut = oriented(self.hpl_strut_lca,self.hpl_strut_chassis)
        self.pt1_jcl_strut = self.hpl_strut_mid
        self.pt1_fal_strut = mirrored(self.pt1_far_strut)
        self.pt2_fal_strut = mirrored(self.pt2_far_strut)
        self.Fs_fal_strut = self.Fs_far_strut
        self.Fd_fal_strut = self.Fd_far_strut
        self.ax1_jcr_strut_lca = oriented(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.ax2_jcr_strut_lca = oriented(self.hpr_strut_lca,self.hpr_strut_chassis)
        self.pt1_jcr_strut_lca = self.hpr_strut_lca
        self.ax1_jcl_strut_lca = oriented(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.ax2_jcl_strut_lca = oriented(self.hpl_strut_lca,self.hpl_strut_chassis)
        self.pt1_jcl_strut_lca = self.hpl_strut_lca
        self.pt1_jcr_tie_upright = self.hpr_tro
        self.ax1_jcl_tie_upright = mirrored(self.ax1_jcr_tie_upright)
        self.pt1_jcl_tie_upright = self.hpl_tro
        self.ax1_jcr_tie_steering = oriented(self.hpr_tri,self.hpr_tro)
        self.ax2_jcr_tie_steering = oriented(self.hpr_tro,self.hpr_tri)
        self.pt1_jcr_tie_steering = self.hpr_tri
        self.ax1_jcl_tie_steering = oriented(self.hpl_tri,self.hpl_tro)
        self.ax2_jcl_tie_steering = oriented(self.hpl_tro,self.hpl_tri)
        self.pt1_jcl_tie_steering = self.hpl_tri


