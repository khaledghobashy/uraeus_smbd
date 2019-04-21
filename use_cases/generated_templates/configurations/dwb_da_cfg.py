
import numpy as np
import pandas as pd
from source.numerical_classes.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.ax1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt2_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Fs_far_strut = lambda t : 0.0
        self.Fd_far_strut = lambda t : 0.0
        self.far_strut_FL = np.array([[0]],dtype=np.float64)
        self.T_rbr_upper_strut_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.T_rbr_lower_strut_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
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
        self.s_links_ro = 1.0
        self.s_strut_outer = 1.0
        self.s_strut_inner = 1.0
        self.s_thickness = 1.0
        self.s_hub_radius = 1.0                       

    
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
        self.hpl_ucao = mirrored(self.hpr_ucao)
        self.hpl_ucar = mirrored(self.hpr_ucar)
        self.hpl_ucaf = mirrored(self.hpr_ucaf)
        self.hpl_lcao = mirrored(self.hpr_lcao)
        self.hpl_lcar = mirrored(self.hpr_lcar)
        self.hpl_lcaf = mirrored(self.hpr_lcaf)
        self.hpl_wc = mirrored(self.hpr_wc)
        self.hpl_strut_lca = mirrored(self.hpr_strut_lca)
        self.hpl_strut_chassis = mirrored(self.hpr_strut_chassis)
        self.hpr_strut_mid = centered(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.hpl_strut_mid = centered(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.hpl_tro = mirrored(self.hpr_tro)
        self.hpl_tri = mirrored(self.hpr_tri)
        self.gmr_uca = triangular_prism(self.hpr_ucaf,self.hpr_ucar,self.hpr_ucao,self.s_thickness)
        self.gml_uca = triangular_prism(self.hpl_ucaf,self.hpl_ucar,self.hpl_ucao,self.s_thickness)
        self.gmr_lca = triangular_prism(self.hpr_lcaf,self.hpr_lcar,self.hpr_lcao,self.s_thickness)
        self.gml_lca = triangular_prism(self.hpl_lcaf,self.hpl_lcar,self.hpl_lcao,self.s_thickness)
        self.gmr_upright = triangular_prism(self.hpr_ucao,self.hpr_wc,self.hpr_lcao,self.s_thickness)
        self.gml_upright = triangular_prism(self.hpl_ucao,self.hpl_wc,self.hpl_lcao,self.s_thickness)
        self.gmr_upper_strut = cylinder_geometry(self.hpr_strut_chassis,self.hpr_strut_mid,self.s_strut_outer)
        self.gml_upper_strut = cylinder_geometry(self.hpl_strut_chassis,self.hpl_strut_mid,self.s_strut_outer)
        self.gmr_lower_strut = cylinder_geometry(self.hpr_strut_mid,self.hpr_strut_lca,self.s_strut_inner)
        self.gml_lower_strut = cylinder_geometry(self.hpl_strut_mid,self.hpl_strut_lca,self.s_strut_inner)
        self.gmr_tie_rod = cylinder_geometry(self.hpr_tri,self.hpr_tro,self.s_links_ro)
        self.gml_tie_rod = cylinder_geometry(self.hpl_tri,self.hpl_tro,self.s_links_ro)
        self.R_rbr_upright = self.gmr_upright.R
        self.gmr_hub = cylinder_geometry(self.hpr_wc,self.R_rbr_upright,self.s_hub_radius)
        self.R_rbl_upright = self.gml_upright.R
        self.gml_hub = cylinder_geometry(self.hpl_wc,self.R_rbl_upright,self.s_hub_radius)
        self.ax1_jcl_uca_upright = mirrored(self.ax1_jcr_uca_upright)
        self.pt1_jcr_uca_upright = self.hpr_ucao
        self.pt1_jcl_uca_upright = self.hpl_ucao
        self.ax1_jcr_uca_chassis = oriented(self.hpr_ucaf,self.hpr_ucar)
        self.ax1_jcl_uca_chassis = oriented(self.hpl_ucaf,self.hpl_ucar)
        self.pt1_jcr_uca_chassis = centered(self.hpr_ucaf,self.hpr_ucar)
        self.pt1_jcl_uca_chassis = centered(self.hpl_ucaf,self.hpl_ucar)
        self.ax1_jcl_lca_upright = mirrored(self.ax1_jcr_lca_upright)
        self.pt1_jcr_lca_upright = self.hpr_lcao
        self.pt1_jcl_lca_upright = self.hpl_lcao
        self.ax1_jcr_lca_chassis = oriented(self.hpr_lcaf,self.hpr_lcar)
        self.ax1_jcl_lca_chassis = oriented(self.hpl_lcaf,self.hpl_lcar)
        self.pt1_jcr_lca_chassis = centered(self.hpr_lcaf,self.hpr_lcar)
        self.pt1_jcl_lca_chassis = centered(self.hpl_lcaf,self.hpl_lcar)
        self.ax1_jcl_hub_bearing = mirrored(self.ax1_jcr_hub_bearing)
        self.pt1_jcr_hub_bearing = self.hpr_wc
        self.pt1_jcl_hub_bearing = self.hpl_wc
        self.ax1_jcr_strut_chassis = oriented(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.ax1_jcl_strut_chassis = oriented(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.ax2_jcr_strut_chassis = oriented(self.hpr_strut_lca,self.hpr_strut_chassis)
        self.ax2_jcl_strut_chassis = oriented(self.hpl_strut_lca,self.hpl_strut_chassis)
        self.pt1_jcr_strut_chassis = self.hpr_strut_chassis
        self.pt1_jcl_strut_chassis = self.hpl_strut_chassis
        self.ax1_jcr_strut = oriented(self.hpr_strut_lca,self.hpr_strut_chassis)
        self.ax1_jcl_strut = oriented(self.hpl_strut_lca,self.hpl_strut_chassis)
        self.pt1_jcr_strut = self.hpr_strut_mid
        self.pt1_jcl_strut = self.hpl_strut_mid
        self.pt1_fal_strut = mirrored(self.pt1_far_strut)
        self.pt2_fal_strut = mirrored(self.pt2_far_strut)
        self.Fs_fal_strut = self.Fs_far_strut
        self.Fd_fal_strut = self.Fd_far_strut
        self.fal_strut_FL = mirrored(self.far_strut_FL)
        self.T_rbl_upper_strut_fal_strut = mirrored(self.T_rbr_upper_strut_far_strut)
        self.T_rbl_lower_strut_fal_strut = mirrored(self.T_rbr_lower_strut_far_strut)
        self.ax1_jcr_strut_lca = oriented(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.ax1_jcl_strut_lca = oriented(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.ax2_jcr_strut_lca = oriented(self.hpr_strut_lca,self.hpr_strut_chassis)
        self.ax2_jcl_strut_lca = oriented(self.hpl_strut_lca,self.hpl_strut_chassis)
        self.pt1_jcr_strut_lca = self.hpr_strut_lca
        self.pt1_jcl_strut_lca = self.hpl_strut_lca
        self.ax1_jcl_tie_upright = mirrored(self.ax1_jcr_tie_upright)
        self.pt1_jcr_tie_upright = self.hpr_tro
        self.pt1_jcl_tie_upright = self.hpl_tro
        self.ax1_jcr_tie_steering = oriented(self.hpr_tri,self.hpr_tro)
        self.ax1_jcl_tie_steering = oriented(self.hpl_tri,self.hpl_tro)
        self.ax2_jcr_tie_steering = oriented(self.hpr_tro,self.hpr_tri)
        self.ax2_jcl_tie_steering = oriented(self.hpl_tro,self.hpl_tri)
        self.pt1_jcr_tie_steering = self.hpr_tri
        self.pt1_jcl_tie_steering = self.hpl_tri
        self.R_rbr_uca = self.gmr_uca.R
        self.R_rbl_uca = self.gml_uca.R
        self.P_rbr_uca = self.gmr_uca.P
        self.P_rbl_uca = self.gml_uca.P
        self.Rd_rbl_uca = mirrored(self.Rd_rbr_uca)
        self.Pd_rbl_uca = mirrored(self.Pd_rbr_uca)
        self.Rdd_rbl_uca = mirrored(self.Rdd_rbr_uca)
        self.Pdd_rbl_uca = mirrored(self.Pdd_rbr_uca)
        self.m_rbr_uca = self.gmr_uca.m
        self.m_rbl_uca = self.gml_uca.m
        self.Jbar_rbr_uca = self.gmr_uca.J
        self.Jbar_rbl_uca = self.gml_uca.J
        self.R_rbr_lca = self.gmr_lca.R
        self.R_rbl_lca = self.gml_lca.R
        self.P_rbr_lca = self.gmr_lca.P
        self.P_rbl_lca = self.gml_lca.P
        self.Rd_rbl_lca = mirrored(self.Rd_rbr_lca)
        self.Pd_rbl_lca = mirrored(self.Pd_rbr_lca)
        self.Rdd_rbl_lca = mirrored(self.Rdd_rbr_lca)
        self.Pdd_rbl_lca = mirrored(self.Pdd_rbr_lca)
        self.m_rbr_lca = self.gmr_lca.m
        self.m_rbl_lca = self.gml_lca.m
        self.Jbar_rbr_lca = self.gmr_lca.J
        self.Jbar_rbl_lca = self.gml_lca.J
        self.P_rbr_upright = self.gmr_upright.P
        self.P_rbl_upright = self.gml_upright.P
        self.Rd_rbl_upright = mirrored(self.Rd_rbr_upright)
        self.Pd_rbl_upright = mirrored(self.Pd_rbr_upright)
        self.Rdd_rbl_upright = mirrored(self.Rdd_rbr_upright)
        self.Pdd_rbl_upright = mirrored(self.Pdd_rbr_upright)
        self.m_rbr_upright = self.gmr_upright.m
        self.m_rbl_upright = self.gml_upright.m
        self.Jbar_rbr_upright = self.gmr_upright.J
        self.Jbar_rbl_upright = self.gml_upright.J
        self.R_rbr_upper_strut = self.gmr_upper_strut.R
        self.R_rbl_upper_strut = self.gml_upper_strut.R
        self.P_rbr_upper_strut = self.gmr_upper_strut.P
        self.P_rbl_upper_strut = self.gml_upper_strut.P
        self.Rd_rbl_upper_strut = mirrored(self.Rd_rbr_upper_strut)
        self.Pd_rbl_upper_strut = mirrored(self.Pd_rbr_upper_strut)
        self.Rdd_rbl_upper_strut = mirrored(self.Rdd_rbr_upper_strut)
        self.Pdd_rbl_upper_strut = mirrored(self.Pdd_rbr_upper_strut)
        self.m_rbr_upper_strut = self.gmr_upper_strut.m
        self.m_rbl_upper_strut = self.gml_upper_strut.m
        self.Jbar_rbr_upper_strut = self.gmr_upper_strut.J
        self.Jbar_rbl_upper_strut = self.gml_upper_strut.J
        self.R_rbr_lower_strut = self.gmr_lower_strut.R
        self.R_rbl_lower_strut = self.gml_lower_strut.R
        self.P_rbr_lower_strut = self.gmr_lower_strut.P
        self.P_rbl_lower_strut = self.gml_lower_strut.P
        self.Rd_rbl_lower_strut = mirrored(self.Rd_rbr_lower_strut)
        self.Pd_rbl_lower_strut = mirrored(self.Pd_rbr_lower_strut)
        self.Rdd_rbl_lower_strut = mirrored(self.Rdd_rbr_lower_strut)
        self.Pdd_rbl_lower_strut = mirrored(self.Pdd_rbr_lower_strut)
        self.m_rbr_lower_strut = self.gmr_lower_strut.m
        self.m_rbl_lower_strut = self.gml_lower_strut.m
        self.Jbar_rbr_lower_strut = self.gmr_lower_strut.J
        self.Jbar_rbl_lower_strut = self.gml_lower_strut.J
        self.R_rbr_tie_rod = self.gmr_tie_rod.R
        self.R_rbl_tie_rod = self.gml_tie_rod.R
        self.P_rbr_tie_rod = self.gmr_tie_rod.P
        self.P_rbl_tie_rod = self.gml_tie_rod.P
        self.Rd_rbl_tie_rod = mirrored(self.Rd_rbr_tie_rod)
        self.Pd_rbl_tie_rod = mirrored(self.Pd_rbr_tie_rod)
        self.Rdd_rbl_tie_rod = mirrored(self.Rdd_rbr_tie_rod)
        self.Pdd_rbl_tie_rod = mirrored(self.Pdd_rbr_tie_rod)
        self.m_rbr_tie_rod = self.gmr_tie_rod.m
        self.m_rbl_tie_rod = self.gml_tie_rod.m
        self.Jbar_rbr_tie_rod = self.gmr_tie_rod.J
        self.Jbar_rbl_tie_rod = self.gml_tie_rod.J
        self.R_rbr_hub = self.gmr_hub.R
        self.R_rbl_hub = self.gml_hub.R
        self.P_rbr_hub = self.gmr_hub.P
        self.P_rbl_hub = self.gml_hub.P
        self.Rd_rbl_hub = mirrored(self.Rd_rbr_hub)
        self.Pd_rbl_hub = mirrored(self.Pd_rbr_hub)
        self.Rdd_rbl_hub = mirrored(self.Rdd_rbr_hub)
        self.Pdd_rbl_hub = mirrored(self.Pdd_rbr_hub)
        self.m_rbr_hub = self.gmr_hub.m
        self.m_rbl_hub = self.gml_hub.m
        self.Jbar_rbr_hub = self.gmr_hub.J
        self.Jbar_rbl_hub = self.gml_hub.J
    

