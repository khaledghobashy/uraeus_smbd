
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.Pdd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_thickness = 1.0
        self.Fs_far_strut = lambda t : 0.0
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_lcao = np.array([[0], [0], [0]],dtype=np.float64)
        self.far_strut_FL = np.array([[0]],dtype=np.float64)
        self.hpr_ucao = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.Fd_far_strut = lambda t : 0.0
        self.hpr_ucaf = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.hpr_ucar = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.hpr_tro = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_hub_radius = 1.0
        self.Pdd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_links_ro = 1.0
        self.s_strut_outer = 1.0
        self.Rdd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_lcaf = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.T_rbr_lower_strut_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.s_strut_inner = 1.0
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_tri = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_wc = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_lcar = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.pt1_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.pt2_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.T_rbr_upper_strut_far_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)                       

    
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
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Centered'>
        <class 'source.mbs_creators.configuration_classes.Centered'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Centered'>
        <class 'source.mbs_creators.configuration_classes.Centered'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Oriented'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Mirrored'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
        <class 'source.mbs_creators.configuration_classes.Equal_to'>
    

