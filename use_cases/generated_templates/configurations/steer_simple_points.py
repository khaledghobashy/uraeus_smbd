
import os
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_rocker_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_rocker_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_links_ro = 1
        self.s_thickness = 1                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_coupler,self.P_rbs_coupler,self.R_rbr_rocker,self.P_rbr_rocker,self.R_rbl_rocker,self.P_rbl_rocker])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_coupler,self.Pd_rbs_coupler,self.Rd_rbr_rocker,self.Pd_rbr_rocker,self.Rd_rbl_rocker,self.Pd_rbl_rocker])
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
        self.hpl_rocker_coupler = mirrored(self.hpr_rocker_coupler)
        self.gms_coupler = cylinder_geometry(self.hpr_rocker_coupler,self.hpl_rocker_coupler,self.s_links_ro)
        self.gmr_rocker = cylinder_geometry(self.hpr_rocker_chassis,self.hpr_rocker_coupler,self.s_links_ro)
        self.hpl_rocker_chassis = mirrored(self.hpr_rocker_chassis)
        self.gml_rocker = cylinder_geometry(self.hpl_rocker_chassis,self.hpl_rocker_coupler,self.s_links_ro)
        self.R_rbs_coupler = self.gms_coupler.R
        self.P_rbs_coupler = self.gms_coupler.P
        self.m_rbs_coupler = self.gms_coupler.m
        self.Jbar_rbs_coupler = self.gms_coupler.J
        self.R_rbr_rocker = self.gmr_rocker.R
        self.P_rbr_rocker = self.gmr_rocker.P
        self.m_rbr_rocker = self.gmr_rocker.m
        self.Jbar_rbr_rocker = self.gmr_rocker.J
        self.R_rbl_rocker = self.gml_rocker.R
        self.P_rbl_rocker = self.gml_rocker.P
        self.Rd_rbl_rocker = mirrored(self.Rd_rbr_rocker)
        self.Pd_rbl_rocker = mirrored(self.Pd_rbr_rocker)
        self.Rdd_rbl_rocker = mirrored(self.Rdd_rbr_rocker)
        self.Pdd_rbl_rocker = mirrored(self.Pdd_rbr_rocker)
        self.m_rbl_rocker = self.gml_rocker.m
        self.Jbar_rbl_rocker = self.gml_rocker.J
        self.ax1_jcr_rocker_ch = oriented(self.hpr_rocker_coupler,self.hpl_rocker_coupler,self.hpr_rocker_chassis)
        self.pt1_jcr_rocker_ch = self.hpr_rocker_chassis
        self.ax1_jcl_rocker_ch = oriented(self.hpl_rocker_coupler,self.hpr_rocker_coupler,self.hpl_rocker_chassis)
        self.pt1_jcl_rocker_ch = self.hpl_rocker_chassis
        self.pt1_jcs_rc_sph = self.hpr_rocker_coupler
        self.ax1_jcs_rc_cyl = oriented(self.hpr_rocker_coupler,self.hpl_rocker_coupler,self.hpr_rocker_chassis)
        self.pt1_jcs_rc_cyl = self.hpl_rocker_coupler
    

