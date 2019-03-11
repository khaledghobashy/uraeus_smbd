
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
        self.Rd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_crank = lambda t : 0
        self.ax1_jcs_rev_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_sph_coupler_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_rev_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_coupler_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_links_ro = 1                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_crank,self.P_rbs_crank,self.R_rbs_rocker,self.P_rbs_rocker,self.R_rbs_coupler,self.P_rbs_coupler])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_crank,self.Pd_rbs_crank,self.Rd_rbs_rocker,self.Pd_rbs_rocker,self.Rd_rbs_coupler,self.Pd_rbs_coupler])
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
        self.gms_crank = cylinder_geometry(self.hps_rev_crank,self.hps_coupler_crank,self.s_links_ro)
        self.gms_rocker = cylinder_geometry(self.hps_rev_rocker,self.hps_coupler_rocker,self.s_links_ro)
        self.gms_coupler = cylinder_geometry(self.hps_coupler_crank,self.hps_coupler_rocker,self.s_links_ro)
        self.R_rbs_crank = self.gms_crank.R
        self.P_rbs_crank = self.gms_crank.P
        self.m_rbs_crank = self.gms_crank.m
        self.Jbar_rbs_crank = self.gms_crank.J
        self.R_rbs_rocker = self.gms_rocker.R
        self.P_rbs_rocker = self.gms_rocker.P
        self.m_rbs_rocker = self.gms_rocker.m
        self.Jbar_rbs_rocker = self.gms_rocker.J
        self.R_rbs_coupler = self.gms_coupler.R
        self.P_rbs_coupler = self.gms_coupler.P
        self.m_rbs_coupler = self.gms_coupler.m
        self.Jbar_rbs_coupler = self.gms_coupler.J
        self.pt1_jcs_rev_crank = self.hps_rev_crank
        self.pt1_jcs_rev_rocker = self.hps_rev_rocker
        self.pt1_jcs_sph_coupler_crank = self.hps_coupler_crank
        self.ax1_jcs_uni_coupler_rocker = oriented(self.hps_coupler_rocker,self.hps_coupler_crank)
        self.ax2_jcs_uni_coupler_rocker = oriented(self.hps_coupler_crank,self.hps_coupler_rocker)
        self.pt1_jcs_uni_coupler_rocker = self.hps_coupler_rocker
    

