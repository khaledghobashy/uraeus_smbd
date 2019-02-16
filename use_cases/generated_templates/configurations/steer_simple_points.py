
import os
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import mirrored, centered, oriented



path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.P_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_coupler = 1
        self.Jbar_rbs_coupler = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_rocker = 1
        self.Jbar_rbr_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_rocker_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.hpr_rocker_coupler = np.array([[0], [0], [0]],dtype=np.float64)                       

    
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
        self.hpl_rocker_chassis = mirrored(self.hpr_rocker_chassis)
        self.R_rbs_coupler = centered(self.hpr_rocker_coupler,self.hpl_rocker_coupler)
        self.R_rbr_rocker = centered(self.hpr_rocker_chassis,self.hpr_rocker_coupler)
        self.R_rbl_rocker = centered(self.hpl_rocker_chassis,self.hpl_rocker_coupler)
        self.P_rbl_rocker = mirrored(self.P_rbr_rocker)
        self.Rd_rbl_rocker = mirrored(self.Rd_rbr_rocker)
        self.Pd_rbl_rocker = mirrored(self.Pd_rbr_rocker)
        self.m_rbl_rocker = self.m_rbr_rocker
        self.Jbar_rbl_rocker = mirrored(self.Jbar_rbr_rocker)
        self.ax1_jcr_rocker_ch = oriented(self.hpr_rocker_coupler,self.hpl_rocker_coupler,self.hpr_rocker_chassis)
        self.pt1_jcr_rocker_ch = self.hpr_rocker_chassis
        self.ax1_jcl_rocker_ch = oriented(self.hpl_rocker_coupler,self.hpr_rocker_coupler,self.hpl_rocker_chassis)
        self.pt1_jcl_rocker_ch = self.hpl_rocker_chassis
        self.pt1_jcs_rc_sph = self.hpr_rocker_coupler
        self.ax1_jcs_rc_cyl = oriented(self.hpr_rocker_coupler,self.hpl_rocker_coupler,self.hpr_rocker_chassis)
        self.pt1_jcs_rc_cyl = self.hpl_rocker_coupler


