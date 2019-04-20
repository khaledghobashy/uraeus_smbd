
import numpy as np
import pandas as pd
from source.numerical_classes.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.ax1_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rc_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rc_cyl = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_rocker_ch = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_rocker_ch = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_coupler = 1.0
        self.Jbar_rbs_coupler = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbr_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbr_rocker = 1.0
        self.Jbar_rbr_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_coupler,self.P_rbs_coupler,self.R_rbr_rocker,self.P_rbr_rocker,self.R_rbl_rocker,self.P_rbl_rocker])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_coupler,self.Pd_rbs_coupler,self.Rd_rbr_rocker,self.Pd_rbr_rocker,self.Rd_rbl_rocker,self.Pd_rbl_rocker])
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
        self.ax1_jcl_rocker_ch = mirrored(self.ax1_jcr_rocker_ch)
        self.pt1_jcl_rocker_ch = mirrored(self.pt1_jcr_rocker_ch)
        self.R_rbl_rocker = mirrored(self.R_rbr_rocker)
        self.P_rbl_rocker = mirrored(self.P_rbr_rocker)
        self.Rd_rbl_rocker = mirrored(self.Rd_rbr_rocker)
        self.Pd_rbl_rocker = mirrored(self.Pd_rbr_rocker)
        self.Rdd_rbl_rocker = mirrored(self.Rdd_rbr_rocker)
        self.Pdd_rbl_rocker = mirrored(self.Pdd_rbr_rocker)
        self.m_rbl_rocker = self.m_rbr_rocker
        self.Jbar_rbl_rocker = mirrored(self.Jbar_rbr_rocker)
    

