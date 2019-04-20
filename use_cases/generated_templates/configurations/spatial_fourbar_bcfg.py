
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.ax1_jcs_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_crank = lambda t : 0.0
        self.ax1_jcs_rev_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_rev_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_sph_coupler_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_sph_coupler_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_uni_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcs_uni_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_uni_coupler_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_crank = 1.0
        self.Jbar_rbs_crank = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker = 1.0
        self.Jbar_rbs_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.R_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_coupler = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_coupler = 1.0
        self.Jbar_rbs_coupler = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_crank,self.P_rbs_crank,self.R_rbs_rocker,self.P_rbs_rocker,self.R_rbs_coupler,self.P_rbs_coupler])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_crank,self.Pd_rbs_crank,self.Rd_rbs_rocker,self.Pd_rbs_rocker,self.Rd_rbs_coupler,self.Pd_rbs_coupler])
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
    
        pass

