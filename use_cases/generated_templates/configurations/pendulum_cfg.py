
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.Rd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_crank = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_crank = lambda t : 0
        self.hps_rev_crank = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_end_point = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_links_ro = 1                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_crank,self.P_rbs_crank])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_crank,self.Pd_rbs_crank])
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
        self.gms_crank = cylinder_geometry(self.hps_rev_crank,self.hps_end_point,self.s_links_ro)
        self.R_rbs_crank = self.gms_crank.R
        self.P_rbs_crank = self.gms_crank.P
        self.m_rbs_crank = self.gms_crank.m
        self.Jbar_rbs_crank = self.gms_crank.J
        self.pt1_jcs_rev_crank = self.hps_rev_crank
    

