
import os
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import mirrored, centered, oriented



path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.P_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rocker = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rocker = 1
        self.Jbar_rbs_rocker = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbs_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_rod = 1
        self.Jbar_rbs_rod = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.P_rbs_slider = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_slider = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_slider = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_slider = 1
        self.Jbar_rbs_slider = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.ax1_jcs_trans = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_A = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_B = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_C = np.array([[0], [0], [0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_rocker,self.P_rbs_rocker,self.R_rbs_rod,self.P_rbs_rod,self.R_rbs_slider,self.P_rbs_slider])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_rocker,self.Pd_rbs_rocker,self.Rd_rbs_rod,self.Pd_rbs_rod,self.Rd_rbs_slider,self.Pd_rbs_slider])
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
        self.ax1_jcs_rev = oriented(self.hps_A,self.hps_B,self.hps_C)
        self.R_rbs_rocker = centered(self.hps_A,self.hps_B)
        self.R_rbs_rod = centered(self.hps_B,self.hps_C)
        self.R_rbs_slider = self.hps_C
        self.pt1_jcs_rev = self.hps_A
        self.ax1_jcs_cyl = self.ax1_jcs_rev
        self.pt1_jcs_cyl = self.hps_B
        self.ax1_jcs_uni = oriented(self.hps_B,self.hps_C)
        self.ax2_jcs_uni = self.ax1_jcs_trans
        self.pt1_jcs_uni = self.hps_B
        self.pt1_jcs_trans = self.hps_C
    

