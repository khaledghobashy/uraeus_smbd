
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.Rd_rbs_block = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_block = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_block = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_block = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbs_dummy = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_dummy = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_dummy = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_dummy = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_dummy = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_dummy = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_dummy = 1
        self.Jbar_rbs_dummy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)
        self.pt1_fas_spring = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt2_fas_spring = np.array([[0], [0], [0]],dtype=np.float64)
        self.Fs_fas_spring = lambda t : 0
        self.Fd_fas_spring = lambda t : 0
        self.fas_spring_FL = np.array([[0]],dtype=np.float64)
        self.T_rbs_block_fas_spring = np.array([[0], [0], [0]],dtype=np.float64)
        self.T_vbs_ground_fas_spring = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_cyl_joint = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcs_fixed = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcs_fixed = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_origin = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_top = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_bottom = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_outer_raduis = 1                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_block,self.P_rbs_block,self.R_rbs_dummy,self.P_rbs_dummy])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_block,self.Pd_rbs_block,self.Rd_rbs_dummy,self.Pd_rbs_dummy])
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
        self.gms_block = cylinder_geometry(self.hps_top,self.hps_bottom,self.s_outer_raduis)
        self.hps_block_center = centered(self.hps_top,self.hps_bottom)
        self.R_rbs_block = self.gms_block.R
        self.P_rbs_block = self.gms_block.P
        self.m_rbs_block = self.gms_block.m
        self.Jbar_rbs_block = self.gms_block.J
        self.pt1_jcs_cyl_joint = centered(self.hps_origin,self.hps_block_center)
    

