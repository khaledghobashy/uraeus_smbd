
import numpy as np
import pandas as pd
from source.numerical_classes.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.ax1_jcs_steer_gear = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_steer_gear = lambda t : 0.0
        self.ax1_jcr_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcr_rev = lambda t : 0.0
        self.pt1_mcr_ver_act = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_mcr_ver_act = lambda t : 0.0                       

    
    @property
    def q(self):
        q = []
        return q

    @property
    def qd(self):
        qd = []
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
        self.ax1_jcl_rev = mirrored(self.ax1_jcr_rev)
        self.AF_jcl_rev = self.AF_jcr_rev
        self.pt1_mcl_ver_act = mirrored(self.pt1_mcr_ver_act)
        self.AF_mcl_ver_act = self.AF_mcr_ver_act
    

