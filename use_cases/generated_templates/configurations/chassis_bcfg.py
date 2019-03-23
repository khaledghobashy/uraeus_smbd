
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.R_rbs_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbs_chassis = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbs_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbs_chassis = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rdd_rbs_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pdd_rbs_chassis = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.m_rbs_chassis = 1
        self.Jbar_rbs_chassis = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]],dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_rbs_chassis,self.P_rbs_chassis])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbs_chassis,self.Pd_rbs_chassis])
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

