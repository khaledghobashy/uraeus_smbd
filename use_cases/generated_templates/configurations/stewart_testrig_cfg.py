
import numpy as np
import pandas as pd
from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                   cylinder_geometry,
                                                   composite_geometry,
                                                   triangular_prism)



class configuration(object):

    def __init__(self):
        self.ax1_jcs_rev_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_1 = lambda t : 0
        self.ax1_jcs_rev_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_2 = lambda t : 0
        self.ax1_jcs_rev_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev_3 = lambda t : 0                       

    
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
    
        pass

