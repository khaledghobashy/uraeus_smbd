
import numpy as np

from smbd.numenv.python.numerics.numcls import num_config
from smbd.numenv.python.numerics.misc import (mirrored, centered, oriented,
                                               cylinder_geometry,
                                               composite_geometry,
                                               triangular_prism,
                                               sphere_geometry)



class configuration(num_config):

    def __init__(self):
        self.ax1_jcs_a = np.array([[0], [0], [0]], dtype=np.float64)
        self.pt1_jcs_a = np.array([[0], [0], [0]], dtype=np.float64)
        self.UF_jcs_a = lambda t : 0.0
        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.R_rbs_body = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_rbs_body = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_rbs_body = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_body = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_body = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_body = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.m_rbs_body = 1.0
        self.Jbar_rbs_body = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_ground,
        self.P_ground,
        self.R_rbs_body,
        self.P_rbs_body])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_ground,
        self.Pd_ground,
        self.Rd_rbs_body,
        self.Pd_rbs_body])
        return qd

    def evaluate(self):
    
        pass

