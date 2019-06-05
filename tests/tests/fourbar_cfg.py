
import numpy as np

from smbd.numenv.python.numerics.numcls import num_config
from smbd.numenv.python.numerics.misc import (mirrored, centered, oriented,
                                               cylinder_geometry,
                                               composite_geometry,
                                               triangular_prism,
                                               sphere_geometry)



class configuration(num_config):

    def __init__(self):
        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.R_rbs_crank = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_rbs_crank = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_rbs_crank = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_crank = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_crank = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_crank = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.m_rbs_crank = 1.0
        self.Jbar_rbs_crank = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        self.R_rbs_conct = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_rbs_conct = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_rbs_conct = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_conct = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_conct = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_conct = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.m_rbs_conct = 1.0
        self.Jbar_rbs_conct = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        self.R_rbs_rockr = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_rbs_rockr = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_rbs_rockr = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_rockr = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_rockr = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_rockr = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.m_rbs_rockr = 1.0
        self.Jbar_rbs_rockr = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        self.hps_a = np.array([[0], [0], [0]], dtype=np.float64)
        self.hps_b = np.array([[0], [0], [0]], dtype=np.float64)
        self.hps_c = np.array([[0], [0], [0]], dtype=np.float64)
        self.hps_d = np.array([[0], [0], [0]], dtype=np.float64)
        self.vcs_x = np.array([[0], [0], [0]], dtype=np.float64)
        self.vcs_y = np.array([[0], [0], [0]], dtype=np.float64)
        self.vcs_z = np.array([[0], [0], [0]], dtype=np.float64)                       

    
    @property
    def q(self):
        q = np.concatenate([self.R_ground,
        self.P_ground,
        self.R_rbs_crank,
        self.P_rbs_crank,
        self.R_rbs_conct,
        self.P_rbs_conct,
        self.R_rbs_rockr,
        self.P_rbs_rockr])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_ground,
        self.Pd_ground,
        self.Rd_rbs_crank,
        self.Pd_rbs_crank,
        self.Rd_rbs_conct,
        self.Pd_rbs_conct,
        self.Rd_rbs_rockr,
        self.Pd_rbs_rockr])
        return qd

    def evaluate(self):
        self.ax1_jcs_a = self.vcs_x
        self.pt1_jcs_a = self.hps_a
        self.ax1_jcs_b = self.vcs_z
        self.pt1_jcs_b = self.hps_b
        self.ax1_jcs_c = oriented(self.hps_b,self.hps_c)
        self.ax2_jcs_c = oriented(self.hps_c,self.hps_b)
        self.pt1_jcs_c = self.hps_c
        self.ax1_jcs_d = self.vcs_y
        self.pt1_jcs_d = self.hps_d
    

