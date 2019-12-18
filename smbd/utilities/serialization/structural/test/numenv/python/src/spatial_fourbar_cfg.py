
import numpy as np

from smbd.numenv.python.numerics.core.shared_classes import config_inputs
from smbd.numenv.python.numerics.core.math_funcs.misc import (mirrored, centered, oriented,
                                               cylinder_geometry,
                                               composite_geometry,
                                               triangular_prism,
                                               sphere_geometry)



class inputs(config_inputs):

    _inputs_names = ('UF_mcs_act',
        'Rd_rbs_l1',
        'Pd_rbs_l1',
        'Rdd_rbs_l1',
        'Pdd_rbs_l1',
        'Rd_rbs_l2',
        'Pd_rbs_l2',
        'Rdd_rbs_l2',
        'Pdd_rbs_l2',
        'Rd_rbs_l3',
        'Pd_rbs_l3',
        'Rdd_rbs_l3',
        'Pdd_rbs_l3',
        'hps_a',
        'hps_b',
        'hps_c',
        'hps_d',
        'vcs_x',
        'vcs_y',
        'vcs_z',
        's_links_ro')

    def __init__(self, name):
        self.name = name

        self.UF_mcs_act = lambda t : 0.0
        self.Rd_rbs_l1 = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_l1 = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_l1 = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_l1 = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_rbs_l2 = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_l2 = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_l2 = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_l2 = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rd_rbs_l3 = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_rbs_l3 = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.Rdd_rbs_l3 = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pdd_rbs_l3 = np.array([[0], [0], [0], [0]], dtype=np.float64)
        self.hps_a = np.array([[0], [0], [0]], dtype=np.float64)
        self.hps_b = np.array([[0], [0], [0]], dtype=np.float64)
        self.hps_c = np.array([[0], [0], [0]], dtype=np.float64)
        self.hps_d = np.array([[0], [0], [0]], dtype=np.float64)
        self.vcs_x = np.array([[0], [0], [0]], dtype=np.float64)
        self.vcs_y = np.array([[0], [0], [0]], dtype=np.float64)
        self.vcs_z = np.array([[0], [0], [0]], dtype=np.float64)
        self.s_links_ro = 1.0


class configuration(object):

    def __init__(self):
        pass

    
    def assemble(self, inputs_instance):

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)

        self.Rd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)

        for arg in inputs_instance._inputs_names:
            value = getattr(inputs_instance, arg)
            setattr(self, arg, value)

        self.gms_l1 = cylinder_geometry(self.hps_a, self.hps_b, self.s_links_ro)
        self.gms_l2 = cylinder_geometry(self.hps_b, self.hps_c, self.s_links_ro)
        self.gms_l3 = cylinder_geometry(self.hps_c, self.hps_d, self.s_links_ro)
        self.ax1_jcs_a = self.vcs_x
        self.pt1_jcs_a = self.hps_a
        self.ax1_jcs_b = self.vcs_z
        self.pt1_jcs_b = self.hps_b
        self.ax1_jcs_c = oriented(self.hps_b,self.hps_c)
        self.ax2_jcs_c = oriented(self.hps_c,self.hps_b)
        self.pt1_jcs_c = self.hps_c
        self.ax1_jcs_d = self.vcs_y
        self.pt1_jcs_d = self.hps_d
        self.R_rbs_l1 = self.gms_l1.R
        self.P_rbs_l1 = self.gms_l1.P
        self.m_rbs_l1 = self.gms_l1.m
        self.Jbar_rbs_l1 = self.gms_l1.J
        self.R_rbs_l2 = self.gms_l2.R
        self.P_rbs_l2 = self.gms_l2.P
        self.m_rbs_l2 = self.gms_l2.m
        self.Jbar_rbs_l2 = self.gms_l2.J
        self.R_rbs_l3 = self.gms_l3.R
        self.P_rbs_l3 = self.gms_l3.P
        self.m_rbs_l3 = self.gms_l3.m
        self.Jbar_rbs_l3 = self.gms_l3.J

        self.q  = np.concatenate([self.R_ground,
        self.P_ground,
        self.R_rbs_l1,
        self.P_rbs_l1,
        self.R_rbs_l2,
        self.P_rbs_l2,
        self.R_rbs_l3,
        self.P_rbs_l3])

        self.qd = np.concatenate([self.Rd_ground,
        self.Pd_ground,
        self.Rd_rbs_l1,
        self.Pd_rbs_l1,
        self.Rd_rbs_l2,
        self.Pd_rbs_l2,
        self.Rd_rbs_l3,
        self.Pd_rbs_l3])
    

