
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from matrix_funcs import A, B, sparse_assembler, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.pt_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_rocker_ch = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_rocker_ch = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_rc_uni = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_rc_uni = np.array([[0], [0], [1]],dtype=np.float64)
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU1_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU2_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU2_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU2_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU2_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)


class numerical_assembly(object):

    def __init__(self,config):
        self.F = config.F
        self.t = 0

        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130])
        self.pos_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130])
        self.vel_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130])
        self.acc_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jacobian_rows_blocks = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,54,54,55,55,55,55,56,56,56,56,57,57,57,57,58,58,58,58,59,59,59,59,60,60,60,60,61,61,61,61,62,62,62,62,63,63,63,63,64,64,64,64,65,65,65,65,66,66,66,66,67,67,67,67,68,68,68,68,69,69,69,69,70,70,70,70,71,71,71,71,72,72,72,72,73,73,73,73,74,74,74,74,75,75,75,75,76,76,76,76,77,77,77,77,78,78,78,78,79,79,79,79,80,80,80,80,81,81,81,81,82,82,82,82,83,83,83,83,84,84,84,84,85,85,85,85,86,86,86,86,87,87,87,87,88,88,88,88,89,89,89,89,90,90,90,90,91,91,91,91,92,92,92,92,93,93,93,93,94,94,94,94,95,95,95,95,96,96,96,96,97,97,97,97,98,98,99,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130])
        self.jacobian_cols_blocks = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,4,5,0,1,4,5,0,1,4,5,0,1,6,7,0,1,6,7,0,1,6,7,0,1,8,9,0,1,8,9,0,1,8,9,0,1,14,15,0,1,14,15,0,1,16,17,0,1,16,17,0,1,50,51,0,1,50,51,0,1,52,53,0,1,52,53,0,1,30,31,0,1,30,31,0,1,30,31,0,1,32,33,0,1,32,33,0,1,32,33,0,1,34,35,0,1,34,35,0,1,34,35,0,1,36,37,0,1,36,37,0,1,36,37,0,1,42,43,0,1,42,43,0,1,44,45,0,1,44,45,0,1,60,61,0,1,60,61,0,1,60,61,0,1,62,63,0,1,62,63,0,1,62,63,2,3,10,11,4,5,12,13,6,7,10,11,6,7,18,19,6,7,18,19,8,9,12,13,8,9,20,21,8,9,20,21,10,11,22,23,10,11,26,27,10,11,26,27,10,11,26,27,12,13,24,25,12,13,28,29,12,13,28,29,12,13,28,29,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,20,21,16,17,20,21,16,17,20,21,22,23,60,61,22,23,60,61,24,25,62,63,24,25,62,63,30,31,38,39,32,33,40,41,34,35,38,39,34,35,46,47,34,35,46,47,36,37,40,41,36,37,48,49,36,37,48,49,38,39,50,51,38,39,54,55,38,39,54,55,38,39,54,55,40,41,52,53,40,41,56,57,40,41,56,57,40,41,56,57,42,43,46,47,42,43,46,47,42,43,46,47,42,43,46,47,44,45,48,49,44,45,48,49,44,45,48,49,44,45,48,49,58,59,60,61,58,59,62,63,58,59,62,63,58,59,62,63,0,1,0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63])

    def eval_constants(self):
        config = self.config

        c0 = A(config.P_ground).T
        c1 = config.pt_jcr_uca_chassis
        c2 = -1.0*multi_dot([c0,config.R_ground])
        c3 = ubar_ground_jcr_uca_chassis = (multi_dot([c0,c1]) + c2)
        c4 = A(config.P_SU1_rbr_uca).T
        c5 = -1.0*multi_dot([c4,config.R_SU1_rbr_uca])
        c6 = Triad(config.ax_jcr_uca_chassis)
        c7 = Mbar_ground_jcr_uca_chassis = multi_dot([c0,c6])
        c8 = config.pt_jcl_uca_chassis
        c9 = ubar_ground_jcl_uca_chassis = (multi_dot([c0,c8]) + c2)
        c10 = A(config.P_SU1_rbl_uca).T
        c11 = -1.0*multi_dot([c10,config.R_SU1_rbl_uca])
        c12 = Triad(config.ax_jcl_uca_chassis)
        c13 = Mbar_ground_jcl_uca_chassis = multi_dot([c0,c12])
        c14 = config.pt_jcr_lca_chassis
        c15 = ubar_ground_jcr_lca_chassis = (multi_dot([c0,c14]) + c2)
        c16 = A(config.P_SU1_rbr_lca).T
        c17 = -1.0*multi_dot([c16,config.R_SU1_rbr_lca])
        c18 = Triad(config.ax_jcr_lca_chassis)
        c19 = Mbar_ground_jcr_lca_chassis = multi_dot([c0,c18])
        c20 = config.pt_jcl_lca_chassis
        c21 = ubar_ground_jcl_lca_chassis = (multi_dot([c0,c20]) + c2)
        c22 = A(config.P_SU1_rbl_lca).T
        c23 = -1.0*multi_dot([c22,config.R_SU1_rbl_lca])
        c24 = Triad(config.ax_jcl_lca_chassis)
        c25 = Mbar_ground_jcl_lca_chassis = multi_dot([c0,c24])
        c26 = config.pt_jcr_strut_chassis
        c27 = ubar_ground_jcr_strut_chassis = (multi_dot([c0,c26]) + c2)
        c28 = A(config.P_SU1_rbr_upper_strut).T
        c29 = -1.0*multi_dot([c28,config.R_SU1_rbr_upper_strut])
        c30 = Triad(config.ax_jcr_strut_chassis)
        c31 = Mbar_ground_jcr_strut_chassis = multi_dot([c0,c30])
        c32 = Triad("config.ax2_jcr_strut_chassis", 'c30[0:3,1:2]')
        c33 = config.pt_jcl_strut_chassis
        c34 = ubar_ground_jcl_strut_chassis = (multi_dot([c0,c33]) + c2)
        c35 = A(config.P_SU1_rbl_upper_strut).T
        c36 = -1.0*multi_dot([c35,config.R_SU1_rbl_upper_strut])
        c37 = Triad(config.ax_jcl_strut_chassis)
        c38 = Mbar_ground_jcl_strut_chassis = multi_dot([c0,c37])
        c39 = Triad("config.ax2_jcl_strut_chassis", 'c37[0:3,1:2]')
        c40 = config.pt_jcr_tie_steering
        c41 = A(config.P_SU2_rbr_tie_rod).T
        c42 = -1.0*multi_dot([c41,config.R_SU2_rbr_tie_rod])
        c43 = Triad(config.ax_jcr_tie_steering)
        c44 = Triad("config.ax2_jcr_tie_steering", 'c43[0:3,1:2]')
        c45 = config.pt_jcl_tie_steering
        c46 = A(config.P_SU2_rbl_tie_rod).T
        c47 = -1.0*multi_dot([c46,config.R_SU2_rbl_tie_rod])
        c48 = Triad(config.ax_jcl_tie_steering)
        c49 = Triad("config.ax2_jcl_tie_steering", 'c48[0:3,1:2]')
        c50 = A(config.P_SU2_rbr_uca).T
        c51 = -1.0*multi_dot([c50,config.R_SU2_rbr_uca])
        c52 = A(config.P_SU2_rbl_uca).T
        c53 = -1.0*multi_dot([c52,config.R_SU2_rbl_uca])
        c54 = A(config.P_SU2_rbr_lca).T
        c55 = -1.0*multi_dot([c54,config.R_SU2_rbr_lca])
        c56 = A(config.P_SU2_rbl_lca).T
        c57 = -1.0*multi_dot([c56,config.R_SU2_rbl_lca])
        c58 = A(config.P_SU2_rbr_upper_strut).T
        c59 = -1.0*multi_dot([c58,config.R_SU2_rbr_upper_strut])
        c60 = A(config.P_SU2_rbl_upper_strut).T
        c61 = -1.0*multi_dot([c60,config.R_SU2_rbl_upper_strut])
        c62 = config.pt_jcr_rocker_ch
        c63 = A(config.P_ST_rbr_rocker).T
        c64 = -1.0*multi_dot([c63,config.R_ST_rbr_rocker])
        c65 = Triad(config.ax_jcr_rocker_ch)
        c66 = config.pt_jcl_rocker_ch
        c67 = A(config.P_ST_rbl_rocker).T
        c68 = -1.0*multi_dot([c67,config.R_ST_rbl_rocker])
        c69 = Triad(config.ax_jcl_rocker_ch)
        c70 = config.pt_jcr_uca_upright
        c71 = A(config.P_SU1_rbr_upright).T
        c72 = -1.0*multi_dot([c71,config.R_SU1_rbr_upright])
        c73 = Triad(config.ax_jcr_uca_upright)
        c74 = config.pt_jcl_uca_upright
        c75 = A(config.P_SU1_rbl_upright).T
        c76 = -1.0*multi_dot([c75,config.R_SU1_rbl_upright])
        c77 = Triad(config.ax_jcl_uca_upright)
        c78 = config.pt_jcr_lca_upright
        c79 = Triad(config.ax_jcr_lca_upright)
        c80 = config.pt_jcr_strut_lca
        c81 = A(config.P_SU1_rbr_lower_strut).T
        c82 = -1.0*multi_dot([c81,config.R_SU1_rbr_lower_strut])
        c83 = Triad(config.ax_jcr_strut_lca)
        c84 = Triad("config.ax2_jcr_strut_lca", 'c83[0:3,1:2]')
        c85 = config.pt_jcl_lca_upright
        c86 = Triad(config.ax_jcl_lca_upright)
        c87 = config.pt_jcl_strut_lca
        c88 = A(config.P_SU1_rbl_lower_strut).T
        c89 = -1.0*multi_dot([c88,config.R_SU1_rbl_lower_strut])
        c90 = Triad(config.ax_jcl_strut_lca)
        c91 = Triad("config.ax2_jcl_strut_lca", 'c90[0:3,1:2]')
        c92 = config.pt_jcr_tie_upright
        c93 = A(config.P_SU1_rbr_tie_rod).T
        c94 = -1.0*multi_dot([c93,config.R_SU1_rbr_tie_rod])
        c95 = Triad(config.ax_jcr_tie_upright)
        c96 = config.pt_jcr_hub_bearing
        c97 = A(config.P_SU1_rbr_hub).T
        c98 = Triad(config.ax_jcr_hub_bearing)
        c99 = config.pt_jcl_tie_upright
        c100 = A(config.P_SU1_rbl_tie_rod).T
        c101 = -1.0*multi_dot([c100,config.R_SU1_rbl_tie_rod])
        c102 = Triad(config.ax_jcl_tie_upright)
        c103 = config.pt_jcl_hub_bearing
        c104 = A(config.P_SU1_rbl_hub).T
        c105 = Triad(config.ax_jcl_hub_bearing)
        c106 = config.pt_jcr_strut
        c107 = Triad(config.ax_jcr_strut)
        c108 = config.pt_jcl_strut
        c109 = Triad(config.ax_jcl_strut)
        c110 = A(config.P_SU2_rbr_upright).T
        c111 = -1.0*multi_dot([c110,config.R_SU2_rbr_upright])
        c112 = A(config.P_SU2_rbl_upright).T
        c113 = -1.0*multi_dot([c112,config.R_SU2_rbl_upright])
        c114 = A(config.P_SU2_rbr_lower_strut).T
        c115 = -1.0*multi_dot([c114,config.R_SU2_rbr_lower_strut])
        c116 = A(config.P_SU2_rbl_lower_strut).T
        c117 = -1.0*multi_dot([c116,config.R_SU2_rbl_lower_strut])
        c118 = A(config.P_SU2_rbr_hub).T
        c119 = A(config.P_SU2_rbl_hub).T
        c120 = A(config.P_ST_rbs_coupler).T
        c121 = config.pt_jcs_rc_sph
        c122 = -1.0*multi_dot([c120,config.R_ST_rbs_coupler])
        c123 = Triad(config.ax_jcs_rc_sph)
        c124 = config.pt_jcs_rc_uni
        c125 = Triad(config.ax_jcs_rc_uni)

        self.c3
        self.ubar_SU1_rbr_uca_jcr_uca_chassis = (multi_dot([c4,c1]) + c5)
        self.c7
        self.Mbar_SU1_rbr_uca_jcr_uca_chassis = multi_dot([c4,c6])
        self.c9
        self.ubar_SU1_rbl_uca_jcl_uca_chassis = (multi_dot([c10,c8]) + c11)
        self.c13
        self.Mbar_SU1_rbl_uca_jcl_uca_chassis = multi_dot([c10,c12])
        self.c15
        self.ubar_SU1_rbr_lca_jcr_lca_chassis = (multi_dot([c16,c14]) + c17)
        self.c19
        self.Mbar_SU1_rbr_lca_jcr_lca_chassis = multi_dot([c16,c18])
        self.c21
        self.ubar_SU1_rbl_lca_jcl_lca_chassis = (multi_dot([c22,c20]) + c23)
        self.c25
        self.Mbar_SU1_rbl_lca_jcl_lca_chassis = multi_dot([c22,c24])
        self.c27
        self.ubar_SU1_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c28,c26]) + c29)
        self.c31
        self.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis = multi_dot([c28,c32])
        self.c34
        self.ubar_SU1_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c35,c33]) + c36)
        self.c38
        self.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis = multi_dot([c35,c39])
        self.ubar_ground_jcr_tie_steering = (multi_dot([c0,c40]) + c2)
        self.ubar_SU2_rbr_tie_rod_jcr_tie_steering = (multi_dot([c41,c40]) + c42)
        self.Mbar_ground_jcr_tie_steering = multi_dot([c0,c43])
        self.Mbar_SU2_rbr_tie_rod_jcr_tie_steering = multi_dot([c41,c44])
        self.ubar_ground_jcl_tie_steering = (multi_dot([c0,c45]) + c2)
        self.ubar_SU2_rbl_tie_rod_jcl_tie_steering = (multi_dot([c46,c45]) + c47)
        self.Mbar_ground_jcl_tie_steering = multi_dot([c0,c48])
        self.Mbar_SU2_rbl_tie_rod_jcl_tie_steering = multi_dot([c46,c49])
        self.c3
        self.ubar_SU2_rbr_uca_jcr_uca_chassis = (multi_dot([c50,c1]) + c51)
        self.c7
        self.Mbar_SU2_rbr_uca_jcr_uca_chassis = multi_dot([c50,c6])
        self.c9
        self.ubar_SU2_rbl_uca_jcl_uca_chassis = (multi_dot([c52,c8]) + c53)
        self.c13
        self.Mbar_SU2_rbl_uca_jcl_uca_chassis = multi_dot([c52,c12])
        self.c15
        self.ubar_SU2_rbr_lca_jcr_lca_chassis = (multi_dot([c54,c14]) + c55)
        self.c19
        self.Mbar_SU2_rbr_lca_jcr_lca_chassis = multi_dot([c54,c18])
        self.c21
        self.ubar_SU2_rbl_lca_jcl_lca_chassis = (multi_dot([c56,c20]) + c57)
        self.c25
        self.Mbar_SU2_rbl_lca_jcl_lca_chassis = multi_dot([c56,c24])
        self.c27
        self.ubar_SU2_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c58,c26]) + c59)
        self.c31
        self.Mbar_SU2_rbr_upper_strut_jcr_strut_chassis = multi_dot([c58,c32])
        self.c34
        self.ubar_SU2_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c60,c33]) + c61)
        self.c38
        self.Mbar_SU2_rbl_upper_strut_jcl_strut_chassis = multi_dot([c60,c39])
        self.ubar_ground_jcr_rocker_ch = (multi_dot([c0,c62]) + c2)
        self.ubar_ST_rbr_rocker_jcr_rocker_ch = (multi_dot([c63,c62]) + c64)
        self.Mbar_ground_jcr_rocker_ch = multi_dot([c0,c65])
        self.Mbar_ST_rbr_rocker_jcr_rocker_ch = multi_dot([c63,c65])
        self.ubar_ground_jcl_rocker_ch = (multi_dot([c0,c66]) + c2)
        self.ubar_ST_rbl_rocker_jcl_rocker_ch = (multi_dot([c67,c66]) + c68)
        self.Mbar_ground_jcl_rocker_ch = multi_dot([c0,c69])
        self.Mbar_ST_rbl_rocker_jcl_rocker_ch = multi_dot([c67,c69])
        self.ubar_SU1_rbr_uca_jcr_uca_upright = (multi_dot([c4,c70]) + c5)
        self.ubar_SU1_rbr_upright_jcr_uca_upright = (multi_dot([c71,c70]) + c72)
        self.Mbar_SU1_rbr_uca_jcr_uca_upright = multi_dot([c4,c73])
        self.Mbar_SU1_rbr_upright_jcr_uca_upright = multi_dot([c71,c73])
        self.ubar_SU1_rbl_uca_jcl_uca_upright = (multi_dot([c10,c74]) + c11)
        self.ubar_SU1_rbl_upright_jcl_uca_upright = (multi_dot([c75,c74]) + c76)
        self.Mbar_SU1_rbl_uca_jcl_uca_upright = multi_dot([c10,c77])
        self.Mbar_SU1_rbl_upright_jcl_uca_upright = multi_dot([c75,c77])
        self.ubar_SU1_rbr_lca_jcr_lca_upright = (multi_dot([c16,c78]) + c17)
        self.ubar_SU1_rbr_upright_jcr_lca_upright = (multi_dot([c71,c78]) + c72)
        self.Mbar_SU1_rbr_lca_jcr_lca_upright = multi_dot([c16,c79])
        self.Mbar_SU1_rbr_upright_jcr_lca_upright = multi_dot([c71,c79])
        self.ubar_SU1_rbr_lca_jcr_strut_lca = (multi_dot([c16,c80]) + c17)
        self.ubar_SU1_rbr_lower_strut_jcr_strut_lca = (multi_dot([c81,c80]) + c82)
        self.Mbar_SU1_rbr_lca_jcr_strut_lca = multi_dot([c16,c83])
        self.Mbar_SU1_rbr_lower_strut_jcr_strut_lca = multi_dot([c81,c84])
        self.ubar_SU1_rbl_lca_jcl_lca_upright = (multi_dot([c22,c85]) + c23)
        self.ubar_SU1_rbl_upright_jcl_lca_upright = (multi_dot([c75,c85]) + c76)
        self.Mbar_SU1_rbl_lca_jcl_lca_upright = multi_dot([c22,c86])
        self.Mbar_SU1_rbl_upright_jcl_lca_upright = multi_dot([c75,c86])
        self.ubar_SU1_rbl_lca_jcl_strut_lca = (multi_dot([c22,c87]) + c23)
        self.ubar_SU1_rbl_lower_strut_jcl_strut_lca = (multi_dot([c88,c87]) + c89)
        self.Mbar_SU1_rbl_lca_jcl_strut_lca = multi_dot([c22,c90])
        self.Mbar_SU1_rbl_lower_strut_jcl_strut_lca = multi_dot([c88,c91])
        self.ubar_SU1_rbr_upright_jcr_tie_upright = (multi_dot([c71,c92]) + c72)
        self.ubar_SU1_rbr_tie_rod_jcr_tie_upright = (multi_dot([c93,c92]) + c94)
        self.Mbar_SU1_rbr_upright_jcr_tie_upright = multi_dot([c71,c95])
        self.Mbar_SU1_rbr_tie_rod_jcr_tie_upright = multi_dot([c93,c95])
        self.ubar_SU1_rbr_upright_jcr_hub_bearing = (multi_dot([c71,c96]) + c72)
        self.ubar_SU1_rbr_hub_jcr_hub_bearing = (multi_dot([c97,c96]) + -1.0*multi_dot([c97,'R_SU1_rbr_hub']))
        self.Mbar_SU1_rbr_upright_jcr_hub_bearing = multi_dot([c71,c98])
        self.Mbar_SU1_rbr_hub_jcr_hub_bearing = multi_dot([c97,c98])
        self.ubar_SU1_rbl_upright_jcl_tie_upright = (multi_dot([c75,c99]) + c76)
        self.ubar_SU1_rbl_tie_rod_jcl_tie_upright = (multi_dot([c100,c99]) + c101)
        self.Mbar_SU1_rbl_upright_jcl_tie_upright = multi_dot([c75,c102])
        self.Mbar_SU1_rbl_tie_rod_jcl_tie_upright = multi_dot([c100,c102])
        self.ubar_SU1_rbl_upright_jcl_hub_bearing = (multi_dot([c75,c103]) + c76)
        self.ubar_SU1_rbl_hub_jcl_hub_bearing = (multi_dot([c104,c103]) + -1.0*multi_dot([c104,'R_SU1_rbl_hub']))
        self.Mbar_SU1_rbl_upright_jcl_hub_bearing = multi_dot([c75,c105])
        self.Mbar_SU1_rbl_hub_jcl_hub_bearing = multi_dot([c104,c105])
        self.ubar_SU1_rbr_upper_strut_jcr_strut = (multi_dot([c28,c106]) + c29)
        self.ubar_SU1_rbr_lower_strut_jcr_strut = (multi_dot([c81,c106]) + c82)
        self.Mbar_SU1_rbr_upper_strut_jcr_strut = multi_dot([c28,c107])
        self.Mbar_SU1_rbr_lower_strut_jcr_strut = multi_dot([c81,c107])
        self.ubar_SU1_rbl_upper_strut_jcl_strut = (multi_dot([c35,c108]) + c36)
        self.ubar_SU1_rbl_lower_strut_jcl_strut = (multi_dot([c88,c108]) + c89)
        self.Mbar_SU1_rbl_upper_strut_jcl_strut = multi_dot([c35,c109])
        self.Mbar_SU1_rbl_lower_strut_jcl_strut = multi_dot([c88,c109])
        self.ubar_SU1_rbr_tie_rod_jcr_tie_steering = (multi_dot([c93,c40]) + c94)
        self.ubar_ST_rbr_rocker_jcr_tie_steering = (multi_dot([c63,c40]) + c64)
        self.Mbar_SU1_rbr_tie_rod_jcr_tie_steering = multi_dot([c93,c43])
        self.Mbar_ST_rbr_rocker_jcr_tie_steering = multi_dot([c63,c44])
        self.ubar_SU1_rbl_tie_rod_jcl_tie_steering = (multi_dot([c100,c45]) + c101)
        self.ubar_ST_rbl_rocker_jcl_tie_steering = (multi_dot([c67,c45]) + c68)
        self.Mbar_SU1_rbl_tie_rod_jcl_tie_steering = multi_dot([c100,c48])
        self.Mbar_ST_rbl_rocker_jcl_tie_steering = multi_dot([c67,c49])
        self.ubar_SU2_rbr_uca_jcr_uca_upright = (multi_dot([c50,c70]) + c51)
        self.ubar_SU2_rbr_upright_jcr_uca_upright = (multi_dot([c110,c70]) + c111)
        self.Mbar_SU2_rbr_uca_jcr_uca_upright = multi_dot([c50,c73])
        self.Mbar_SU2_rbr_upright_jcr_uca_upright = multi_dot([c110,c73])
        self.ubar_SU2_rbl_uca_jcl_uca_upright = (multi_dot([c52,c74]) + c53)
        self.ubar_SU2_rbl_upright_jcl_uca_upright = (multi_dot([c112,c74]) + c113)
        self.Mbar_SU2_rbl_uca_jcl_uca_upright = multi_dot([c52,c77])
        self.Mbar_SU2_rbl_upright_jcl_uca_upright = multi_dot([c112,c77])
        self.ubar_SU2_rbr_lca_jcr_lca_upright = (multi_dot([c54,c78]) + c55)
        self.ubar_SU2_rbr_upright_jcr_lca_upright = (multi_dot([c110,c78]) + c111)
        self.Mbar_SU2_rbr_lca_jcr_lca_upright = multi_dot([c54,c79])
        self.Mbar_SU2_rbr_upright_jcr_lca_upright = multi_dot([c110,c79])
        self.ubar_SU2_rbr_lca_jcr_strut_lca = (multi_dot([c54,c80]) + c55)
        self.ubar_SU2_rbr_lower_strut_jcr_strut_lca = (multi_dot([c114,c80]) + c115)
        self.Mbar_SU2_rbr_lca_jcr_strut_lca = multi_dot([c54,c83])
        self.Mbar_SU2_rbr_lower_strut_jcr_strut_lca = multi_dot([c114,c84])
        self.ubar_SU2_rbl_lca_jcl_lca_upright = (multi_dot([c56,c85]) + c57)
        self.ubar_SU2_rbl_upright_jcl_lca_upright = (multi_dot([c112,c85]) + c113)
        self.Mbar_SU2_rbl_lca_jcl_lca_upright = multi_dot([c56,c86])
        self.Mbar_SU2_rbl_upright_jcl_lca_upright = multi_dot([c112,c86])
        self.ubar_SU2_rbl_lca_jcl_strut_lca = (multi_dot([c56,c87]) + c57)
        self.ubar_SU2_rbl_lower_strut_jcl_strut_lca = (multi_dot([c116,c87]) + c117)
        self.Mbar_SU2_rbl_lca_jcl_strut_lca = multi_dot([c56,c90])
        self.Mbar_SU2_rbl_lower_strut_jcl_strut_lca = multi_dot([c116,c91])
        self.ubar_SU2_rbr_upright_jcr_tie_upright = (multi_dot([c110,c92]) + c111)
        self.ubar_SU2_rbr_tie_rod_jcr_tie_upright = (multi_dot([c41,c92]) + c42)
        self.Mbar_SU2_rbr_upright_jcr_tie_upright = multi_dot([c110,c95])
        self.Mbar_SU2_rbr_tie_rod_jcr_tie_upright = multi_dot([c41,c95])
        self.ubar_SU2_rbr_upright_jcr_hub_bearing = (multi_dot([c110,c96]) + c111)
        self.ubar_SU2_rbr_hub_jcr_hub_bearing = (multi_dot([c118,c96]) + -1.0*multi_dot([c118,'R_SU2_rbr_hub']))
        self.Mbar_SU2_rbr_upright_jcr_hub_bearing = multi_dot([c110,c98])
        self.Mbar_SU2_rbr_hub_jcr_hub_bearing = multi_dot([c118,c98])
        self.ubar_SU2_rbl_upright_jcl_tie_upright = (multi_dot([c112,c99]) + c113)
        self.ubar_SU2_rbl_tie_rod_jcl_tie_upright = (multi_dot([c46,c99]) + c47)
        self.Mbar_SU2_rbl_upright_jcl_tie_upright = multi_dot([c112,c102])
        self.Mbar_SU2_rbl_tie_rod_jcl_tie_upright = multi_dot([c46,c102])
        self.ubar_SU2_rbl_upright_jcl_hub_bearing = (multi_dot([c112,c103]) + c113)
        self.ubar_SU2_rbl_hub_jcl_hub_bearing = (multi_dot([c119,c103]) + -1.0*multi_dot([c119,'R_SU2_rbl_hub']))
        self.Mbar_SU2_rbl_upright_jcl_hub_bearing = multi_dot([c112,c105])
        self.Mbar_SU2_rbl_hub_jcl_hub_bearing = multi_dot([c119,c105])
        self.ubar_SU2_rbr_upper_strut_jcr_strut = (multi_dot([c58,c106]) + c59)
        self.ubar_SU2_rbr_lower_strut_jcr_strut = (multi_dot([c114,c106]) + c115)
        self.Mbar_SU2_rbr_upper_strut_jcr_strut = multi_dot([c58,c107])
        self.Mbar_SU2_rbr_lower_strut_jcr_strut = multi_dot([c114,c107])
        self.ubar_SU2_rbl_upper_strut_jcl_strut = (multi_dot([c60,c108]) + c61)
        self.ubar_SU2_rbl_lower_strut_jcl_strut = (multi_dot([c116,c108]) + c117)
        self.Mbar_SU2_rbl_upper_strut_jcl_strut = multi_dot([c60,c109])
        self.Mbar_SU2_rbl_lower_strut_jcl_strut = multi_dot([c116,c109])
        self.ubar_ST_rbs_coupler_jcs_rc_sph = (multi_dot([c120,c121]) + c122)
        self.ubar_ST_rbr_rocker_jcs_rc_sph = (multi_dot([c63,c121]) + c64)
        self.Mbar_ST_rbs_coupler_jcs_rc_sph = multi_dot([c120,c123])
        self.Mbar_ST_rbr_rocker_jcs_rc_sph = multi_dot([c63,c123])
        self.ubar_ST_rbs_coupler_jcs_rc_uni = (multi_dot([c120,c124]) + c122)
        self.ubar_ST_rbl_rocker_jcs_rc_uni = (multi_dot([c67,c124]) + c68)
        self.Mbar_ST_rbs_coupler_jcs_rc_uni = multi_dot([c120,c125])
        self.Mbar_ST_rbl_rocker_jcs_rc_uni = multi_dot([c67,c125])

    
def set_coordinates(self,q):
    self.R_ground = q[0:3,0:1]
    self.P_ground = q[3:7,0:1]
    self.R_SU1_rbr_uca = q[7:10,0:1]
    self.P_SU1_rbr_uca = q[10:14,0:1]
    self.R_SU1_rbl_uca = q[14:17,0:1]
    self.P_SU1_rbl_uca = q[17:21,0:1]
    self.R_SU1_rbr_lca = q[21:24,0:1]
    self.P_SU1_rbr_lca = q[24:28,0:1]
    self.R_SU1_rbl_lca = q[28:31,0:1]
    self.P_SU1_rbl_lca = q[31:35,0:1]
    self.R_SU1_rbr_upright = q[35:38,0:1]
    self.P_SU1_rbr_upright = q[38:42,0:1]
    self.R_SU1_rbl_upright = q[42:45,0:1]
    self.P_SU1_rbl_upright = q[45:49,0:1]
    self.R_SU1_rbr_upper_strut = q[49:52,0:1]
    self.P_SU1_rbr_upper_strut = q[52:56,0:1]
    self.R_SU1_rbl_upper_strut = q[56:59,0:1]
    self.P_SU1_rbl_upper_strut = q[59:63,0:1]
    self.R_SU1_rbr_lower_strut = q[63:66,0:1]
    self.P_SU1_rbr_lower_strut = q[66:70,0:1]
    self.R_SU1_rbl_lower_strut = q[70:73,0:1]
    self.P_SU1_rbl_lower_strut = q[73:77,0:1]
    self.R_SU1_rbr_tie_rod = q[77:80,0:1]
    self.P_SU1_rbr_tie_rod = q[80:84,0:1]
    self.R_SU1_rbl_tie_rod = q[84:87,0:1]
    self.P_SU1_rbl_tie_rod = q[87:91,0:1]
    self.R_SU1_rbr_hub = q[91:94,0:1]
    self.P_SU1_rbr_hub = q[94:98,0:1]
    self.R_SU1_rbl_hub = q[98:101,0:1]
    self.P_SU1_rbl_hub = q[101:105,0:1]
    self.R_SU2_rbr_uca = q[105:108,0:1]
    self.P_SU2_rbr_uca = q[108:112,0:1]
    self.R_SU2_rbl_uca = q[112:115,0:1]
    self.P_SU2_rbl_uca = q[115:119,0:1]
    self.R_SU2_rbr_lca = q[119:122,0:1]
    self.P_SU2_rbr_lca = q[122:126,0:1]
    self.R_SU2_rbl_lca = q[126:129,0:1]
    self.P_SU2_rbl_lca = q[129:133,0:1]
    self.R_SU2_rbr_upright = q[133:136,0:1]
    self.P_SU2_rbr_upright = q[136:140,0:1]
    self.R_SU2_rbl_upright = q[140:143,0:1]
    self.P_SU2_rbl_upright = q[143:147,0:1]
    self.R_SU2_rbr_upper_strut = q[147:150,0:1]
    self.P_SU2_rbr_upper_strut = q[150:154,0:1]
    self.R_SU2_rbl_upper_strut = q[154:157,0:1]
    self.P_SU2_rbl_upper_strut = q[157:161,0:1]
    self.R_SU2_rbr_lower_strut = q[161:164,0:1]
    self.P_SU2_rbr_lower_strut = q[164:168,0:1]
    self.R_SU2_rbl_lower_strut = q[168:171,0:1]
    self.P_SU2_rbl_lower_strut = q[171:175,0:1]
    self.R_SU2_rbr_tie_rod = q[175:178,0:1]
    self.P_SU2_rbr_tie_rod = q[178:182,0:1]
    self.R_SU2_rbl_tie_rod = q[182:185,0:1]
    self.P_SU2_rbl_tie_rod = q[185:189,0:1]
    self.R_SU2_rbr_hub = q[189:192,0:1]
    self.P_SU2_rbr_hub = q[192:196,0:1]
    self.R_SU2_rbl_hub = q[196:199,0:1]
    self.P_SU2_rbl_hub = q[199:203,0:1]
    self.R_ST_rbs_coupler = q[203:206,0:1]
    self.P_ST_rbs_coupler = q[206:210,0:1]
    self.R_ST_rbr_rocker = q[210:213,0:1]
    self.P_ST_rbr_rocker = q[213:217,0:1]
    self.R_ST_rbl_rocker = q[217:220,0:1]
    self.P_ST_rbl_rocker = q[220:224,0:1]


    
def set_velocities(self,qd):
    self.Rd_ground = qd[0:3,0:1]
    self.Pd_ground = qd[3:7,0:1]
    self.Rd_SU1_rbr_uca = qd[7:10,0:1]
    self.Pd_SU1_rbr_uca = qd[10:14,0:1]
    self.Rd_SU1_rbl_uca = qd[14:17,0:1]
    self.Pd_SU1_rbl_uca = qd[17:21,0:1]
    self.Rd_SU1_rbr_lca = qd[21:24,0:1]
    self.Pd_SU1_rbr_lca = qd[24:28,0:1]
    self.Rd_SU1_rbl_lca = qd[28:31,0:1]
    self.Pd_SU1_rbl_lca = qd[31:35,0:1]
    self.Rd_SU1_rbr_upright = qd[35:38,0:1]
    self.Pd_SU1_rbr_upright = qd[38:42,0:1]
    self.Rd_SU1_rbl_upright = qd[42:45,0:1]
    self.Pd_SU1_rbl_upright = qd[45:49,0:1]
    self.Rd_SU1_rbr_upper_strut = qd[49:52,0:1]
    self.Pd_SU1_rbr_upper_strut = qd[52:56,0:1]
    self.Rd_SU1_rbl_upper_strut = qd[56:59,0:1]
    self.Pd_SU1_rbl_upper_strut = qd[59:63,0:1]
    self.Rd_SU1_rbr_lower_strut = qd[63:66,0:1]
    self.Pd_SU1_rbr_lower_strut = qd[66:70,0:1]
    self.Rd_SU1_rbl_lower_strut = qd[70:73,0:1]
    self.Pd_SU1_rbl_lower_strut = qd[73:77,0:1]
    self.Rd_SU1_rbr_tie_rod = qd[77:80,0:1]
    self.Pd_SU1_rbr_tie_rod = qd[80:84,0:1]
    self.Rd_SU1_rbl_tie_rod = qd[84:87,0:1]
    self.Pd_SU1_rbl_tie_rod = qd[87:91,0:1]
    self.Rd_SU1_rbr_hub = qd[91:94,0:1]
    self.Pd_SU1_rbr_hub = qd[94:98,0:1]
    self.Rd_SU1_rbl_hub = qd[98:101,0:1]
    self.Pd_SU1_rbl_hub = qd[101:105,0:1]
    self.Rd_SU2_rbr_uca = qd[105:108,0:1]
    self.Pd_SU2_rbr_uca = qd[108:112,0:1]
    self.Rd_SU2_rbl_uca = qd[112:115,0:1]
    self.Pd_SU2_rbl_uca = qd[115:119,0:1]
    self.Rd_SU2_rbr_lca = qd[119:122,0:1]
    self.Pd_SU2_rbr_lca = qd[122:126,0:1]
    self.Rd_SU2_rbl_lca = qd[126:129,0:1]
    self.Pd_SU2_rbl_lca = qd[129:133,0:1]
    self.Rd_SU2_rbr_upright = qd[133:136,0:1]
    self.Pd_SU2_rbr_upright = qd[136:140,0:1]
    self.Rd_SU2_rbl_upright = qd[140:143,0:1]
    self.Pd_SU2_rbl_upright = qd[143:147,0:1]
    self.Rd_SU2_rbr_upper_strut = qd[147:150,0:1]
    self.Pd_SU2_rbr_upper_strut = qd[150:154,0:1]
    self.Rd_SU2_rbl_upper_strut = qd[154:157,0:1]
    self.Pd_SU2_rbl_upper_strut = qd[157:161,0:1]
    self.Rd_SU2_rbr_lower_strut = qd[161:164,0:1]
    self.Pd_SU2_rbr_lower_strut = qd[164:168,0:1]
    self.Rd_SU2_rbl_lower_strut = qd[168:171,0:1]
    self.Pd_SU2_rbl_lower_strut = qd[171:175,0:1]
    self.Rd_SU2_rbr_tie_rod = qd[175:178,0:1]
    self.Pd_SU2_rbr_tie_rod = qd[178:182,0:1]
    self.Rd_SU2_rbl_tie_rod = qd[182:185,0:1]
    self.Pd_SU2_rbl_tie_rod = qd[185:189,0:1]
    self.Rd_SU2_rbr_hub = qd[189:192,0:1]
    self.Pd_SU2_rbr_hub = qd[192:196,0:1]
    self.Rd_SU2_rbl_hub = qd[196:199,0:1]
    self.Pd_SU2_rbl_hub = qd[199:203,0:1]
    self.Rd_ST_rbs_coupler = qd[203:206,0:1]
    self.Pd_ST_rbs_coupler = qd[206:210,0:1]
    self.Rd_ST_rbr_rocker = qd[210:213,0:1]
    self.Pd_ST_rbr_rocker = qd[213:217,0:1]
    self.Rd_ST_rbl_rocker = qd[217:220,0:1]
    self.Pd_ST_rbl_rocker = qd[220:224,0:1]


    
def set_initial_configuration(self):
    config = self.config

    q = np.concatenate([config.R_ground,
    config.P_ground,
    config.R_SU1_rbr_uca,
    config.P_SU1_rbr_uca,
    config.R_SU1_rbl_uca,
    config.P_SU1_rbl_uca,
    config.R_SU1_rbr_lca,
    config.P_SU1_rbr_lca,
    config.R_SU1_rbl_lca,
    config.P_SU1_rbl_lca,
    config.R_SU1_rbr_upright,
    config.P_SU1_rbr_upright,
    config.R_SU1_rbl_upright,
    config.P_SU1_rbl_upright,
    config.R_SU1_rbr_upper_strut,
    config.P_SU1_rbr_upper_strut,
    config.R_SU1_rbl_upper_strut,
    config.P_SU1_rbl_upper_strut,
    config.R_SU1_rbr_lower_strut,
    config.P_SU1_rbr_lower_strut,
    config.R_SU1_rbl_lower_strut,
    config.P_SU1_rbl_lower_strut,
    config.R_SU1_rbr_tie_rod,
    config.P_SU1_rbr_tie_rod,
    config.R_SU1_rbl_tie_rod,
    config.P_SU1_rbl_tie_rod,
    config.R_SU1_rbr_hub,
    config.P_SU1_rbr_hub,
    config.R_SU1_rbl_hub,
    config.P_SU1_rbl_hub,
    config.R_SU2_rbr_uca,
    config.P_SU2_rbr_uca,
    config.R_SU2_rbl_uca,
    config.P_SU2_rbl_uca,
    config.R_SU2_rbr_lca,
    config.P_SU2_rbr_lca,
    config.R_SU2_rbl_lca,
    config.P_SU2_rbl_lca,
    config.R_SU2_rbr_upright,
    config.P_SU2_rbr_upright,
    config.R_SU2_rbl_upright,
    config.P_SU2_rbl_upright,
    config.R_SU2_rbr_upper_strut,
    config.P_SU2_rbr_upper_strut,
    config.R_SU2_rbl_upper_strut,
    config.P_SU2_rbl_upper_strut,
    config.R_SU2_rbr_lower_strut,
    config.P_SU2_rbr_lower_strut,
    config.R_SU2_rbl_lower_strut,
    config.P_SU2_rbl_lower_strut,
    config.R_SU2_rbr_tie_rod,
    config.P_SU2_rbr_tie_rod,
    config.R_SU2_rbl_tie_rod,
    config.P_SU2_rbl_tie_rod,
    config.R_SU2_rbr_hub,
    config.P_SU2_rbr_hub,
    config.R_SU2_rbl_hub,
    config.P_SU2_rbl_hub,
    config.R_ST_rbs_coupler,
    config.P_ST_rbs_coupler,
    config.R_ST_rbr_rocker,
    config.P_ST_rbr_rocker,
    config.R_ST_rbl_rocker,
    config.P_ST_rbl_rocker])

    qd = np.concatenate([config.Rd_ground,
    config.Pd_ground,
    config.Rd_SU1_rbr_uca,
    config.Pd_SU1_rbr_uca,
    config.Rd_SU1_rbl_uca,
    config.Pd_SU1_rbl_uca,
    config.Rd_SU1_rbr_lca,
    config.Pd_SU1_rbr_lca,
    config.Rd_SU1_rbl_lca,
    config.Pd_SU1_rbl_lca,
    config.Rd_SU1_rbr_upright,
    config.Pd_SU1_rbr_upright,
    config.Rd_SU1_rbl_upright,
    config.Pd_SU1_rbl_upright,
    config.Rd_SU1_rbr_upper_strut,
    config.Pd_SU1_rbr_upper_strut,
    config.Rd_SU1_rbl_upper_strut,
    config.Pd_SU1_rbl_upper_strut,
    config.Rd_SU1_rbr_lower_strut,
    config.Pd_SU1_rbr_lower_strut,
    config.Rd_SU1_rbl_lower_strut,
    config.Pd_SU1_rbl_lower_strut,
    config.Rd_SU1_rbr_tie_rod,
    config.Pd_SU1_rbr_tie_rod,
    config.Rd_SU1_rbl_tie_rod,
    config.Pd_SU1_rbl_tie_rod,
    config.Rd_SU1_rbr_hub,
    config.Pd_SU1_rbr_hub,
    config.Rd_SU1_rbl_hub,
    config.Pd_SU1_rbl_hub,
    config.Rd_SU2_rbr_uca,
    config.Pd_SU2_rbr_uca,
    config.Rd_SU2_rbl_uca,
    config.Pd_SU2_rbl_uca,
    config.Rd_SU2_rbr_lca,
    config.Pd_SU2_rbr_lca,
    config.Rd_SU2_rbl_lca,
    config.Pd_SU2_rbl_lca,
    config.Rd_SU2_rbr_upright,
    config.Pd_SU2_rbr_upright,
    config.Rd_SU2_rbl_upright,
    config.Pd_SU2_rbl_upright,
    config.Rd_SU2_rbr_upper_strut,
    config.Pd_SU2_rbr_upper_strut,
    config.Rd_SU2_rbl_upper_strut,
    config.Pd_SU2_rbl_upper_strut,
    config.Rd_SU2_rbr_lower_strut,
    config.Pd_SU2_rbr_lower_strut,
    config.Rd_SU2_rbl_lower_strut,
    config.Pd_SU2_rbl_lower_strut,
    config.Rd_SU2_rbr_tie_rod,
    config.Pd_SU2_rbr_tie_rod,
    config.Rd_SU2_rbl_tie_rod,
    config.Pd_SU2_rbl_tie_rod,
    config.Rd_SU2_rbr_hub,
    config.Pd_SU2_rbr_hub,
    config.Rd_SU2_rbl_hub,
    config.Pd_SU2_rbl_hub,
    config.Rd_ST_rbs_coupler,
    config.Pd_ST_rbs_coupler,
    config.Rd_ST_rbr_rocker,
    config.Pd_ST_rbr_rocker,
    config.Rd_ST_rbl_rocker,
    config.Pd_ST_rbl_rocker])

    self.set_coordinates(q)
    self.set_velocities(qd)
    self.q_initial = q.copy()



    
def eval_pos_eq(self):
    F = self.F
    t = self.t

    x0 = self.R_SU1_rbr_uca
    x1 = self.P_SU1_rbr_uca
    x2 = A(x1)
    x3 = self.P_ground
    x4 = A(x3)
    x5 = self.R_ground
    x6 = (multi_dot([x4,self.ubar_ground_jcr_uca_chassis]) + x5)
    x7 = x4.T
    x8 = self.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
    x9 = self.R_SU1_rbl_uca
    x10 = self.P_SU1_rbl_uca
    x11 = A(x10)
    x12 = (multi_dot([x4,self.ubar_ground_jcl_uca_chassis]) + x5)
    x13 = self.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
    x14 = self.R_SU1_rbr_lca
    x15 = self.P_SU1_rbr_lca
    x16 = A(x15)
    x17 = (multi_dot([x4,self.ubar_ground_jcr_lca_chassis]) + x5)
    x18 = self.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
    x19 = self.R_SU1_rbl_lca
    x20 = self.P_SU1_rbl_lca
    x21 = A(x20)
    x22 = (multi_dot([x4,self.ubar_ground_jcl_lca_chassis]) + x5)
    x23 = self.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
    x24 = self.R_SU1_rbr_upper_strut
    x25 = self.P_SU1_rbr_upper_strut
    x26 = A(x25)
    x27 = (multi_dot([x4,self.ubar_ground_jcr_strut_chassis]) + x5)
    x28 = self.R_SU1_rbl_upper_strut
    x29 = self.P_SU1_rbl_upper_strut
    x30 = A(x29)
    x31 = (multi_dot([x4,self.ubar_ground_jcl_strut_chassis]) + x5)
    x32 = -1.0*self.R_SU2_rbr_tie_rod
    x33 = self.P_SU2_rbr_tie_rod
    x34 = A(x33)
    x35 = -1.0*self.R_SU2_rbl_tie_rod
    x36 = self.P_SU2_rbl_tie_rod
    x37 = A(x36)
    x38 = self.R_SU2_rbr_uca
    x39 = self.P_SU2_rbr_uca
    x40 = A(x39)
    x41 = self.Mbar_SU2_rbr_uca_jcr_uca_chassis[:,2:3]
    x42 = self.R_SU2_rbl_uca
    x43 = self.P_SU2_rbl_uca
    x44 = A(x43)
    x45 = self.Mbar_SU2_rbl_uca_jcl_uca_chassis[:,2:3]
    x46 = self.R_SU2_rbr_lca
    x47 = self.P_SU2_rbr_lca
    x48 = A(x47)
    x49 = self.Mbar_SU2_rbr_lca_jcr_lca_chassis[:,2:3]
    x50 = self.R_SU2_rbl_lca
    x51 = self.P_SU2_rbl_lca
    x52 = A(x51)
    x53 = self.Mbar_SU2_rbl_lca_jcl_lca_chassis[:,2:3]
    x54 = self.R_SU2_rbr_upper_strut
    x55 = self.P_SU2_rbr_upper_strut
    x56 = A(x55)
    x57 = self.R_SU2_rbl_upper_strut
    x58 = self.P_SU2_rbl_upper_strut
    x59 = A(x58)
    x60 = -1.0*self.R_ST_rbr_rocker
    x61 = self.P_ST_rbr_rocker
    x62 = A(x61)
    x63 = self.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
    x64 = -1.0*self.R_ST_rbl_rocker
    x65 = self.P_ST_rbl_rocker
    x66 = A(x65)
    x67 = self.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
    x68 = self.R_SU1_rbr_upright
    x69 = -1.0*x68
    x70 = self.P_SU1_rbr_upright
    x71 = A(x70)
    x72 = self.R_SU1_rbl_upright
    x73 = -1.0*x72
    x74 = self.P_SU1_rbl_upright
    x75 = A(x74)
    x76 = -1.0*self.R_SU1_rbr_lower_strut
    x77 = self.P_SU1_rbr_lower_strut
    x78 = A(x77)
    x79 = -1.0*self.R_SU1_rbl_lower_strut
    x80 = self.P_SU1_rbl_lower_strut
    x81 = A(x80)
    x82 = self.R_SU1_rbr_tie_rod
    x83 = self.P_SU1_rbr_tie_rod
    x84 = A(x83)
    x85 = self.P_SU1_rbr_hub
    x86 = A(x85)
    x87 = x71.T
    x88 = self.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
    x89 = self.R_SU1_rbl_tie_rod
    x90 = self.P_SU1_rbl_tie_rod
    x91 = A(x90)
    x92 = self.P_SU1_rbl_hub
    x93 = A(x92)
    x94 = x75.T
    x95 = self.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
    x96 = self.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1].T
    x97 = x26.T
    x98 = self.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
    x99 = self.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2].T
    x100 = (x24 + x76 + multi_dot([x26,self.ubar_SU1_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x78,self.ubar_SU1_rbr_lower_strut_jcr_strut]))
    x101 = self.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1].T
    x102 = x30.T
    x103 = self.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
    x104 = self.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2].T
    x105 = (x28 + x79 + multi_dot([x30,self.ubar_SU1_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x81,self.ubar_SU1_rbl_lower_strut_jcl_strut]))
    x106 = self.R_SU2_rbr_upright
    x107 = -1.0*x106
    x108 = self.P_SU2_rbr_upright
    x109 = A(x108)
    x110 = self.R_SU2_rbl_upright
    x111 = -1.0*x110
    x112 = self.P_SU2_rbl_upright
    x113 = A(x112)
    x114 = -1.0*self.R_SU2_rbr_lower_strut
    x115 = self.P_SU2_rbr_lower_strut
    x116 = A(x115)
    x117 = -1.0*self.R_SU2_rbl_lower_strut
    x118 = self.P_SU2_rbl_lower_strut
    x119 = A(x118)
    x120 = self.P_SU2_rbr_hub
    x121 = A(x120)
    x122 = x109.T
    x123 = self.Mbar_SU2_rbr_hub_jcr_hub_bearing[:,2:3]
    x124 = self.P_SU2_rbl_hub
    x125 = A(x124)
    x126 = x113.T
    x127 = self.Mbar_SU2_rbl_hub_jcl_hub_bearing[:,2:3]
    x128 = self.Mbar_SU2_rbr_upper_strut_jcr_strut[:,0:1].T
    x129 = x56.T
    x130 = self.Mbar_SU2_rbr_lower_strut_jcr_strut[:,2:3]
    x131 = self.Mbar_SU2_rbr_upper_strut_jcr_strut[:,1:2].T
    x132 = (x54 + x114 + multi_dot([x56,self.ubar_SU2_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x116,self.ubar_SU2_rbr_lower_strut_jcr_strut]))
    x133 = self.Mbar_SU2_rbl_upper_strut_jcl_strut[:,0:1].T
    x134 = x59.T
    x135 = self.Mbar_SU2_rbl_lower_strut_jcl_strut[:,2:3]
    x136 = self.Mbar_SU2_rbl_upper_strut_jcl_strut[:,1:2].T
    x137 = (x57 + x117 + multi_dot([x59,self.ubar_SU2_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x119,self.ubar_SU2_rbl_lower_strut_jcl_strut]))
    x138 = self.R_ST_rbs_coupler
    x139 = self.P_ST_rbs_coupler
    x140 = A(x139)
    x141 = x140.T
    x142 = self.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
    x143 = -1.0*np.eye(1,dtype=np.float64)

    self.pos_level_data_blocks = [x6 + (-1.0*x0 + -1.0*multi_dot([x2,self.ubar_SU1_rbr_uca_jcr_uca_chassis])),multi_dot([self.Mbar_ground_jcr_uca_chassis[:,0:1].T,x7,x2,x8]),multi_dot([self.Mbar_ground_jcr_uca_chassis[:,1:2].T,x7,x2,x8]),x12 + (-1.0*x9 + -1.0*multi_dot([x11,self.ubar_SU1_rbl_uca_jcl_uca_chassis])),multi_dot([self.Mbar_ground_jcl_uca_chassis[:,0:1].T,x7,x11,x13]),multi_dot([self.Mbar_ground_jcl_uca_chassis[:,1:2].T,x7,x11,x13]),x17 + (-1.0*x14 + -1.0*multi_dot([x16,self.ubar_SU1_rbr_lca_jcr_lca_chassis])),multi_dot([self.Mbar_ground_jcr_lca_chassis[:,0:1].T,x7,x16,x18]),multi_dot([self.Mbar_ground_jcr_lca_chassis[:,1:2].T,x7,x16,x18]),x22 + (-1.0*x19 + -1.0*multi_dot([x21,self.ubar_SU1_rbl_lca_jcl_lca_chassis])),multi_dot([self.Mbar_ground_jcl_lca_chassis[:,0:1].T,x7,x21,x23]),multi_dot([self.Mbar_ground_jcl_lca_chassis[:,1:2].T,x7,x21,x23]),x27 + (-1.0*x24 + -1.0*multi_dot([x26,self.ubar_SU1_rbr_upper_strut_jcr_strut_chassis])),multi_dot([self.Mbar_ground_jcr_strut_chassis[:,0:1].T,x7,x26,self.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]]),x31 + (-1.0*x28 + -1.0*multi_dot([x30,self.ubar_SU1_rbl_upper_strut_jcl_strut_chassis])),multi_dot([self.Mbar_ground_jcl_strut_chassis[:,0:1].T,x7,x30,self.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]]),(x5 + x32 + multi_dot([x4,self.ubar_ground_jcr_tie_steering]) + -1.0*multi_dot([x34,self.ubar_SU2_rbr_tie_rod_jcr_tie_steering])),multi_dot([self.Mbar_ground_jcr_tie_steering[:,0:1].T,x7,x34,self.Mbar_SU2_rbr_tie_rod_jcr_tie_steering[:,0:1]]),(x5 + x35 + multi_dot([x4,self.ubar_ground_jcl_tie_steering]) + -1.0*multi_dot([x37,self.ubar_SU2_rbl_tie_rod_jcl_tie_steering])),multi_dot([self.Mbar_ground_jcl_tie_steering[:,0:1].T,x7,x37,self.Mbar_SU2_rbl_tie_rod_jcl_tie_steering[:,0:1]]),x6 + (-1.0*x38 + -1.0*multi_dot([x40,self.ubar_SU2_rbr_uca_jcr_uca_chassis])),multi_dot([self.Mbar_ground_jcr_uca_chassis[:,0:1].T,x7,x40,x41]),multi_dot([self.Mbar_ground_jcr_uca_chassis[:,1:2].T,x7,x40,x41]),x12 + (-1.0*x42 + -1.0*multi_dot([x44,self.ubar_SU2_rbl_uca_jcl_uca_chassis])),multi_dot([self.Mbar_ground_jcl_uca_chassis[:,0:1].T,x7,x44,x45]),multi_dot([self.Mbar_ground_jcl_uca_chassis[:,1:2].T,x7,x44,x45]),x17 + (-1.0*x46 + -1.0*multi_dot([x48,self.ubar_SU2_rbr_lca_jcr_lca_chassis])),multi_dot([self.Mbar_ground_jcr_lca_chassis[:,0:1].T,x7,x48,x49]),multi_dot([self.Mbar_ground_jcr_lca_chassis[:,1:2].T,x7,x48,x49]),x22 + (-1.0*x50 + -1.0*multi_dot([x52,self.ubar_SU2_rbl_lca_jcl_lca_chassis])),multi_dot([self.Mbar_ground_jcl_lca_chassis[:,0:1].T,x7,x52,x53]),multi_dot([self.Mbar_ground_jcl_lca_chassis[:,1:2].T,x7,x52,x53]),x27 + (-1.0*x54 + -1.0*multi_dot([x56,self.ubar_SU2_rbr_upper_strut_jcr_strut_chassis])),multi_dot([self.Mbar_ground_jcr_strut_chassis[:,0:1].T,x7,x56,self.Mbar_SU2_rbr_upper_strut_jcr_strut_chassis[:,0:1]]),x31 + (-1.0*x57 + -1.0*multi_dot([x59,self.ubar_SU2_rbl_upper_strut_jcl_strut_chassis])),multi_dot([self.Mbar_ground_jcl_strut_chassis[:,0:1].T,x7,x59,self.Mbar_SU2_rbl_upper_strut_jcl_strut_chassis[:,0:1]]),(x5 + x60 + multi_dot([x4,self.ubar_ground_jcr_rocker_ch]) + -1.0*multi_dot([x62,self.ubar_ST_rbr_rocker_jcr_rocker_ch])),multi_dot([self.Mbar_ground_jcr_rocker_ch[:,0:1].T,x7,x62,x63]),multi_dot([self.Mbar_ground_jcr_rocker_ch[:,1:2].T,x7,x62,x63]),(x5 + x64 + multi_dot([x4,self.ubar_ground_jcl_rocker_ch]) + -1.0*multi_dot([x66,self.ubar_ST_rbl_rocker_jcl_rocker_ch])),multi_dot([self.Mbar_ground_jcl_rocker_ch[:,0:1].T,x7,x66,x67]),multi_dot([self.Mbar_ground_jcl_rocker_ch[:,1:2].T,x7,x66,x67]),(x0 + x69 + multi_dot([x2,self.ubar_SU1_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x71,self.ubar_SU1_rbr_upright_jcr_uca_upright])),(x9 + x73 + multi_dot([x11,self.ubar_SU1_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x75,self.ubar_SU1_rbl_upright_jcl_uca_upright])),(x14 + x69 + multi_dot([x16,self.ubar_SU1_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x71,self.ubar_SU1_rbr_upright_jcr_lca_upright])),(x14 + x76 + multi_dot([x16,self.ubar_SU1_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x78,self.ubar_SU1_rbr_lower_strut_jcr_strut_lca])),multi_dot([self.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1].T,x16.T,x78,self.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x19 + x73 + multi_dot([x21,self.ubar_SU1_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x75,self.ubar_SU1_rbl_upright_jcl_lca_upright])),(x19 + x79 + multi_dot([x21,self.ubar_SU1_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x81,self.ubar_SU1_rbl_lower_strut_jcl_strut_lca])),multi_dot([self.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1].T,x21.T,x81,self.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x68 + -1.0*x82 + multi_dot([x71,self.ubar_SU1_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x84,self.ubar_SU1_rbr_tie_rod_jcr_tie_upright])),(x68 + -1.0*self.R_SU1_rbr_hub + multi_dot([x71,self.ubar_SU1_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x86,self.ubar_SU1_rbr_hub_jcr_hub_bearing])),multi_dot([self.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1].T,x87,x86,x88]),multi_dot([self.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2].T,x87,x86,x88]),(x72 + -1.0*x89 + multi_dot([x75,self.ubar_SU1_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x91,self.ubar_SU1_rbl_tie_rod_jcl_tie_upright])),(x72 + -1.0*self.R_SU1_rbl_hub + multi_dot([x75,self.ubar_SU1_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x93,self.ubar_SU1_rbl_hub_jcl_hub_bearing])),multi_dot([self.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1].T,x94,x93,x95]),multi_dot([self.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2].T,x94,x93,x95]),multi_dot([x96,x97,x78,x98]),multi_dot([x99,x97,x78,x98]),multi_dot([x96,x97,x100]),multi_dot([x99,x97,x100]),multi_dot([x101,x102,x81,x103]),multi_dot([x104,x102,x81,x103]),multi_dot([x101,x102,x105]),multi_dot([x104,x102,x105]),(x82 + x60 + multi_dot([x84,self.ubar_SU1_rbr_tie_rod_jcr_tie_steering]) + -1.0*multi_dot([x62,self.ubar_ST_rbr_rocker_jcr_tie_steering])),multi_dot([self.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x84.T,x62,self.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]]),(x89 + x64 + multi_dot([x91,self.ubar_SU1_rbl_tie_rod_jcl_tie_steering]) + -1.0*multi_dot([x66,self.ubar_ST_rbl_rocker_jcl_tie_steering])),multi_dot([self.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x91.T,x66,self.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]]),(x38 + x107 + multi_dot([x40,self.ubar_SU2_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x109,self.ubar_SU2_rbr_upright_jcr_uca_upright])),(x42 + x111 + multi_dot([x44,self.ubar_SU2_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x113,self.ubar_SU2_rbl_upright_jcl_uca_upright])),(x46 + x107 + multi_dot([x48,self.ubar_SU2_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x109,self.ubar_SU2_rbr_upright_jcr_lca_upright])),(x46 + x114 + multi_dot([x48,self.ubar_SU2_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x116,self.ubar_SU2_rbr_lower_strut_jcr_strut_lca])),multi_dot([self.Mbar_SU2_rbr_lca_jcr_strut_lca[:,0:1].T,x48.T,x116,self.Mbar_SU2_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x50 + x111 + multi_dot([x52,self.ubar_SU2_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x113,self.ubar_SU2_rbl_upright_jcl_lca_upright])),(x50 + x117 + multi_dot([x52,self.ubar_SU2_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x119,self.ubar_SU2_rbl_lower_strut_jcl_strut_lca])),multi_dot([self.Mbar_SU2_rbl_lca_jcl_strut_lca[:,0:1].T,x52.T,x119,self.Mbar_SU2_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x106 + x32 + multi_dot([x109,self.ubar_SU2_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x34,self.ubar_SU2_rbr_tie_rod_jcr_tie_upright])),(x106 + -1.0*self.R_SU2_rbr_hub + multi_dot([x109,self.ubar_SU2_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x121,self.ubar_SU2_rbr_hub_jcr_hub_bearing])),multi_dot([self.Mbar_SU2_rbr_upright_jcr_hub_bearing[:,0:1].T,x122,x121,x123]),multi_dot([self.Mbar_SU2_rbr_upright_jcr_hub_bearing[:,1:2].T,x122,x121,x123]),(x110 + x35 + multi_dot([x113,self.ubar_SU2_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x37,self.ubar_SU2_rbl_tie_rod_jcl_tie_upright])),(x110 + -1.0*self.R_SU2_rbl_hub + multi_dot([x113,self.ubar_SU2_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x125,self.ubar_SU2_rbl_hub_jcl_hub_bearing])),multi_dot([self.Mbar_SU2_rbl_upright_jcl_hub_bearing[:,0:1].T,x126,x125,x127]),multi_dot([self.Mbar_SU2_rbl_upright_jcl_hub_bearing[:,1:2].T,x126,x125,x127]),multi_dot([x128,x129,x116,x130]),multi_dot([x131,x129,x116,x130]),multi_dot([x128,x129,x132]),multi_dot([x131,x129,x132]),multi_dot([x133,x134,x119,x135]),multi_dot([x136,x134,x119,x135]),multi_dot([x133,x134,x137]),multi_dot([x136,x134,x137]),(x138 + x60 + multi_dot([x140,self.ubar_ST_rbs_coupler_jcs_rc_sph]) + -1.0*multi_dot([x62,self.ubar_ST_rbr_rocker_jcs_rc_sph])),(x138 + x64 + multi_dot([x140,self.ubar_ST_rbs_coupler_jcs_rc_uni]) + -1.0*multi_dot([x66,self.ubar_ST_rbl_rocker_jcs_rc_uni])),multi_dot([self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1].T,x141,x66,x142]),multi_dot([self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2].T,x141,x66,x142]),x5,(x3 + -1.0*'Pg_ground'),(x143 + (multi_dot([x1.T,x1]))**(1.0/2.0)),(x143 + (multi_dot([x10.T,x10]))**(1.0/2.0)),(x143 + (multi_dot([x15.T,x15]))**(1.0/2.0)),(x143 + (multi_dot([x20.T,x20]))**(1.0/2.0)),(x143 + (multi_dot([x70.T,x70]))**(1.0/2.0)),(x143 + (multi_dot([x74.T,x74]))**(1.0/2.0)),(x143 + (multi_dot([x25.T,x25]))**(1.0/2.0)),(x143 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x143 + (multi_dot([x77.T,x77]))**(1.0/2.0)),(x143 + (multi_dot([x80.T,x80]))**(1.0/2.0)),(x143 + (multi_dot([x83.T,x83]))**(1.0/2.0)),(x143 + (multi_dot([x90.T,x90]))**(1.0/2.0)),(x143 + (multi_dot([x85.T,x85]))**(1.0/2.0)),(x143 + (multi_dot([x92.T,x92]))**(1.0/2.0)),(x143 + (multi_dot([x39.T,x39]))**(1.0/2.0)),(x143 + (multi_dot([x43.T,x43]))**(1.0/2.0)),(x143 + (multi_dot([x47.T,x47]))**(1.0/2.0)),(x143 + (multi_dot([x51.T,x51]))**(1.0/2.0)),(x143 + (multi_dot([x108.T,x108]))**(1.0/2.0)),(x143 + (multi_dot([x112.T,x112]))**(1.0/2.0)),(x143 + (multi_dot([x55.T,x55]))**(1.0/2.0)),(x143 + (multi_dot([x58.T,x58]))**(1.0/2.0)),(x143 + (multi_dot([x115.T,x115]))**(1.0/2.0)),(x143 + (multi_dot([x118.T,x118]))**(1.0/2.0)),(x143 + (multi_dot([x33.T,x33]))**(1.0/2.0)),(x143 + (multi_dot([x36.T,x36]))**(1.0/2.0)),(x143 + (multi_dot([x120.T,x120]))**(1.0/2.0)),(x143 + (multi_dot([x124.T,x124]))**(1.0/2.0)),(x143 + (multi_dot([x139.T,x139]))**(1.0/2.0)),(x143 + (multi_dot([x61.T,x61]))**(1.0/2.0)),(x143 + (multi_dot([x65.T,x65]))**(1.0/2.0))]

    self.pos_level_rows_explicit = []
    self.pos_level_cols_explicit = []
    self.pos_level_data_explicit = []

    sparse_assembler(self.pos_level_data_blocks, self.pos_level_rows_blocks, self.pos_level_cols_blocks,
                     self.pos_level_data_explicit, self.pos_level_rows_explicit, self.pos_level_cols_explicit)

    self.pos_rhs = sc.sparse.coo_matrix(
    (self.pos_level_data_explicit,
    (self.pos_level_rows_explicit,self.pos_level_cols_explicit)),
    (28,1))



    
def eval_vel_eq(self):
    F = self.F
    t = self.t

    x0 = np.zeros((3,1),dtype=np.float64)
    x1 = np.zeros((1,1),dtype=np.float64)

    self.vel_level_data_blocks = [x0,x1,x1,x0,x1,x1,x0,x1,x1,x0,x1,x1,x0,x1,x0,x1,x0,x1,x0,x1,x0,x1,x1,x0,x1,x1,x0,x1,x1,x0,x1,x1,x0,x1,x0,x1,x0,x1,x1,x0,x1,x1,x0,x0,x0,x0,x1,x0,x0,x1,x0,x0,x1,x1,x0,x0,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x0,x1,x0,x1,x0,x0,x0,x0,x1,x0,x0,x1,x0,x0,x1,x1,x0,x0,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x0,x0,x1,x1,x0,np.zeros((4,1),dtype=np.float64),x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1,x1]

    self.vel_level_rows_explicit = []
    self.vel_level_cols_explicit = []
    self.vel_level_data_explicit = []

    sparse_assembler(self.vel_level_data_blocks, self.vel_level_rows_blocks, self.vel_level_cols_blocks,
                     self.vel_level_data_explicit, self.vel_level_rows_explicit, self.vel_level_cols_explicit)

    self.vel_rhs = sc.sparse.coo_matrix(
    (self.vel_level_data_explicit,
    (self.vel_level_rows_explicit,self.vel_level_cols_explicit)),
    (28,1))



    
def eval_acc_eq(self):
    F = self.F
    t = self.t

    a0 = self.Pd_ground
    a1 = multi_dot([B(a0,self.ubar_ground_jcr_uca_chassis),a0])
    a2 = self.Pd_SU1_rbr_uca
    a3 = self.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
    a4 = a3.T
    a5 = self.P_SU1_rbr_uca
    a6 = A(a5).T
    a7 = self.Mbar_ground_jcr_uca_chassis[:,0:1]
    a8 = self.P_ground
    a9 = A(a8).T
    a10 = B(a2,a3)
    a11 = a0.T
    a12 = B(a5,a3)
    a13 = self.Mbar_ground_jcr_uca_chassis[:,1:2]
    a14 = multi_dot([B(a0,self.ubar_ground_jcl_uca_chassis),a0])
    a15 = self.Pd_SU1_rbl_uca
    a16 = self.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
    a17 = a16.T
    a18 = self.P_SU1_rbl_uca
    a19 = A(a18).T
    a20 = self.Mbar_ground_jcl_uca_chassis[:,0:1]
    a21 = B(a15,a16)
    a22 = B(a18,a16)
    a23 = self.Mbar_ground_jcl_uca_chassis[:,1:2]
    a24 = multi_dot([B(a0,self.ubar_ground_jcr_lca_chassis),a0])
    a25 = self.Pd_SU1_rbr_lca
    a26 = self.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
    a27 = a26.T
    a28 = self.P_SU1_rbr_lca
    a29 = A(a28).T
    a30 = self.Mbar_ground_jcr_lca_chassis[:,0:1]
    a31 = B(a25,a26)
    a32 = B(a28,a26)
    a33 = self.Mbar_ground_jcr_lca_chassis[:,1:2]
    a34 = multi_dot([B(a0,self.ubar_ground_jcl_lca_chassis),a0])
    a35 = self.Pd_SU1_rbl_lca
    a36 = self.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
    a37 = a36.T
    a38 = self.P_SU1_rbl_lca
    a39 = A(a38).T
    a40 = self.Mbar_ground_jcl_lca_chassis[:,0:1]
    a41 = B(a35,a36)
    a42 = B(a38,a36)
    a43 = self.Mbar_ground_jcl_lca_chassis[:,1:2]
    a44 = multi_dot([B(a0,self.ubar_ground_jcr_strut_chassis),a0])
    a45 = self.Pd_SU1_rbr_upper_strut
    a46 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
    a47 = self.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]
    a48 = self.P_SU1_rbr_upper_strut
    a49 = A(a48).T
    a50 = multi_dot([B(a0,self.ubar_ground_jcl_strut_chassis),a0])
    a51 = self.Pd_SU1_rbl_upper_strut
    a52 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
    a53 = self.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]
    a54 = self.P_SU1_rbl_upper_strut
    a55 = A(a54).T
    a56 = self.Pd_SU2_rbr_tie_rod
    a57 = self.Mbar_ground_jcr_tie_steering[:,0:1]
    a58 = self.Mbar_SU2_rbr_tie_rod_jcr_tie_steering[:,0:1]
    a59 = self.P_SU2_rbr_tie_rod
    a60 = self.Pd_SU2_rbl_tie_rod
    a61 = self.Mbar_ground_jcl_tie_steering[:,0:1]
    a62 = self.Mbar_SU2_rbl_tie_rod_jcl_tie_steering[:,0:1]
    a63 = self.P_SU2_rbl_tie_rod
    a64 = self.Pd_SU2_rbr_uca
    a65 = self.Mbar_SU2_rbr_uca_jcr_uca_chassis[:,2:3]
    a66 = a65.T
    a67 = self.P_SU2_rbr_uca
    a68 = A(a67).T
    a69 = self.Mbar_ground_jcr_uca_chassis[:,0:1]
    a70 = B(a64,a65)
    a71 = B(a67,a65)
    a72 = self.Mbar_ground_jcr_uca_chassis[:,1:2]
    a73 = self.Pd_SU2_rbl_uca
    a74 = self.Mbar_ground_jcl_uca_chassis[:,0:1]
    a75 = self.Mbar_SU2_rbl_uca_jcl_uca_chassis[:,2:3]
    a76 = B(a73,a75)
    a77 = a75.T
    a78 = self.P_SU2_rbl_uca
    a79 = A(a78).T
    a80 = B(a78,a75)
    a81 = self.Mbar_ground_jcl_uca_chassis[:,1:2]
    a82 = self.Pd_SU2_rbr_lca
    a83 = self.Mbar_SU2_rbr_lca_jcr_lca_chassis[:,2:3]
    a84 = a83.T
    a85 = self.P_SU2_rbr_lca
    a86 = A(a85).T
    a87 = self.Mbar_ground_jcr_lca_chassis[:,0:1]
    a88 = B(a82,a83)
    a89 = B(a85,a83)
    a90 = self.Mbar_ground_jcr_lca_chassis[:,1:2]
    a91 = self.Pd_SU2_rbl_lca
    a92 = self.Mbar_SU2_rbl_lca_jcl_lca_chassis[:,2:3]
    a93 = a92.T
    a94 = self.P_SU2_rbl_lca
    a95 = A(a94).T
    a96 = self.Mbar_ground_jcl_lca_chassis[:,0:1]
    a97 = B(a91,a92)
    a98 = B(a94,a92)
    a99 = self.Mbar_ground_jcl_lca_chassis[:,1:2]
    a100 = self.Pd_SU2_rbr_upper_strut
    a101 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
    a102 = self.Mbar_SU2_rbr_upper_strut_jcr_strut_chassis[:,0:1]
    a103 = self.P_SU2_rbr_upper_strut
    a104 = A(a103).T
    a105 = self.Pd_SU2_rbl_upper_strut
    a106 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
    a107 = self.Mbar_SU2_rbl_upper_strut_jcl_strut_chassis[:,0:1]
    a108 = self.P_SU2_rbl_upper_strut
    a109 = A(a108).T
    a110 = self.Pd_ST_rbr_rocker
    a111 = self.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
    a112 = a111.T
    a113 = self.P_ST_rbr_rocker
    a114 = A(a113).T
    a115 = self.Mbar_ground_jcr_rocker_ch[:,0:1]
    a116 = B(a110,a111)
    a117 = B(a113,a111)
    a118 = self.Mbar_ground_jcr_rocker_ch[:,1:2]
    a119 = self.Pd_ST_rbl_rocker
    a120 = self.Mbar_ground_jcl_rocker_ch[:,0:1]
    a121 = self.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
    a122 = B(a119,a121)
    a123 = a121.T
    a124 = self.P_ST_rbl_rocker
    a125 = A(a124).T
    a126 = B(a124,a121)
    a127 = self.Mbar_ground_jcl_rocker_ch[:,1:2]
    a128 = self.Pd_SU1_rbr_upright
    a129 = self.Pd_SU1_rbl_upright
    a130 = self.Pd_SU1_rbr_lower_strut
    a131 = self.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
    a132 = self.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
    a133 = self.P_SU1_rbr_lower_strut
    a134 = A(a133).T
    a135 = a25.T
    a136 = self.Pd_SU1_rbl_lower_strut
    a137 = self.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
    a138 = self.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
    a139 = self.P_SU1_rbl_lower_strut
    a140 = A(a139).T
    a141 = a35.T
    a142 = self.Pd_SU1_rbr_tie_rod
    a143 = self.Pd_SU1_rbr_hub
    a144 = self.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
    a145 = self.P_SU1_rbr_upright
    a146 = A(a145).T
    a147 = self.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
    a148 = B(a143,a147)
    a149 = a147.T
    a150 = self.P_SU1_rbr_hub
    a151 = A(a150).T
    a152 = a128.T
    a153 = B(a150,a147)
    a154 = self.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
    a155 = self.Pd_SU1_rbl_tie_rod
    a156 = self.Pd_SU1_rbl_hub
    a157 = self.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
    a158 = a157.T
    a159 = self.P_SU1_rbl_hub
    a160 = A(a159).T
    a161 = self.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
    a162 = self.P_SU1_rbl_upright
    a163 = A(a162).T
    a164 = B(a156,a157)
    a165 = a129.T
    a166 = B(a159,a157)
    a167 = self.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
    a168 = self.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
    a169 = a168.T
    a170 = self.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
    a171 = B(a130,a170)
    a172 = a170.T
    a173 = B(a45,a168)
    a174 = a45.T
    a175 = B(a48,a168).T
    a176 = B(a133,a170)
    a177 = self.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
    a178 = a177.T
    a179 = B(a45,a177)
    a180 = B(a48,a177).T
    a181 = self.ubar_SU1_rbr_lower_strut_jcr_strut
    a182 = self.ubar_SU1_rbr_upper_strut_jcr_strut
    a183 = (multi_dot([B(a130,a181),a130]) + -1.0*multi_dot([B(a45,a182),a45]))
    a184 = (self.Rd_SU1_rbr_upper_strut + -1.0*self.Rd_SU1_rbr_lower_strut + multi_dot([B(a133,a181),a130]) + multi_dot([B(a48,a182),a45]))
    a185 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([a182.T,a49]) + -1.0*multi_dot([a181.T,a134]))
    a186 = self.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
    a187 = a186.T
    a188 = self.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
    a189 = B(a136,a188)
    a190 = a188.T
    a191 = B(a51,a186)
    a192 = a51.T
    a193 = B(a54,a186).T
    a194 = B(a139,a188)
    a195 = self.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
    a196 = a195.T
    a197 = B(a51,a195)
    a198 = B(a54,a195).T
    a199 = self.ubar_SU1_rbl_lower_strut_jcl_strut
    a200 = self.ubar_SU1_rbl_upper_strut_jcl_strut
    a201 = (multi_dot([B(a136,a199),a136]) + -1.0*multi_dot([B(a51,a200),a51]))
    a202 = (self.Rd_SU1_rbl_upper_strut + -1.0*self.Rd_SU1_rbl_lower_strut + multi_dot([B(a139,a199),a136]) + multi_dot([B(a54,a200),a51]))
    a203 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([a200.T,a55]) + -1.0*multi_dot([a199.T,a140]))
    a204 = self.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]
    a205 = self.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]
    a206 = self.P_SU1_rbr_tie_rod
    a207 = a142.T
    a208 = self.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]
    a209 = self.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]
    a210 = self.P_SU1_rbl_tie_rod
    a211 = a155.T
    a212 = self.Pd_SU2_rbr_upright
    a213 = self.Pd_SU2_rbl_upright
    a214 = self.Pd_SU2_rbr_lower_strut
    a215 = self.Mbar_SU2_rbr_lca_jcr_strut_lca[:,0:1]
    a216 = self.Mbar_SU2_rbr_lower_strut_jcr_strut_lca[:,0:1]
    a217 = self.P_SU2_rbr_lower_strut
    a218 = A(a217).T
    a219 = a82.T
    a220 = self.Pd_SU2_rbl_lower_strut
    a221 = self.Mbar_SU2_rbl_lca_jcl_strut_lca[:,0:1]
    a222 = self.Mbar_SU2_rbl_lower_strut_jcl_strut_lca[:,0:1]
    a223 = self.P_SU2_rbl_lower_strut
    a224 = A(a223).T
    a225 = a91.T
    a226 = self.Pd_SU2_rbr_hub
    a227 = self.Mbar_SU2_rbr_upright_jcr_hub_bearing[:,0:1]
    a228 = self.P_SU2_rbr_upright
    a229 = A(a228).T
    a230 = self.Mbar_SU2_rbr_hub_jcr_hub_bearing[:,2:3]
    a231 = B(a226,a230)
    a232 = a230.T
    a233 = self.P_SU2_rbr_hub
    a234 = A(a233).T
    a235 = a212.T
    a236 = B(a233,a230)
    a237 = self.Mbar_SU2_rbr_upright_jcr_hub_bearing[:,1:2]
    a238 = self.Pd_SU2_rbl_hub
    a239 = self.Mbar_SU2_rbl_hub_jcl_hub_bearing[:,2:3]
    a240 = a239.T
    a241 = self.P_SU2_rbl_hub
    a242 = A(a241).T
    a243 = self.Mbar_SU2_rbl_upright_jcl_hub_bearing[:,0:1]
    a244 = self.P_SU2_rbl_upright
    a245 = A(a244).T
    a246 = B(a238,a239)
    a247 = a213.T
    a248 = B(a241,a239)
    a249 = self.Mbar_SU2_rbl_upright_jcl_hub_bearing[:,1:2]
    a250 = self.Mbar_SU2_rbr_lower_strut_jcr_strut[:,2:3]
    a251 = a250.T
    a252 = self.Mbar_SU2_rbr_upper_strut_jcr_strut[:,0:1]
    a253 = B(a100,a252)
    a254 = a252.T
    a255 = B(a214,a250)
    a256 = a100.T
    a257 = B(a103,a252).T
    a258 = B(a217,a250)
    a259 = self.Mbar_SU2_rbr_upper_strut_jcr_strut[:,1:2]
    a260 = B(a100,a259)
    a261 = a259.T
    a262 = B(a103,a259).T
    a263 = self.ubar_SU2_rbr_lower_strut_jcr_strut
    a264 = self.ubar_SU2_rbr_upper_strut_jcr_strut
    a265 = (multi_dot([B(a214,a263),a214]) + -1.0*multi_dot([B(a100,a264),a100]))
    a266 = (self.Rd_SU2_rbr_upper_strut + -1.0*self.Rd_SU2_rbr_lower_strut + multi_dot([B(a217,a263),a214]) + multi_dot([B(a103,a264),a100]))
    a267 = (self.R_SU2_rbr_upper_strut.T + -1.0*self.R_SU2_rbr_lower_strut.T + multi_dot([a264.T,a104]) + -1.0*multi_dot([a263.T,a218]))
    a268 = self.Mbar_SU2_rbl_upper_strut_jcl_strut[:,0:1]
    a269 = a268.T
    a270 = self.Mbar_SU2_rbl_lower_strut_jcl_strut[:,2:3]
    a271 = B(a220,a270)
    a272 = a270.T
    a273 = B(a105,a268)
    a274 = a105.T
    a275 = B(a108,a268).T
    a276 = B(a223,a270)
    a277 = self.Mbar_SU2_rbl_upper_strut_jcl_strut[:,1:2]
    a278 = a277.T
    a279 = B(a105,a277)
    a280 = B(a108,a277).T
    a281 = self.ubar_SU2_rbl_lower_strut_jcl_strut
    a282 = self.ubar_SU2_rbl_upper_strut_jcl_strut
    a283 = (multi_dot([B(a220,a281),a220]) + -1.0*multi_dot([B(a105,a282),a105]))
    a284 = (self.Rd_SU2_rbl_upper_strut + -1.0*self.Rd_SU2_rbl_lower_strut + multi_dot([B(a223,a281),a220]) + multi_dot([B(a108,a282),a105]))
    a285 = (self.R_SU2_rbl_upper_strut.T + -1.0*self.R_SU2_rbl_lower_strut.T + multi_dot([a282.T,a109]) + -1.0*multi_dot([a281.T,a224]))
    a286 = self.Pd_ST_rbs_coupler
    a287 = self.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
    a288 = a287.T
    a289 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1]
    a290 = self.P_ST_rbs_coupler
    a291 = A(a290).T
    a292 = B(a119,a287)
    a293 = a286.T
    a294 = B(a124,a287)
    a295 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2]

    self.acc_level_data_blocks = [(a1 + -1.0*multi_dot([B(a2,self.ubar_SU1_rbr_uca_jcr_uca_chassis),a2])),(multi_dot([a4,a6,B(a0,a7),a0]) + multi_dot([a7.T,a9,a10,a2]) + 2.0*multi_dot([a11,B(a8,a7).T,a12,a2])),(multi_dot([a4,a6,B(a0,a13),a0]) + multi_dot([a13.T,a9,a10,a2]) + 2.0*multi_dot([a11,B(a8,a13).T,a12,a2])),(a14 + -1.0*multi_dot([B(a15,self.ubar_SU1_rbl_uca_jcl_uca_chassis),a15])),(multi_dot([a17,a19,B(a0,a20),a0]) + multi_dot([a20.T,a9,a21,a15]) + 2.0*multi_dot([a11,B(a8,a20).T,a22,a15])),(multi_dot([a17,a19,B(a0,a23),a0]) + multi_dot([a23.T,a9,a21,a15]) + 2.0*multi_dot([a11,B(a8,a23).T,a22,a15])),(a24 + -1.0*multi_dot([B(a25,self.ubar_SU1_rbr_lca_jcr_lca_chassis),a25])),(multi_dot([a27,a29,B(a0,a30),a0]) + multi_dot([a30.T,a9,a31,a25]) + 2.0*multi_dot([a11,B(a8,a30).T,a32,a25])),(multi_dot([a27,a29,B(a0,a33),a0]) + multi_dot([a33.T,a9,a31,a25]) + 2.0*multi_dot([a11,B(a8,a33).T,a32,a25])),(a34 + -1.0*multi_dot([B(a35,self.ubar_SU1_rbl_lca_jcl_lca_chassis),a35])),(multi_dot([a37,a39,B(a0,a40),a0]) + multi_dot([a40.T,a9,a41,a35]) + 2.0*multi_dot([a11,B(a8,a40).T,a42,a35])),(multi_dot([a37,a39,B(a0,a43),a0]) + multi_dot([a43.T,a9,a41,a35]) + 2.0*multi_dot([a11,B(a8,a43).T,a42,a35])),(a44 + -1.0*multi_dot([B(a45,self.ubar_SU1_rbr_upper_strut_jcr_strut_chassis),a45])),(multi_dot([a46.T,a9,B(a45,a47),a45]) + multi_dot([a47.T,a49,B(a0,a46),a0]) + 2.0*multi_dot([a11,B(a8,a46).T,B(a48,a47),a45])),(a50 + -1.0*multi_dot([B(a51,self.ubar_SU1_rbl_upper_strut_jcl_strut_chassis),a51])),(multi_dot([a52.T,a9,B(a51,a53),a51]) + multi_dot([a53.T,a55,B(a0,a52),a0]) + 2.0*multi_dot([a11,B(a8,a52).T,B(a54,a53),a51])),(multi_dot([B(a0,self.ubar_ground_jcr_tie_steering),a0]) + -1.0*multi_dot([B(a56,self.ubar_SU2_rbr_tie_rod_jcr_tie_steering),a56])),(multi_dot([a57.T,a9,B(a56,a58),a56]) + multi_dot([a58.T,A(a59).T,B(a0,a57),a0]) + 2.0*multi_dot([a11,B(a8,a57).T,B(a59,a58),a56])),(multi_dot([B(a0,self.ubar_ground_jcl_tie_steering),a0]) + -1.0*multi_dot([B(a60,self.ubar_SU2_rbl_tie_rod_jcl_tie_steering),a60])),(multi_dot([a61.T,a9,B(a60,a62),a60]) + multi_dot([a62.T,A(a63).T,B(a0,a61),a0]) + 2.0*multi_dot([a11,B(a8,a61).T,B(a63,a62),a60])),(a1 + -1.0*multi_dot([B(a64,self.ubar_SU2_rbr_uca_jcr_uca_chassis),a64])),(multi_dot([a66,a68,B(a0,a69),a0]) + multi_dot([a69.T,a9,a70,a64]) + 2.0*multi_dot([a11,B(a8,a69).T,a71,a64])),(multi_dot([a66,a68,B(a0,a72),a0]) + multi_dot([a72.T,a9,a70,a64]) + 2.0*multi_dot([a11,B(a8,a72).T,a71,a64])),(a14 + -1.0*multi_dot([B(a73,self.ubar_SU2_rbl_uca_jcl_uca_chassis),a73])),(multi_dot([a74.T,a9,a76,a73]) + multi_dot([a77,a79,B(a0,a74),a0]) + 2.0*multi_dot([a11,B(a8,a74).T,a80,a73])),(multi_dot([a81.T,a9,a76,a73]) + multi_dot([a77,a79,B(a0,a81),a0]) + 2.0*multi_dot([a11,B(a8,a81).T,a80,a73])),(a24 + -1.0*multi_dot([B(a82,self.ubar_SU2_rbr_lca_jcr_lca_chassis),a82])),(multi_dot([a84,a86,B(a0,a87),a0]) + multi_dot([a87.T,a9,a88,a82]) + 2.0*multi_dot([a11,B(a8,a87).T,a89,a82])),(multi_dot([a84,a86,B(a0,a90),a0]) + multi_dot([a90.T,a9,a88,a82]) + 2.0*multi_dot([a11,B(a8,a90).T,a89,a82])),(a34 + -1.0*multi_dot([B(a91,self.ubar_SU2_rbl_lca_jcl_lca_chassis),a91])),(multi_dot([a93,a95,B(a0,a96),a0]) + multi_dot([a96.T,a9,a97,a91]) + 2.0*multi_dot([a11,B(a8,a96).T,a98,a91])),(multi_dot([a93,a95,B(a0,a99),a0]) + multi_dot([a99.T,a9,a97,a91]) + 2.0*multi_dot([a11,B(a8,a99).T,a98,a91])),(a44 + -1.0*multi_dot([B(a100,self.ubar_SU2_rbr_upper_strut_jcr_strut_chassis),a100])),(multi_dot([a101.T,a9,B(a100,a102),a100]) + multi_dot([a102.T,a104,B(a0,a101),a0]) + 2.0*multi_dot([a11,B(a8,a101).T,B(a103,a102),a100])),(a50 + -1.0*multi_dot([B(a105,self.ubar_SU2_rbl_upper_strut_jcl_strut_chassis),a105])),(multi_dot([a106.T,a9,B(a105,a107),a105]) + multi_dot([a107.T,a109,B(a0,a106),a0]) + 2.0*multi_dot([a11,B(a8,a106).T,B(a108,a107),a105])),(multi_dot([B(a0,self.ubar_ground_jcr_rocker_ch),a0]) + -1.0*multi_dot([B(a110,self.ubar_ST_rbr_rocker_jcr_rocker_ch),a110])),(multi_dot([a112,a114,B(a0,a115),a0]) + multi_dot([a115.T,a9,a116,a110]) + 2.0*multi_dot([a11,B(a8,a115).T,a117,a110])),(multi_dot([a112,a114,B(a0,a118),a0]) + multi_dot([a118.T,a9,a116,a110]) + 2.0*multi_dot([a11,B(a8,a118).T,a117,a110])),(multi_dot([B(a0,self.ubar_ground_jcl_rocker_ch),a0]) + -1.0*multi_dot([B(a119,self.ubar_ST_rbl_rocker_jcl_rocker_ch),a119])),(multi_dot([a120.T,a9,a122,a119]) + multi_dot([a123,a125,B(a0,a120),a0]) + 2.0*multi_dot([a11,B(a8,a120).T,a126,a119])),(multi_dot([a127.T,a9,a122,a119]) + multi_dot([a123,a125,B(a0,a127),a0]) + 2.0*multi_dot([a11,B(a8,a127).T,a126,a119])),(multi_dot([B(a2,self.ubar_SU1_rbr_uca_jcr_uca_upright),a2]) + -1.0*multi_dot([B(a128,self.ubar_SU1_rbr_upright_jcr_uca_upright),a128])),(multi_dot([B(a15,self.ubar_SU1_rbl_uca_jcl_uca_upright),a15]) + -1.0*multi_dot([B(a129,self.ubar_SU1_rbl_upright_jcl_uca_upright),a129])),(multi_dot([B(a25,self.ubar_SU1_rbr_lca_jcr_lca_upright),a25]) + -1.0*multi_dot([B(a128,self.ubar_SU1_rbr_upright_jcr_lca_upright),a128])),(multi_dot([B(a25,self.ubar_SU1_rbr_lca_jcr_strut_lca),a25]) + -1.0*multi_dot([B(a130,self.ubar_SU1_rbr_lower_strut_jcr_strut_lca),a130])),(multi_dot([a131.T,a29,B(a130,a132),a130]) + multi_dot([a132.T,a134,B(a25,a131),a25]) + 2.0*multi_dot([a135,B(a28,a131).T,B(a133,a132),a130])),(multi_dot([B(a35,self.ubar_SU1_rbl_lca_jcl_lca_upright),a35]) + -1.0*multi_dot([B(a129,self.ubar_SU1_rbl_upright_jcl_lca_upright),a129])),(multi_dot([B(a35,self.ubar_SU1_rbl_lca_jcl_strut_lca),a35]) + -1.0*multi_dot([B(a136,self.ubar_SU1_rbl_lower_strut_jcl_strut_lca),a136])),(multi_dot([a137.T,a39,B(a136,a138),a136]) + multi_dot([a138.T,a140,B(a35,a137),a35]) + 2.0*multi_dot([a141,B(a38,a137).T,B(a139,a138),a136])),(multi_dot([B(a128,self.ubar_SU1_rbr_upright_jcr_tie_upright),a128]) + -1.0*multi_dot([B(a142,self.ubar_SU1_rbr_tie_rod_jcr_tie_upright),a142])),(multi_dot([B(a128,self.ubar_SU1_rbr_upright_jcr_hub_bearing),a128]) + -1.0*multi_dot([B(a143,self.ubar_SU1_rbr_hub_jcr_hub_bearing),a143])),(multi_dot([a144.T,a146,a148,a143]) + multi_dot([a149,a151,B(a128,a144),a128]) + 2.0*multi_dot([a152,B(a145,a144).T,a153,a143])),(multi_dot([a154.T,a146,a148,a143]) + multi_dot([a149,a151,B(a128,a154),a128]) + 2.0*multi_dot([a152,B(a145,a154).T,a153,a143])),(multi_dot([B(a129,self.ubar_SU1_rbl_upright_jcl_tie_upright),a129]) + -1.0*multi_dot([B(a155,self.ubar_SU1_rbl_tie_rod_jcl_tie_upright),a155])),(multi_dot([B(a129,self.ubar_SU1_rbl_upright_jcl_hub_bearing),a129]) + -1.0*multi_dot([B(a156,self.ubar_SU1_rbl_hub_jcl_hub_bearing),a156])),(multi_dot([a158,a160,B(a129,a161),a129]) + multi_dot([a161.T,a163,a164,a156]) + 2.0*multi_dot([a165,B(a162,a161).T,a166,a156])),(multi_dot([a158,a160,B(a129,a167),a129]) + multi_dot([a167.T,a163,a164,a156]) + 2.0*multi_dot([a165,B(a162,a167).T,a166,a156])),(multi_dot([a169,a49,a171,a130]) + multi_dot([a172,a134,a173,a45]) + 2.0*multi_dot([a174,a175,a176,a130])),(multi_dot([a178,a49,a171,a130]) + multi_dot([a172,a134,a179,a45]) + 2.0*multi_dot([a174,a180,a176,a130])),(multi_dot([a169,a49,a183]) + 2.0*multi_dot([a174,a175,a184]) + multi_dot([a185,a173,a45])),(multi_dot([a178,a49,a183]) + 2.0*multi_dot([a174,a180,a184]) + multi_dot([a185,a179,a45])),(multi_dot([a187,a55,a189,a136]) + multi_dot([a190,a140,a191,a51]) + 2.0*multi_dot([a192,a193,a194,a136])),(multi_dot([a196,a55,a189,a136]) + multi_dot([a190,a140,a197,a51]) + 2.0*multi_dot([a192,a198,a194,a136])),(multi_dot([a187,a55,a201]) + 2.0*multi_dot([a192,a193,a202]) + multi_dot([a203,a191,a51])),(multi_dot([a196,a55,a201]) + 2.0*multi_dot([a192,a198,a202]) + multi_dot([a203,a197,a51])),(multi_dot([B(a142,self.ubar_SU1_rbr_tie_rod_jcr_tie_steering),a142]) + -1.0*multi_dot([B(a110,self.ubar_ST_rbr_rocker_jcr_tie_steering),a110])),(multi_dot([a204.T,a114,B(a142,a205),a142]) + multi_dot([a205.T,A(a206).T,B(a110,a204),a110]) + 2.0*multi_dot([a207,B(a206,a205).T,B(a113,a204),a110])),(multi_dot([B(a155,self.ubar_SU1_rbl_tie_rod_jcl_tie_steering),a155]) + -1.0*multi_dot([B(a119,self.ubar_ST_rbl_rocker_jcl_tie_steering),a119])),(multi_dot([a208.T,a125,B(a155,a209),a155]) + multi_dot([a209.T,A(a210).T,B(a119,a208),a119]) + 2.0*multi_dot([a211,B(a210,a209).T,B(a124,a208),a119])),(multi_dot([B(a64,self.ubar_SU2_rbr_uca_jcr_uca_upright),a64]) + -1.0*multi_dot([B(a212,self.ubar_SU2_rbr_upright_jcr_uca_upright),a212])),(multi_dot([B(a73,self.ubar_SU2_rbl_uca_jcl_uca_upright),a73]) + -1.0*multi_dot([B(a213,self.ubar_SU2_rbl_upright_jcl_uca_upright),a213])),(multi_dot([B(a82,self.ubar_SU2_rbr_lca_jcr_lca_upright),a82]) + -1.0*multi_dot([B(a212,self.ubar_SU2_rbr_upright_jcr_lca_upright),a212])),(multi_dot([B(a82,self.ubar_SU2_rbr_lca_jcr_strut_lca),a82]) + -1.0*multi_dot([B(a214,self.ubar_SU2_rbr_lower_strut_jcr_strut_lca),a214])),(multi_dot([a215.T,a86,B(a214,a216),a214]) + multi_dot([a216.T,a218,B(a82,a215),a82]) + 2.0*multi_dot([a219,B(a85,a215).T,B(a217,a216),a214])),(multi_dot([B(a91,self.ubar_SU2_rbl_lca_jcl_lca_upright),a91]) + -1.0*multi_dot([B(a213,self.ubar_SU2_rbl_upright_jcl_lca_upright),a213])),(multi_dot([B(a91,self.ubar_SU2_rbl_lca_jcl_strut_lca),a91]) + -1.0*multi_dot([B(a220,self.ubar_SU2_rbl_lower_strut_jcl_strut_lca),a220])),(multi_dot([a221.T,a95,B(a220,a222),a220]) + multi_dot([a222.T,a224,B(a91,a221),a91]) + 2.0*multi_dot([a225,B(a94,a221).T,B(a223,a222),a220])),(multi_dot([B(a212,self.ubar_SU2_rbr_upright_jcr_tie_upright),a212]) + -1.0*multi_dot([B(a56,self.ubar_SU2_rbr_tie_rod_jcr_tie_upright),a56])),(multi_dot([B(a212,self.ubar_SU2_rbr_upright_jcr_hub_bearing),a212]) + -1.0*multi_dot([B(a226,self.ubar_SU2_rbr_hub_jcr_hub_bearing),a226])),(multi_dot([a227.T,a229,a231,a226]) + multi_dot([a232,a234,B(a212,a227),a212]) + 2.0*multi_dot([a235,B(a228,a227).T,a236,a226])),(multi_dot([a237.T,a229,a231,a226]) + multi_dot([a232,a234,B(a212,a237),a212]) + 2.0*multi_dot([a235,B(a228,a237).T,a236,a226])),(multi_dot([B(a213,self.ubar_SU2_rbl_upright_jcl_tie_upright),a213]) + -1.0*multi_dot([B(a60,self.ubar_SU2_rbl_tie_rod_jcl_tie_upright),a60])),(multi_dot([B(a213,self.ubar_SU2_rbl_upright_jcl_hub_bearing),a213]) + -1.0*multi_dot([B(a238,self.ubar_SU2_rbl_hub_jcl_hub_bearing),a238])),(multi_dot([a240,a242,B(a213,a243),a213]) + multi_dot([a243.T,a245,a246,a238]) + 2.0*multi_dot([a247,B(a244,a243).T,a248,a238])),(multi_dot([a240,a242,B(a213,a249),a213]) + multi_dot([a249.T,a245,a246,a238]) + 2.0*multi_dot([a247,B(a244,a249).T,a248,a238])),(multi_dot([a251,a218,a253,a100]) + multi_dot([a254,a104,a255,a214]) + 2.0*multi_dot([a256,a257,a258,a214])),(multi_dot([a251,a218,a260,a100]) + multi_dot([a261,a104,a255,a214]) + 2.0*multi_dot([a256,a262,a258,a214])),(multi_dot([a254,a104,a265]) + 2.0*multi_dot([a256,a257,a266]) + multi_dot([a267,a253,a100])),(multi_dot([a261,a104,a265]) + 2.0*multi_dot([a256,a262,a266]) + multi_dot([a267,a260,a100])),(multi_dot([a269,a109,a271,a220]) + multi_dot([a272,a224,a273,a105]) + 2.0*multi_dot([a274,a275,a276,a220])),(multi_dot([a278,a109,a271,a220]) + multi_dot([a272,a224,a279,a105]) + 2.0*multi_dot([a274,a280,a276,a220])),(multi_dot([a269,a109,a283]) + 2.0*multi_dot([a274,a275,a284]) + multi_dot([a285,a273,a105])),(multi_dot([a278,a109,a283]) + 2.0*multi_dot([a274,a280,a284]) + multi_dot([a285,a279,a105])),(multi_dot([B(a286,self.ubar_ST_rbs_coupler_jcs_rc_sph),a286]) + -1.0*multi_dot([B(a110,self.ubar_ST_rbr_rocker_jcs_rc_sph),a110])),(multi_dot([B(a286,self.ubar_ST_rbs_coupler_jcs_rc_uni),a286]) + -1.0*multi_dot([B(a119,self.ubar_ST_rbl_rocker_jcs_rc_uni),a119])),(multi_dot([a288,a125,B(a286,a289),a286]) + multi_dot([a289.T,a291,a292,a119]) + 2.0*multi_dot([a293,B(a290,a289).T,a294,a119])),(multi_dot([a288,a125,B(a286,a295),a286]) + multi_dot([a295.T,a291,a292,a119]) + 2.0*multi_dot([a293,B(a290,a295).T,a294,a119])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a2.T,a2]))**(1.0/2.0),2.0*(multi_dot([a15.T,a15]))**(1.0/2.0),2.0*(multi_dot([a135,a25]))**(1.0/2.0),2.0*(multi_dot([a141,a35]))**(1.0/2.0),2.0*(multi_dot([a152,a128]))**(1.0/2.0),2.0*(multi_dot([a165,a129]))**(1.0/2.0),2.0*(multi_dot([a174,a45]))**(1.0/2.0),2.0*(multi_dot([a192,a51]))**(1.0/2.0),2.0*(multi_dot([a130.T,a130]))**(1.0/2.0),2.0*(multi_dot([a136.T,a136]))**(1.0/2.0),2.0*(multi_dot([a207,a142]))**(1.0/2.0),2.0*(multi_dot([a211,a155]))**(1.0/2.0),2.0*(multi_dot([a143.T,a143]))**(1.0/2.0),2.0*(multi_dot([a156.T,a156]))**(1.0/2.0),2.0*(multi_dot([a64.T,a64]))**(1.0/2.0),2.0*(multi_dot([a73.T,a73]))**(1.0/2.0),2.0*(multi_dot([a219,a82]))**(1.0/2.0),2.0*(multi_dot([a225,a91]))**(1.0/2.0),2.0*(multi_dot([a235,a212]))**(1.0/2.0),2.0*(multi_dot([a247,a213]))**(1.0/2.0),2.0*(multi_dot([a256,a100]))**(1.0/2.0),2.0*(multi_dot([a274,a105]))**(1.0/2.0),2.0*(multi_dot([a214.T,a214]))**(1.0/2.0),2.0*(multi_dot([a220.T,a220]))**(1.0/2.0),2.0*(multi_dot([a56.T,a56]))**(1.0/2.0),2.0*(multi_dot([a60.T,a60]))**(1.0/2.0),2.0*(multi_dot([a226.T,a226]))**(1.0/2.0),2.0*(multi_dot([a238.T,a238]))**(1.0/2.0),2.0*(multi_dot([a293,a286]))**(1.0/2.0),2.0*(multi_dot([a110.T,a110]))**(1.0/2.0),2.0*(multi_dot([a119.T,a119]))**(1.0/2.0)]

    self.acc_level_rows_explicit = []
    self.acc_level_cols_explicit = []
    self.acc_level_data_explicit = []

    sparse_assembler(self.acc_level_data_blocks, self.acc_level_rows_blocks, self.acc_level_cols_blocks,
                     self.acc_level_data_explicit, self.acc_level_rows_explicit, self.acc_level_cols_explicit)

    self.acc_rhs = sc.sparse.coo_matrix(
    (self.acc_level_data_explicit,
    (self.acc_level_rows_explicit,self.acc_level_cols_explicit)),
    (28,1))



    
def eval_jacobian(self):

    j0 = np.eye(3,dtype=np.float64)
    j1 = self.P_ground
    j2 = B(j1,self.ubar_ground_jcr_uca_chassis)
    j3 = np.zeros((1,3),dtype=np.float64)
    j4 = self.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
    j5 = j4.T
    j6 = self.P_SU1_rbr_uca
    j7 = A(j6).T
    j8 = self.Mbar_ground_jcr_uca_chassis[:,0:1]
    j9 = self.Mbar_ground_jcr_uca_chassis[:,1:2]
    j10 = -1.0*j0
    j11 = A(j1).T
    j12 = B(j6,j4)
    j13 = B(j1,self.ubar_ground_jcl_uca_chassis)
    j14 = self.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
    j15 = j14.T
    j16 = self.P_SU1_rbl_uca
    j17 = A(j16).T
    j18 = self.Mbar_ground_jcl_uca_chassis[:,0:1]
    j19 = self.Mbar_ground_jcl_uca_chassis[:,1:2]
    j20 = B(j16,j14)
    j21 = B(j1,self.ubar_ground_jcr_lca_chassis)
    j22 = self.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
    j23 = j22.T
    j24 = self.P_SU1_rbr_lca
    j25 = A(j24).T
    j26 = self.Mbar_ground_jcr_lca_chassis[:,0:1]
    j27 = self.Mbar_ground_jcr_lca_chassis[:,1:2]
    j28 = B(j24,j22)
    j29 = B(j1,self.ubar_ground_jcl_lca_chassis)
    j30 = self.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
    j31 = j30.T
    j32 = self.P_SU1_rbl_lca
    j33 = A(j32).T
    j34 = self.Mbar_ground_jcl_lca_chassis[:,0:1]
    j35 = self.Mbar_ground_jcl_lca_chassis[:,1:2]
    j36 = B(j32,j30)
    j37 = B(j1,self.ubar_ground_jcr_strut_chassis)
    j38 = self.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]
    j39 = self.P_SU1_rbr_upper_strut
    j40 = A(j39).T
    j41 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
    j42 = B(j1,self.ubar_ground_jcl_strut_chassis)
    j43 = self.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]
    j44 = self.P_SU1_rbl_upper_strut
    j45 = A(j44).T
    j46 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
    j47 = self.Mbar_SU2_rbr_tie_rod_jcr_tie_steering[:,0:1]
    j48 = self.P_SU2_rbr_tie_rod
    j49 = self.Mbar_ground_jcr_tie_steering[:,0:1]
    j50 = self.Mbar_SU2_rbl_tie_rod_jcl_tie_steering[:,0:1]
    j51 = self.P_SU2_rbl_tie_rod
    j52 = self.Mbar_ground_jcl_tie_steering[:,0:1]
    j53 = self.Mbar_SU2_rbr_uca_jcr_uca_chassis[:,2:3]
    j54 = j53.T
    j55 = self.P_SU2_rbr_uca
    j56 = A(j55).T
    j57 = self.Mbar_ground_jcr_uca_chassis[:,0:1]
    j58 = self.Mbar_ground_jcr_uca_chassis[:,1:2]
    j59 = B(j55,j53)
    j60 = self.Mbar_SU2_rbl_uca_jcl_uca_chassis[:,2:3]
    j61 = j60.T
    j62 = self.P_SU2_rbl_uca
    j63 = A(j62).T
    j64 = self.Mbar_ground_jcl_uca_chassis[:,0:1]
    j65 = self.Mbar_ground_jcl_uca_chassis[:,1:2]
    j66 = B(j62,j60)
    j67 = self.Mbar_SU2_rbr_lca_jcr_lca_chassis[:,2:3]
    j68 = j67.T
    j69 = self.P_SU2_rbr_lca
    j70 = A(j69).T
    j71 = self.Mbar_ground_jcr_lca_chassis[:,0:1]
    j72 = self.Mbar_ground_jcr_lca_chassis[:,1:2]
    j73 = B(j69,j67)
    j74 = self.Mbar_SU2_rbl_lca_jcl_lca_chassis[:,2:3]
    j75 = j74.T
    j76 = self.P_SU2_rbl_lca
    j77 = A(j76).T
    j78 = self.Mbar_ground_jcl_lca_chassis[:,0:1]
    j79 = self.Mbar_ground_jcl_lca_chassis[:,1:2]
    j80 = B(j76,j74)
    j81 = self.Mbar_SU2_rbr_upper_strut_jcr_strut_chassis[:,0:1]
    j82 = self.P_SU2_rbr_upper_strut
    j83 = A(j82).T
    j84 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
    j85 = self.Mbar_SU2_rbl_upper_strut_jcl_strut_chassis[:,0:1]
    j86 = self.P_SU2_rbl_upper_strut
    j87 = A(j86).T
    j88 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
    j89 = self.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
    j90 = j89.T
    j91 = self.P_ST_rbr_rocker
    j92 = A(j91).T
    j93 = self.Mbar_ground_jcr_rocker_ch[:,0:1]
    j94 = self.Mbar_ground_jcr_rocker_ch[:,1:2]
    j95 = B(j91,j89)
    j96 = self.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
    j97 = j96.T
    j98 = self.P_ST_rbl_rocker
    j99 = A(j98).T
    j100 = self.Mbar_ground_jcl_rocker_ch[:,0:1]
    j101 = self.Mbar_ground_jcl_rocker_ch[:,1:2]
    j102 = B(j98,j96)
    j103 = self.P_SU1_rbr_upright
    j104 = self.P_SU1_rbl_upright
    j105 = self.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
    j106 = self.P_SU1_rbr_lower_strut
    j107 = A(j106).T
    j108 = self.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
    j109 = self.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
    j110 = self.P_SU1_rbl_lower_strut
    j111 = A(j110).T
    j112 = self.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
    j113 = self.P_SU1_rbr_tie_rod
    j114 = self.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
    j115 = j114.T
    j116 = self.P_SU1_rbr_hub
    j117 = A(j116).T
    j118 = self.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
    j119 = self.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
    j120 = A(j103).T
    j121 = B(j116,j114)
    j122 = self.P_SU1_rbl_tie_rod
    j123 = self.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
    j124 = j123.T
    j125 = self.P_SU1_rbl_hub
    j126 = A(j125).T
    j127 = self.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
    j128 = self.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
    j129 = A(j104).T
    j130 = B(j125,j123)
    j131 = self.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
    j132 = j131.T
    j133 = self.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
    j134 = B(j39,j133)
    j135 = self.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
    j136 = B(j39,j135)
    j137 = j133.T
    j138 = multi_dot([j137,j40])
    j139 = self.ubar_SU1_rbr_upper_strut_jcr_strut
    j140 = B(j39,j139)
    j141 = self.ubar_SU1_rbr_lower_strut_jcr_strut
    j142 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([j139.T,j40]) + -1.0*multi_dot([j141.T,j107]))
    j143 = j135.T
    j144 = multi_dot([j143,j40])
    j145 = B(j106,j131)
    j146 = B(j106,j141)
    j147 = self.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
    j148 = j147.T
    j149 = self.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
    j150 = B(j44,j149)
    j151 = self.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
    j152 = B(j44,j151)
    j153 = j149.T
    j154 = multi_dot([j153,j45])
    j155 = self.ubar_SU1_rbl_upper_strut_jcl_strut
    j156 = B(j44,j155)
    j157 = self.ubar_SU1_rbl_lower_strut_jcl_strut
    j158 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([j155.T,j45]) + -1.0*multi_dot([j157.T,j111]))
    j159 = j151.T
    j160 = multi_dot([j159,j45])
    j161 = B(j110,j147)
    j162 = B(j110,j157)
    j163 = self.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]
    j164 = self.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]
    j165 = self.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]
    j166 = self.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]
    j167 = self.P_SU2_rbr_upright
    j168 = self.P_SU2_rbl_upright
    j169 = self.Mbar_SU2_rbr_lower_strut_jcr_strut_lca[:,0:1]
    j170 = self.P_SU2_rbr_lower_strut
    j171 = A(j170).T
    j172 = self.Mbar_SU2_rbr_lca_jcr_strut_lca[:,0:1]
    j173 = self.Mbar_SU2_rbl_lower_strut_jcl_strut_lca[:,0:1]
    j174 = self.P_SU2_rbl_lower_strut
    j175 = A(j174).T
    j176 = self.Mbar_SU2_rbl_lca_jcl_strut_lca[:,0:1]
    j177 = self.Mbar_SU2_rbr_hub_jcr_hub_bearing[:,2:3]
    j178 = j177.T
    j179 = self.P_SU2_rbr_hub
    j180 = A(j179).T
    j181 = self.Mbar_SU2_rbr_upright_jcr_hub_bearing[:,0:1]
    j182 = self.Mbar_SU2_rbr_upright_jcr_hub_bearing[:,1:2]
    j183 = A(j167).T
    j184 = B(j179,j177)
    j185 = self.Mbar_SU2_rbl_hub_jcl_hub_bearing[:,2:3]
    j186 = j185.T
    j187 = self.P_SU2_rbl_hub
    j188 = A(j187).T
    j189 = self.Mbar_SU2_rbl_upright_jcl_hub_bearing[:,0:1]
    j190 = self.Mbar_SU2_rbl_upright_jcl_hub_bearing[:,1:2]
    j191 = A(j168).T
    j192 = B(j187,j185)
    j193 = self.Mbar_SU2_rbr_lower_strut_jcr_strut[:,2:3]
    j194 = j193.T
    j195 = self.Mbar_SU2_rbr_upper_strut_jcr_strut[:,0:1]
    j196 = B(j82,j195)
    j197 = self.Mbar_SU2_rbr_upper_strut_jcr_strut[:,1:2]
    j198 = B(j82,j197)
    j199 = j195.T
    j200 = multi_dot([j199,j83])
    j201 = self.ubar_SU2_rbr_upper_strut_jcr_strut
    j202 = B(j82,j201)
    j203 = self.ubar_SU2_rbr_lower_strut_jcr_strut
    j204 = (self.R_SU2_rbr_upper_strut.T + -1.0*self.R_SU2_rbr_lower_strut.T + multi_dot([j201.T,j83]) + -1.0*multi_dot([j203.T,j171]))
    j205 = j197.T
    j206 = multi_dot([j205,j83])
    j207 = B(j170,j193)
    j208 = B(j170,j203)
    j209 = self.Mbar_SU2_rbl_lower_strut_jcl_strut[:,2:3]
    j210 = j209.T
    j211 = self.Mbar_SU2_rbl_upper_strut_jcl_strut[:,0:1]
    j212 = B(j86,j211)
    j213 = self.Mbar_SU2_rbl_upper_strut_jcl_strut[:,1:2]
    j214 = B(j86,j213)
    j215 = j211.T
    j216 = multi_dot([j215,j87])
    j217 = self.ubar_SU2_rbl_upper_strut_jcl_strut
    j218 = B(j86,j217)
    j219 = self.ubar_SU2_rbl_lower_strut_jcl_strut
    j220 = (self.R_SU2_rbl_upper_strut.T + -1.0*self.R_SU2_rbl_lower_strut.T + multi_dot([j217.T,j87]) + -1.0*multi_dot([j219.T,j175]))
    j221 = j213.T
    j222 = multi_dot([j221,j87])
    j223 = B(j174,j209)
    j224 = B(j174,j219)
    j225 = self.P_ST_rbs_coupler
    j226 = self.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
    j227 = j226.T
    j228 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1]
    j229 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2]
    j230 = A(j225).T
    j231 = B(j98,j226)

    self.jacobian_data_blocks = [j0,j2,j10,-1.0*B(j6,self.ubar_SU1_rbr_uca_jcr_uca_chassis),j3,multi_dot([j5,j7,B(j1,j8)]),j3,multi_dot([j8.T,j11,j12]),j3,multi_dot([j5,j7,B(j1,j9)]),j3,multi_dot([j9.T,j11,j12]),j0,j13,j10,-1.0*B(j16,self.ubar_SU1_rbl_uca_jcl_uca_chassis),j3,multi_dot([j15,j17,B(j1,j18)]),j3,multi_dot([j18.T,j11,j20]),j3,multi_dot([j15,j17,B(j1,j19)]),j3,multi_dot([j19.T,j11,j20]),j0,j21,j10,-1.0*B(j24,self.ubar_SU1_rbr_lca_jcr_lca_chassis),j3,multi_dot([j23,j25,B(j1,j26)]),j3,multi_dot([j26.T,j11,j28]),j3,multi_dot([j23,j25,B(j1,j27)]),j3,multi_dot([j27.T,j11,j28]),j0,j29,j10,-1.0*B(j32,self.ubar_SU1_rbl_lca_jcl_lca_chassis),j3,multi_dot([j31,j33,B(j1,j34)]),j3,multi_dot([j34.T,j11,j36]),j3,multi_dot([j31,j33,B(j1,j35)]),j3,multi_dot([j35.T,j11,j36]),j0,j37,j10,-1.0*B(j39,self.ubar_SU1_rbr_upper_strut_jcr_strut_chassis),j3,multi_dot([j38.T,j40,B(j1,j41)]),j3,multi_dot([j41.T,j11,B(j39,j38)]),j0,j42,j10,-1.0*B(j44,self.ubar_SU1_rbl_upper_strut_jcl_strut_chassis),j3,multi_dot([j43.T,j45,B(j1,j46)]),j3,multi_dot([j46.T,j11,B(j44,j43)]),j0,B(j1,self.ubar_ground_jcr_tie_steering),j10,-1.0*B(j48,self.ubar_SU2_rbr_tie_rod_jcr_tie_steering),j3,multi_dot([j47.T,A(j48).T,B(j1,j49)]),j3,multi_dot([j49.T,j11,B(j48,j47)]),j0,B(j1,self.ubar_ground_jcl_tie_steering),j10,-1.0*B(j51,self.ubar_SU2_rbl_tie_rod_jcl_tie_steering),j3,multi_dot([j50.T,A(j51).T,B(j1,j52)]),j3,multi_dot([j52.T,j11,B(j51,j50)]),j0,j2,j10,-1.0*B(j55,self.ubar_SU2_rbr_uca_jcr_uca_chassis),j3,multi_dot([j54,j56,B(j1,j57)]),j3,multi_dot([j57.T,j11,j59]),j3,multi_dot([j54,j56,B(j1,j58)]),j3,multi_dot([j58.T,j11,j59]),j0,j13,j10,-1.0*B(j62,self.ubar_SU2_rbl_uca_jcl_uca_chassis),j3,multi_dot([j61,j63,B(j1,j64)]),j3,multi_dot([j64.T,j11,j66]),j3,multi_dot([j61,j63,B(j1,j65)]),j3,multi_dot([j65.T,j11,j66]),j0,j21,j10,-1.0*B(j69,self.ubar_SU2_rbr_lca_jcr_lca_chassis),j3,multi_dot([j68,j70,B(j1,j71)]),j3,multi_dot([j71.T,j11,j73]),j3,multi_dot([j68,j70,B(j1,j72)]),j3,multi_dot([j72.T,j11,j73]),j0,j29,j10,-1.0*B(j76,self.ubar_SU2_rbl_lca_jcl_lca_chassis),j3,multi_dot([j75,j77,B(j1,j78)]),j3,multi_dot([j78.T,j11,j80]),j3,multi_dot([j75,j77,B(j1,j79)]),j3,multi_dot([j79.T,j11,j80]),j0,j37,j10,-1.0*B(j82,self.ubar_SU2_rbr_upper_strut_jcr_strut_chassis),j3,multi_dot([j81.T,j83,B(j1,j84)]),j3,multi_dot([j84.T,j11,B(j82,j81)]),j0,j42,j10,-1.0*B(j86,self.ubar_SU2_rbl_upper_strut_jcl_strut_chassis),j3,multi_dot([j85.T,j87,B(j1,j88)]),j3,multi_dot([j88.T,j11,B(j86,j85)]),j0,B(j1,self.ubar_ground_jcr_rocker_ch),j10,-1.0*B(j91,self.ubar_ST_rbr_rocker_jcr_rocker_ch),j3,multi_dot([j90,j92,B(j1,j93)]),j3,multi_dot([j93.T,j11,j95]),j3,multi_dot([j90,j92,B(j1,j94)]),j3,multi_dot([j94.T,j11,j95]),j0,B(j1,self.ubar_ground_jcl_rocker_ch),j10,-1.0*B(j98,self.ubar_ST_rbl_rocker_jcl_rocker_ch),j3,multi_dot([j97,j99,B(j1,j100)]),j3,multi_dot([j100.T,j11,j102]),j3,multi_dot([j97,j99,B(j1,j101)]),j3,multi_dot([j101.T,j11,j102]),j0,B(j6,self.ubar_SU1_rbr_uca_jcr_uca_upright),j10,-1.0*B(j103,self.ubar_SU1_rbr_upright_jcr_uca_upright),j0,B(j16,self.ubar_SU1_rbl_uca_jcl_uca_upright),j10,-1.0*B(j104,self.ubar_SU1_rbl_upright_jcl_uca_upright),j0,B(j24,self.ubar_SU1_rbr_lca_jcr_lca_upright),j10,-1.0*B(j103,self.ubar_SU1_rbr_upright_jcr_lca_upright),j0,B(j24,self.ubar_SU1_rbr_lca_jcr_strut_lca),j10,-1.0*B(j106,self.ubar_SU1_rbr_lower_strut_jcr_strut_lca),j3,multi_dot([j105.T,j107,B(j24,j108)]),j3,multi_dot([j108.T,j25,B(j106,j105)]),j0,B(j32,self.ubar_SU1_rbl_lca_jcl_lca_upright),j10,-1.0*B(j104,self.ubar_SU1_rbl_upright_jcl_lca_upright),j0,B(j32,self.ubar_SU1_rbl_lca_jcl_strut_lca),j10,-1.0*B(j110,self.ubar_SU1_rbl_lower_strut_jcl_strut_lca),j3,multi_dot([j109.T,j111,B(j32,j112)]),j3,multi_dot([j112.T,j33,B(j110,j109)]),j0,B(j103,self.ubar_SU1_rbr_upright_jcr_tie_upright),j10,-1.0*B(j113,self.ubar_SU1_rbr_tie_rod_jcr_tie_upright),j0,B(j103,self.ubar_SU1_rbr_upright_jcr_hub_bearing),j10,-1.0*B(j116,self.ubar_SU1_rbr_hub_jcr_hub_bearing),j3,multi_dot([j115,j117,B(j103,j118)]),j3,multi_dot([j118.T,j120,j121]),j3,multi_dot([j115,j117,B(j103,j119)]),j3,multi_dot([j119.T,j120,j121]),j0,B(j104,self.ubar_SU1_rbl_upright_jcl_tie_upright),j10,-1.0*B(j122,self.ubar_SU1_rbl_tie_rod_jcl_tie_upright),j0,B(j104,self.ubar_SU1_rbl_upright_jcl_hub_bearing),j10,-1.0*B(j125,self.ubar_SU1_rbl_hub_jcl_hub_bearing),j3,multi_dot([j124,j126,B(j104,j127)]),j3,multi_dot([j127.T,j129,j130]),j3,multi_dot([j124,j126,B(j104,j128)]),j3,multi_dot([j128.T,j129,j130]),j3,multi_dot([j132,j107,j134]),j3,multi_dot([j137,j40,j145]),j3,multi_dot([j132,j107,j136]),j3,multi_dot([j143,j40,j145]),-1.0*j138,(-1.0*multi_dot([j137,j40,j140]) + multi_dot([j142,j134])),j138,multi_dot([j137,j40,j146]),-1.0*j144,(-1.0*multi_dot([j143,j40,j140]) + multi_dot([j142,j136])),j144,multi_dot([j143,j40,j146]),j3,multi_dot([j148,j111,j150]),j3,multi_dot([j153,j45,j161]),j3,multi_dot([j148,j111,j152]),j3,multi_dot([j159,j45,j161]),-1.0*j154,(-1.0*multi_dot([j153,j45,j156]) + multi_dot([j158,j150])),j154,multi_dot([j153,j45,j162]),-1.0*j160,(-1.0*multi_dot([j159,j45,j156]) + multi_dot([j158,j152])),j160,multi_dot([j159,j45,j162]),j0,B(j113,self.ubar_SU1_rbr_tie_rod_jcr_tie_steering),j10,-1.0*B(j91,self.ubar_ST_rbr_rocker_jcr_tie_steering),j3,multi_dot([j163.T,j92,B(j113,j164)]),j3,multi_dot([j164.T,A(j113).T,B(j91,j163)]),j0,B(j122,self.ubar_SU1_rbl_tie_rod_jcl_tie_steering),j10,-1.0*B(j98,self.ubar_ST_rbl_rocker_jcl_tie_steering),j3,multi_dot([j165.T,j99,B(j122,j166)]),j3,multi_dot([j166.T,A(j122).T,B(j98,j165)]),j0,B(j55,self.ubar_SU2_rbr_uca_jcr_uca_upright),j10,-1.0*B(j167,self.ubar_SU2_rbr_upright_jcr_uca_upright),j0,B(j62,self.ubar_SU2_rbl_uca_jcl_uca_upright),j10,-1.0*B(j168,self.ubar_SU2_rbl_upright_jcl_uca_upright),j0,B(j69,self.ubar_SU2_rbr_lca_jcr_lca_upright),j10,-1.0*B(j167,self.ubar_SU2_rbr_upright_jcr_lca_upright),j0,B(j69,self.ubar_SU2_rbr_lca_jcr_strut_lca),j10,-1.0*B(j170,self.ubar_SU2_rbr_lower_strut_jcr_strut_lca),j3,multi_dot([j169.T,j171,B(j69,j172)]),j3,multi_dot([j172.T,j70,B(j170,j169)]),j0,B(j76,self.ubar_SU2_rbl_lca_jcl_lca_upright),j10,-1.0*B(j168,self.ubar_SU2_rbl_upright_jcl_lca_upright),j0,B(j76,self.ubar_SU2_rbl_lca_jcl_strut_lca),j10,-1.0*B(j174,self.ubar_SU2_rbl_lower_strut_jcl_strut_lca),j3,multi_dot([j173.T,j175,B(j76,j176)]),j3,multi_dot([j176.T,j77,B(j174,j173)]),j0,B(j167,self.ubar_SU2_rbr_upright_jcr_tie_upright),j10,-1.0*B(j48,self.ubar_SU2_rbr_tie_rod_jcr_tie_upright),j0,B(j167,self.ubar_SU2_rbr_upright_jcr_hub_bearing),j10,-1.0*B(j179,self.ubar_SU2_rbr_hub_jcr_hub_bearing),j3,multi_dot([j178,j180,B(j167,j181)]),j3,multi_dot([j181.T,j183,j184]),j3,multi_dot([j178,j180,B(j167,j182)]),j3,multi_dot([j182.T,j183,j184]),j0,B(j168,self.ubar_SU2_rbl_upright_jcl_tie_upright),j10,-1.0*B(j51,self.ubar_SU2_rbl_tie_rod_jcl_tie_upright),j0,B(j168,self.ubar_SU2_rbl_upright_jcl_hub_bearing),j10,-1.0*B(j187,self.ubar_SU2_rbl_hub_jcl_hub_bearing),j3,multi_dot([j186,j188,B(j168,j189)]),j3,multi_dot([j189.T,j191,j192]),j3,multi_dot([j186,j188,B(j168,j190)]),j3,multi_dot([j190.T,j191,j192]),j3,multi_dot([j194,j171,j196]),j3,multi_dot([j199,j83,j207]),j3,multi_dot([j194,j171,j198]),j3,multi_dot([j205,j83,j207]),-1.0*j200,(-1.0*multi_dot([j199,j83,j202]) + multi_dot([j204,j196])),j200,multi_dot([j199,j83,j208]),-1.0*j206,(-1.0*multi_dot([j205,j83,j202]) + multi_dot([j204,j198])),j206,multi_dot([j205,j83,j208]),j3,multi_dot([j210,j175,j212]),j3,multi_dot([j215,j87,j223]),j3,multi_dot([j210,j175,j214]),j3,multi_dot([j221,j87,j223]),-1.0*j216,(-1.0*multi_dot([j215,j87,j218]) + multi_dot([j220,j212])),j216,multi_dot([j215,j87,j224]),-1.0*j222,(-1.0*multi_dot([j221,j87,j218]) + multi_dot([j220,j214])),j222,multi_dot([j221,j87,j224]),j0,B(j225,self.ubar_ST_rbs_coupler_jcs_rc_sph),j10,-1.0*B(j91,self.ubar_ST_rbr_rocker_jcs_rc_sph),j0,B(j225,self.ubar_ST_rbs_coupler_jcs_rc_uni),j10,-1.0*B(j98,self.ubar_ST_rbl_rocker_jcs_rc_uni),j3,multi_dot([j227,j99,B(j225,j228)]),j3,multi_dot([j228.T,j230,j231]),j3,multi_dot([j227,j99,B(j225,j229)]),j3,multi_dot([j229.T,j230,j231]),j0,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j6.T,2.0*j16.T,2.0*j24.T,2.0*j32.T,2.0*j103.T,2.0*j104.T,2.0*j39.T,2.0*j44.T,2.0*j106.T,2.0*j110.T,2.0*j113.T,2.0*j122.T,2.0*j116.T,2.0*j125.T,2.0*j55.T,2.0*j62.T,2.0*j69.T,2.0*j76.T,2.0*j167.T,2.0*j168.T,2.0*j82.T,2.0*j86.T,2.0*j170.T,2.0*j174.T,2.0*j48.T,2.0*j51.T,2.0*j179.T,2.0*j187.T,2.0*j225.T,2.0*j91.T,2.0*j98.T]

    self.jacobian_rows_explicit = []
    self.jacobian_cols_explicit = []
    self.jacobian_data_explicit = []

    sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
                     self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)

    self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))



