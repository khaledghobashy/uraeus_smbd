
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from matrix_funcs import A, B, sparse_assembler, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.F_mcr_zact = lambda t : 0
        self.J_mcr_zact = np.array([[0, 0, 1]],dtype=np.float64)
        self.F_mcl_zact = lambda t : 0
        self.J_mcl_zact = np.array([[0, 0, 1]],dtype=np.float64)
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
        self.F_jcr_hub_bearing = lambda t : 0
        self.pt_jcl_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.F_jcl_hub_bearing = lambda t : 0
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
        self.pt_jcs_rc_sph = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_rc_uni = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax_jcs_rc_uni = np.array([[0], [0], [1]],dtype=np.float64)
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_SU_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
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

        self.pos_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76])
        self.pos_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76])
        self.vel_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_level_rows_blocks = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76])
        self.acc_level_cols_blocks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jacobian_rows_blocks = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,54,54,55,55,55,55,56,56,56,56,57,57,57,57,58,58,59,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76])
        self.jacobian_cols_blocks = np.array([0,1,26,27,0,1,28,29,0,1,2,3,0,1,2,3,0,1,2,3,0,1,4,5,0,1,4,5,0,1,4,5,0,1,6,7,0,1,6,7,0,1,6,7,0,1,8,9,0,1,8,9,0,1,8,9,0,1,14,15,0,1,14,15,0,1,16,17,0,1,16,17,0,1,32,33,0,1,32,33,0,1,32,33,0,1,34,35,0,1,34,35,0,1,34,35,2,3,10,11,4,5,12,13,6,7,10,11,6,7,18,19,6,7,18,19,8,9,12,13,8,9,20,21,8,9,20,21,10,11,22,23,10,11,26,27,10,11,26,27,10,11,26,27,10,11,26,27,12,13,24,25,12,13,28,29,12,13,28,29,12,13,28,29,12,13,28,29,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,20,21,16,17,20,21,16,17,20,21,22,23,32,33,22,23,32,33,24,25,34,35,24,25,34,35,30,31,32,33,30,31,34,35,30,31,34,35,30,31,34,35,0,1,0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35])

    def eval_constants(self):
        config = self.config

        c0 = A(config.P_ground).T
        c1 = config.pt_jcr_uca_chassis
        c2 = -1.0*multi_dot([c0,config.R_ground])
        c3 = A(config.P_SU_rbr_uca).T
        c4 = -1.0*multi_dot([c3,config.R_SU_rbr_uca])
        c5 = Triad(config.ax_jcr_uca_chassis)
        c6 = config.pt_jcl_uca_chassis
        c7 = A(config.P_SU_rbl_uca).T
        c8 = -1.0*multi_dot([c7,config.R_SU_rbl_uca])
        c9 = Triad(config.ax_jcl_uca_chassis)
        c10 = config.pt_jcr_lca_chassis
        c11 = A(config.P_SU_rbr_lca).T
        c12 = -1.0*multi_dot([c11,config.R_SU_rbr_lca])
        c13 = Triad(config.ax_jcr_lca_chassis)
        c14 = config.pt_jcl_lca_chassis
        c15 = A(config.P_SU_rbl_lca).T
        c16 = -1.0*multi_dot([c15,config.R_SU_rbl_lca])
        c17 = Triad(config.ax_jcl_lca_chassis)
        c18 = config.pt_jcr_strut_chassis
        c19 = A(config.P_SU_rbr_upper_strut).T
        c20 = -1.0*multi_dot([c19,config.R_SU_rbr_upper_strut])
        c21 = Triad(config.ax_jcr_strut_chassis)
        c22 = config.pt_jcl_strut_chassis
        c23 = A(config.P_SU_rbl_upper_strut).T
        c24 = -1.0*multi_dot([c23,config.R_SU_rbl_upper_strut])
        c25 = Triad(config.ax_jcl_strut_chassis)
        c26 = config.pt_jcr_rocker_ch
        c27 = A(config.P_ST_rbr_rocker).T
        c28 = -1.0*multi_dot([c27,config.R_ST_rbr_rocker])
        c29 = Triad(config.ax_jcr_rocker_ch)
        c30 = config.pt_jcl_rocker_ch
        c31 = A(config.P_ST_rbl_rocker).T
        c32 = -1.0*multi_dot([c31,config.R_ST_rbl_rocker])
        c33 = Triad(config.ax_jcl_rocker_ch)
        c34 = config.pt_jcr_uca_upright
        c35 = A(config.P_SU_rbr_upright).T
        c36 = -1.0*multi_dot([c35,config.R_SU_rbr_upright])
        c37 = Triad(config.ax_jcr_uca_upright)
        c38 = config.pt_jcl_uca_upright
        c39 = A(config.P_SU_rbl_upright).T
        c40 = -1.0*multi_dot([c39,config.R_SU_rbl_upright])
        c41 = Triad(config.ax_jcl_uca_upright)
        c42 = config.pt_jcr_lca_upright
        c43 = Triad(config.ax_jcr_lca_upright)
        c44 = config.pt_jcr_strut_lca
        c45 = A(config.P_SU_rbr_lower_strut).T
        c46 = -1.0*multi_dot([c45,config.R_SU_rbr_lower_strut])
        c47 = Triad(config.ax_jcr_strut_lca)
        c48 = config.pt_jcl_lca_upright
        c49 = Triad(config.ax_jcl_lca_upright)
        c50 = config.pt_jcl_strut_lca
        c51 = A(config.P_SU_rbl_lower_strut).T
        c52 = -1.0*multi_dot([c51,config.R_SU_rbl_lower_strut])
        c53 = Triad(config.ax_jcl_strut_lca)
        c54 = config.pt_jcr_tie_upright
        c55 = A(config.P_SU_rbr_tie_rod).T
        c56 = -1.0*multi_dot([c55,config.R_SU_rbr_tie_rod])
        c57 = Triad(config.ax_jcr_tie_upright)
        c58 = config.pt_jcr_hub_bearing
        c59 = A(config.P_SU_rbr_hub).T
        c60 = Triad(config.ax_jcr_hub_bearing)
        c61 = config.pt_jcl_tie_upright
        c62 = A(config.P_SU_rbl_tie_rod).T
        c63 = -1.0*multi_dot([c62,config.R_SU_rbl_tie_rod])
        c64 = Triad(config.ax_jcl_tie_upright)
        c65 = config.pt_jcl_hub_bearing
        c66 = A(config.P_SU_rbl_hub).T
        c67 = Triad(config.ax_jcl_hub_bearing)
        c68 = config.pt_jcr_strut
        c69 = Triad(config.ax_jcr_strut)
        c70 = config.pt_jcl_strut
        c71 = Triad(config.ax_jcl_strut)
        c72 = config.pt_jcr_tie_steering
        c73 = Triad(config.ax_jcr_tie_steering)
        c74 = config.pt_jcl_tie_steering
        c75 = Triad(config.ax_jcl_tie_steering)
        c76 = A(config.P_ST_rbs_coupler).T
        c77 = config.pt_jcs_rc_sph
        c78 = -1.0*multi_dot([c76,config.R_ST_rbs_coupler])
        c79 = Triad(config.ax_jcs_rc_sph)
        c80 = config.pt_jcs_rc_uni
        c81 = Triad(config.ax_jcs_rc_uni)

        self.ubar_ground_jcr_uca_chassis = (multi_dot([c0,c1]) + c2)
        self.ubar_SU_rbr_uca_jcr_uca_chassis = (multi_dot([c3,c1]) + c4)
        self.Mbar_ground_jcr_uca_chassis = multi_dot([c0,c5])
        self.Mbar_SU_rbr_uca_jcr_uca_chassis = multi_dot([c3,c5])
        self.ubar_ground_jcl_uca_chassis = (multi_dot([c0,c6]) + c2)
        self.ubar_SU_rbl_uca_jcl_uca_chassis = (multi_dot([c7,c6]) + c8)
        self.Mbar_ground_jcl_uca_chassis = multi_dot([c0,c9])
        self.Mbar_SU_rbl_uca_jcl_uca_chassis = multi_dot([c7,c9])
        self.ubar_ground_jcr_lca_chassis = (multi_dot([c0,c10]) + c2)
        self.ubar_SU_rbr_lca_jcr_lca_chassis = (multi_dot([c11,c10]) + c12)
        self.Mbar_ground_jcr_lca_chassis = multi_dot([c0,c13])
        self.Mbar_SU_rbr_lca_jcr_lca_chassis = multi_dot([c11,c13])
        self.ubar_ground_jcl_lca_chassis = (multi_dot([c0,c14]) + c2)
        self.ubar_SU_rbl_lca_jcl_lca_chassis = (multi_dot([c15,c14]) + c16)
        self.Mbar_ground_jcl_lca_chassis = multi_dot([c0,c17])
        self.Mbar_SU_rbl_lca_jcl_lca_chassis = multi_dot([c15,c17])
        self.ubar_ground_jcr_strut_chassis = (multi_dot([c0,c18]) + c2)
        self.ubar_SU_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c19,c18]) + c20)
        self.Mbar_ground_jcr_strut_chassis = multi_dot([c0,c21])
        self.Mbar_SU_rbr_upper_strut_jcr_strut_chassis = multi_dot([c19,Triad("'ax2_jcr_strut_chassis'", 'c21[0:3,1:2]')])
        self.ubar_ground_jcl_strut_chassis = (multi_dot([c0,c22]) + c2)
        self.ubar_SU_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c23,c22]) + c24)
        self.Mbar_ground_jcl_strut_chassis = multi_dot([c0,c25])
        self.Mbar_SU_rbl_upper_strut_jcl_strut_chassis = multi_dot([c23,Triad("'ax2_jcl_strut_chassis'", 'c25[0:3,1:2]')])
        self.ubar_ground_jcr_rocker_ch = (multi_dot([c0,c26]) + c2)
        self.ubar_ST_rbr_rocker_jcr_rocker_ch = (multi_dot([c27,c26]) + c28)
        self.Mbar_ground_jcr_rocker_ch = multi_dot([c0,c29])
        self.Mbar_ST_rbr_rocker_jcr_rocker_ch = multi_dot([c27,c29])
        self.ubar_ground_jcl_rocker_ch = (multi_dot([c0,c30]) + c2)
        self.ubar_ST_rbl_rocker_jcl_rocker_ch = (multi_dot([c31,c30]) + c32)
        self.Mbar_ground_jcl_rocker_ch = multi_dot([c0,c33])
        self.Mbar_ST_rbl_rocker_jcl_rocker_ch = multi_dot([c31,c33])
        self.ubar_SU_rbr_uca_jcr_uca_upright = (multi_dot([c3,c34]) + c4)
        self.ubar_SU_rbr_upright_jcr_uca_upright = (multi_dot([c35,c34]) + c36)
        self.Mbar_SU_rbr_uca_jcr_uca_upright = multi_dot([c3,c37])
        self.Mbar_SU_rbr_upright_jcr_uca_upright = multi_dot([c35,c37])
        self.ubar_SU_rbl_uca_jcl_uca_upright = (multi_dot([c7,c38]) + c8)
        self.ubar_SU_rbl_upright_jcl_uca_upright = (multi_dot([c39,c38]) + c40)
        self.Mbar_SU_rbl_uca_jcl_uca_upright = multi_dot([c7,c41])
        self.Mbar_SU_rbl_upright_jcl_uca_upright = multi_dot([c39,c41])
        self.ubar_SU_rbr_lca_jcr_lca_upright = (multi_dot([c11,c42]) + c12)
        self.ubar_SU_rbr_upright_jcr_lca_upright = (multi_dot([c35,c42]) + c36)
        self.Mbar_SU_rbr_lca_jcr_lca_upright = multi_dot([c11,c43])
        self.Mbar_SU_rbr_upright_jcr_lca_upright = multi_dot([c35,c43])
        self.ubar_SU_rbr_lca_jcr_strut_lca = (multi_dot([c11,c44]) + c12)
        self.ubar_SU_rbr_lower_strut_jcr_strut_lca = (multi_dot([c45,c44]) + c46)
        self.Mbar_SU_rbr_lca_jcr_strut_lca = multi_dot([c11,c47])
        self.Mbar_SU_rbr_lower_strut_jcr_strut_lca = multi_dot([c45,Triad("'ax2_jcr_strut_lca'", 'c47[0:3,1:2]')])
        self.ubar_SU_rbl_lca_jcl_lca_upright = (multi_dot([c15,c48]) + c16)
        self.ubar_SU_rbl_upright_jcl_lca_upright = (multi_dot([c39,c48]) + c40)
        self.Mbar_SU_rbl_lca_jcl_lca_upright = multi_dot([c15,c49])
        self.Mbar_SU_rbl_upright_jcl_lca_upright = multi_dot([c39,c49])
        self.ubar_SU_rbl_lca_jcl_strut_lca = (multi_dot([c15,c50]) + c16)
        self.ubar_SU_rbl_lower_strut_jcl_strut_lca = (multi_dot([c51,c50]) + c52)
        self.Mbar_SU_rbl_lca_jcl_strut_lca = multi_dot([c15,c53])
        self.Mbar_SU_rbl_lower_strut_jcl_strut_lca = multi_dot([c51,Triad("'ax2_jcl_strut_lca'", 'c53[0:3,1:2]')])
        self.ubar_SU_rbr_upright_jcr_tie_upright = (multi_dot([c35,c54]) + c36)
        self.ubar_SU_rbr_tie_rod_jcr_tie_upright = (multi_dot([c55,c54]) + c56)
        self.Mbar_SU_rbr_upright_jcr_tie_upright = multi_dot([c35,c57])
        self.Mbar_SU_rbr_tie_rod_jcr_tie_upright = multi_dot([c55,c57])
        self.ubar_SU_rbr_upright_jcr_hub_bearing = (multi_dot([c35,c58]) + c36)
        self.ubar_SU_rbr_hub_jcr_hub_bearing = (multi_dot([c59,c58]) + -1.0*multi_dot([c59,'R_SU_rbr_hub']))
        self.Mbar_SU_rbr_upright_jcr_hub_bearing = multi_dot([c35,c60])
        self.Mbar_SU_rbr_hub_jcr_hub_bearing = multi_dot([c59,c60])
        self.ubar_SU_rbl_upright_jcl_tie_upright = (multi_dot([c39,c61]) + c40)
        self.ubar_SU_rbl_tie_rod_jcl_tie_upright = (multi_dot([c62,c61]) + c63)
        self.Mbar_SU_rbl_upright_jcl_tie_upright = multi_dot([c39,c64])
        self.Mbar_SU_rbl_tie_rod_jcl_tie_upright = multi_dot([c62,c64])
        self.ubar_SU_rbl_upright_jcl_hub_bearing = (multi_dot([c39,c65]) + c40)
        self.ubar_SU_rbl_hub_jcl_hub_bearing = (multi_dot([c66,c65]) + -1.0*multi_dot([c66,'R_SU_rbl_hub']))
        self.Mbar_SU_rbl_upright_jcl_hub_bearing = multi_dot([c39,c67])
        self.Mbar_SU_rbl_hub_jcl_hub_bearing = multi_dot([c66,c67])
        self.ubar_SU_rbr_upper_strut_jcr_strut = (multi_dot([c19,c68]) + c20)
        self.ubar_SU_rbr_lower_strut_jcr_strut = (multi_dot([c45,c68]) + c46)
        self.Mbar_SU_rbr_upper_strut_jcr_strut = multi_dot([c19,c69])
        self.Mbar_SU_rbr_lower_strut_jcr_strut = multi_dot([c45,c69])
        self.ubar_SU_rbl_upper_strut_jcl_strut = (multi_dot([c23,c70]) + c24)
        self.ubar_SU_rbl_lower_strut_jcl_strut = (multi_dot([c51,c70]) + c52)
        self.Mbar_SU_rbl_upper_strut_jcl_strut = multi_dot([c23,c71])
        self.Mbar_SU_rbl_lower_strut_jcl_strut = multi_dot([c51,c71])
        self.ubar_SU_rbr_tie_rod_jcr_tie_steering = (multi_dot([c55,c72]) + c56)
        self.ubar_ST_rbr_rocker_jcr_tie_steering = (multi_dot([c27,c72]) + c28)
        self.Mbar_SU_rbr_tie_rod_jcr_tie_steering = multi_dot([c55,c73])
        self.Mbar_ST_rbr_rocker_jcr_tie_steering = multi_dot([c27,Triad("'ax2_jcr_tie_steering'", 'c73[0:3,1:2]')])
        self.ubar_SU_rbl_tie_rod_jcl_tie_steering = (multi_dot([c62,c74]) + c63)
        self.ubar_ST_rbl_rocker_jcl_tie_steering = (multi_dot([c31,c74]) + c32)
        self.Mbar_SU_rbl_tie_rod_jcl_tie_steering = multi_dot([c62,c75])
        self.Mbar_ST_rbl_rocker_jcl_tie_steering = multi_dot([c31,Triad("'ax2_jcl_tie_steering'", 'c75[0:3,1:2]')])
        self.ubar_ST_rbs_coupler_jcs_rc_sph = (multi_dot([c76,c77]) + c78)
        self.ubar_ST_rbr_rocker_jcs_rc_sph = (multi_dot([c27,c77]) + c28)
        self.Mbar_ST_rbs_coupler_jcs_rc_sph = multi_dot([c76,c79])
        self.Mbar_ST_rbr_rocker_jcs_rc_sph = multi_dot([c27,c79])
        self.ubar_ST_rbs_coupler_jcs_rc_uni = (multi_dot([c76,c80]) + c78)
        self.ubar_ST_rbl_rocker_jcs_rc_uni = (multi_dot([c31,c80]) + c32)
        self.Mbar_ST_rbs_coupler_jcs_rc_uni = multi_dot([c76,c81])
        self.Mbar_ST_rbl_rocker_jcs_rc_uni = multi_dot([c31,c81])

    
def set_coordinates(self,q):
    self.R_ground = q[0:3,0:1]
    self.P_ground = q[3:7,0:1]
    self.R_SU_rbr_uca = q[7:10,0:1]
    self.P_SU_rbr_uca = q[10:14,0:1]
    self.R_SU_rbl_uca = q[14:17,0:1]
    self.P_SU_rbl_uca = q[17:21,0:1]
    self.R_SU_rbr_lca = q[21:24,0:1]
    self.P_SU_rbr_lca = q[24:28,0:1]
    self.R_SU_rbl_lca = q[28:31,0:1]
    self.P_SU_rbl_lca = q[31:35,0:1]
    self.R_SU_rbr_upright = q[35:38,0:1]
    self.P_SU_rbr_upright = q[38:42,0:1]
    self.R_SU_rbl_upright = q[42:45,0:1]
    self.P_SU_rbl_upright = q[45:49,0:1]
    self.R_SU_rbr_upper_strut = q[49:52,0:1]
    self.P_SU_rbr_upper_strut = q[52:56,0:1]
    self.R_SU_rbl_upper_strut = q[56:59,0:1]
    self.P_SU_rbl_upper_strut = q[59:63,0:1]
    self.R_SU_rbr_lower_strut = q[63:66,0:1]
    self.P_SU_rbr_lower_strut = q[66:70,0:1]
    self.R_SU_rbl_lower_strut = q[70:73,0:1]
    self.P_SU_rbl_lower_strut = q[73:77,0:1]
    self.R_SU_rbr_tie_rod = q[77:80,0:1]
    self.P_SU_rbr_tie_rod = q[80:84,0:1]
    self.R_SU_rbl_tie_rod = q[84:87,0:1]
    self.P_SU_rbl_tie_rod = q[87:91,0:1]
    self.R_SU_rbr_hub = q[91:94,0:1]
    self.P_SU_rbr_hub = q[94:98,0:1]
    self.R_SU_rbl_hub = q[98:101,0:1]
    self.P_SU_rbl_hub = q[101:105,0:1]
    self.R_ST_rbs_coupler = q[105:108,0:1]
    self.P_ST_rbs_coupler = q[108:112,0:1]
    self.R_ST_rbr_rocker = q[112:115,0:1]
    self.P_ST_rbr_rocker = q[115:119,0:1]
    self.R_ST_rbl_rocker = q[119:122,0:1]
    self.P_ST_rbl_rocker = q[122:126,0:1]


    
def set_velocities(self,qd):
    self.Rd_ground = qd[0:3,0:1]
    self.Pd_ground = qd[3:7,0:1]
    self.Rd_SU_rbr_uca = qd[7:10,0:1]
    self.Pd_SU_rbr_uca = qd[10:14,0:1]
    self.Rd_SU_rbl_uca = qd[14:17,0:1]
    self.Pd_SU_rbl_uca = qd[17:21,0:1]
    self.Rd_SU_rbr_lca = qd[21:24,0:1]
    self.Pd_SU_rbr_lca = qd[24:28,0:1]
    self.Rd_SU_rbl_lca = qd[28:31,0:1]
    self.Pd_SU_rbl_lca = qd[31:35,0:1]
    self.Rd_SU_rbr_upright = qd[35:38,0:1]
    self.Pd_SU_rbr_upright = qd[38:42,0:1]
    self.Rd_SU_rbl_upright = qd[42:45,0:1]
    self.Pd_SU_rbl_upright = qd[45:49,0:1]
    self.Rd_SU_rbr_upper_strut = qd[49:52,0:1]
    self.Pd_SU_rbr_upper_strut = qd[52:56,0:1]
    self.Rd_SU_rbl_upper_strut = qd[56:59,0:1]
    self.Pd_SU_rbl_upper_strut = qd[59:63,0:1]
    self.Rd_SU_rbr_lower_strut = qd[63:66,0:1]
    self.Pd_SU_rbr_lower_strut = qd[66:70,0:1]
    self.Rd_SU_rbl_lower_strut = qd[70:73,0:1]
    self.Pd_SU_rbl_lower_strut = qd[73:77,0:1]
    self.Rd_SU_rbr_tie_rod = qd[77:80,0:1]
    self.Pd_SU_rbr_tie_rod = qd[80:84,0:1]
    self.Rd_SU_rbl_tie_rod = qd[84:87,0:1]
    self.Pd_SU_rbl_tie_rod = qd[87:91,0:1]
    self.Rd_SU_rbr_hub = qd[91:94,0:1]
    self.Pd_SU_rbr_hub = qd[94:98,0:1]
    self.Rd_SU_rbl_hub = qd[98:101,0:1]
    self.Pd_SU_rbl_hub = qd[101:105,0:1]
    self.Rd_ST_rbs_coupler = qd[105:108,0:1]
    self.Pd_ST_rbs_coupler = qd[108:112,0:1]
    self.Rd_ST_rbr_rocker = qd[112:115,0:1]
    self.Pd_ST_rbr_rocker = qd[115:119,0:1]
    self.Rd_ST_rbl_rocker = qd[119:122,0:1]
    self.Pd_ST_rbl_rocker = qd[122:126,0:1]


    
def set_initial_configuration(self):
    config = self.config

    q = np.concatenate([config.R_ground,
    config.P_ground,
    config.R_SU_rbr_uca,
    config.P_SU_rbr_uca,
    config.R_SU_rbl_uca,
    config.P_SU_rbl_uca,
    config.R_SU_rbr_lca,
    config.P_SU_rbr_lca,
    config.R_SU_rbl_lca,
    config.P_SU_rbl_lca,
    config.R_SU_rbr_upright,
    config.P_SU_rbr_upright,
    config.R_SU_rbl_upright,
    config.P_SU_rbl_upright,
    config.R_SU_rbr_upper_strut,
    config.P_SU_rbr_upper_strut,
    config.R_SU_rbl_upper_strut,
    config.P_SU_rbl_upper_strut,
    config.R_SU_rbr_lower_strut,
    config.P_SU_rbr_lower_strut,
    config.R_SU_rbl_lower_strut,
    config.P_SU_rbl_lower_strut,
    config.R_SU_rbr_tie_rod,
    config.P_SU_rbr_tie_rod,
    config.R_SU_rbl_tie_rod,
    config.P_SU_rbl_tie_rod,
    config.R_SU_rbr_hub,
    config.P_SU_rbr_hub,
    config.R_SU_rbl_hub,
    config.P_SU_rbl_hub,
    config.R_ST_rbs_coupler,
    config.P_ST_rbs_coupler,
    config.R_ST_rbr_rocker,
    config.P_ST_rbr_rocker,
    config.R_ST_rbl_rocker,
    config.P_ST_rbl_rocker])

    qd = np.concatenate([config.Rd_ground,
    config.Pd_ground,
    config.Rd_SU_rbr_uca,
    config.Pd_SU_rbr_uca,
    config.Rd_SU_rbl_uca,
    config.Pd_SU_rbl_uca,
    config.Rd_SU_rbr_lca,
    config.Pd_SU_rbr_lca,
    config.Rd_SU_rbl_lca,
    config.Pd_SU_rbl_lca,
    config.Rd_SU_rbr_upright,
    config.Pd_SU_rbr_upright,
    config.Rd_SU_rbl_upright,
    config.Pd_SU_rbl_upright,
    config.Rd_SU_rbr_upper_strut,
    config.Pd_SU_rbr_upper_strut,
    config.Rd_SU_rbl_upper_strut,
    config.Pd_SU_rbl_upper_strut,
    config.Rd_SU_rbr_lower_strut,
    config.Pd_SU_rbr_lower_strut,
    config.Rd_SU_rbl_lower_strut,
    config.Pd_SU_rbl_lower_strut,
    config.Rd_SU_rbr_tie_rod,
    config.Pd_SU_rbr_tie_rod,
    config.Rd_SU_rbl_tie_rod,
    config.Pd_SU_rbl_tie_rod,
    config.Rd_SU_rbr_hub,
    config.Pd_SU_rbr_hub,
    config.Rd_SU_rbl_hub,
    config.Pd_SU_rbl_hub,
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

    x0 = self.R_ground
    x1 = x0[2]
    x2 = np.eye(1,dtype=np.float64)
    x3 = self.R_SU_rbr_uca
    x4 = self.P_ground
    x5 = A(x4)
    x6 = self.P_SU_rbr_uca
    x7 = A(x6)
    x8 = x5.T
    x9 = self.Mbar_SU_rbr_uca_jcr_uca_chassis[:,2:3]
    x10 = self.R_SU_rbl_uca
    x11 = self.P_SU_rbl_uca
    x12 = A(x11)
    x13 = self.Mbar_SU_rbl_uca_jcl_uca_chassis[:,2:3]
    x14 = self.R_SU_rbr_lca
    x15 = self.P_SU_rbr_lca
    x16 = A(x15)
    x17 = self.Mbar_SU_rbr_lca_jcr_lca_chassis[:,2:3]
    x18 = self.R_SU_rbl_lca
    x19 = self.P_SU_rbl_lca
    x20 = A(x19)
    x21 = self.Mbar_SU_rbl_lca_jcl_lca_chassis[:,2:3]
    x22 = self.R_SU_rbr_upper_strut
    x23 = self.P_SU_rbr_upper_strut
    x24 = A(x23)
    x25 = self.R_SU_rbl_upper_strut
    x26 = self.P_SU_rbl_upper_strut
    x27 = A(x26)
    x28 = -1.0*self.R_ST_rbr_rocker
    x29 = self.P_ST_rbr_rocker
    x30 = A(x29)
    x31 = self.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
    x32 = -1.0*self.R_ST_rbl_rocker
    x33 = self.P_ST_rbl_rocker
    x34 = A(x33)
    x35 = self.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
    x36 = self.R_SU_rbr_upright
    x37 = -1.0*x36
    x38 = self.P_SU_rbr_upright
    x39 = A(x38)
    x40 = self.R_SU_rbl_upright
    x41 = -1.0*x40
    x42 = self.P_SU_rbl_upright
    x43 = A(x42)
    x44 = -1.0*self.R_SU_rbr_lower_strut
    x45 = self.P_SU_rbr_lower_strut
    x46 = A(x45)
    x47 = -1.0*self.R_SU_rbl_lower_strut
    x48 = self.P_SU_rbl_lower_strut
    x49 = A(x48)
    x50 = self.R_SU_rbr_tie_rod
    x51 = self.P_SU_rbr_tie_rod
    x52 = A(x51)
    x53 = self.P_SU_rbr_hub
    x54 = A(x53)
    x55 = x39.T
    x56 = self.Mbar_SU_rbr_hub_jcr_hub_bearing[:,2:3]
    x57 = F_jcr_hub_bearing(t)
    x58 = self.Mbar_SU_rbr_hub_jcr_hub_bearing[:,0:1]
    x59 = self.R_SU_rbl_tie_rod
    x60 = self.P_SU_rbl_tie_rod
    x61 = A(x60)
    x62 = self.P_SU_rbl_hub
    x63 = A(x62)
    x64 = x43.T
    x65 = self.Mbar_SU_rbl_hub_jcl_hub_bearing[:,2:3]
    x66 = F_jcl_hub_bearing(t)
    x67 = self.Mbar_SU_rbl_hub_jcl_hub_bearing[:,0:1]
    x68 = self.Mbar_SU_rbr_upper_strut_jcr_strut[:,0:1].T
    x69 = x24.T
    x70 = self.Mbar_SU_rbr_lower_strut_jcr_strut[:,2:3]
    x71 = self.Mbar_SU_rbr_upper_strut_jcr_strut[:,1:2].T
    x72 = (x22 + x44 + multi_dot([x24,self.ubar_SU_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x46,self.ubar_SU_rbr_lower_strut_jcr_strut]))
    x73 = self.Mbar_SU_rbl_upper_strut_jcl_strut[:,0:1].T
    x74 = x27.T
    x75 = self.Mbar_SU_rbl_lower_strut_jcl_strut[:,2:3]
    x76 = self.Mbar_SU_rbl_upper_strut_jcl_strut[:,1:2].T
    x77 = (x25 + x47 + multi_dot([x27,self.ubar_SU_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x49,self.ubar_SU_rbl_lower_strut_jcl_strut]))
    x78 = self.R_ST_rbs_coupler
    x79 = self.P_ST_rbs_coupler
    x80 = A(x79)
    x81 = x80.T
    x82 = self.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
    x83 = -1.0*x2

    self.pos_level_data_blocks = [x1 - 1*F_mcr_zact(t)*x2,x1 - 1*F_mcl_zact(t)*x2,(x0 + -1.0*x3 + multi_dot([x5,self.ubar_ground_jcr_uca_chassis]) + -1.0*multi_dot([x7,self.ubar_SU_rbr_uca_jcr_uca_chassis])),multi_dot([self.Mbar_ground_jcr_uca_chassis[:,0:1].T,x8,x7,x9]),multi_dot([self.Mbar_ground_jcr_uca_chassis[:,1:2].T,x8,x7,x9]),(x0 + -1.0*x10 + multi_dot([x5,self.ubar_ground_jcl_uca_chassis]) + -1.0*multi_dot([x12,self.ubar_SU_rbl_uca_jcl_uca_chassis])),multi_dot([self.Mbar_ground_jcl_uca_chassis[:,0:1].T,x8,x12,x13]),multi_dot([self.Mbar_ground_jcl_uca_chassis[:,1:2].T,x8,x12,x13]),(x0 + -1.0*x14 + multi_dot([x5,self.ubar_ground_jcr_lca_chassis]) + -1.0*multi_dot([x16,self.ubar_SU_rbr_lca_jcr_lca_chassis])),multi_dot([self.Mbar_ground_jcr_lca_chassis[:,0:1].T,x8,x16,x17]),multi_dot([self.Mbar_ground_jcr_lca_chassis[:,1:2].T,x8,x16,x17]),(x0 + -1.0*x18 + multi_dot([x5,self.ubar_ground_jcl_lca_chassis]) + -1.0*multi_dot([x20,self.ubar_SU_rbl_lca_jcl_lca_chassis])),multi_dot([self.Mbar_ground_jcl_lca_chassis[:,0:1].T,x8,x20,x21]),multi_dot([self.Mbar_ground_jcl_lca_chassis[:,1:2].T,x8,x20,x21]),(x0 + -1.0*x22 + multi_dot([x5,self.ubar_ground_jcr_strut_chassis]) + -1.0*multi_dot([x24,self.ubar_SU_rbr_upper_strut_jcr_strut_chassis])),multi_dot([self.Mbar_ground_jcr_strut_chassis[:,0:1].T,x8,x24,self.Mbar_SU_rbr_upper_strut_jcr_strut_chassis[:,0:1]]),(x0 + -1.0*x25 + multi_dot([x5,self.ubar_ground_jcl_strut_chassis]) + -1.0*multi_dot([x27,self.ubar_SU_rbl_upper_strut_jcl_strut_chassis])),multi_dot([self.Mbar_ground_jcl_strut_chassis[:,0:1].T,x8,x27,self.Mbar_SU_rbl_upper_strut_jcl_strut_chassis[:,0:1]]),(x0 + x28 + multi_dot([x5,self.ubar_ground_jcr_rocker_ch]) + -1.0*multi_dot([x30,self.ubar_ST_rbr_rocker_jcr_rocker_ch])),multi_dot([self.Mbar_ground_jcr_rocker_ch[:,0:1].T,x8,x30,x31]),multi_dot([self.Mbar_ground_jcr_rocker_ch[:,1:2].T,x8,x30,x31]),(x0 + x32 + multi_dot([x5,self.ubar_ground_jcl_rocker_ch]) + -1.0*multi_dot([x34,self.ubar_ST_rbl_rocker_jcl_rocker_ch])),multi_dot([self.Mbar_ground_jcl_rocker_ch[:,0:1].T,x8,x34,x35]),multi_dot([self.Mbar_ground_jcl_rocker_ch[:,1:2].T,x8,x34,x35]),(x3 + x37 + multi_dot([x7,self.ubar_SU_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x39,self.ubar_SU_rbr_upright_jcr_uca_upright])),(x10 + x41 + multi_dot([x12,self.ubar_SU_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x43,self.ubar_SU_rbl_upright_jcl_uca_upright])),(x14 + x37 + multi_dot([x16,self.ubar_SU_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x39,self.ubar_SU_rbr_upright_jcr_lca_upright])),(x14 + x44 + multi_dot([x16,self.ubar_SU_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x46,self.ubar_SU_rbr_lower_strut_jcr_strut_lca])),multi_dot([self.Mbar_SU_rbr_lca_jcr_strut_lca[:,0:1].T,x16.T,x46,self.Mbar_SU_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x18 + x41 + multi_dot([x20,self.ubar_SU_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x43,self.ubar_SU_rbl_upright_jcl_lca_upright])),(x18 + x47 + multi_dot([x20,self.ubar_SU_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x49,self.ubar_SU_rbl_lower_strut_jcl_strut_lca])),multi_dot([self.Mbar_SU_rbl_lca_jcl_strut_lca[:,0:1].T,x20.T,x49,self.Mbar_SU_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x36 + -1.0*x50 + multi_dot([x39,self.ubar_SU_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x52,self.ubar_SU_rbr_tie_rod_jcr_tie_upright])),(x36 + -1.0*self.R_SU_rbr_hub + multi_dot([x39,self.ubar_SU_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x54,self.ubar_SU_rbr_hub_jcr_hub_bearing])),multi_dot([self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,0:1].T,x55,x54,x56]),multi_dot([self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,1:2].T,x55,x54,x56]),(cos(x57)*multi_dot([self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,1:2].T,x55,x54,x58]) + sin(x57)*-1.0*multi_dot([self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,0:1].T,x55,x54,x58])),(x40 + -1.0*x59 + multi_dot([x43,self.ubar_SU_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x61,self.ubar_SU_rbl_tie_rod_jcl_tie_upright])),(x40 + -1.0*self.R_SU_rbl_hub + multi_dot([x43,self.ubar_SU_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x63,self.ubar_SU_rbl_hub_jcl_hub_bearing])),multi_dot([self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,0:1].T,x64,x63,x65]),multi_dot([self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,1:2].T,x64,x63,x65]),(cos(x66)*multi_dot([self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,1:2].T,x64,x63,x67]) + sin(x66)*-1.0*multi_dot([self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,0:1].T,x64,x63,x67])),multi_dot([x68,x69,x46,x70]),multi_dot([x71,x69,x46,x70]),multi_dot([x68,x69,x72]),multi_dot([x71,x69,x72]),multi_dot([x73,x74,x49,x75]),multi_dot([x76,x74,x49,x75]),multi_dot([x73,x74,x77]),multi_dot([x76,x74,x77]),(x50 + x28 + multi_dot([x52,self.ubar_SU_rbr_tie_rod_jcr_tie_steering]) + -1.0*multi_dot([x30,self.ubar_ST_rbr_rocker_jcr_tie_steering])),multi_dot([self.Mbar_SU_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x52.T,x30,self.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]]),(x59 + x32 + multi_dot([x61,self.ubar_SU_rbl_tie_rod_jcl_tie_steering]) + -1.0*multi_dot([x34,self.ubar_ST_rbl_rocker_jcl_tie_steering])),multi_dot([self.Mbar_SU_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x61.T,x34,self.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]]),(x78 + x28 + multi_dot([x80,self.ubar_ST_rbs_coupler_jcs_rc_sph]) + -1.0*multi_dot([x30,self.ubar_ST_rbr_rocker_jcs_rc_sph])),(x78 + x32 + multi_dot([x80,self.ubar_ST_rbs_coupler_jcs_rc_uni]) + -1.0*multi_dot([x34,self.ubar_ST_rbl_rocker_jcs_rc_uni])),multi_dot([self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1].T,x81,x34,x82]),multi_dot([self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2].T,x81,x34,x82]),x0,(x4 + -1.0*'Pg_ground'),(x83 + (multi_dot([x6.T,x6]))**(1.0/2.0)),(x83 + (multi_dot([x11.T,x11]))**(1.0/2.0)),(x83 + (multi_dot([x15.T,x15]))**(1.0/2.0)),(x83 + (multi_dot([x19.T,x19]))**(1.0/2.0)),(x83 + (multi_dot([x38.T,x38]))**(1.0/2.0)),(x83 + (multi_dot([x42.T,x42]))**(1.0/2.0)),(x83 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x83 + (multi_dot([x26.T,x26]))**(1.0/2.0)),(x83 + (multi_dot([x45.T,x45]))**(1.0/2.0)),(x83 + (multi_dot([x48.T,x48]))**(1.0/2.0)),(x83 + (multi_dot([x51.T,x51]))**(1.0/2.0)),(x83 + (multi_dot([x60.T,x60]))**(1.0/2.0)),(x83 + (multi_dot([x53.T,x53]))**(1.0/2.0)),(x83 + (multi_dot([x62.T,x62]))**(1.0/2.0)),(x83 + (multi_dot([x79.T,x79]))**(1.0/2.0)),(x83 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x83 + (multi_dot([x33.T,x33]))**(1.0/2.0))]

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

    x0 = np.zeros((1,1),dtype=np.float64)
    x1 = np.eye(1,dtype=np.float64)
    x2 = np.zeros((3,1),dtype=np.float64)

    self.vel_level_data_blocks = [(x0 + derivative(F,t,0.1,1)*-1.0*x1),(x0 + derivative(F,t,0.1,1)*-1.0*x1),x2,x0,x0,x2,x0,x0,x2,x0,x0,x2,x0,x0,x2,x0,x2,x0,x2,x0,x0,x2,x0,x0,x2,x2,x2,x2,x0,x2,x2,x0,x2,x2,x0,x0,(x0 + derivative(F,t,0.1,1)*-1.0*x1),x2,x2,x0,x0,(x0 + derivative(F,t,0.1,1)*-1.0*x1),x0,x0,x0,x0,x0,x0,x0,x0,x2,x0,x2,x0,x2,x2,x0,x0,x2,np.zeros((4,1),dtype=np.float64),x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0,x0]

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

    a0 = np.zeros((1,1),dtype=np.float64)
    a1 = np.eye(1,dtype=np.float64)
    a2 = self.Pd_ground
    a3 = self.Pd_SU_rbr_uca
    a4 = self.Mbar_SU_rbr_uca_jcr_uca_chassis[:,2:3]
    a5 = a4.T
    a6 = self.P_SU_rbr_uca
    a7 = A(a6).T
    a8 = self.Mbar_ground_jcr_uca_chassis[:,0:1]
    a9 = self.P_ground
    a10 = A(a9).T
    a11 = B(a3,a4)
    a12 = a2.T
    a13 = B(a6,a4)
    a14 = self.Mbar_ground_jcr_uca_chassis[:,1:2]
    a15 = self.Pd_SU_rbl_uca
    a16 = self.Mbar_ground_jcl_uca_chassis[:,0:1]
    a17 = self.Mbar_SU_rbl_uca_jcl_uca_chassis[:,2:3]
    a18 = B(a15,a17)
    a19 = a17.T
    a20 = self.P_SU_rbl_uca
    a21 = A(a20).T
    a22 = B(a20,a17)
    a23 = self.Mbar_ground_jcl_uca_chassis[:,1:2]
    a24 = self.Pd_SU_rbr_lca
    a25 = self.Mbar_ground_jcr_lca_chassis[:,0:1]
    a26 = self.Mbar_SU_rbr_lca_jcr_lca_chassis[:,2:3]
    a27 = B(a24,a26)
    a28 = a26.T
    a29 = self.P_SU_rbr_lca
    a30 = A(a29).T
    a31 = B(a29,a26)
    a32 = self.Mbar_ground_jcr_lca_chassis[:,1:2]
    a33 = self.Pd_SU_rbl_lca
    a34 = self.Mbar_SU_rbl_lca_jcl_lca_chassis[:,2:3]
    a35 = a34.T
    a36 = self.P_SU_rbl_lca
    a37 = A(a36).T
    a38 = self.Mbar_ground_jcl_lca_chassis[:,0:1]
    a39 = B(a33,a34)
    a40 = B(a36,a34)
    a41 = self.Mbar_ground_jcl_lca_chassis[:,1:2]
    a42 = self.Pd_SU_rbr_upper_strut
    a43 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
    a44 = self.Mbar_SU_rbr_upper_strut_jcr_strut_chassis[:,0:1]
    a45 = self.P_SU_rbr_upper_strut
    a46 = A(a45).T
    a47 = self.Pd_SU_rbl_upper_strut
    a48 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
    a49 = self.Mbar_SU_rbl_upper_strut_jcl_strut_chassis[:,0:1]
    a50 = self.P_SU_rbl_upper_strut
    a51 = A(a50).T
    a52 = self.Pd_ST_rbr_rocker
    a53 = self.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
    a54 = a53.T
    a55 = self.P_ST_rbr_rocker
    a56 = A(a55).T
    a57 = self.Mbar_ground_jcr_rocker_ch[:,0:1]
    a58 = B(a52,a53)
    a59 = B(a55,a53)
    a60 = self.Mbar_ground_jcr_rocker_ch[:,1:2]
    a61 = self.Pd_ST_rbl_rocker
    a62 = self.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
    a63 = a62.T
    a64 = self.P_ST_rbl_rocker
    a65 = A(a64).T
    a66 = self.Mbar_ground_jcl_rocker_ch[:,0:1]
    a67 = B(a61,a62)
    a68 = B(a64,a62)
    a69 = self.Mbar_ground_jcl_rocker_ch[:,1:2]
    a70 = self.Pd_SU_rbr_upright
    a71 = self.Pd_SU_rbl_upright
    a72 = self.Pd_SU_rbr_lower_strut
    a73 = self.Mbar_SU_rbr_lower_strut_jcr_strut_lca[:,0:1]
    a74 = self.P_SU_rbr_lower_strut
    a75 = A(a74).T
    a76 = self.Mbar_SU_rbr_lca_jcr_strut_lca[:,0:1]
    a77 = a24.T
    a78 = self.Pd_SU_rbl_lower_strut
    a79 = self.Mbar_SU_rbl_lca_jcl_strut_lca[:,0:1]
    a80 = self.Mbar_SU_rbl_lower_strut_jcl_strut_lca[:,0:1]
    a81 = self.P_SU_rbl_lower_strut
    a82 = A(a81).T
    a83 = a33.T
    a84 = self.Pd_SU_rbr_tie_rod
    a85 = self.Pd_SU_rbr_hub
    a86 = self.Mbar_SU_rbr_hub_jcr_hub_bearing[:,2:3]
    a87 = a86.T
    a88 = self.P_SU_rbr_hub
    a89 = A(a88).T
    a90 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,0:1]
    a91 = self.P_SU_rbr_upright
    a92 = A(a91).T
    a93 = B(a85,a86)
    a94 = a70.T
    a95 = B(a88,a86)
    a96 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,1:2]
    a97 = F_jcr_hub_bearing(t)
    a98 = self.Mbar_SU_rbr_hub_jcr_hub_bearing[:,0:1]
    a99 = cos(a97)
    a100 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,1:2]
    a101 = sin(a97)
    a102 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,0:1]
    a103 = self.Pd_SU_rbl_tie_rod
    a104 = self.Pd_SU_rbl_hub
    a105 = self.Mbar_SU_rbl_hub_jcl_hub_bearing[:,2:3]
    a106 = a105.T
    a107 = self.P_SU_rbl_hub
    a108 = A(a107).T
    a109 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,0:1]
    a110 = self.P_SU_rbl_upright
    a111 = A(a110).T
    a112 = B(a104,a105)
    a113 = a71.T
    a114 = B(a107,a105)
    a115 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,1:2]
    a116 = F_jcl_hub_bearing(t)
    a117 = self.Mbar_SU_rbl_hub_jcl_hub_bearing[:,0:1]
    a118 = cos(a116)
    a119 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,1:2]
    a120 = sin(a116)
    a121 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,0:1]
    a122 = self.Mbar_SU_rbr_upper_strut_jcr_strut[:,0:1]
    a123 = a122.T
    a124 = self.Mbar_SU_rbr_lower_strut_jcr_strut[:,2:3]
    a125 = B(a72,a124)
    a126 = a124.T
    a127 = B(a42,a122)
    a128 = a42.T
    a129 = B(a45,a122).T
    a130 = B(a74,a124)
    a131 = self.Mbar_SU_rbr_upper_strut_jcr_strut[:,1:2]
    a132 = a131.T
    a133 = B(a42,a131)
    a134 = B(a45,a131).T
    a135 = self.ubar_SU_rbr_lower_strut_jcr_strut
    a136 = self.ubar_SU_rbr_upper_strut_jcr_strut
    a137 = (multi_dot([B(a72,a135),a72]) + -1.0*multi_dot([B(a42,a136),a42]))
    a138 = (self.Rd_SU_rbr_upper_strut + -1.0*self.Rd_SU_rbr_lower_strut + multi_dot([B(a74,a135),a72]) + multi_dot([B(a45,a136),a42]))
    a139 = (self.R_SU_rbr_upper_strut.T + -1.0*self.R_SU_rbr_lower_strut.T + multi_dot([a136.T,a46]) + -1.0*multi_dot([a135.T,a75]))
    a140 = self.Mbar_SU_rbl_lower_strut_jcl_strut[:,2:3]
    a141 = a140.T
    a142 = self.Mbar_SU_rbl_upper_strut_jcl_strut[:,0:1]
    a143 = B(a47,a142)
    a144 = a142.T
    a145 = B(a78,a140)
    a146 = a47.T
    a147 = B(a50,a142).T
    a148 = B(a81,a140)
    a149 = self.Mbar_SU_rbl_upper_strut_jcl_strut[:,1:2]
    a150 = B(a47,a149)
    a151 = a149.T
    a152 = B(a50,a149).T
    a153 = self.ubar_SU_rbl_lower_strut_jcl_strut
    a154 = self.ubar_SU_rbl_upper_strut_jcl_strut
    a155 = (multi_dot([B(a78,a153),a78]) + -1.0*multi_dot([B(a47,a154),a47]))
    a156 = (self.Rd_SU_rbl_upper_strut + -1.0*self.Rd_SU_rbl_lower_strut + multi_dot([B(a81,a153),a78]) + multi_dot([B(a50,a154),a47]))
    a157 = (self.R_SU_rbl_upper_strut.T + -1.0*self.R_SU_rbl_lower_strut.T + multi_dot([a154.T,a51]) + -1.0*multi_dot([a153.T,a82]))
    a158 = self.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]
    a159 = self.Mbar_SU_rbr_tie_rod_jcr_tie_steering[:,0:1]
    a160 = self.P_SU_rbr_tie_rod
    a161 = a84.T
    a162 = self.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]
    a163 = self.Mbar_SU_rbl_tie_rod_jcl_tie_steering[:,0:1]
    a164 = self.P_SU_rbl_tie_rod
    a165 = a103.T
    a166 = self.Pd_ST_rbs_coupler
    a167 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1]
    a168 = self.P_ST_rbs_coupler
    a169 = A(a168).T
    a170 = self.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
    a171 = B(a61,a170)
    a172 = a170.T
    a173 = a166.T
    a174 = B(a64,a170)
    a175 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2]

    self.acc_level_data_blocks = [(a0 + derivative(F,t,0.1,2)*-1.0*a1),(a0 + derivative(F,t,0.1,2)*-1.0*a1),(multi_dot([B(a2,self.ubar_ground_jcr_uca_chassis),a2]) + -1.0*multi_dot([B(a3,self.ubar_SU_rbr_uca_jcr_uca_chassis),a3])),(multi_dot([a5,a7,B(a2,a8),a2]) + multi_dot([a8.T,a10,a11,a3]) + 2.0*multi_dot([a12,B(a9,a8).T,a13,a3])),(multi_dot([a5,a7,B(a2,a14),a2]) + multi_dot([a14.T,a10,a11,a3]) + 2.0*multi_dot([a12,B(a9,a14).T,a13,a3])),(multi_dot([B(a2,self.ubar_ground_jcl_uca_chassis),a2]) + -1.0*multi_dot([B(a15,self.ubar_SU_rbl_uca_jcl_uca_chassis),a15])),(multi_dot([a16.T,a10,a18,a15]) + multi_dot([a19,a21,B(a2,a16),a2]) + 2.0*multi_dot([a12,B(a9,a16).T,a22,a15])),(multi_dot([a23.T,a10,a18,a15]) + multi_dot([a19,a21,B(a2,a23),a2]) + 2.0*multi_dot([a12,B(a9,a23).T,a22,a15])),(multi_dot([B(a2,self.ubar_ground_jcr_lca_chassis),a2]) + -1.0*multi_dot([B(a24,self.ubar_SU_rbr_lca_jcr_lca_chassis),a24])),(multi_dot([a25.T,a10,a27,a24]) + multi_dot([a28,a30,B(a2,a25),a2]) + 2.0*multi_dot([a12,B(a9,a25).T,a31,a24])),(multi_dot([a32.T,a10,a27,a24]) + multi_dot([a28,a30,B(a2,a32),a2]) + 2.0*multi_dot([a12,B(a9,a32).T,a31,a24])),(multi_dot([B(a2,self.ubar_ground_jcl_lca_chassis),a2]) + -1.0*multi_dot([B(a33,self.ubar_SU_rbl_lca_jcl_lca_chassis),a33])),(multi_dot([a35,a37,B(a2,a38),a2]) + multi_dot([a38.T,a10,a39,a33]) + 2.0*multi_dot([a12,B(a9,a38).T,a40,a33])),(multi_dot([a35,a37,B(a2,a41),a2]) + multi_dot([a41.T,a10,a39,a33]) + 2.0*multi_dot([a12,B(a9,a41).T,a40,a33])),(multi_dot([B(a2,self.ubar_ground_jcr_strut_chassis),a2]) + -1.0*multi_dot([B(a42,self.ubar_SU_rbr_upper_strut_jcr_strut_chassis),a42])),(multi_dot([a43.T,a10,B(a42,a44),a42]) + multi_dot([a44.T,a46,B(a2,a43),a2]) + 2.0*multi_dot([a12,B(a9,a43).T,B(a45,a44),a42])),(multi_dot([B(a2,self.ubar_ground_jcl_strut_chassis),a2]) + -1.0*multi_dot([B(a47,self.ubar_SU_rbl_upper_strut_jcl_strut_chassis),a47])),(multi_dot([a48.T,a10,B(a47,a49),a47]) + multi_dot([a49.T,a51,B(a2,a48),a2]) + 2.0*multi_dot([a12,B(a9,a48).T,B(a50,a49),a47])),(multi_dot([B(a2,self.ubar_ground_jcr_rocker_ch),a2]) + -1.0*multi_dot([B(a52,self.ubar_ST_rbr_rocker_jcr_rocker_ch),a52])),(multi_dot([a54,a56,B(a2,a57),a2]) + multi_dot([a57.T,a10,a58,a52]) + 2.0*multi_dot([a12,B(a9,a57).T,a59,a52])),(multi_dot([a54,a56,B(a2,a60),a2]) + multi_dot([a60.T,a10,a58,a52]) + 2.0*multi_dot([a12,B(a9,a60).T,a59,a52])),(multi_dot([B(a2,self.ubar_ground_jcl_rocker_ch),a2]) + -1.0*multi_dot([B(a61,self.ubar_ST_rbl_rocker_jcl_rocker_ch),a61])),(multi_dot([a63,a65,B(a2,a66),a2]) + multi_dot([a66.T,a10,a67,a61]) + 2.0*multi_dot([a12,B(a9,a66).T,a68,a61])),(multi_dot([a63,a65,B(a2,a69),a2]) + multi_dot([a69.T,a10,a67,a61]) + 2.0*multi_dot([a12,B(a9,a69).T,a68,a61])),(multi_dot([B(a3,self.ubar_SU_rbr_uca_jcr_uca_upright),a3]) + -1.0*multi_dot([B(a70,self.ubar_SU_rbr_upright_jcr_uca_upright),a70])),(multi_dot([B(a15,self.ubar_SU_rbl_uca_jcl_uca_upright),a15]) + -1.0*multi_dot([B(a71,self.ubar_SU_rbl_upright_jcl_uca_upright),a71])),(multi_dot([B(a24,self.ubar_SU_rbr_lca_jcr_lca_upright),a24]) + -1.0*multi_dot([B(a70,self.ubar_SU_rbr_upright_jcr_lca_upright),a70])),(multi_dot([B(a24,self.ubar_SU_rbr_lca_jcr_strut_lca),a24]) + -1.0*multi_dot([B(a72,self.ubar_SU_rbr_lower_strut_jcr_strut_lca),a72])),(multi_dot([a73.T,a75,B(a24,a76),a24]) + multi_dot([a76.T,a30,B(a72,a73),a72]) + 2.0*multi_dot([a77,B(a29,a76).T,B(a74,a73),a72])),(multi_dot([B(a33,self.ubar_SU_rbl_lca_jcl_lca_upright),a33]) + -1.0*multi_dot([B(a71,self.ubar_SU_rbl_upright_jcl_lca_upright),a71])),(multi_dot([B(a33,self.ubar_SU_rbl_lca_jcl_strut_lca),a33]) + -1.0*multi_dot([B(a78,self.ubar_SU_rbl_lower_strut_jcl_strut_lca),a78])),(multi_dot([a79.T,a37,B(a78,a80),a78]) + multi_dot([a80.T,a82,B(a33,a79),a33]) + 2.0*multi_dot([a83,B(a36,a79).T,B(a81,a80),a78])),(multi_dot([B(a70,self.ubar_SU_rbr_upright_jcr_tie_upright),a70]) + -1.0*multi_dot([B(a84,self.ubar_SU_rbr_tie_rod_jcr_tie_upright),a84])),(multi_dot([B(a70,self.ubar_SU_rbr_upright_jcr_hub_bearing),a70]) + -1.0*multi_dot([B(a85,self.ubar_SU_rbr_hub_jcr_hub_bearing),a85])),(multi_dot([a87,a89,B(a70,a90),a70]) + multi_dot([a90.T,a92,a93,a85]) + 2.0*multi_dot([a94,B(a91,a90).T,a95,a85])),(multi_dot([a87,a89,B(a70,a96),a70]) + multi_dot([a96.T,a92,a93,a85]) + 2.0*multi_dot([a94,B(a91,a96).T,a95,a85])),(derivative(F,t,0.1,2)*-1.0*a1 + multi_dot([a98.T,a89,(a99*B(a70,a100) + a101*-1.0*B(a70,a102)),a70]) + multi_dot([(a99*multi_dot([a100.T,a92]) + a101*-1.0*multi_dot([a102.T,a92])),B(a85,a98),a85]) + 2.0*multi_dot([((a99*multi_dot([B(a91,a100),a70])).T + transpose(a101)*-1.0*multi_dot([a94,B(a91,a102).T])),B(a88,a98),a85])),(multi_dot([B(a71,self.ubar_SU_rbl_upright_jcl_tie_upright),a71]) + -1.0*multi_dot([B(a103,self.ubar_SU_rbl_tie_rod_jcl_tie_upright),a103])),(multi_dot([B(a71,self.ubar_SU_rbl_upright_jcl_hub_bearing),a71]) + -1.0*multi_dot([B(a104,self.ubar_SU_rbl_hub_jcl_hub_bearing),a104])),(multi_dot([a106,a108,B(a71,a109),a71]) + multi_dot([a109.T,a111,a112,a104]) + 2.0*multi_dot([a113,B(a110,a109).T,a114,a104])),(multi_dot([a106,a108,B(a71,a115),a71]) + multi_dot([a115.T,a111,a112,a104]) + 2.0*multi_dot([a113,B(a110,a115).T,a114,a104])),(derivative(F,t,0.1,2)*-1.0*a1 + multi_dot([a117.T,a108,(a118*B(a71,a119) + a120*-1.0*B(a71,a121)),a71]) + multi_dot([(a118*multi_dot([a119.T,a111]) + a120*-1.0*multi_dot([a121.T,a111])),B(a104,a117),a104]) + 2.0*multi_dot([((a118*multi_dot([B(a110,a119),a71])).T + transpose(a120)*-1.0*multi_dot([a113,B(a110,a121).T])),B(a107,a117),a104])),(multi_dot([a123,a46,a125,a72]) + multi_dot([a126,a75,a127,a42]) + 2.0*multi_dot([a128,a129,a130,a72])),(multi_dot([a132,a46,a125,a72]) + multi_dot([a126,a75,a133,a42]) + 2.0*multi_dot([a128,a134,a130,a72])),(multi_dot([a123,a46,a137]) + 2.0*multi_dot([a128,a129,a138]) + multi_dot([a139,a127,a42])),(multi_dot([a132,a46,a137]) + 2.0*multi_dot([a128,a134,a138]) + multi_dot([a139,a133,a42])),(multi_dot([a141,a82,a143,a47]) + multi_dot([a144,a51,a145,a78]) + 2.0*multi_dot([a146,a147,a148,a78])),(multi_dot([a141,a82,a150,a47]) + multi_dot([a151,a51,a145,a78]) + 2.0*multi_dot([a146,a152,a148,a78])),(multi_dot([a144,a51,a155]) + 2.0*multi_dot([a146,a147,a156]) + multi_dot([a157,a143,a47])),(multi_dot([a151,a51,a155]) + 2.0*multi_dot([a146,a152,a156]) + multi_dot([a157,a150,a47])),(multi_dot([B(a84,self.ubar_SU_rbr_tie_rod_jcr_tie_steering),a84]) + -1.0*multi_dot([B(a52,self.ubar_ST_rbr_rocker_jcr_tie_steering),a52])),(multi_dot([a158.T,a56,B(a84,a159),a84]) + multi_dot([a159.T,A(a160).T,B(a52,a158),a52]) + 2.0*multi_dot([a161,B(a160,a159).T,B(a55,a158),a52])),(multi_dot([B(a103,self.ubar_SU_rbl_tie_rod_jcl_tie_steering),a103]) + -1.0*multi_dot([B(a61,self.ubar_ST_rbl_rocker_jcl_tie_steering),a61])),(multi_dot([a162.T,a65,B(a103,a163),a103]) + multi_dot([a163.T,A(a164).T,B(a61,a162),a61]) + 2.0*multi_dot([a165,B(a164,a163).T,B(a64,a162),a61])),(multi_dot([B(a166,self.ubar_ST_rbs_coupler_jcs_rc_sph),a166]) + -1.0*multi_dot([B(a52,self.ubar_ST_rbr_rocker_jcs_rc_sph),a52])),(multi_dot([B(a166,self.ubar_ST_rbs_coupler_jcs_rc_uni),a166]) + -1.0*multi_dot([B(a61,self.ubar_ST_rbl_rocker_jcs_rc_uni),a61])),(multi_dot([a167.T,a169,a171,a61]) + multi_dot([a172,a65,B(a166,a167),a166]) + 2.0*multi_dot([a173,B(a168,a167).T,a174,a61])),(multi_dot([a175.T,a169,a171,a61]) + multi_dot([a172,a65,B(a166,a175),a166]) + 2.0*multi_dot([a173,B(a168,a175).T,a174,a61])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a3.T,a3]))**(1.0/2.0),2.0*(multi_dot([a15.T,a15]))**(1.0/2.0),2.0*(multi_dot([a77,a24]))**(1.0/2.0),2.0*(multi_dot([a83,a33]))**(1.0/2.0),2.0*(multi_dot([a94,a70]))**(1.0/2.0),2.0*(multi_dot([a113,a71]))**(1.0/2.0),2.0*(multi_dot([a128,a42]))**(1.0/2.0),2.0*(multi_dot([a146,a47]))**(1.0/2.0),2.0*(multi_dot([a72.T,a72]))**(1.0/2.0),2.0*(multi_dot([a78.T,a78]))**(1.0/2.0),2.0*(multi_dot([a161,a84]))**(1.0/2.0),2.0*(multi_dot([a165,a103]))**(1.0/2.0),2.0*(multi_dot([a85.T,a85]))**(1.0/2.0),2.0*(multi_dot([a104.T,a104]))**(1.0/2.0),2.0*(multi_dot([a173,a166]))**(1.0/2.0),2.0*(multi_dot([a52.T,a52]))**(1.0/2.0),2.0*(multi_dot([a61.T,a61]))**(1.0/2.0)]

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

    j0 = np.zeros((1,4),dtype=np.float64)
    j1 = np.zeros((1,3),dtype=np.float64)
    j2 = np.eye(3,dtype=np.float64)
    j3 = self.P_ground
    j4 = self.Mbar_SU_rbr_uca_jcr_uca_chassis[:,2:3]
    j5 = j4.T
    j6 = self.P_SU_rbr_uca
    j7 = A(j6).T
    j8 = self.Mbar_ground_jcr_uca_chassis[:,0:1]
    j9 = self.Mbar_ground_jcr_uca_chassis[:,1:2]
    j10 = -1.0*j2
    j11 = A(j3).T
    j12 = B(j6,j4)
    j13 = self.Mbar_SU_rbl_uca_jcl_uca_chassis[:,2:3]
    j14 = j13.T
    j15 = self.P_SU_rbl_uca
    j16 = A(j15).T
    j17 = self.Mbar_ground_jcl_uca_chassis[:,0:1]
    j18 = self.Mbar_ground_jcl_uca_chassis[:,1:2]
    j19 = B(j15,j13)
    j20 = self.Mbar_SU_rbr_lca_jcr_lca_chassis[:,2:3]
    j21 = j20.T
    j22 = self.P_SU_rbr_lca
    j23 = A(j22).T
    j24 = self.Mbar_ground_jcr_lca_chassis[:,0:1]
    j25 = self.Mbar_ground_jcr_lca_chassis[:,1:2]
    j26 = B(j22,j20)
    j27 = self.Mbar_SU_rbl_lca_jcl_lca_chassis[:,2:3]
    j28 = j27.T
    j29 = self.P_SU_rbl_lca
    j30 = A(j29).T
    j31 = self.Mbar_ground_jcl_lca_chassis[:,0:1]
    j32 = self.Mbar_ground_jcl_lca_chassis[:,1:2]
    j33 = B(j29,j27)
    j34 = self.Mbar_SU_rbr_upper_strut_jcr_strut_chassis[:,0:1]
    j35 = self.P_SU_rbr_upper_strut
    j36 = A(j35).T
    j37 = self.Mbar_ground_jcr_strut_chassis[:,0:1]
    j38 = self.Mbar_SU_rbl_upper_strut_jcl_strut_chassis[:,0:1]
    j39 = self.P_SU_rbl_upper_strut
    j40 = A(j39).T
    j41 = self.Mbar_ground_jcl_strut_chassis[:,0:1]
    j42 = self.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
    j43 = j42.T
    j44 = self.P_ST_rbr_rocker
    j45 = A(j44).T
    j46 = self.Mbar_ground_jcr_rocker_ch[:,0:1]
    j47 = self.Mbar_ground_jcr_rocker_ch[:,1:2]
    j48 = B(j44,j42)
    j49 = self.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
    j50 = j49.T
    j51 = self.P_ST_rbl_rocker
    j52 = A(j51).T
    j53 = self.Mbar_ground_jcl_rocker_ch[:,0:1]
    j54 = self.Mbar_ground_jcl_rocker_ch[:,1:2]
    j55 = B(j51,j49)
    j56 = self.P_SU_rbr_upright
    j57 = self.P_SU_rbl_upright
    j58 = self.Mbar_SU_rbr_lower_strut_jcr_strut_lca[:,0:1]
    j59 = self.P_SU_rbr_lower_strut
    j60 = A(j59).T
    j61 = self.Mbar_SU_rbr_lca_jcr_strut_lca[:,0:1]
    j62 = self.Mbar_SU_rbl_lower_strut_jcl_strut_lca[:,0:1]
    j63 = self.P_SU_rbl_lower_strut
    j64 = A(j63).T
    j65 = self.Mbar_SU_rbl_lca_jcl_strut_lca[:,0:1]
    j66 = self.P_SU_rbr_tie_rod
    j67 = self.Mbar_SU_rbr_hub_jcr_hub_bearing[:,2:3]
    j68 = j67.T
    j69 = self.P_SU_rbr_hub
    j70 = A(j69).T
    j71 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,0:1]
    j72 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,1:2]
    j73 = A(j56).T
    j74 = B(j69,j67)
    j75 = self.Mbar_SU_rbr_hub_jcr_hub_bearing[:,0:1]
    j76 = F_jcr_hub_bearing(t)
    j77 = cos(j76)
    j78 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,1:2]
    j79 = self.Mbar_SU_rbr_upright_jcr_hub_bearing[:,0:1]
    j80 = self.P_SU_rbl_tie_rod
    j81 = self.Mbar_SU_rbl_hub_jcl_hub_bearing[:,2:3]
    j82 = j81.T
    j83 = self.P_SU_rbl_hub
    j84 = A(j83).T
    j85 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,0:1]
    j86 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,1:2]
    j87 = A(j57).T
    j88 = B(j83,j81)
    j89 = self.Mbar_SU_rbl_hub_jcl_hub_bearing[:,0:1]
    j90 = F_jcl_hub_bearing(t)
    j91 = cos(j90)
    j92 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,1:2]
    j93 = self.Mbar_SU_rbl_upright_jcl_hub_bearing[:,0:1]
    j94 = self.Mbar_SU_rbr_lower_strut_jcr_strut[:,2:3]
    j95 = j94.T
    j96 = self.Mbar_SU_rbr_upper_strut_jcr_strut[:,0:1]
    j97 = B(j35,j96)
    j98 = self.Mbar_SU_rbr_upper_strut_jcr_strut[:,1:2]
    j99 = B(j35,j98)
    j100 = j96.T
    j101 = multi_dot([j100,j36])
    j102 = self.ubar_SU_rbr_upper_strut_jcr_strut
    j103 = B(j35,j102)
    j104 = self.ubar_SU_rbr_lower_strut_jcr_strut
    j105 = (self.R_SU_rbr_upper_strut.T + -1.0*self.R_SU_rbr_lower_strut.T + multi_dot([j102.T,j36]) + -1.0*multi_dot([j104.T,j60]))
    j106 = j98.T
    j107 = multi_dot([j106,j36])
    j108 = B(j59,j94)
    j109 = B(j59,j104)
    j110 = self.Mbar_SU_rbl_lower_strut_jcl_strut[:,2:3]
    j111 = j110.T
    j112 = self.Mbar_SU_rbl_upper_strut_jcl_strut[:,0:1]
    j113 = B(j39,j112)
    j114 = self.Mbar_SU_rbl_upper_strut_jcl_strut[:,1:2]
    j115 = B(j39,j114)
    j116 = j112.T
    j117 = multi_dot([j116,j40])
    j118 = self.ubar_SU_rbl_upper_strut_jcl_strut
    j119 = B(j39,j118)
    j120 = self.ubar_SU_rbl_lower_strut_jcl_strut
    j121 = (self.R_SU_rbl_upper_strut.T + -1.0*self.R_SU_rbl_lower_strut.T + multi_dot([j118.T,j40]) + -1.0*multi_dot([j120.T,j64]))
    j122 = j114.T
    j123 = multi_dot([j122,j40])
    j124 = B(j63,j110)
    j125 = B(j63,j120)
    j126 = self.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]
    j127 = self.Mbar_SU_rbr_tie_rod_jcr_tie_steering[:,0:1]
    j128 = self.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]
    j129 = self.Mbar_SU_rbl_tie_rod_jcl_tie_steering[:,0:1]
    j130 = self.P_ST_rbs_coupler
    j131 = self.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
    j132 = j131.T
    j133 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1]
    j134 = self.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2]
    j135 = A(j130).T
    j136 = B(j51,j131)

    self.jacobian_data_blocks = [J_mcr_zact,j0,j1,j0,J_mcl_zact,j0,j1,j0,j2,B(j3,self.ubar_ground_jcr_uca_chassis),j10,-1.0*B(j6,self.ubar_SU_rbr_uca_jcr_uca_chassis),j1,multi_dot([j5,j7,B(j3,j8)]),j1,multi_dot([j8.T,j11,j12]),j1,multi_dot([j5,j7,B(j3,j9)]),j1,multi_dot([j9.T,j11,j12]),j2,B(j3,self.ubar_ground_jcl_uca_chassis),j10,-1.0*B(j15,self.ubar_SU_rbl_uca_jcl_uca_chassis),j1,multi_dot([j14,j16,B(j3,j17)]),j1,multi_dot([j17.T,j11,j19]),j1,multi_dot([j14,j16,B(j3,j18)]),j1,multi_dot([j18.T,j11,j19]),j2,B(j3,self.ubar_ground_jcr_lca_chassis),j10,-1.0*B(j22,self.ubar_SU_rbr_lca_jcr_lca_chassis),j1,multi_dot([j21,j23,B(j3,j24)]),j1,multi_dot([j24.T,j11,j26]),j1,multi_dot([j21,j23,B(j3,j25)]),j1,multi_dot([j25.T,j11,j26]),j2,B(j3,self.ubar_ground_jcl_lca_chassis),j10,-1.0*B(j29,self.ubar_SU_rbl_lca_jcl_lca_chassis),j1,multi_dot([j28,j30,B(j3,j31)]),j1,multi_dot([j31.T,j11,j33]),j1,multi_dot([j28,j30,B(j3,j32)]),j1,multi_dot([j32.T,j11,j33]),j2,B(j3,self.ubar_ground_jcr_strut_chassis),j10,-1.0*B(j35,self.ubar_SU_rbr_upper_strut_jcr_strut_chassis),j1,multi_dot([j34.T,j36,B(j3,j37)]),j1,multi_dot([j37.T,j11,B(j35,j34)]),j2,B(j3,self.ubar_ground_jcl_strut_chassis),j10,-1.0*B(j39,self.ubar_SU_rbl_upper_strut_jcl_strut_chassis),j1,multi_dot([j38.T,j40,B(j3,j41)]),j1,multi_dot([j41.T,j11,B(j39,j38)]),j2,B(j3,self.ubar_ground_jcr_rocker_ch),j10,-1.0*B(j44,self.ubar_ST_rbr_rocker_jcr_rocker_ch),j1,multi_dot([j43,j45,B(j3,j46)]),j1,multi_dot([j46.T,j11,j48]),j1,multi_dot([j43,j45,B(j3,j47)]),j1,multi_dot([j47.T,j11,j48]),j2,B(j3,self.ubar_ground_jcl_rocker_ch),j10,-1.0*B(j51,self.ubar_ST_rbl_rocker_jcl_rocker_ch),j1,multi_dot([j50,j52,B(j3,j53)]),j1,multi_dot([j53.T,j11,j55]),j1,multi_dot([j50,j52,B(j3,j54)]),j1,multi_dot([j54.T,j11,j55]),j2,B(j6,self.ubar_SU_rbr_uca_jcr_uca_upright),j10,-1.0*B(j56,self.ubar_SU_rbr_upright_jcr_uca_upright),j2,B(j15,self.ubar_SU_rbl_uca_jcl_uca_upright),j10,-1.0*B(j57,self.ubar_SU_rbl_upright_jcl_uca_upright),j2,B(j22,self.ubar_SU_rbr_lca_jcr_lca_upright),j10,-1.0*B(j56,self.ubar_SU_rbr_upright_jcr_lca_upright),j2,B(j22,self.ubar_SU_rbr_lca_jcr_strut_lca),j10,-1.0*B(j59,self.ubar_SU_rbr_lower_strut_jcr_strut_lca),j1,multi_dot([j58.T,j60,B(j22,j61)]),j1,multi_dot([j61.T,j23,B(j59,j58)]),j2,B(j29,self.ubar_SU_rbl_lca_jcl_lca_upright),j10,-1.0*B(j57,self.ubar_SU_rbl_upright_jcl_lca_upright),j2,B(j29,self.ubar_SU_rbl_lca_jcl_strut_lca),j10,-1.0*B(j63,self.ubar_SU_rbl_lower_strut_jcl_strut_lca),j1,multi_dot([j62.T,j64,B(j29,j65)]),j1,multi_dot([j65.T,j30,B(j63,j62)]),j2,B(j56,self.ubar_SU_rbr_upright_jcr_tie_upright),j10,-1.0*B(j66,self.ubar_SU_rbr_tie_rod_jcr_tie_upright),j2,B(j56,self.ubar_SU_rbr_upright_jcr_hub_bearing),j10,-1.0*B(j69,self.ubar_SU_rbr_hub_jcr_hub_bearing),j1,multi_dot([j68,j70,B(j56,j71)]),j1,multi_dot([j71.T,j73,j74]),j1,multi_dot([j68,j70,B(j56,j72)]),j1,multi_dot([j72.T,j73,j74]),j1,multi_dot([j75.T,j70,(j77*B(j56,j78) + sin(j76)*-1.0*B(j56,j79))]),j1,multi_dot([(j77*multi_dot([j78.T,j73]) + sin(j76)*-1.0*multi_dot([j79.T,j73])),B(j69,j75)]),j2,B(j57,self.ubar_SU_rbl_upright_jcl_tie_upright),j10,-1.0*B(j80,self.ubar_SU_rbl_tie_rod_jcl_tie_upright),j2,B(j57,self.ubar_SU_rbl_upright_jcl_hub_bearing),j10,-1.0*B(j83,self.ubar_SU_rbl_hub_jcl_hub_bearing),j1,multi_dot([j82,j84,B(j57,j85)]),j1,multi_dot([j85.T,j87,j88]),j1,multi_dot([j82,j84,B(j57,j86)]),j1,multi_dot([j86.T,j87,j88]),j1,multi_dot([j89.T,j84,(j91*B(j57,j92) + sin(j90)*-1.0*B(j57,j93))]),j1,multi_dot([(j91*multi_dot([j92.T,j87]) + sin(j90)*-1.0*multi_dot([j93.T,j87])),B(j83,j89)]),j1,multi_dot([j95,j60,j97]),j1,multi_dot([j100,j36,j108]),j1,multi_dot([j95,j60,j99]),j1,multi_dot([j106,j36,j108]),-1.0*j101,(-1.0*multi_dot([j100,j36,j103]) + multi_dot([j105,j97])),j101,multi_dot([j100,j36,j109]),-1.0*j107,(-1.0*multi_dot([j106,j36,j103]) + multi_dot([j105,j99])),j107,multi_dot([j106,j36,j109]),j1,multi_dot([j111,j64,j113]),j1,multi_dot([j116,j40,j124]),j1,multi_dot([j111,j64,j115]),j1,multi_dot([j122,j40,j124]),-1.0*j117,(-1.0*multi_dot([j116,j40,j119]) + multi_dot([j121,j113])),j117,multi_dot([j116,j40,j125]),-1.0*j123,(-1.0*multi_dot([j122,j40,j119]) + multi_dot([j121,j115])),j123,multi_dot([j122,j40,j125]),j2,B(j66,self.ubar_SU_rbr_tie_rod_jcr_tie_steering),j10,-1.0*B(j44,self.ubar_ST_rbr_rocker_jcr_tie_steering),j1,multi_dot([j126.T,j45,B(j66,j127)]),j1,multi_dot([j127.T,A(j66).T,B(j44,j126)]),j2,B(j80,self.ubar_SU_rbl_tie_rod_jcl_tie_steering),j10,-1.0*B(j51,self.ubar_ST_rbl_rocker_jcl_tie_steering),j1,multi_dot([j128.T,j52,B(j80,j129)]),j1,multi_dot([j129.T,A(j80).T,B(j51,j128)]),j2,B(j130,self.ubar_ST_rbs_coupler_jcs_rc_sph),j10,-1.0*B(j44,self.ubar_ST_rbr_rocker_jcs_rc_sph),j2,B(j130,self.ubar_ST_rbs_coupler_jcs_rc_uni),j10,-1.0*B(j51,self.ubar_ST_rbl_rocker_jcs_rc_uni),j1,multi_dot([j132,j52,B(j130,j133)]),j1,multi_dot([j133.T,j135,j136]),j1,multi_dot([j132,j52,B(j130,j134)]),j1,multi_dot([j134.T,j135,j136]),j2,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j6.T,2.0*j15.T,2.0*j22.T,2.0*j29.T,2.0*j56.T,2.0*j57.T,2.0*j35.T,2.0*j39.T,2.0*j59.T,2.0*j63.T,2.0*j66.T,2.0*j80.T,2.0*j69.T,2.0*j83.T,2.0*j130.T,2.0*j44.T,2.0*j51.T]

    self.jacobian_rows_explicit = []
    self.jacobian_cols_explicit = []
    self.jacobian_data_explicit = []

    sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
                     self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)

    self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))



