
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin
from source.solvers.python_solver import solver


class inputs(object):

    def __init__(self):
        self.F_mcr_zact = lambda t : 0
        self.J_mcr_zact = np.array([[0, 0, 1]],dtype=np.float64)
        self.F_mcl_zact = lambda t : 0
        self.J_mcl_zact = np.array([[0, 0, 1]],dtype=np.float64)
        
        self.pt_jcr_uca_chassis = np.array([[0], [ 294], [180]],dtype=np.float64)
        self.ax_jcr_uca_chassis = np.array([[1], [0], [0]],dtype=np.float64)
        self.pt_jcl_uca_chassis = np.array([[0], [-294], [180]],dtype=np.float64)
        self.ax_jcl_uca_chassis = np.array([[1], [0], [0]],dtype=np.float64)
        
        self.pt_jcr_lca_chassis = np.array([[0], [ 245], [-106]],dtype=np.float64)
        self.ax_jcr_lca_chassis = np.array([[1], [0], [0]],dtype=np.float64)
        self.pt_jcl_lca_chassis = np.array([[0], [-245], [-106]],dtype=np.float64)
        self.ax_jcl_lca_chassis = np.array([[1], [0], [0]],dtype=np.float64)
        
        self.pt_jcr_strut_chassis = np.array([[-165], [534], [639]],dtype=np.float64)
        self.ax_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [-1]],dtype=np.float64)
        self.pt_jcl_strut_chassis = np.array([[-165], [-534], [639]],dtype=np.float64)
        self.ax_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_chassis = np.array([[0], [0], [-1]],dtype=np.float64)
       
        self.pt_jcl_tie_steering = np.array([[402], [-267], [108]],dtype=np.float64)
        self.ax_jcl_tie_steering = np.array([[0], [1], [0]],dtype=np.float64)
        self.ax2_jcl_tie_steering = np.array([[0], [-1], [0]],dtype=np.float64)
        self.pt_jcr_tie_steering = np.array([[402], [267], [108]],dtype=np.float64)
        self.ax_jcr_tie_steering = np.array([[0], [1], [0]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [-1], [0]],dtype=np.float64)
        
        self.pt_jcr_rocker_ch = np.array([[500], [ 722], [187]],dtype=np.float64)
        self.ax_jcr_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_rocker_ch = np.array([[500], [-722], [187]],dtype=np.float64)
        self.ax_jcl_rocker_ch = np.array([[0], [0], [1]],dtype=np.float64)
        self.F_jcr_rocker_ch = lambda t : 0
                
        self.pt_jcr_uca_upright = np.array([[0], [ 722], [187]],dtype=np.float64)
        self.ax_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_uca_upright = np.array([[0], [-722], [187]],dtype=np.float64)
        self.ax_jcl_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
       
        self.pt_jcr_lca_upright = np.array([[0], [ 776], [-181]],dtype=np.float64)
        self.ax_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_lca_upright = np.array([[0], [-776], [-181]],dtype=np.float64)
        self.ax_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)

        self.pt_jcr_strut_lca = np.array([[-165], [ 534], [-79]],dtype=np.float64)
        self.ax_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [-1]],dtype=np.float64)
        self.pt_jcl_strut_lca = np.array([[-165], [-534], [-79]],dtype=np.float64)
        self.ax_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [-1]],dtype=np.float64)
        
        self.pt_jcr_tie_upright = np.array([[399], [ 720], [110]],dtype=np.float64)
        self.ax_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_tie_upright = np.array([[399], [-720], [110]],dtype=np.float64)
        self.ax_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)

        self.pt_jcr_hub_bearing = np.array([[0], [ 1100], [0]],dtype=np.float64)
        self.ax_jcr_hub_bearing = np.array([[0], [1], [0]],dtype=np.float64)
        self.pt_jcl_hub_bearing = np.array([[0], [-1100], [0]],dtype=np.float64)
        self.ax_jcl_hub_bearing = np.array([[0], [1], [0]],dtype=np.float64)

        self.F_jcr_hub_bearing = lambda t : 0
        self.F_jcl_hub_bearing = lambda t : 0
        
        self.pt_jcr_strut = np.array([[-165], [ 534], [300]],dtype=np.float64)
        self.ax_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcl_strut = np.array([[-165], [-534], [300]],dtype=np.float64)
        self.ax_jcl_strut = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.pt_jcs_rc_sph = np.array([[200], [ 267], [187]],dtype=np.float64)
        self.ax_jcs_rc_sph = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt_jcs_rc_cyl = np.array([[200], [-267], [187]],dtype=np.float64)
        self.ax_jcs_rc_cyl = np.array([[0], [0], [1]],dtype=np.float64)
        
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_uca = np.array([[0], [ 428], [182]],dtype=np.float64)
        self.P_SU1_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_uca = np.array([[0], [-428], [182]],dtype=np.float64)
        self.P_SU1_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_lca = np.array([[0], [422], [-131]],dtype=np.float64)
        self.P_SU1_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_lca = np.array([[0], [-422], [-131]],dtype=np.float64)
        self.P_SU1_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_upright = np.array([[133], [739], [38]],dtype=np.float64)
        self.P_SU1_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_upright = np.array([[133], [-739], [38]],dtype=np.float64)
        self.P_SU1_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_upper_strut = np.array([[-165], [534], [459]],dtype=np.float64)
        self.P_SU1_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_upper_strut = np.array([[-165], [-534], [459]],dtype=np.float64)
        self.P_SU1_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_lower_strut = np.array([[-165], [534], [100]],dtype=np.float64)
        self.P_SU1_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_lower_strut = np.array([[-165], [-534], [100]],dtype=np.float64)
        self.P_SU1_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_tie_rod = np.array([[400], [493], [109]],dtype=np.float64)
        self.P_SU1_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_tie_rod = np.array([[400], [-493], [109]],dtype=np.float64)
        self.P_SU1_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_SU1_rbr_hub = np.array([[0], [1050], [0]],dtype=np.float64)
        self.P_SU1_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_SU1_rbl_hub = np.array([[0], [-1050], [0]],dtype=np.float64)
        self.P_SU1_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_SU1_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_SU1_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_ST_rbs_coupler = np.array([[200], [0], [187]],dtype=np.float64)
        self.P_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbs_coupler = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbs_coupler = np.array([[1], [0], [0], [0]],dtype=np.float64)
        
        self.R_ST_rbr_rocker = np.array([[300], [ 267], [187]],dtype=np.float64)
        self.P_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbr_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbr_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.R_ST_rbl_rocker = np.array([[300], [-267], [187]],dtype=np.float64)
        self.P_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ST_rbl_rocker = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ST_rbl_rocker = np.array([[1], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.P_ground).T
        c1 = self.pt_jcr_uca_chassis
        c2 = -1.0*multi_dot([c0,self.R_ground])
        c3 = A(self.P_SU1_rbr_uca).T
        c4 = -1.0*multi_dot([c3,self.R_SU1_rbr_uca])
        c5 = Triad(self.ax_jcr_uca_chassis,)
        c6 = self.pt_jcl_uca_chassis
        c7 = A(self.P_SU1_rbl_uca).T
        c8 = -1.0*multi_dot([c7,self.R_SU1_rbl_uca])
        c9 = Triad(self.ax_jcl_uca_chassis,)
        c10 = self.pt_jcr_lca_chassis
        c11 = A(self.P_SU1_rbr_lca).T
        c12 = -1.0*multi_dot([c11,self.R_SU1_rbr_lca])
        c13 = Triad(self.ax_jcr_lca_chassis,)
        c14 = self.pt_jcl_lca_chassis
        c15 = A(self.P_SU1_rbl_lca).T
        c16 = -1.0*multi_dot([c15,self.R_SU1_rbl_lca])
        c17 = Triad(self.ax_jcl_lca_chassis,)
        c18 = self.pt_jcr_strut_chassis
        c19 = A(self.P_SU1_rbr_upper_strut).T
        c20 = -1.0*multi_dot([c19,self.R_SU1_rbr_upper_strut])
        c21 = Triad(self.ax_jcr_strut_chassis,)
        c22 = self.pt_jcl_strut_chassis
        c23 = A(self.P_SU1_rbl_upper_strut).T
        c24 = -1.0*multi_dot([c23,self.R_SU1_rbl_upper_strut])
        c25 = Triad(self.ax_jcl_strut_chassis,)
        c26 = self.pt_jcl_tie_steering
        c27 = A(self.P_SU1_rbl_tie_rod).T
        c28 = -1.0*multi_dot([c27,self.R_SU1_rbl_tie_rod])
        c29 = Triad(self.ax_jcl_tie_steering,)
        c30 = self.pt_jcr_tie_steering
        c31 = A(self.P_SU1_rbr_tie_rod).T
        c32 = -1.0*multi_dot([c31,self.R_SU1_rbr_tie_rod])
        c33 = Triad(self.ax_jcr_tie_steering,)
        c34 = self.pt_jcr_rocker_ch
        c35 = A(self.P_ST_rbr_rocker).T
        c36 = -1.0*multi_dot([c35,self.R_ST_rbr_rocker])
        c37 = Triad(self.ax_jcr_rocker_ch,)
        c38 = self.pt_jcl_rocker_ch
        c39 = A(self.P_ST_rbl_rocker).T
        c40 = -1.0*multi_dot([c39,self.R_ST_rbl_rocker])
        c41 = Triad(self.ax_jcl_rocker_ch,)
        c42 = self.pt_jcr_uca_upright
        c43 = A(self.P_SU1_rbr_upright).T
        c44 = -1.0*multi_dot([c43,self.R_SU1_rbr_upright])
        c45 = Triad(self.ax_jcr_uca_upright,)
        c46 = self.pt_jcl_uca_upright
        c47 = A(self.P_SU1_rbl_upright).T
        c48 = -1.0*multi_dot([c47,self.R_SU1_rbl_upright])
        c49 = Triad(self.ax_jcl_uca_upright,)
        c50 = self.pt_jcr_lca_upright
        c51 = Triad(self.ax_jcr_lca_upright,)
        c52 = self.pt_jcr_strut_lca
        c53 = A(self.P_SU1_rbr_lower_strut).T
        c54 = -1.0*multi_dot([c53,self.R_SU1_rbr_lower_strut])
        c55 = Triad(self.ax_jcr_strut_lca,)
        c56 = self.pt_jcl_lca_upright
        c57 = Triad(self.ax_jcl_lca_upright,)
        c58 = self.pt_jcl_strut_lca
        c59 = A(self.P_SU1_rbl_lower_strut).T
        c60 = -1.0*multi_dot([c59,self.R_SU1_rbl_lower_strut])
        c61 = Triad(self.ax_jcl_strut_lca,)
        c62 = self.pt_jcr_tie_upright
        c63 = Triad(self.ax_jcr_tie_upright,)
        c64 = self.pt_jcr_hub_bearing
        c65 = A(self.P_SU1_rbr_hub).T
        c66 = Triad(self.ax_jcr_hub_bearing,)
        c67 = self.pt_jcl_tie_upright
        c68 = Triad(self.ax_jcl_tie_upright,)
        c69 = self.pt_jcl_hub_bearing
        c70 = A(self.P_SU1_rbl_hub).T
        c71 = Triad(self.ax_jcl_hub_bearing,)
        c72 = self.pt_jcr_strut
        c73 = Triad(self.ax_jcr_strut,)
        c74 = self.pt_jcl_strut
        c75 = Triad(self.ax_jcl_strut,)
        c76 = A(self.P_ST_rbs_coupler).T
        c77 = self.pt_jcs_rc_sph
        c78 = -1.0*multi_dot([c76,self.R_ST_rbs_coupler])
        c79 = Triad(self.ax_jcs_rc_sph,)
        c80 = self.pt_jcs_rc_cyl
        c81 = Triad(self.ax_jcs_rc_cyl,)

        self.ubar_ground_jcr_uca_chassis = (multi_dot([c0,c1]) + c2)
        self.ubar_SU1_rbr_uca_jcr_uca_chassis = (multi_dot([c3,c1]) + c4)
        self.Mbar_ground_jcr_uca_chassis = multi_dot([c0,c5])
        self.Mbar_SU1_rbr_uca_jcr_uca_chassis = multi_dot([c3,c5])
        self.ubar_ground_jcl_uca_chassis = (multi_dot([c0,c6]) + c2)
        self.ubar_SU1_rbl_uca_jcl_uca_chassis = (multi_dot([c7,c6]) + c8)
        self.Mbar_ground_jcl_uca_chassis = multi_dot([c0,c9])
        self.Mbar_SU1_rbl_uca_jcl_uca_chassis = multi_dot([c7,c9])
        self.ubar_ground_jcr_lca_chassis = (multi_dot([c0,c10]) + c2)
        self.ubar_SU1_rbr_lca_jcr_lca_chassis = (multi_dot([c11,c10]) + c12)
        self.Mbar_ground_jcr_lca_chassis = multi_dot([c0,c13])
        self.Mbar_SU1_rbr_lca_jcr_lca_chassis = multi_dot([c11,c13])
        self.ubar_ground_jcl_lca_chassis = (multi_dot([c0,c14]) + c2)
        self.ubar_SU1_rbl_lca_jcl_lca_chassis = (multi_dot([c15,c14]) + c16)
        self.Mbar_ground_jcl_lca_chassis = multi_dot([c0,c17])
        self.Mbar_SU1_rbl_lca_jcl_lca_chassis = multi_dot([c15,c17])
        self.ubar_ground_jcr_strut_chassis = (multi_dot([c0,c18]) + c2)
        self.ubar_SU1_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c19,c18]) + c20)
        self.Mbar_ground_jcr_strut_chassis = multi_dot([c0,c21])
        self.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis = multi_dot([c19,Triad(self.ax2_jcr_strut_chassis,c21[0:3,1:2])])
        self.ubar_ground_jcl_strut_chassis = (multi_dot([c0,c22]) + c2)
        self.ubar_SU1_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c23,c22]) + c24)
        self.Mbar_ground_jcl_strut_chassis = multi_dot([c0,c25])
        self.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis = multi_dot([c23,Triad(self.ax2_jcl_strut_chassis,c25[0:3,1:2])])
        self.ubar_ground_jcl_tie_steering = (multi_dot([c0,c26]) + c2)
        self.ubar_SU1_rbl_tie_rod_jcl_tie_steering = (multi_dot([c27,c26]) + c28)
        self.Mbar_ground_jcl_tie_steering = multi_dot([c0,c29])
        self.Mbar_SU1_rbl_tie_rod_jcl_tie_steering = multi_dot([c27,Triad(self.ax2_jcl_tie_steering,c29[0:3,1:2])])
        self.ubar_ground_jcr_tie_steering = (multi_dot([c0,c30]) + c2)
        self.ubar_SU1_rbr_tie_rod_jcr_tie_steering = (multi_dot([c31,c30]) + c32)
        self.Mbar_ground_jcr_tie_steering = multi_dot([c0,c33])
        self.Mbar_SU1_rbr_tie_rod_jcr_tie_steering = multi_dot([c31,Triad(self.ax2_jcr_tie_steering,c33[0:3,1:2])])
        self.ubar_ground_jcr_rocker_ch = (multi_dot([c0,c34]) + c2)
        self.ubar_ST_rbr_rocker_jcr_rocker_ch = (multi_dot([c35,c34]) + c36)
        self.Mbar_ground_jcr_rocker_ch = multi_dot([c0,c37])
        self.Mbar_ST_rbr_rocker_jcr_rocker_ch = multi_dot([c35,c37])
        self.ubar_ground_jcl_rocker_ch = (multi_dot([c0,c38]) + c2)
        self.ubar_ST_rbl_rocker_jcl_rocker_ch = (multi_dot([c39,c38]) + c40)
        self.Mbar_ground_jcl_rocker_ch = multi_dot([c0,c41])
        self.Mbar_ST_rbl_rocker_jcl_rocker_ch = multi_dot([c39,c41])
        self.ubar_SU1_rbr_uca_jcr_uca_upright = (multi_dot([c3,c42]) + c4)
        self.ubar_SU1_rbr_upright_jcr_uca_upright = (multi_dot([c43,c42]) + c44)
        self.Mbar_SU1_rbr_uca_jcr_uca_upright = multi_dot([c3,c45])
        self.Mbar_SU1_rbr_upright_jcr_uca_upright = multi_dot([c43,c45])
        self.ubar_SU1_rbl_uca_jcl_uca_upright = (multi_dot([c7,c46]) + c8)
        self.ubar_SU1_rbl_upright_jcl_uca_upright = (multi_dot([c47,c46]) + c48)
        self.Mbar_SU1_rbl_uca_jcl_uca_upright = multi_dot([c7,c49])
        self.Mbar_SU1_rbl_upright_jcl_uca_upright = multi_dot([c47,c49])
        self.ubar_SU1_rbr_lca_jcr_lca_upright = (multi_dot([c11,c50]) + c12)
        self.ubar_SU1_rbr_upright_jcr_lca_upright = (multi_dot([c43,c50]) + c44)
        self.Mbar_SU1_rbr_lca_jcr_lca_upright = multi_dot([c11,c51])
        self.Mbar_SU1_rbr_upright_jcr_lca_upright = multi_dot([c43,c51])
        self.ubar_SU1_rbr_lca_jcr_strut_lca = (multi_dot([c11,c52]) + c12)
        self.ubar_SU1_rbr_lower_strut_jcr_strut_lca = (multi_dot([c53,c52]) + c54)
        self.Mbar_SU1_rbr_lca_jcr_strut_lca = multi_dot([c11,c55])
        self.Mbar_SU1_rbr_lower_strut_jcr_strut_lca = multi_dot([c53,Triad(self.ax2_jcr_strut_lca,c55[0:3,1:2])])
        self.ubar_SU1_rbl_lca_jcl_lca_upright = (multi_dot([c15,c56]) + c16)
        self.ubar_SU1_rbl_upright_jcl_lca_upright = (multi_dot([c47,c56]) + c48)
        self.Mbar_SU1_rbl_lca_jcl_lca_upright = multi_dot([c15,c57])
        self.Mbar_SU1_rbl_upright_jcl_lca_upright = multi_dot([c47,c57])
        self.ubar_SU1_rbl_lca_jcl_strut_lca = (multi_dot([c15,c58]) + c16)
        self.ubar_SU1_rbl_lower_strut_jcl_strut_lca = (multi_dot([c59,c58]) + c60)
        self.Mbar_SU1_rbl_lca_jcl_strut_lca = multi_dot([c15,c61])
        self.Mbar_SU1_rbl_lower_strut_jcl_strut_lca = multi_dot([c59,Triad(self.ax2_jcl_strut_lca,c61[0:3,1:2])])
        self.ubar_SU1_rbr_upright_jcr_tie_upright = (multi_dot([c43,c62]) + c44)
        self.ubar_SU1_rbr_tie_rod_jcr_tie_upright = (multi_dot([c31,c62]) + c32)
        self.Mbar_SU1_rbr_upright_jcr_tie_upright = multi_dot([c43,c63])
        self.Mbar_SU1_rbr_tie_rod_jcr_tie_upright = multi_dot([c31,c63])
        self.ubar_SU1_rbr_upright_jcr_hub_bearing = (multi_dot([c43,c64]) + c44)
        self.ubar_SU1_rbr_hub_jcr_hub_bearing = (multi_dot([c65,c64]) + -1.0*multi_dot([c65,self.R_SU1_rbr_hub]))
        self.Mbar_SU1_rbr_upright_jcr_hub_bearing = multi_dot([c43,c66])
        self.Mbar_SU1_rbr_hub_jcr_hub_bearing = multi_dot([c65,c66])
        self.ubar_SU1_rbl_upright_jcl_tie_upright = (multi_dot([c47,c67]) + c48)
        self.ubar_SU1_rbl_tie_rod_jcl_tie_upright = (multi_dot([c27,c67]) + c28)
        self.Mbar_SU1_rbl_upright_jcl_tie_upright = multi_dot([c47,c68])
        self.Mbar_SU1_rbl_tie_rod_jcl_tie_upright = multi_dot([c27,c68])
        self.ubar_SU1_rbl_upright_jcl_hub_bearing = (multi_dot([c47,c69]) + c48)
        self.ubar_SU1_rbl_hub_jcl_hub_bearing = (multi_dot([c70,c69]) + -1.0*multi_dot([c70,self.R_SU1_rbl_hub]))
        self.Mbar_SU1_rbl_upright_jcl_hub_bearing = multi_dot([c47,c71])
        self.Mbar_SU1_rbl_hub_jcl_hub_bearing = multi_dot([c70,c71])
        self.ubar_SU1_rbr_upper_strut_jcr_strut = (multi_dot([c19,c72]) + c20)
        self.ubar_SU1_rbr_lower_strut_jcr_strut = (multi_dot([c53,c72]) + c54)
        self.Mbar_SU1_rbr_upper_strut_jcr_strut = multi_dot([c19,c73])
        self.Mbar_SU1_rbr_lower_strut_jcr_strut = multi_dot([c53,c73])
        self.ubar_SU1_rbl_upper_strut_jcl_strut = (multi_dot([c23,c74]) + c24)
        self.ubar_SU1_rbl_lower_strut_jcl_strut = (multi_dot([c59,c74]) + c60)
        self.Mbar_SU1_rbl_upper_strut_jcl_strut = multi_dot([c23,c75])
        self.Mbar_SU1_rbl_lower_strut_jcl_strut = multi_dot([c59,c75])
        self.ubar_ST_rbs_coupler_jcs_rc_sph = (multi_dot([c76,c77]) + c78)
        self.ubar_ST_rbr_rocker_jcs_rc_sph = (multi_dot([c35,c77]) + c36)
        self.Mbar_ST_rbs_coupler_jcs_rc_sph = multi_dot([c76,c79])
        self.Mbar_ST_rbr_rocker_jcs_rc_sph = multi_dot([c35,c79])
        self.ubar_ST_rbs_coupler_jcs_rc_cyl = (multi_dot([c76,c80]) + c78)
        self.ubar_ST_rbl_rocker_jcs_rc_cyl = (multi_dot([c39,c80]) + c40)
        self.Mbar_ST_rbs_coupler_jcs_rc_cyl = multi_dot([c76,c81])
        self.Mbar_ST_rbl_rocker_jcs_rc_cyl = multi_dot([c39,c81])

    @property
    def q_initial(self):
        q = np.concatenate([self.R_ground,self.P_ground,self.R_SU1_rbr_uca,self.P_SU1_rbr_uca,self.R_SU1_rbl_uca,self.P_SU1_rbl_uca,self.R_SU1_rbr_lca,self.P_SU1_rbr_lca,self.R_SU1_rbl_lca,self.P_SU1_rbl_lca,self.R_SU1_rbr_upright,self.P_SU1_rbr_upright,self.R_SU1_rbl_upright,self.P_SU1_rbl_upright,self.R_SU1_rbr_upper_strut,self.P_SU1_rbr_upper_strut,self.R_SU1_rbl_upper_strut,self.P_SU1_rbl_upper_strut,self.R_SU1_rbr_lower_strut,self.P_SU1_rbr_lower_strut,self.R_SU1_rbl_lower_strut,self.P_SU1_rbl_lower_strut,self.R_SU1_rbr_tie_rod,self.P_SU1_rbr_tie_rod,self.R_SU1_rbl_tie_rod,self.P_SU1_rbl_tie_rod,self.R_SU1_rbr_hub,self.P_SU1_rbr_hub,self.R_SU1_rbl_hub,self.P_SU1_rbl_hub,self.R_ST_rbs_coupler,self.P_ST_rbs_coupler,self.R_ST_rbr_rocker,self.P_ST_rbr_rocker,self.R_ST_rbl_rocker,self.P_ST_rbl_rocker])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.Rd_ground,self.Pd_ground,self.Rd_SU1_rbr_uca,self.Pd_SU1_rbr_uca,self.Rd_SU1_rbl_uca,self.Pd_SU1_rbl_uca,self.Rd_SU1_rbr_lca,self.Pd_SU1_rbr_lca,self.Rd_SU1_rbl_lca,self.Pd_SU1_rbl_lca,self.Rd_SU1_rbr_upright,self.Pd_SU1_rbr_upright,self.Rd_SU1_rbl_upright,self.Pd_SU1_rbl_upright,self.Rd_SU1_rbr_upper_strut,self.Pd_SU1_rbr_upper_strut,self.Rd_SU1_rbl_upper_strut,self.Pd_SU1_rbl_upper_strut,self.Rd_SU1_rbr_lower_strut,self.Pd_SU1_rbr_lower_strut,self.Rd_SU1_rbl_lower_strut,self.Pd_SU1_rbl_lower_strut,self.Rd_SU1_rbr_tie_rod,self.Pd_SU1_rbr_tie_rod,self.Rd_SU1_rbl_tie_rod,self.Pd_SU1_rbl_tie_rod,self.Rd_SU1_rbr_hub,self.Pd_SU1_rbr_hub,self.Rd_SU1_rbl_hub,self.Pd_SU1_rbl_hub,self.Rd_ST_rbs_coupler,self.Pd_ST_rbs_coupler,self.Rd_ST_rbr_rocker,self.Pd_ST_rbr_rocker,self.Rd_ST_rbl_rocker,self.Pd_ST_rbl_rocker])
        return qd



class numerical_assembly(object):

    def __init__(self,config):
        self.t = 0.0
        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,54,54,55,55,55,55,56,56,56,56,57,57,57,57,58,58,58,58,59,59,59,59,60,60,61,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78])
        self.jac_cols = np.array([0,1,26,27,0,1,28,29,0,1,2,3,0,1,2,3,0,1,2,3,0,1,4,5,0,1,4,5,0,1,4,5,0,1,6,7,0,1,6,7,0,1,6,7,0,1,8,9,0,1,8,9,0,1,8,9,0,1,14,15,0,1,14,15,0,1,16,17,0,1,16,17,0,1,24,25,0,1,24,25,0,1,22,23,0,1,22,23,0,1,32,33,0,1,32,33,0,1,32,33,0,1,32,33,0,1,34,35,0,1,34,35,0,1,34,35,2,3,10,11,4,5,12,13,6,7,10,11,6,7,18,19,6,7,18,19,8,9,12,13,8,9,20,21,8,9,20,21,10,11,22,23,10,11,26,27,10,11,26,27,10,11,26,27,10,11,26,27,12,13,24,25,12,13,28,29,12,13,28,29,12,13,28,29,12,13,28,29,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,20,21,16,17,20,21,16,17,20,21,30,31,32,33,30,31,34,35,30,31,34,35,30,31,34,35,30,31,34,35,0,1,0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35])

    
    def set_q(self,q):
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
        self.R_ST_rbs_coupler = q[105:108,0:1]
        self.P_ST_rbs_coupler = q[108:112,0:1]
        self.R_ST_rbr_rocker = q[112:115,0:1]
        self.P_ST_rbr_rocker = q[115:119,0:1]
        self.R_ST_rbl_rocker = q[119:122,0:1]
        self.P_ST_rbl_rocker = q[122:126,0:1]

    
    def set_qd(self,qd):
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
        self.Rd_ST_rbs_coupler = qd[105:108,0:1]
        self.Pd_ST_rbs_coupler = qd[108:112,0:1]
        self.Rd_ST_rbr_rocker = qd[112:115,0:1]
        self.Pd_ST_rbr_rocker = qd[115:119,0:1]
        self.Rd_ST_rbl_rocker = qd[119:122,0:1]
        self.Pd_ST_rbl_rocker = qd[122:126,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = x0[2]
        x2 = np.eye(1,dtype=np.float64)
        x3 = self.R_SU1_rbr_uca
        x4 = self.P_ground
        x5 = A(x4)
        x6 = self.P_SU1_rbr_uca
        x7 = A(x6)
        x8 = x5.T
        x9 = config.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
        x10 = self.R_SU1_rbl_uca
        x11 = self.P_SU1_rbl_uca
        x12 = A(x11)
        x13 = config.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
        x14 = self.R_SU1_rbr_lca
        x15 = self.P_SU1_rbr_lca
        x16 = A(x15)
        x17 = config.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
        x18 = self.R_SU1_rbl_lca
        x19 = self.P_SU1_rbl_lca
        x20 = A(x19)
        x21 = config.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
        x22 = self.R_SU1_rbr_upper_strut
        x23 = self.P_SU1_rbr_upper_strut
        x24 = A(x23)
        x25 = self.R_SU1_rbl_upper_strut
        x26 = self.P_SU1_rbl_upper_strut
        x27 = A(x26)
        x28 = -1.0*self.R_SU1_rbl_tie_rod
        x29 = self.P_SU1_rbl_tie_rod
        x30 = A(x29)
        x31 = -1.0*self.R_SU1_rbr_tie_rod
        x32 = self.P_SU1_rbr_tie_rod
        x33 = A(x32)
        x34 = -1.0*self.R_ST_rbr_rocker
        x35 = self.P_ST_rbr_rocker
        x36 = A(x35)
        x37 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        x38 = config.F_jcr_rocker_ch(t,)
        x39 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,0:1]
        x40 = -1.0*self.R_ST_rbl_rocker
        x41 = self.P_ST_rbl_rocker
        x42 = A(x41)
        x43 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        x44 = self.R_SU1_rbr_upright
        x45 = -1.0*x44
        x46 = self.P_SU1_rbr_upright
        x47 = A(x46)
        x48 = self.R_SU1_rbl_upright
        x49 = -1.0*x48
        x50 = self.P_SU1_rbl_upright
        x51 = A(x50)
        x52 = -1.0*self.R_SU1_rbr_lower_strut
        x53 = self.P_SU1_rbr_lower_strut
        x54 = A(x53)
        x55 = -1.0*self.R_SU1_rbl_lower_strut
        x56 = self.P_SU1_rbl_lower_strut
        x57 = A(x56)
        x58 = self.P_SU1_rbr_hub
        x59 = A(x58)
        x60 = x47.T
        x61 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        x62 = config.F_jcr_hub_bearing(t,)
        x63 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,0:1]
        x64 = self.P_SU1_rbl_hub
        x65 = A(x64)
        x66 = x51.T
        x67 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        x68 = config.F_jcl_hub_bearing(t,)
        x69 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,0:1]
        x70 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1].T
        x71 = x24.T
        x72 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        x73 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2].T
        x74 = (x22 + x52 + multi_dot([x24,config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,2:3]]) + multi_dot([x24,config.ubar_SU1_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x54,config.ubar_SU1_rbr_lower_strut_jcr_strut]))
        x75 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1].T
        x76 = x27.T
        x77 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        x78 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2].T
        x79 = (x25 + x55 + multi_dot([x27,config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,2:3]]) + multi_dot([x27,config.ubar_SU1_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x57,config.ubar_SU1_rbl_lower_strut_jcl_strut]))
        x80 = self.R_ST_rbs_coupler
        x81 = self.P_ST_rbs_coupler
        x82 = A(x81)
        x83 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1].T
        x84 = x82.T
        x85 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        x86 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2].T
        x87 = (x80 + x40 + multi_dot([x82,config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3]]) + multi_dot([x82,config.ubar_ST_rbs_coupler_jcs_rc_cyl]) + -1.0*multi_dot([x42,config.ubar_ST_rbl_rocker_jcs_rc_cyl]))
        x88 = -1.0*x2

        self.pos_eq_blocks = [x44[2] - 1*config.F_mcr_zact(t,)*x2,x48[2] - 1*config.F_mcl_zact(t,)*x2,(x0 + -1.0*x3 + multi_dot([x5,config.ubar_ground_jcr_uca_chassis]) + -1.0*multi_dot([x7,config.ubar_SU1_rbr_uca_jcr_uca_chassis])),multi_dot([config.Mbar_ground_jcr_uca_chassis[:,0:1].T,x8,x7,x9]),multi_dot([config.Mbar_ground_jcr_uca_chassis[:,1:2].T,x8,x7,x9]),(x0 + -1.0*x10 + multi_dot([x5,config.ubar_ground_jcl_uca_chassis]) + -1.0*multi_dot([x12,config.ubar_SU1_rbl_uca_jcl_uca_chassis])),multi_dot([config.Mbar_ground_jcl_uca_chassis[:,0:1].T,x8,x12,x13]),multi_dot([config.Mbar_ground_jcl_uca_chassis[:,1:2].T,x8,x12,x13]),(x0 + -1.0*x14 + multi_dot([x5,config.ubar_ground_jcr_lca_chassis]) + -1.0*multi_dot([x16,config.ubar_SU1_rbr_lca_jcr_lca_chassis])),multi_dot([config.Mbar_ground_jcr_lca_chassis[:,0:1].T,x8,x16,x17]),multi_dot([config.Mbar_ground_jcr_lca_chassis[:,1:2].T,x8,x16,x17]),(x0 + -1.0*x18 + multi_dot([x5,config.ubar_ground_jcl_lca_chassis]) + -1.0*multi_dot([x20,config.ubar_SU1_rbl_lca_jcl_lca_chassis])),multi_dot([config.Mbar_ground_jcl_lca_chassis[:,0:1].T,x8,x20,x21]),multi_dot([config.Mbar_ground_jcl_lca_chassis[:,1:2].T,x8,x20,x21]),(x0 + -1.0*x22 + multi_dot([x5,config.ubar_ground_jcr_strut_chassis]) + -1.0*multi_dot([x24,config.ubar_SU1_rbr_upper_strut_jcr_strut_chassis])),multi_dot([config.Mbar_ground_jcr_strut_chassis[:,0:1].T,x8,x24,config.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]]),(x0 + -1.0*x25 + multi_dot([x5,config.ubar_ground_jcl_strut_chassis]) + -1.0*multi_dot([x27,config.ubar_SU1_rbl_upper_strut_jcl_strut_chassis])),multi_dot([config.Mbar_ground_jcl_strut_chassis[:,0:1].T,x8,x27,config.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]]),(x0 + x28 + multi_dot([x5,config.ubar_ground_jcl_tie_steering]) + -1.0*multi_dot([x30,config.ubar_SU1_rbl_tie_rod_jcl_tie_steering])),multi_dot([config.Mbar_ground_jcl_tie_steering[:,0:1].T,x8,x30,config.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]]),(x0 + x31 + multi_dot([x5,config.ubar_ground_jcr_tie_steering]) + -1.0*multi_dot([x33,config.ubar_SU1_rbr_tie_rod_jcr_tie_steering])),multi_dot([config.Mbar_ground_jcr_tie_steering[:,0:1].T,x8,x33,config.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]]),(x0 + x34 + multi_dot([x5,config.ubar_ground_jcr_rocker_ch]) + -1.0*multi_dot([x36,config.ubar_ST_rbr_rocker_jcr_rocker_ch])),multi_dot([config.Mbar_ground_jcr_rocker_ch[:,0:1].T,x8,x36,x37]),multi_dot([config.Mbar_ground_jcr_rocker_ch[:,1:2].T,x8,x36,x37]),(cos(x38)*multi_dot([config.Mbar_ground_jcr_rocker_ch[:,1:2].T,x8,x36,x39]) + sin(x38)*-1.0*multi_dot([config.Mbar_ground_jcr_rocker_ch[:,0:1].T,x8,x36,x39])),(x0 + x40 + multi_dot([x5,config.ubar_ground_jcl_rocker_ch]) + -1.0*multi_dot([x42,config.ubar_ST_rbl_rocker_jcl_rocker_ch])),multi_dot([config.Mbar_ground_jcl_rocker_ch[:,0:1].T,x8,x42,x43]),multi_dot([config.Mbar_ground_jcl_rocker_ch[:,1:2].T,x8,x42,x43]),(x3 + x45 + multi_dot([x7,config.ubar_SU1_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x47,config.ubar_SU1_rbr_upright_jcr_uca_upright])),(x10 + x49 + multi_dot([x12,config.ubar_SU1_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x51,config.ubar_SU1_rbl_upright_jcl_uca_upright])),(x14 + x45 + multi_dot([x16,config.ubar_SU1_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x47,config.ubar_SU1_rbr_upright_jcr_lca_upright])),(x14 + x52 + multi_dot([x16,config.ubar_SU1_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x54,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca])),multi_dot([config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1].T,x16.T,x54,config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x18 + x49 + multi_dot([x20,config.ubar_SU1_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x51,config.ubar_SU1_rbl_upright_jcl_lca_upright])),(x18 + x55 + multi_dot([x20,config.ubar_SU1_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x57,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1].T,x20.T,x57,config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x44 + x31 + multi_dot([x47,config.ubar_SU1_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x33,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright])),(x44 + -1.0*self.R_SU1_rbr_hub + multi_dot([x47,config.ubar_SU1_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x59,config.ubar_SU1_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1].T,x60,x59,x61]),multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2].T,x60,x59,x61]),(cos(x62)*multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2].T,x60,x59,x63]) + sin(x62)*-1.0*multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1].T,x60,x59,x63])),(x48 + x28 + multi_dot([x51,config.ubar_SU1_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x30,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright])),(x48 + -1.0*self.R_SU1_rbl_hub + multi_dot([x51,config.ubar_SU1_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x65,config.ubar_SU1_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1].T,x66,x65,x67]),multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2].T,x66,x65,x67]),(cos(x68)*multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2].T,x66,x65,x69]) + sin(x68)*-1.0*multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1].T,x66,x65,x69])),multi_dot([x70,x71,x54,x72]),multi_dot([x73,x71,x54,x72]),multi_dot([x70,x71,x74]),multi_dot([x73,x71,x74]),multi_dot([x75,x76,x57,x77]),multi_dot([x78,x76,x57,x77]),multi_dot([x75,x76,x79]),multi_dot([x78,x76,x79]),(x80 + x34 + multi_dot([x82,config.ubar_ST_rbs_coupler_jcs_rc_sph]) + -1.0*multi_dot([x36,config.ubar_ST_rbr_rocker_jcs_rc_sph])),multi_dot([x83,x84,x42,x85]),multi_dot([x86,x84,x42,x85]),multi_dot([x83,x84,x87]),multi_dot([x86,x84,x87]),x0,(x4 + -1.0*config.Pg_ground),(x88 + (multi_dot([x6.T,x6]))**(1.0/2.0)),(x88 + (multi_dot([x11.T,x11]))**(1.0/2.0)),(x88 + (multi_dot([x15.T,x15]))**(1.0/2.0)),(x88 + (multi_dot([x19.T,x19]))**(1.0/2.0)),(x88 + (multi_dot([x46.T,x46]))**(1.0/2.0)),(x88 + (multi_dot([x50.T,x50]))**(1.0/2.0)),(x88 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x88 + (multi_dot([x26.T,x26]))**(1.0/2.0)),(x88 + (multi_dot([x53.T,x53]))**(1.0/2.0)),(x88 + (multi_dot([x56.T,x56]))**(1.0/2.0)),(x88 + (multi_dot([x32.T,x32]))**(1.0/2.0)),(x88 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x88 + (multi_dot([x58.T,x58]))**(1.0/2.0)),(x88 + (multi_dot([x64.T,x64]))**(1.0/2.0)),(x88 + (multi_dot([x81.T,x81]))**(1.0/2.0)),(x88 + (multi_dot([x35.T,x35]))**(1.0/2.0)),(x88 + (multi_dot([x41.T,x41]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)
        v1 = np.eye(1,dtype=np.float64)
        v2 = np.zeros((3,1),dtype=np.float64)

        self.vel_eq_blocks = [(v0 + derivative(config.F_mcr_zact,t,0.1,1)*-1.0*v1),(v0 + derivative(config.F_mcl_zact,t,0.1,1)*-1.0*v1),v2,v0,v0,v2,v0,v0,v2,v0,v0,v2,v0,v0,v2,v0,v2,v0,v2,v0,v2,v0,v2,v0,v0,(v0 + derivative(config.F_jcr_rocker_ch,t,0.1,1)*-1.0*v1),v2,v0,v0,v2,v2,v2,v2,v0,v2,v2,v0,v2,v2,v0,v0,(v0 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v1),v2,v2,v0,v0,(v0 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v1),v0,v0,v0,v0,v0,v0,v0,v0,v2,v0,v0,v0,v0,v2,np.zeros((4,1),dtype=np.float64),v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.zeros((1,1),dtype=np.float64)
        a1 = np.eye(1,dtype=np.float64)
        a2 = self.Pd_ground
        a3 = self.Pd_SU1_rbr_uca
        a4 = config.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
        a5 = a4.T
        a6 = self.P_SU1_rbr_uca
        a7 = A(a6).T
        a8 = config.Mbar_ground_jcr_uca_chassis[:,0:1]
        a9 = self.P_ground
        a10 = A(a9).T
        a11 = B(a3,a4)
        a12 = a2.T
        a13 = B(a6,a4)
        a14 = config.Mbar_ground_jcr_uca_chassis[:,1:2]
        a15 = self.Pd_SU1_rbl_uca
        a16 = config.Mbar_ground_jcl_uca_chassis[:,0:1]
        a17 = config.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
        a18 = B(a15,a17)
        a19 = a17.T
        a20 = self.P_SU1_rbl_uca
        a21 = A(a20).T
        a22 = B(a20,a17)
        a23 = config.Mbar_ground_jcl_uca_chassis[:,1:2]
        a24 = self.Pd_SU1_rbr_lca
        a25 = config.Mbar_ground_jcr_lca_chassis[:,0:1]
        a26 = config.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
        a27 = B(a24,a26)
        a28 = a26.T
        a29 = self.P_SU1_rbr_lca
        a30 = A(a29).T
        a31 = B(a29,a26)
        a32 = config.Mbar_ground_jcr_lca_chassis[:,1:2]
        a33 = self.Pd_SU1_rbl_lca
        a34 = config.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
        a35 = a34.T
        a36 = self.P_SU1_rbl_lca
        a37 = A(a36).T
        a38 = config.Mbar_ground_jcl_lca_chassis[:,0:1]
        a39 = B(a33,a34)
        a40 = B(a36,a34)
        a41 = config.Mbar_ground_jcl_lca_chassis[:,1:2]
        a42 = self.Pd_SU1_rbr_upper_strut
        a43 = config.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a44 = self.P_SU1_rbr_upper_strut
        a45 = A(a44).T
        a46 = config.Mbar_ground_jcr_strut_chassis[:,0:1]
        a47 = self.Pd_SU1_rbl_upper_strut
        a48 = config.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a49 = self.P_SU1_rbl_upper_strut
        a50 = A(a49).T
        a51 = config.Mbar_ground_jcl_strut_chassis[:,0:1]
        a52 = self.Pd_SU1_rbl_tie_rod
        a53 = config.Mbar_ground_jcl_tie_steering[:,0:1]
        a54 = config.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a55 = self.P_SU1_rbl_tie_rod
        a56 = self.Pd_SU1_rbr_tie_rod
        a57 = config.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a58 = self.P_SU1_rbr_tie_rod
        a59 = config.Mbar_ground_jcr_tie_steering[:,0:1]
        a60 = self.Pd_ST_rbr_rocker
        a61 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        a62 = a61.T
        a63 = self.P_ST_rbr_rocker
        a64 = A(a63).T
        a65 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        a66 = B(a60,a61)
        a67 = B(a63,a61)
        a68 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        a69 = config.F_jcr_rocker_ch(t,)
        a70 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,0:1]
        a71 = cos(a69)
        a72 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        a73 = sin(a69)
        a74 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        a75 = self.Pd_ST_rbl_rocker
        a76 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        a77 = a76.T
        a78 = self.P_ST_rbl_rocker
        a79 = A(a78).T
        a80 = config.Mbar_ground_jcl_rocker_ch[:,0:1]
        a81 = B(a75,a76)
        a82 = B(a78,a76)
        a83 = config.Mbar_ground_jcl_rocker_ch[:,1:2]
        a84 = self.Pd_SU1_rbr_upright
        a85 = self.Pd_SU1_rbl_upright
        a86 = self.Pd_SU1_rbr_lower_strut
        a87 = config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
        a88 = config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a89 = self.P_SU1_rbr_lower_strut
        a90 = A(a89).T
        a91 = a24.T
        a92 = self.Pd_SU1_rbl_lower_strut
        a93 = config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a94 = self.P_SU1_rbl_lower_strut
        a95 = A(a94).T
        a96 = config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
        a97 = a33.T
        a98 = self.Pd_SU1_rbr_hub
        a99 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        a100 = a99.T
        a101 = self.P_SU1_rbr_hub
        a102 = A(a101).T
        a103 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        a104 = self.P_SU1_rbr_upright
        a105 = A(a104).T
        a106 = B(a98,a99)
        a107 = a84.T
        a108 = B(a101,a99)
        a109 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        a110 = config.F_jcr_hub_bearing(t,)
        a111 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,0:1]
        a112 = cos(a110)
        a113 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        a114 = sin(a110)
        a115 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        a116 = self.Pd_SU1_rbl_hub
        a117 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        a118 = a117.T
        a119 = self.P_SU1_rbl_hub
        a120 = A(a119).T
        a121 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        a122 = self.P_SU1_rbl_upright
        a123 = A(a122).T
        a124 = B(a116,a117)
        a125 = a85.T
        a126 = B(a119,a117)
        a127 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        a128 = config.F_jcl_hub_bearing(t,)
        a129 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,0:1]
        a130 = cos(a128)
        a131 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        a132 = sin(a128)
        a133 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        a134 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        a135 = a134.T
        a136 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
        a137 = B(a42,a136)
        a138 = a136.T
        a139 = B(a86,a134)
        a140 = a42.T
        a141 = B(a44,a136).T
        a142 = B(a89,a134)
        a143 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
        a144 = B(a42,a143)
        a145 = a143.T
        a146 = B(a44,a143).T
        a147 = config.ubar_SU1_rbr_upper_strut_jcr_strut
        a148 = config.ubar_SU1_rbr_lower_strut_jcr_strut
        a149 = (multi_dot([B(a42,a147),a42]) + -1.0*multi_dot([B(a86,a148),a86]))
        a150 = (self.Rd_SU1_rbr_upper_strut + -1.0*self.Rd_SU1_rbr_lower_strut + multi_dot([B(a89,a148),a86]) + multi_dot([B(a44,a147),a42]))
        a151 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,2:3].T,a45]) + multi_dot([a147.T,a45]) + -1.0*multi_dot([a148.T,a90]))
        a152 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        a153 = a152.T
        a154 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
        a155 = B(a47,a154)
        a156 = a154.T
        a157 = B(a92,a152)
        a158 = a47.T
        a159 = B(a49,a154).T
        a160 = B(a94,a152)
        a161 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
        a162 = B(a47,a161)
        a163 = a161.T
        a164 = B(a49,a161).T
        a165 = config.ubar_SU1_rbl_upper_strut_jcl_strut
        a166 = config.ubar_SU1_rbl_lower_strut_jcl_strut
        a167 = (multi_dot([B(a47,a165),a47]) + -1.0*multi_dot([B(a92,a166),a92]))
        a168 = (self.Rd_SU1_rbl_upper_strut + -1.0*self.Rd_SU1_rbl_lower_strut + multi_dot([B(a94,a166),a92]) + multi_dot([B(a49,a165),a47]))
        a169 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,2:3].T,a50]) + multi_dot([a165.T,a50]) + -1.0*multi_dot([a166.T,a95]))
        a170 = self.Pd_ST_rbs_coupler
        a171 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        a172 = a171.T
        a173 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1]
        a174 = B(a170,a173)
        a175 = a173.T
        a176 = self.P_ST_rbs_coupler
        a177 = A(a176).T
        a178 = B(a75,a171)
        a179 = a170.T
        a180 = B(a176,a173).T
        a181 = B(a78,a171)
        a182 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2]
        a183 = B(a170,a182)
        a184 = a182.T
        a185 = B(a176,a182).T
        a186 = config.ubar_ST_rbs_coupler_jcs_rc_cyl
        a187 = config.ubar_ST_rbl_rocker_jcs_rc_cyl
        a188 = (multi_dot([B(a170,a186),a170]) + -1.0*multi_dot([B(a75,a187),a75]))
        a189 = (self.Rd_ST_rbs_coupler + -1.0*self.Rd_ST_rbl_rocker + multi_dot([B(a78,a187),a75]) + multi_dot([B(a176,a186),a170]))
        a190 = (self.R_ST_rbs_coupler.T + -1.0*self.R_ST_rbl_rocker.T + multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3].T,a177]) + multi_dot([a186.T,a177]) + -1.0*multi_dot([a187.T,a79]))

        self.acc_eq_blocks = [(a0 + derivative(config.F_mcr_zact,t,0.1,2)*-1.0*a1),(a0 + derivative(config.F_mcl_zact,t,0.1,2)*-1.0*a1),(multi_dot([B(a2,config.ubar_ground_jcr_uca_chassis),a2]) + -1.0*multi_dot([B(a3,config.ubar_SU1_rbr_uca_jcr_uca_chassis),a3])),(multi_dot([a5,a7,B(a2,a8),a2]) + multi_dot([a8.T,a10,a11,a3]) + 2.0*multi_dot([a12,B(a9,a8).T,a13,a3])),(multi_dot([a5,a7,B(a2,a14),a2]) + multi_dot([a14.T,a10,a11,a3]) + 2.0*multi_dot([a12,B(a9,a14).T,a13,a3])),(multi_dot([B(a2,config.ubar_ground_jcl_uca_chassis),a2]) + -1.0*multi_dot([B(a15,config.ubar_SU1_rbl_uca_jcl_uca_chassis),a15])),(multi_dot([a16.T,a10,a18,a15]) + multi_dot([a19,a21,B(a2,a16),a2]) + 2.0*multi_dot([a12,B(a9,a16).T,a22,a15])),(multi_dot([a23.T,a10,a18,a15]) + multi_dot([a19,a21,B(a2,a23),a2]) + 2.0*multi_dot([a12,B(a9,a23).T,a22,a15])),(multi_dot([B(a2,config.ubar_ground_jcr_lca_chassis),a2]) + -1.0*multi_dot([B(a24,config.ubar_SU1_rbr_lca_jcr_lca_chassis),a24])),(multi_dot([a25.T,a10,a27,a24]) + multi_dot([a28,a30,B(a2,a25),a2]) + 2.0*multi_dot([a12,B(a9,a25).T,a31,a24])),(multi_dot([a32.T,a10,a27,a24]) + multi_dot([a28,a30,B(a2,a32),a2]) + 2.0*multi_dot([a12,B(a9,a32).T,a31,a24])),(multi_dot([B(a2,config.ubar_ground_jcl_lca_chassis),a2]) + -1.0*multi_dot([B(a33,config.ubar_SU1_rbl_lca_jcl_lca_chassis),a33])),(multi_dot([a35,a37,B(a2,a38),a2]) + multi_dot([a38.T,a10,a39,a33]) + 2.0*multi_dot([a12,B(a9,a38).T,a40,a33])),(multi_dot([a35,a37,B(a2,a41),a2]) + multi_dot([a41.T,a10,a39,a33]) + 2.0*multi_dot([a12,B(a9,a41).T,a40,a33])),(multi_dot([B(a2,config.ubar_ground_jcr_strut_chassis),a2]) + -1.0*multi_dot([B(a42,config.ubar_SU1_rbr_upper_strut_jcr_strut_chassis),a42])),(multi_dot([a43.T,a45,B(a2,a46),a2]) + multi_dot([a46.T,a10,B(a42,a43),a42]) + 2.0*multi_dot([a12,B(a9,a46).T,B(a44,a43),a42])),(multi_dot([B(a2,config.ubar_ground_jcl_strut_chassis),a2]) + -1.0*multi_dot([B(a47,config.ubar_SU1_rbl_upper_strut_jcl_strut_chassis),a47])),(multi_dot([a48.T,a50,B(a2,a51),a2]) + multi_dot([a51.T,a10,B(a47,a48),a47]) + 2.0*multi_dot([a12,B(a9,a51).T,B(a49,a48),a47])),(multi_dot([B(a2,config.ubar_ground_jcl_tie_steering),a2]) + -1.0*multi_dot([B(a52,config.ubar_SU1_rbl_tie_rod_jcl_tie_steering),a52])),(multi_dot([a53.T,a10,B(a52,a54),a52]) + multi_dot([a54.T,A(a55).T,B(a2,a53),a2]) + 2.0*multi_dot([a12,B(a9,a53).T,B(a55,a54),a52])),(multi_dot([B(a2,config.ubar_ground_jcr_tie_steering),a2]) + -1.0*multi_dot([B(a56,config.ubar_SU1_rbr_tie_rod_jcr_tie_steering),a56])),(multi_dot([a57.T,A(a58).T,B(a2,a59),a2]) + multi_dot([a59.T,a10,B(a56,a57),a56]) + 2.0*multi_dot([a12,B(a9,a59).T,B(a58,a57),a56])),(multi_dot([B(a2,config.ubar_ground_jcr_rocker_ch),a2]) + -1.0*multi_dot([B(a60,config.ubar_ST_rbr_rocker_jcr_rocker_ch),a60])),(multi_dot([a62,a64,B(a2,a65),a2]) + multi_dot([a65.T,a10,a66,a60]) + 2.0*multi_dot([a12,B(a9,a65).T,a67,a60])),(multi_dot([a62,a64,B(a2,a68),a2]) + multi_dot([a68.T,a10,a66,a60]) + 2.0*multi_dot([a12,B(a9,a68).T,a67,a60])),(derivative(config.F_jcr_rocker_ch,t,0.1,2)*-1.0*a1 + multi_dot([a70.T,a64,(a71*B(a2,a72) + a73*-1.0*B(a2,a74)),a2]) + multi_dot([(a71*multi_dot([a72.T,a10]) + a73*-1.0*multi_dot([a74.T,a10])),B(a60,a70),a60]) + 2.0*multi_dot([((a71*multi_dot([B(a9,a72),a2])).T + a73*-1.0*multi_dot([a12,B(a9,a74).T])),B(a63,a70),a60])),(multi_dot([B(a2,config.ubar_ground_jcl_rocker_ch),a2]) + -1.0*multi_dot([B(a75,config.ubar_ST_rbl_rocker_jcl_rocker_ch),a75])),(multi_dot([a77,a79,B(a2,a80),a2]) + multi_dot([a80.T,a10,a81,a75]) + 2.0*multi_dot([a12,B(a9,a80).T,a82,a75])),(multi_dot([a77,a79,B(a2,a83),a2]) + multi_dot([a83.T,a10,a81,a75]) + 2.0*multi_dot([a12,B(a9,a83).T,a82,a75])),(multi_dot([B(a3,config.ubar_SU1_rbr_uca_jcr_uca_upright),a3]) + -1.0*multi_dot([B(a84,config.ubar_SU1_rbr_upright_jcr_uca_upright),a84])),(multi_dot([B(a15,config.ubar_SU1_rbl_uca_jcl_uca_upright),a15]) + -1.0*multi_dot([B(a85,config.ubar_SU1_rbl_upright_jcl_uca_upright),a85])),(multi_dot([B(a24,config.ubar_SU1_rbr_lca_jcr_lca_upright),a24]) + -1.0*multi_dot([B(a84,config.ubar_SU1_rbr_upright_jcr_lca_upright),a84])),(multi_dot([B(a24,config.ubar_SU1_rbr_lca_jcr_strut_lca),a24]) + -1.0*multi_dot([B(a86,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca),a86])),(multi_dot([a87.T,a30,B(a86,a88),a86]) + multi_dot([a88.T,a90,B(a24,a87),a24]) + 2.0*multi_dot([a91,B(a29,a87).T,B(a89,a88),a86])),(multi_dot([B(a33,config.ubar_SU1_rbl_lca_jcl_lca_upright),a33]) + -1.0*multi_dot([B(a85,config.ubar_SU1_rbl_upright_jcl_lca_upright),a85])),(multi_dot([B(a33,config.ubar_SU1_rbl_lca_jcl_strut_lca),a33]) + -1.0*multi_dot([B(a92,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca),a92])),(multi_dot([a93.T,a95,B(a33,a96),a33]) + multi_dot([a96.T,a37,B(a92,a93),a92]) + 2.0*multi_dot([a97,B(a36,a96).T,B(a94,a93),a92])),(multi_dot([B(a84,config.ubar_SU1_rbr_upright_jcr_tie_upright),a84]) + -1.0*multi_dot([B(a56,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright),a56])),(multi_dot([B(a84,config.ubar_SU1_rbr_upright_jcr_hub_bearing),a84]) + -1.0*multi_dot([B(a98,config.ubar_SU1_rbr_hub_jcr_hub_bearing),a98])),(multi_dot([a100,a102,B(a84,a103),a84]) + multi_dot([a103.T,a105,a106,a98]) + 2.0*multi_dot([a107,B(a104,a103).T,a108,a98])),(multi_dot([a100,a102,B(a84,a109),a84]) + multi_dot([a109.T,a105,a106,a98]) + 2.0*multi_dot([a107,B(a104,a109).T,a108,a98])),(derivative(config.F_jcr_hub_bearing,t,0.1,2)*-1.0*a1 + multi_dot([a111.T,a102,(a112*B(a84,a113) + a114*-1.0*B(a84,a115)),a84]) + multi_dot([(a112*multi_dot([a113.T,a105]) + a114*-1.0*multi_dot([a115.T,a105])),B(a98,a111),a98]) + 2.0*multi_dot([((a112*multi_dot([B(a104,a113),a84])).T + a114*-1.0*multi_dot([a107,B(a104,a115).T])),B(a101,a111),a98])),(multi_dot([B(a85,config.ubar_SU1_rbl_upright_jcl_tie_upright),a85]) + -1.0*multi_dot([B(a52,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright),a52])),(multi_dot([B(a85,config.ubar_SU1_rbl_upright_jcl_hub_bearing),a85]) + -1.0*multi_dot([B(a116,config.ubar_SU1_rbl_hub_jcl_hub_bearing),a116])),(multi_dot([a118,a120,B(a85,a121),a85]) + multi_dot([a121.T,a123,a124,a116]) + 2.0*multi_dot([a125,B(a122,a121).T,a126,a116])),(multi_dot([a118,a120,B(a85,a127),a85]) + multi_dot([a127.T,a123,a124,a116]) + 2.0*multi_dot([a125,B(a122,a127).T,a126,a116])),(derivative(config.F_jcl_hub_bearing,t,0.1,2)*-1.0*a1 + multi_dot([a129.T,a120,(a130*B(a85,a131) + a132*-1.0*B(a85,a133)),a85]) + multi_dot([(a130*multi_dot([a131.T,a123]) + a132*-1.0*multi_dot([a133.T,a123])),B(a116,a129),a116]) + 2.0*multi_dot([((a130*multi_dot([B(a122,a131),a85])).T + a132*-1.0*multi_dot([a125,B(a122,a133).T])),B(a119,a129),a116])),(multi_dot([a135,a90,a137,a42]) + multi_dot([a138,a45,a139,a86]) + 2.0*multi_dot([a140,a141,a142,a86])),(multi_dot([a135,a90,a144,a42]) + multi_dot([a145,a45,a139,a86]) + 2.0*multi_dot([a140,a146,a142,a86])),(multi_dot([a138,a45,a149]) + 2.0*multi_dot([a140,a141,a150]) + multi_dot([a151,a137,a42])),(multi_dot([a145,a45,a149]) + 2.0*multi_dot([a140,a146,a150]) + multi_dot([a151,a144,a42])),(multi_dot([a153,a95,a155,a47]) + multi_dot([a156,a50,a157,a92]) + 2.0*multi_dot([a158,a159,a160,a92])),(multi_dot([a153,a95,a162,a47]) + multi_dot([a163,a50,a157,a92]) + 2.0*multi_dot([a158,a164,a160,a92])),(multi_dot([a156,a50,a167]) + 2.0*multi_dot([a158,a159,a168]) + multi_dot([a169,a155,a47])),(multi_dot([a163,a50,a167]) + 2.0*multi_dot([a158,a164,a168]) + multi_dot([a169,a162,a47])),(multi_dot([B(a170,config.ubar_ST_rbs_coupler_jcs_rc_sph),a170]) + -1.0*multi_dot([B(a60,config.ubar_ST_rbr_rocker_jcs_rc_sph),a60])),(multi_dot([a172,a79,a174,a170]) + multi_dot([a175,a177,a178,a75]) + 2.0*multi_dot([a179,a180,a181,a75])),(multi_dot([a172,a79,a183,a170]) + multi_dot([a184,a177,a178,a75]) + 2.0*multi_dot([a179,a185,a181,a75])),(multi_dot([a175,a177,a188]) + 2.0*multi_dot([a179,a180,a189]) + multi_dot([a190,a174,a170])),(multi_dot([a184,a177,a188]) + 2.0*multi_dot([a179,a185,a189]) + multi_dot([a190,a183,a170])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a3.T,a3]))**(1.0/2.0),2.0*(multi_dot([a15.T,a15]))**(1.0/2.0),2.0*(multi_dot([a91,a24]))**(1.0/2.0),2.0*(multi_dot([a97,a33]))**(1.0/2.0),2.0*(multi_dot([a107,a84]))**(1.0/2.0),2.0*(multi_dot([a125,a85]))**(1.0/2.0),2.0*(multi_dot([a140,a42]))**(1.0/2.0),2.0*(multi_dot([a158,a47]))**(1.0/2.0),2.0*(multi_dot([a86.T,a86]))**(1.0/2.0),2.0*(multi_dot([a92.T,a92]))**(1.0/2.0),2.0*(multi_dot([a56.T,a56]))**(1.0/2.0),2.0*(multi_dot([a52.T,a52]))**(1.0/2.0),2.0*(multi_dot([a98.T,a98]))**(1.0/2.0),2.0*(multi_dot([a116.T,a116]))**(1.0/2.0),2.0*(multi_dot([a179,a170]))**(1.0/2.0),2.0*(multi_dot([a60.T,a60]))**(1.0/2.0),2.0*(multi_dot([a75.T,a75]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,4),dtype=np.float64)
        j1 = np.zeros((1,3),dtype=np.float64)
        j2 = np.eye(3,dtype=np.float64)
        j3 = self.P_ground
        j4 = config.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
        j5 = j4.T
        j6 = self.P_SU1_rbr_uca
        j7 = A(j6).T
        j8 = config.Mbar_ground_jcr_uca_chassis[:,0:1]
        j9 = config.Mbar_ground_jcr_uca_chassis[:,1:2]
        j10 = -1.0*j2
        j11 = A(j3).T
        j12 = B(j6,j4)
        j13 = config.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
        j14 = j13.T
        j15 = self.P_SU1_rbl_uca
        j16 = A(j15).T
        j17 = config.Mbar_ground_jcl_uca_chassis[:,0:1]
        j18 = config.Mbar_ground_jcl_uca_chassis[:,1:2]
        j19 = B(j15,j13)
        j20 = config.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
        j21 = j20.T
        j22 = self.P_SU1_rbr_lca
        j23 = A(j22).T
        j24 = config.Mbar_ground_jcr_lca_chassis[:,0:1]
        j25 = config.Mbar_ground_jcr_lca_chassis[:,1:2]
        j26 = B(j22,j20)
        j27 = config.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
        j28 = j27.T
        j29 = self.P_SU1_rbl_lca
        j30 = A(j29).T
        j31 = config.Mbar_ground_jcl_lca_chassis[:,0:1]
        j32 = config.Mbar_ground_jcl_lca_chassis[:,1:2]
        j33 = B(j29,j27)
        j34 = config.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j35 = self.P_SU1_rbr_upper_strut
        j36 = A(j35).T
        j37 = config.Mbar_ground_jcr_strut_chassis[:,0:1]
        j38 = config.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j39 = self.P_SU1_rbl_upper_strut
        j40 = A(j39).T
        j41 = config.Mbar_ground_jcl_strut_chassis[:,0:1]
        j42 = config.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]
        j43 = self.P_SU1_rbl_tie_rod
        j44 = config.Mbar_ground_jcl_tie_steering[:,0:1]
        j45 = config.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j46 = self.P_SU1_rbr_tie_rod
        j47 = config.Mbar_ground_jcr_tie_steering[:,0:1]
        j48 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        j49 = j48.T
        j50 = self.P_ST_rbr_rocker
        j51 = A(j50).T
        j52 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        j53 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        j54 = B(j50,j48)
        j55 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,0:1]
        j56 = config.F_jcr_rocker_ch(t,)
        j57 = cos(j56)
        j58 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        j59 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        j60 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        j61 = j60.T
        j62 = self.P_ST_rbl_rocker
        j63 = A(j62).T
        j64 = config.Mbar_ground_jcl_rocker_ch[:,0:1]
        j65 = config.Mbar_ground_jcl_rocker_ch[:,1:2]
        j66 = B(j62,j60)
        j67 = self.P_SU1_rbr_upright
        j68 = self.P_SU1_rbl_upright
        j69 = config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j70 = self.P_SU1_rbr_lower_strut
        j71 = A(j70).T
        j72 = config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
        j73 = config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j74 = self.P_SU1_rbl_lower_strut
        j75 = A(j74).T
        j76 = config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
        j77 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        j78 = j77.T
        j79 = self.P_SU1_rbr_hub
        j80 = A(j79).T
        j81 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        j82 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        j83 = A(j67).T
        j84 = B(j79,j77)
        j85 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,0:1]
        j86 = config.F_jcr_hub_bearing(t,)
        j87 = cos(j86)
        j88 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        j89 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        j90 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        j91 = j90.T
        j92 = self.P_SU1_rbl_hub
        j93 = A(j92).T
        j94 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        j95 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        j96 = A(j68).T
        j97 = B(j92,j90)
        j98 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,0:1]
        j99 = config.F_jcl_hub_bearing(t,)
        j100 = cos(j99)
        j101 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        j102 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        j103 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        j104 = j103.T
        j105 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
        j106 = B(j35,j105)
        j107 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
        j108 = B(j35,j107)
        j109 = j105.T
        j110 = multi_dot([j109,j36])
        j111 = config.ubar_SU1_rbr_upper_strut_jcr_strut
        j112 = B(j35,j111)
        j113 = config.ubar_SU1_rbr_lower_strut_jcr_strut
        j114 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,2:3].T,j36]) + multi_dot([j111.T,j36]) + -1.0*multi_dot([j113.T,j71]))
        j115 = j107.T
        j116 = multi_dot([j115,j36])
        j117 = B(j70,j103)
        j118 = B(j70,j113)
        j119 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        j120 = j119.T
        j121 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
        j122 = B(j39,j121)
        j123 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
        j124 = B(j39,j123)
        j125 = j121.T
        j126 = multi_dot([j125,j40])
        j127 = config.ubar_SU1_rbl_upper_strut_jcl_strut
        j128 = B(j39,j127)
        j129 = config.ubar_SU1_rbl_lower_strut_jcl_strut
        j130 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,2:3].T,j40]) + multi_dot([j127.T,j40]) + -1.0*multi_dot([j129.T,j75]))
        j131 = j123.T
        j132 = multi_dot([j131,j40])
        j133 = B(j74,j119)
        j134 = B(j74,j129)
        j135 = self.P_ST_rbs_coupler
        j136 = config.Mbar_ST_rbl_rocker_jcs_rc_cyl[:,2:3]
        j137 = j136.T
        j138 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,0:1]
        j139 = B(j135,j138)
        j140 = config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,1:2]
        j141 = B(j135,j140)
        j142 = j138.T
        j143 = A(j135).T
        j144 = multi_dot([j142,j143])
        j145 = config.ubar_ST_rbs_coupler_jcs_rc_cyl
        j146 = B(j135,j145)
        j147 = config.ubar_ST_rbl_rocker_jcs_rc_cyl
        j148 = (self.R_ST_rbs_coupler.T + -1.0*self.R_ST_rbl_rocker.T + multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_cyl[:,2:3].T,j143]) + multi_dot([j145.T,j143]) + -1.0*multi_dot([j147.T,j63]))
        j149 = j140.T
        j150 = multi_dot([j149,j143])
        j151 = B(j62,j136)
        j152 = B(j62,j147)

        self.jac_eq_blocks = [j1,j0,config.J_mcr_zact,j0,j1,j0,config.J_mcl_zact,j0,j2,B(j3,config.ubar_ground_jcr_uca_chassis),j10,-1.0*B(j6,config.ubar_SU1_rbr_uca_jcr_uca_chassis),j1,multi_dot([j5,j7,B(j3,j8)]),j1,multi_dot([j8.T,j11,j12]),j1,multi_dot([j5,j7,B(j3,j9)]),j1,multi_dot([j9.T,j11,j12]),j2,B(j3,config.ubar_ground_jcl_uca_chassis),j10,-1.0*B(j15,config.ubar_SU1_rbl_uca_jcl_uca_chassis),j1,multi_dot([j14,j16,B(j3,j17)]),j1,multi_dot([j17.T,j11,j19]),j1,multi_dot([j14,j16,B(j3,j18)]),j1,multi_dot([j18.T,j11,j19]),j2,B(j3,config.ubar_ground_jcr_lca_chassis),j10,-1.0*B(j22,config.ubar_SU1_rbr_lca_jcr_lca_chassis),j1,multi_dot([j21,j23,B(j3,j24)]),j1,multi_dot([j24.T,j11,j26]),j1,multi_dot([j21,j23,B(j3,j25)]),j1,multi_dot([j25.T,j11,j26]),j2,B(j3,config.ubar_ground_jcl_lca_chassis),j10,-1.0*B(j29,config.ubar_SU1_rbl_lca_jcl_lca_chassis),j1,multi_dot([j28,j30,B(j3,j31)]),j1,multi_dot([j31.T,j11,j33]),j1,multi_dot([j28,j30,B(j3,j32)]),j1,multi_dot([j32.T,j11,j33]),j2,B(j3,config.ubar_ground_jcr_strut_chassis),j10,-1.0*B(j35,config.ubar_SU1_rbr_upper_strut_jcr_strut_chassis),j1,multi_dot([j34.T,j36,B(j3,j37)]),j1,multi_dot([j37.T,j11,B(j35,j34)]),j2,B(j3,config.ubar_ground_jcl_strut_chassis),j10,-1.0*B(j39,config.ubar_SU1_rbl_upper_strut_jcl_strut_chassis),j1,multi_dot([j38.T,j40,B(j3,j41)]),j1,multi_dot([j41.T,j11,B(j39,j38)]),j2,B(j3,config.ubar_ground_jcl_tie_steering),j10,-1.0*B(j43,config.ubar_SU1_rbl_tie_rod_jcl_tie_steering),j1,multi_dot([j42.T,A(j43).T,B(j3,j44)]),j1,multi_dot([j44.T,j11,B(j43,j42)]),j2,B(j3,config.ubar_ground_jcr_tie_steering),j10,-1.0*B(j46,config.ubar_SU1_rbr_tie_rod_jcr_tie_steering),j1,multi_dot([j45.T,A(j46).T,B(j3,j47)]),j1,multi_dot([j47.T,j11,B(j46,j45)]),j2,B(j3,config.ubar_ground_jcr_rocker_ch),j10,-1.0*B(j50,config.ubar_ST_rbr_rocker_jcr_rocker_ch),j1,multi_dot([j49,j51,B(j3,j52)]),j1,multi_dot([j52.T,j11,j54]),j1,multi_dot([j49,j51,B(j3,j53)]),j1,multi_dot([j53.T,j11,j54]),j1,multi_dot([j55.T,j51,(j57*B(j3,j58) + sin(j56)*-1.0*B(j3,j59))]),j1,multi_dot([(j57*multi_dot([j58.T,j11]) + sin(j56)*-1.0*multi_dot([j59.T,j11])),B(j50,j55)]),j2,B(j3,config.ubar_ground_jcl_rocker_ch),j10,-1.0*B(j62,config.ubar_ST_rbl_rocker_jcl_rocker_ch),j1,multi_dot([j61,j63,B(j3,j64)]),j1,multi_dot([j64.T,j11,j66]),j1,multi_dot([j61,j63,B(j3,j65)]),j1,multi_dot([j65.T,j11,j66]),j2,B(j6,config.ubar_SU1_rbr_uca_jcr_uca_upright),j10,-1.0*B(j67,config.ubar_SU1_rbr_upright_jcr_uca_upright),j2,B(j15,config.ubar_SU1_rbl_uca_jcl_uca_upright),j10,-1.0*B(j68,config.ubar_SU1_rbl_upright_jcl_uca_upright),j2,B(j22,config.ubar_SU1_rbr_lca_jcr_lca_upright),j10,-1.0*B(j67,config.ubar_SU1_rbr_upright_jcr_lca_upright),j2,B(j22,config.ubar_SU1_rbr_lca_jcr_strut_lca),j10,-1.0*B(j70,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca),j1,multi_dot([j69.T,j71,B(j22,j72)]),j1,multi_dot([j72.T,j23,B(j70,j69)]),j2,B(j29,config.ubar_SU1_rbl_lca_jcl_lca_upright),j10,-1.0*B(j68,config.ubar_SU1_rbl_upright_jcl_lca_upright),j2,B(j29,config.ubar_SU1_rbl_lca_jcl_strut_lca),j10,-1.0*B(j74,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca),j1,multi_dot([j73.T,j75,B(j29,j76)]),j1,multi_dot([j76.T,j30,B(j74,j73)]),j2,B(j67,config.ubar_SU1_rbr_upright_jcr_tie_upright),j10,-1.0*B(j46,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright),j2,B(j67,config.ubar_SU1_rbr_upright_jcr_hub_bearing),j10,-1.0*B(j79,config.ubar_SU1_rbr_hub_jcr_hub_bearing),j1,multi_dot([j78,j80,B(j67,j81)]),j1,multi_dot([j81.T,j83,j84]),j1,multi_dot([j78,j80,B(j67,j82)]),j1,multi_dot([j82.T,j83,j84]),j1,multi_dot([j85.T,j80,(j87*B(j67,j88) + sin(j86)*-1.0*B(j67,j89))]),j1,multi_dot([(j87*multi_dot([j88.T,j83]) + sin(j86)*-1.0*multi_dot([j89.T,j83])),B(j79,j85)]),j2,B(j68,config.ubar_SU1_rbl_upright_jcl_tie_upright),j10,-1.0*B(j43,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright),j2,B(j68,config.ubar_SU1_rbl_upright_jcl_hub_bearing),j10,-1.0*B(j92,config.ubar_SU1_rbl_hub_jcl_hub_bearing),j1,multi_dot([j91,j93,B(j68,j94)]),j1,multi_dot([j94.T,j96,j97]),j1,multi_dot([j91,j93,B(j68,j95)]),j1,multi_dot([j95.T,j96,j97]),j1,multi_dot([j98.T,j93,(j100*B(j68,j101) + sin(j99)*-1.0*B(j68,j102))]),j1,multi_dot([(j100*multi_dot([j101.T,j96]) + sin(j99)*-1.0*multi_dot([j102.T,j96])),B(j92,j98)]),j1,multi_dot([j104,j71,j106]),j1,multi_dot([j109,j36,j117]),j1,multi_dot([j104,j71,j108]),j1,multi_dot([j115,j36,j117]),j110,(multi_dot([j109,j36,j112]) + multi_dot([j114,j106])),-1.0*j110,-1.0*multi_dot([j109,j36,j118]),j116,(multi_dot([j115,j36,j112]) + multi_dot([j114,j108])),-1.0*j116,-1.0*multi_dot([j115,j36,j118]),j1,multi_dot([j120,j75,j122]),j1,multi_dot([j125,j40,j133]),j1,multi_dot([j120,j75,j124]),j1,multi_dot([j131,j40,j133]),j126,(multi_dot([j125,j40,j128]) + multi_dot([j130,j122])),-1.0*j126,-1.0*multi_dot([j125,j40,j134]),j132,(multi_dot([j131,j40,j128]) + multi_dot([j130,j124])),-1.0*j132,-1.0*multi_dot([j131,j40,j134]),j2,B(j135,config.ubar_ST_rbs_coupler_jcs_rc_sph),j10,-1.0*B(j50,config.ubar_ST_rbr_rocker_jcs_rc_sph),j1,multi_dot([j137,j63,j139]),j1,multi_dot([j142,j143,j151]),j1,multi_dot([j137,j63,j141]),j1,multi_dot([j149,j143,j151]),j144,(multi_dot([j142,j143,j146]) + multi_dot([j148,j139])),-1.0*j144,-1.0*multi_dot([j142,j143,j152]),j150,(multi_dot([j149,j143,j146]) + multi_dot([j148,j141])),-1.0*j150,-1.0*multi_dot([j149,j143,j152]),j2,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j6.T,2.0*j15.T,2.0*j22.T,2.0*j29.T,2.0*j67.T,2.0*j68.T,2.0*j35.T,2.0*j39.T,2.0*j70.T,2.0*j74.T,2.0*j46.T,2.0*j43.T,2.0*j79.T,2.0*j92.T,2.0*j135.T,2.0*j50.T,2.0*j62.T]
  
