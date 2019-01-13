
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
        self.R_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_ground = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
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

    def eval_constants(self):

        c0 = A(self.P_SU1_rbr_uca).T
        c1 = self.pt_jcr_uca_upright
        c2 = A(self.P_SU1_rbr_upright).T
        c3 = -1.0*multi_dot([c2,self.R_SU1_rbr_upright])
        c4 = Triad(self.ax_jcr_uca_upright,)
        c5 = A(self.P_SU1_rbl_uca).T
        c6 = self.pt_jcl_uca_upright
        c7 = A(self.P_SU1_rbl_upright).T
        c8 = -1.0*multi_dot([c7,self.R_SU1_rbl_upright])
        c9 = Triad(self.ax_jcl_uca_upright,)
        c10 = A(self.P_SU1_rbr_lca).T
        c11 = self.pt_jcr_lca_upright
        c12 = -1.0*multi_dot([c10,self.R_SU1_rbr_lca])
        c13 = Triad(self.ax_jcr_lca_upright,)
        c14 = self.pt_jcr_strut_lca
        c15 = A(self.P_SU1_rbr_lower_strut).T
        c16 = -1.0*multi_dot([c15,self.R_SU1_rbr_lower_strut])
        c17 = Triad(self.ax_jcr_strut_lca,)
        c18 = A(self.P_SU1_rbl_lca).T
        c19 = self.pt_jcl_lca_upright
        c20 = -1.0*multi_dot([c18,self.R_SU1_rbl_lca])
        c21 = Triad(self.ax_jcl_lca_upright,)
        c22 = self.pt_jcl_strut_lca
        c23 = A(self.P_SU1_rbl_lower_strut).T
        c24 = -1.0*multi_dot([c23,self.R_SU1_rbl_lower_strut])
        c25 = Triad(self.ax_jcl_strut_lca,)
        c26 = self.pt_jcr_tie_upright
        c27 = A(self.P_SU1_rbr_tie_rod).T
        c28 = Triad(self.ax_jcr_tie_upright,)
        c29 = self.pt_jcr_hub_bearing
        c30 = A(self.P_SU1_rbr_hub).T
        c31 = Triad(self.ax_jcr_hub_bearing,)
        c32 = self.pt_jcl_tie_upright
        c33 = A(self.P_SU1_rbl_tie_rod).T
        c34 = Triad(self.ax_jcl_tie_upright,)
        c35 = self.pt_jcl_hub_bearing
        c36 = A(self.P_SU1_rbl_hub).T
        c37 = Triad(self.ax_jcl_hub_bearing,)
        c38 = A(self.P_SU1_rbr_upper_strut).T
        c39 = self.pt_jcr_strut
        c40 = Triad(self.ax_jcr_strut,)
        c41 = A(self.P_SU1_rbl_upper_strut).T
        c42 = self.pt_jcl_strut
        c43 = Triad(self.ax_jcl_strut,)

        self.ubar_SU1_rbr_uca_jcr_uca_upright = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,self.R_SU1_rbr_uca]))
        self.ubar_SU1_rbr_upright_jcr_uca_upright = (multi_dot([c2,c1]) + c3)
        self.Mbar_SU1_rbr_uca_jcr_uca_upright = multi_dot([c0,c4])
        self.Mbar_SU1_rbr_upright_jcr_uca_upright = multi_dot([c2,c4])
        self.ubar_SU1_rbl_uca_jcl_uca_upright = (multi_dot([c5,c6]) + -1.0*multi_dot([c5,self.R_SU1_rbl_uca]))
        self.ubar_SU1_rbl_upright_jcl_uca_upright = (multi_dot([c7,c6]) + c8)
        self.Mbar_SU1_rbl_uca_jcl_uca_upright = multi_dot([c5,c9])
        self.Mbar_SU1_rbl_upright_jcl_uca_upright = multi_dot([c7,c9])
        self.ubar_SU1_rbr_lca_jcr_lca_upright = (multi_dot([c10,c11]) + c12)
        self.ubar_SU1_rbr_upright_jcr_lca_upright = (multi_dot([c2,c11]) + c3)
        self.Mbar_SU1_rbr_lca_jcr_lca_upright = multi_dot([c10,c13])
        self.Mbar_SU1_rbr_upright_jcr_lca_upright = multi_dot([c2,c13])
        self.ubar_SU1_rbr_lca_jcr_strut_lca = (multi_dot([c10,c14]) + c12)
        self.ubar_SU1_rbr_lower_strut_jcr_strut_lca = (multi_dot([c15,c14]) + c16)
        self.Mbar_SU1_rbr_lca_jcr_strut_lca = multi_dot([c10,c17])
        self.Mbar_SU1_rbr_lower_strut_jcr_strut_lca = multi_dot([c15,Triad(self.ax2_jcr_strut_lca,c17[0:3,1:2])])
        self.ubar_SU1_rbl_lca_jcl_lca_upright = (multi_dot([c18,c19]) + c20)
        self.ubar_SU1_rbl_upright_jcl_lca_upright = (multi_dot([c7,c19]) + c8)
        self.Mbar_SU1_rbl_lca_jcl_lca_upright = multi_dot([c18,c21])
        self.Mbar_SU1_rbl_upright_jcl_lca_upright = multi_dot([c7,c21])
        self.ubar_SU1_rbl_lca_jcl_strut_lca = (multi_dot([c18,c22]) + c20)
        self.ubar_SU1_rbl_lower_strut_jcl_strut_lca = (multi_dot([c23,c22]) + c24)
        self.Mbar_SU1_rbl_lca_jcl_strut_lca = multi_dot([c18,c25])
        self.Mbar_SU1_rbl_lower_strut_jcl_strut_lca = multi_dot([c23,Triad(self.ax2_jcl_strut_lca,c25[0:3,1:2])])
        self.ubar_SU1_rbr_upright_jcr_tie_upright = (multi_dot([c2,c26]) + c3)
        self.ubar_SU1_rbr_tie_rod_jcr_tie_upright = (multi_dot([c27,c26]) + -1.0*multi_dot([c27,self.R_SU1_rbr_tie_rod]))
        self.Mbar_SU1_rbr_upright_jcr_tie_upright = multi_dot([c2,c28])
        self.Mbar_SU1_rbr_tie_rod_jcr_tie_upright = multi_dot([c27,c28])
        self.ubar_SU1_rbr_upright_jcr_hub_bearing = (multi_dot([c2,c29]) + c3)
        self.ubar_SU1_rbr_hub_jcr_hub_bearing = (multi_dot([c30,c29]) + -1.0*multi_dot([c30,self.R_SU1_rbr_hub]))
        self.Mbar_SU1_rbr_upright_jcr_hub_bearing = multi_dot([c2,c31])
        self.Mbar_SU1_rbr_hub_jcr_hub_bearing = multi_dot([c30,c31])
        self.ubar_SU1_rbl_upright_jcl_tie_upright = (multi_dot([c7,c32]) + c8)
        self.ubar_SU1_rbl_tie_rod_jcl_tie_upright = (multi_dot([c33,c32]) + -1.0*multi_dot([c33,self.R_SU1_rbl_tie_rod]))
        self.Mbar_SU1_rbl_upright_jcl_tie_upright = multi_dot([c7,c34])
        self.Mbar_SU1_rbl_tie_rod_jcl_tie_upright = multi_dot([c33,c34])
        self.ubar_SU1_rbl_upright_jcl_hub_bearing = (multi_dot([c7,c35]) + c8)
        self.ubar_SU1_rbl_hub_jcl_hub_bearing = (multi_dot([c36,c35]) + -1.0*multi_dot([c36,self.R_SU1_rbl_hub]))
        self.Mbar_SU1_rbl_upright_jcl_hub_bearing = multi_dot([c7,c37])
        self.Mbar_SU1_rbl_hub_jcl_hub_bearing = multi_dot([c36,c37])
        self.ubar_SU1_rbr_upper_strut_jcr_strut = (multi_dot([c38,c39]) + -1.0*multi_dot([c38,self.R_SU1_rbr_upper_strut]))
        self.ubar_SU1_rbr_lower_strut_jcr_strut = (multi_dot([c15,c39]) + c16)
        self.Mbar_SU1_rbr_upper_strut_jcr_strut = multi_dot([c38,c40])
        self.Mbar_SU1_rbr_lower_strut_jcr_strut = multi_dot([c15,c40])
        self.ubar_SU1_rbl_upper_strut_jcl_strut = (multi_dot([c41,c42]) + -1.0*multi_dot([c41,self.R_SU1_rbl_upper_strut]))
        self.ubar_SU1_rbl_lower_strut_jcl_strut = (multi_dot([c23,c42]) + c24)
        self.Mbar_SU1_rbl_upper_strut_jcl_strut = multi_dot([c41,c43])
        self.Mbar_SU1_rbl_lower_strut_jcl_strut = multi_dot([c23,c43])

    @property
    def q_initial(self):
        q = np.concatenate([self.R_ground,self.P_ground,self.R_SU1_rbr_uca,self.P_SU1_rbr_uca,self.R_SU1_rbl_uca,self.P_SU1_rbl_uca,self.R_SU1_rbr_lca,self.P_SU1_rbr_lca,self.R_SU1_rbl_lca,self.P_SU1_rbl_lca,self.R_SU1_rbr_upright,self.P_SU1_rbr_upright,self.R_SU1_rbl_upright,self.P_SU1_rbl_upright,self.R_SU1_rbr_upper_strut,self.P_SU1_rbr_upper_strut,self.R_SU1_rbl_upper_strut,self.P_SU1_rbl_upper_strut,self.R_SU1_rbr_lower_strut,self.P_SU1_rbr_lower_strut,self.R_SU1_rbl_lower_strut,self.P_SU1_rbl_lower_strut,self.R_SU1_rbr_tie_rod,self.P_SU1_rbr_tie_rod,self.R_SU1_rbl_tie_rod,self.P_SU1_rbl_tie_rod,self.R_SU1_rbr_hub,self.P_SU1_rbr_hub,self.R_SU1_rbl_hub,self.P_SU1_rbl_hub])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.Rd_ground,self.Pd_ground,self.Rd_SU1_rbr_uca,self.Pd_SU1_rbr_uca,self.Rd_SU1_rbl_uca,self.Pd_SU1_rbl_uca,self.Rd_SU1_rbr_lca,self.Pd_SU1_rbr_lca,self.Rd_SU1_rbl_lca,self.Pd_SU1_rbl_lca,self.Rd_SU1_rbr_upright,self.Pd_SU1_rbr_upright,self.Rd_SU1_rbl_upright,self.Pd_SU1_rbl_upright,self.Rd_SU1_rbr_upper_strut,self.Pd_SU1_rbr_upper_strut,self.Rd_SU1_rbl_upper_strut,self.Pd_SU1_rbl_upper_strut,self.Rd_SU1_rbr_lower_strut,self.Pd_SU1_rbr_lower_strut,self.Rd_SU1_rbl_lower_strut,self.Pd_SU1_rbl_lower_strut,self.Rd_SU1_rbr_tie_rod,self.Pd_SU1_rbr_tie_rod,self.Rd_SU1_rbl_tie_rod,self.Pd_SU1_rbl_tie_rod,self.Rd_SU1_rbr_hub,self.Pd_SU1_rbr_hub,self.Rd_SU1_rbl_hub,self.Pd_SU1_rbl_hub])
        return qd



class numerical_assembly(object):

    def __init__(self,config):
        self.t = 0.0
        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.nve = 70

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,29,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43])
        self.jac_cols = np.array([0,1,26,27,0,1,28,29,2,3,10,11,4,5,12,13,6,7,10,11,6,7,18,19,6,7,18,19,8,9,12,13,8,9,20,21,8,9,20,21,10,11,22,23,10,11,26,27,10,11,26,27,10,11,26,27,10,11,26,27,12,13,24,25,12,13,28,29,12,13,28,29,12,13,28,29,12,13,28,29,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,20,21,16,17,20,21,16,17,20,21,0,1,0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29])

    
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

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = x0[2]
        x2 = np.eye(1,dtype=np.float64)
        x3 = self.R_SU1_rbr_upright
        x4 = -1.0*x3
        x5 = self.P_SU1_rbr_uca
        x6 = self.P_SU1_rbr_upright
        x7 = A(x6)
        x8 = self.R_SU1_rbl_upright
        x9 = -1.0*x8
        x10 = self.P_SU1_rbl_uca
        x11 = self.P_SU1_rbl_upright
        x12 = A(x11)
        x13 = self.R_SU1_rbr_lca
        x14 = self.P_SU1_rbr_lca
        x15 = A(x14)
        x16 = -1.0*self.R_SU1_rbr_lower_strut
        x17 = self.P_SU1_rbr_lower_strut
        x18 = A(x17)
        x19 = self.R_SU1_rbl_lca
        x20 = self.P_SU1_rbl_lca
        x21 = A(x20)
        x22 = -1.0*self.R_SU1_rbl_lower_strut
        x23 = self.P_SU1_rbl_lower_strut
        x24 = A(x23)
        x25 = self.P_SU1_rbr_tie_rod
        x26 = self.P_SU1_rbr_hub
        x27 = A(x26)
        x28 = x7.T
        x29 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        x30 = config.F_jcr_hub_bearing(t,)
        x31 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,0:1]
        x32 = self.P_SU1_rbl_tie_rod
        x33 = self.P_SU1_rbl_hub
        x34 = A(x33)
        x35 = x12.T
        x36 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        x37 = config.F_jcl_hub_bearing(t,)
        x38 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,0:1]
        x39 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1].T
        x40 = self.P_SU1_rbr_upper_strut
        x41 = A(x40)
        x42 = x41.T
        x43 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        x44 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2].T
        x45 = (self.R_SU1_rbr_upper_strut + x16 + multi_dot([x41,config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,2:3]]) + multi_dot([x41,config.ubar_SU1_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x18,config.ubar_SU1_rbr_lower_strut_jcr_strut]))
        x46 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1].T
        x47 = self.P_SU1_rbl_upper_strut
        x48 = A(x47)
        x49 = x48.T
        x50 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        x51 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2].T
        x52 = (self.R_SU1_rbl_upper_strut + x22 + multi_dot([x48,config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,2:3]]) + multi_dot([x48,config.ubar_SU1_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x24,config.ubar_SU1_rbl_lower_strut_jcl_strut]))
        x53 = -1.0*x2

        self.pos_eq_blocks = [x1 - 1*config.F_mcr_zact(t,)*x2,x1 - 1*config.F_mcl_zact(t,)*x2,(self.R_SU1_rbr_uca + x4 + multi_dot([A(x5),config.ubar_SU1_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x7,config.ubar_SU1_rbr_upright_jcr_uca_upright])),(self.R_SU1_rbl_uca + x9 + multi_dot([A(x10),config.ubar_SU1_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x12,config.ubar_SU1_rbl_upright_jcl_uca_upright])),(x13 + x4 + multi_dot([x15,config.ubar_SU1_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x7,config.ubar_SU1_rbr_upright_jcr_lca_upright])),(x13 + x16 + multi_dot([x15,config.ubar_SU1_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x18,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca])),multi_dot([config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1].T,x15.T,x18,config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x19 + x9 + multi_dot([x21,config.ubar_SU1_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x12,config.ubar_SU1_rbl_upright_jcl_lca_upright])),(x19 + x22 + multi_dot([x21,config.ubar_SU1_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x24,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1].T,x21.T,x24,config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x3 + -1.0*self.R_SU1_rbr_tie_rod + multi_dot([x7,config.ubar_SU1_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([A(x25),config.ubar_SU1_rbr_tie_rod_jcr_tie_upright])),(x3 + -1.0*self.R_SU1_rbr_hub + multi_dot([x7,config.ubar_SU1_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x27,config.ubar_SU1_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1].T,x28,x27,x29]),multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2].T,x28,x27,x29]),(cos(x30)*multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2].T,x28,x27,x31]) + sin(x30)*-1.0*multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1].T,x28,x27,x31])),(x8 + -1.0*self.R_SU1_rbl_tie_rod + multi_dot([x12,config.ubar_SU1_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([A(x32),config.ubar_SU1_rbl_tie_rod_jcl_tie_upright])),(x8 + -1.0*self.R_SU1_rbl_hub + multi_dot([x12,config.ubar_SU1_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x34,config.ubar_SU1_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1].T,x35,x34,x36]),multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2].T,x35,x34,x36]),(cos(x37)*multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2].T,x35,x34,x38]) + sin(x37)*-1.0*multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1].T,x35,x34,x38])),multi_dot([x39,x42,x18,x43]),multi_dot([x44,x42,x18,x43]),multi_dot([x39,x42,x45]),multi_dot([x44,x42,x45]),multi_dot([x46,x49,x24,x50]),multi_dot([x51,x49,x24,x50]),multi_dot([x46,x49,x52]),multi_dot([x51,x49,x52]),x0,(self.P_ground + -1.0*config.Pg_ground),(x53 + (multi_dot([x5.T,x5]))**(1.0/2.0)),(x53 + (multi_dot([x10.T,x10]))**(1.0/2.0)),(x53 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x53 + (multi_dot([x20.T,x20]))**(1.0/2.0)),(x53 + (multi_dot([x6.T,x6]))**(1.0/2.0)),(x53 + (multi_dot([x11.T,x11]))**(1.0/2.0)),(x53 + (multi_dot([x40.T,x40]))**(1.0/2.0)),(x53 + (multi_dot([x47.T,x47]))**(1.0/2.0)),(x53 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x53 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x53 + (multi_dot([x25.T,x25]))**(1.0/2.0)),(x53 + (multi_dot([x32.T,x32]))**(1.0/2.0)),(x53 + (multi_dot([x26.T,x26]))**(1.0/2.0)),(x53 + (multi_dot([x33.T,x33]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)
        v1 = np.eye(1,dtype=np.float64)
        v2 = np.zeros((3,1),dtype=np.float64)

        self.vel_eq_blocks = [(v0 + derivative(config.F_mcr_zact,t,0.1,1)*-1.0*v1),(v0 + derivative(config.F_mcl_zact,t,0.1,1)*-1.0*v1),v2,v2,v2,v2,v0,v2,v2,v0,v2,v2,v0,v0,(v0 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v1),v2,v2,v0,v0,(v0 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v1),v0,v0,v0,v0,v0,v0,v0,v0,v2,np.zeros((4,1),dtype=np.float64),v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.zeros((1,1),dtype=np.float64)
        a1 = np.eye(1,dtype=np.float64)
        a2 = self.Pd_SU1_rbr_uca
        a3 = self.Pd_SU1_rbr_upright
        a4 = self.Pd_SU1_rbl_uca
        a5 = self.Pd_SU1_rbl_upright
        a6 = self.Pd_SU1_rbr_lca
        a7 = self.Pd_SU1_rbr_lower_strut
        a8 = config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
        a9 = self.P_SU1_rbr_lca
        a10 = config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a11 = self.P_SU1_rbr_lower_strut
        a12 = A(a11).T
        a13 = a6.T
        a14 = self.Pd_SU1_rbl_lca
        a15 = self.Pd_SU1_rbl_lower_strut
        a16 = config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a17 = self.P_SU1_rbl_lower_strut
        a18 = A(a17).T
        a19 = config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
        a20 = self.P_SU1_rbl_lca
        a21 = a14.T
        a22 = self.Pd_SU1_rbr_tie_rod
        a23 = self.Pd_SU1_rbr_hub
        a24 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        a25 = self.P_SU1_rbr_upright
        a26 = A(a25).T
        a27 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        a28 = B(a23,a27)
        a29 = a27.T
        a30 = self.P_SU1_rbr_hub
        a31 = A(a30).T
        a32 = a3.T
        a33 = B(a30,a27)
        a34 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        a35 = config.F_jcr_hub_bearing(t,)
        a36 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,0:1]
        a37 = cos(a35)
        a38 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        a39 = sin(a35)
        a40 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        a41 = self.Pd_SU1_rbl_tie_rod
        a42 = self.Pd_SU1_rbl_hub
        a43 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        a44 = self.P_SU1_rbl_upright
        a45 = A(a44).T
        a46 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        a47 = B(a42,a46)
        a48 = a46.T
        a49 = self.P_SU1_rbl_hub
        a50 = A(a49).T
        a51 = a5.T
        a52 = B(a49,a46)
        a53 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        a54 = config.F_jcl_hub_bearing(t,)
        a55 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,0:1]
        a56 = cos(a54)
        a57 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        a58 = sin(a54)
        a59 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        a60 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        a61 = a60.T
        a62 = self.Pd_SU1_rbr_upper_strut
        a63 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
        a64 = B(a62,a63)
        a65 = a63.T
        a66 = self.P_SU1_rbr_upper_strut
        a67 = A(a66).T
        a68 = B(a7,a60)
        a69 = a62.T
        a70 = B(a66,a63).T
        a71 = B(a11,a60)
        a72 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
        a73 = B(a62,a72)
        a74 = a72.T
        a75 = B(a66,a72).T
        a76 = config.ubar_SU1_rbr_upper_strut_jcr_strut
        a77 = config.ubar_SU1_rbr_lower_strut_jcr_strut
        a78 = (multi_dot([B(a62,a76),a62]) + -1.0*multi_dot([B(a7,a77),a7]))
        a79 = (self.Rd_SU1_rbr_upper_strut + -1.0*self.Rd_SU1_rbr_lower_strut + multi_dot([B(a11,a77),a7]) + multi_dot([B(a66,a76),a62]))
        a80 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,2:3].T,a67]) + multi_dot([a76.T,a67]) + -1.0*multi_dot([a77.T,a12]))
        a81 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        a82 = a81.T
        a83 = self.Pd_SU1_rbl_upper_strut
        a84 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
        a85 = B(a83,a84)
        a86 = a84.T
        a87 = self.P_SU1_rbl_upper_strut
        a88 = A(a87).T
        a89 = B(a15,a81)
        a90 = a83.T
        a91 = B(a87,a84).T
        a92 = B(a17,a81)
        a93 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
        a94 = B(a83,a93)
        a95 = a93.T
        a96 = B(a87,a93).T
        a97 = config.ubar_SU1_rbl_upper_strut_jcl_strut
        a98 = config.ubar_SU1_rbl_lower_strut_jcl_strut
        a99 = (multi_dot([B(a83,a97),a83]) + -1.0*multi_dot([B(a15,a98),a15]))
        a100 = (self.Rd_SU1_rbl_upper_strut + -1.0*self.Rd_SU1_rbl_lower_strut + multi_dot([B(a17,a98),a15]) + multi_dot([B(a87,a97),a83]))
        a101 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,2:3].T,a88]) + multi_dot([a97.T,a88]) + -1.0*multi_dot([a98.T,a18]))

        self.acc_eq_blocks = [(a0 + derivative(config.F_mcr_zact,t,0.1,2)*-1.0*a1),(a0 + derivative(config.F_mcl_zact,t,0.1,2)*-1.0*a1),(multi_dot([B(a2,config.ubar_SU1_rbr_uca_jcr_uca_upright),a2]) + -1.0*multi_dot([B(a3,config.ubar_SU1_rbr_upright_jcr_uca_upright),a3])),(multi_dot([B(a4,config.ubar_SU1_rbl_uca_jcl_uca_upright),a4]) + -1.0*multi_dot([B(a5,config.ubar_SU1_rbl_upright_jcl_uca_upright),a5])),(multi_dot([B(a6,config.ubar_SU1_rbr_lca_jcr_lca_upright),a6]) + -1.0*multi_dot([B(a3,config.ubar_SU1_rbr_upright_jcr_lca_upright),a3])),(multi_dot([B(a6,config.ubar_SU1_rbr_lca_jcr_strut_lca),a6]) + -1.0*multi_dot([B(a7,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca),a7])),(multi_dot([a8.T,A(a9).T,B(a7,a10),a7]) + multi_dot([a10.T,a12,B(a6,a8),a6]) + 2.0*multi_dot([a13,B(a9,a8).T,B(a11,a10),a7])),(multi_dot([B(a14,config.ubar_SU1_rbl_lca_jcl_lca_upright),a14]) + -1.0*multi_dot([B(a5,config.ubar_SU1_rbl_upright_jcl_lca_upright),a5])),(multi_dot([B(a14,config.ubar_SU1_rbl_lca_jcl_strut_lca),a14]) + -1.0*multi_dot([B(a15,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca),a15])),(multi_dot([a16.T,a18,B(a14,a19),a14]) + multi_dot([a19.T,A(a20).T,B(a15,a16),a15]) + 2.0*multi_dot([a21,B(a20,a19).T,B(a17,a16),a15])),(multi_dot([B(a3,config.ubar_SU1_rbr_upright_jcr_tie_upright),a3]) + -1.0*multi_dot([B(a22,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright),a22])),(multi_dot([B(a3,config.ubar_SU1_rbr_upright_jcr_hub_bearing),a3]) + -1.0*multi_dot([B(a23,config.ubar_SU1_rbr_hub_jcr_hub_bearing),a23])),(multi_dot([a24.T,a26,a28,a23]) + multi_dot([a29,a31,B(a3,a24),a3]) + 2.0*multi_dot([a32,B(a25,a24).T,a33,a23])),(multi_dot([a34.T,a26,a28,a23]) + multi_dot([a29,a31,B(a3,a34),a3]) + 2.0*multi_dot([a32,B(a25,a34).T,a33,a23])),(derivative(a35,t,0.1,2)*-1.0*a1 + multi_dot([a36.T,a31,(a37*B(a3,a38) + a39*-1.0*B(a3,a40)),a3]) + multi_dot([(a37*multi_dot([a38.T,a26]) + a39*-1.0*multi_dot([a40.T,a26])),B(a23,a36),a23]) + 2.0*multi_dot([((a37*multi_dot([B(a25,a38),a3])).T + a39*-1.0*multi_dot([a32,B(a25,a40).T])),B(a30,a36),a23])),(multi_dot([B(a5,config.ubar_SU1_rbl_upright_jcl_tie_upright),a5]) + -1.0*multi_dot([B(a41,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright),a41])),(multi_dot([B(a5,config.ubar_SU1_rbl_upright_jcl_hub_bearing),a5]) + -1.0*multi_dot([B(a42,config.ubar_SU1_rbl_hub_jcl_hub_bearing),a42])),(multi_dot([a43.T,a45,a47,a42]) + multi_dot([a48,a50,B(a5,a43),a5]) + 2.0*multi_dot([a51,B(a44,a43).T,a52,a42])),(multi_dot([a53.T,a45,a47,a42]) + multi_dot([a48,a50,B(a5,a53),a5]) + 2.0*multi_dot([a51,B(a44,a53).T,a52,a42])),(derivative(a54,t,0.1,2)*-1.0*a1 + multi_dot([a55.T,a50,(a56*B(a5,a57) + a58*-1.0*B(a5,a59)),a5]) + multi_dot([(a56*multi_dot([a57.T,a45]) + a58*-1.0*multi_dot([a59.T,a45])),B(a42,a55),a42]) + 2.0*multi_dot([((a56*multi_dot([B(a44,a57),a5])).T + a58*-1.0*multi_dot([a51,B(a44,a59).T])),B(a49,a55),a42])),(multi_dot([a61,a12,a64,a62]) + multi_dot([a65,a67,a68,a7]) + 2.0*multi_dot([a69,a70,a71,a7])),(multi_dot([a61,a12,a73,a62]) + multi_dot([a74,a67,a68,a7]) + 2.0*multi_dot([a69,a75,a71,a7])),(multi_dot([a65,a67,a78]) + 2.0*multi_dot([a69,a70,a79]) + multi_dot([a80,a64,a62])),(multi_dot([a74,a67,a78]) + 2.0*multi_dot([a69,a75,a79]) + multi_dot([a80,a73,a62])),(multi_dot([a82,a18,a85,a83]) + multi_dot([a86,a88,a89,a15]) + 2.0*multi_dot([a90,a91,a92,a15])),(multi_dot([a82,a18,a94,a83]) + multi_dot([a95,a88,a89,a15]) + 2.0*multi_dot([a90,a96,a92,a15])),(multi_dot([a86,a88,a99]) + 2.0*multi_dot([a90,a91,a100]) + multi_dot([a101,a85,a83])),(multi_dot([a95,a88,a99]) + 2.0*multi_dot([a90,a96,a100]) + multi_dot([a101,a94,a83])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a2.T,a2]))**(1.0/2.0),2.0*(multi_dot([a4.T,a4]))**(1.0/2.0),2.0*(multi_dot([a13,a6]))**(1.0/2.0),2.0*(multi_dot([a21,a14]))**(1.0/2.0),2.0*(multi_dot([a32,a3]))**(1.0/2.0),2.0*(multi_dot([a51,a5]))**(1.0/2.0),2.0*(multi_dot([a69,a62]))**(1.0/2.0),2.0*(multi_dot([a90,a83]))**(1.0/2.0),2.0*(multi_dot([a7.T,a7]))**(1.0/2.0),2.0*(multi_dot([a15.T,a15]))**(1.0/2.0),2.0*(multi_dot([a22.T,a22]))**(1.0/2.0),2.0*(multi_dot([a41.T,a41]))**(1.0/2.0),2.0*(multi_dot([a23.T,a23]))**(1.0/2.0),2.0*(multi_dot([a42.T,a42]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,4),dtype=np.float64)
        j1 = np.zeros((1,3),dtype=np.float64)
        j2 = np.eye(3,dtype=np.float64)
        j3 = self.P_SU1_rbr_uca
        j4 = -1.0*j2
        j5 = self.P_SU1_rbr_upright
        j6 = self.P_SU1_rbl_uca
        j7 = self.P_SU1_rbl_upright
        j8 = self.P_SU1_rbr_lca
        j9 = config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j10 = self.P_SU1_rbr_lower_strut
        j11 = A(j10).T
        j12 = config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
        j13 = self.P_SU1_rbl_lca
        j14 = config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j15 = self.P_SU1_rbl_lower_strut
        j16 = A(j15).T
        j17 = config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
        j18 = self.P_SU1_rbr_tie_rod
        j19 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        j20 = j19.T
        j21 = self.P_SU1_rbr_hub
        j22 = A(j21).T
        j23 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        j24 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        j25 = A(j5).T
        j26 = B(j21,j19)
        j27 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,0:1]
        j28 = config.F_jcr_hub_bearing(t,)
        j29 = cos(j28)
        j30 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        j31 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        j32 = self.P_SU1_rbl_tie_rod
        j33 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        j34 = j33.T
        j35 = self.P_SU1_rbl_hub
        j36 = A(j35).T
        j37 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        j38 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        j39 = A(j7).T
        j40 = B(j35,j33)
        j41 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,0:1]
        j42 = config.F_jcl_hub_bearing(t,)
        j43 = cos(j42)
        j44 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        j45 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        j46 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        j47 = j46.T
        j48 = self.P_SU1_rbr_upper_strut
        j49 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
        j50 = B(j48,j49)
        j51 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
        j52 = B(j48,j51)
        j53 = j49.T
        j54 = A(j48).T
        j55 = multi_dot([j53,j54])
        j56 = config.ubar_SU1_rbr_upper_strut_jcr_strut
        j57 = B(j48,j56)
        j58 = config.ubar_SU1_rbr_lower_strut_jcr_strut
        j59 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,2:3].T,j54]) + multi_dot([j56.T,j54]) + -1.0*multi_dot([j58.T,j11]))
        j60 = j51.T
        j61 = multi_dot([j60,j54])
        j62 = B(j10,j46)
        j63 = B(j10,j58)
        j64 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        j65 = j64.T
        j66 = self.P_SU1_rbl_upper_strut
        j67 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
        j68 = B(j66,j67)
        j69 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
        j70 = B(j66,j69)
        j71 = j67.T
        j72 = A(j66).T
        j73 = multi_dot([j71,j72])
        j74 = config.ubar_SU1_rbl_upper_strut_jcl_strut
        j75 = B(j66,j74)
        j76 = config.ubar_SU1_rbl_lower_strut_jcl_strut
        j77 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,2:3].T,j72]) + multi_dot([j74.T,j72]) + -1.0*multi_dot([j76.T,j16]))
        j78 = j69.T
        j79 = multi_dot([j78,j72])
        j80 = B(j15,j64)
        j81 = B(j15,j76)

        self.jac_eq_blocks = [config.J_mcr_zact,j0,j1,j0,config.J_mcl_zact,j0,j1,j0,j2,B(j3,config.ubar_SU1_rbr_uca_jcr_uca_upright),j4,-1.0*B(j5,config.ubar_SU1_rbr_upright_jcr_uca_upright),j2,B(j6,config.ubar_SU1_rbl_uca_jcl_uca_upright),j4,-1.0*B(j7,config.ubar_SU1_rbl_upright_jcl_uca_upright),j2,B(j8,config.ubar_SU1_rbr_lca_jcr_lca_upright),j4,-1.0*B(j5,config.ubar_SU1_rbr_upright_jcr_lca_upright),j2,B(j8,config.ubar_SU1_rbr_lca_jcr_strut_lca),j4,-1.0*B(j10,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca),j1,multi_dot([j9.T,j11,B(j8,j12)]),j1,multi_dot([j12.T,A(j8).T,B(j10,j9)]),j2,B(j13,config.ubar_SU1_rbl_lca_jcl_lca_upright),j4,-1.0*B(j7,config.ubar_SU1_rbl_upright_jcl_lca_upright),j2,B(j13,config.ubar_SU1_rbl_lca_jcl_strut_lca),j4,-1.0*B(j15,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca),j1,multi_dot([j14.T,j16,B(j13,j17)]),j1,multi_dot([j17.T,A(j13).T,B(j15,j14)]),j2,B(j5,config.ubar_SU1_rbr_upright_jcr_tie_upright),j4,-1.0*B(j18,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright),j2,B(j5,config.ubar_SU1_rbr_upright_jcr_hub_bearing),j4,-1.0*B(j21,config.ubar_SU1_rbr_hub_jcr_hub_bearing),j1,multi_dot([j20,j22,B(j5,j23)]),j1,multi_dot([j23.T,j25,j26]),j1,multi_dot([j20,j22,B(j5,j24)]),j1,multi_dot([j24.T,j25,j26]),j1,multi_dot([j27.T,j22,(j29*B(j5,j30) + sin(j28)*-1.0*B(j5,j31))]),j1,multi_dot([(j29*multi_dot([j30.T,j25]) + sin(j28)*-1.0*multi_dot([j31.T,j25])),B(j21,j27)]),j2,B(j7,config.ubar_SU1_rbl_upright_jcl_tie_upright),j4,-1.0*B(j32,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright),j2,B(j7,config.ubar_SU1_rbl_upright_jcl_hub_bearing),j4,-1.0*B(j35,config.ubar_SU1_rbl_hub_jcl_hub_bearing),j1,multi_dot([j34,j36,B(j7,j37)]),j1,multi_dot([j37.T,j39,j40]),j1,multi_dot([j34,j36,B(j7,j38)]),j1,multi_dot([j38.T,j39,j40]),j1,multi_dot([j41.T,j36,(j43*B(j7,j44) + sin(j42)*-1.0*B(j7,j45))]),j1,multi_dot([(j43*multi_dot([j44.T,j39]) + sin(j42)*-1.0*multi_dot([j45.T,j39])),B(j35,j41)]),j1,multi_dot([j47,j11,j50]),j1,multi_dot([j53,j54,j62]),j1,multi_dot([j47,j11,j52]),j1,multi_dot([j60,j54,j62]),j55,(multi_dot([j53,j54,j57]) + multi_dot([j59,j50])),-1.0*j55,-1.0*multi_dot([j53,j54,j63]),j61,(multi_dot([j60,j54,j57]) + multi_dot([j59,j52])),-1.0*j61,-1.0*multi_dot([j60,j54,j63]),j1,multi_dot([j65,j16,j68]),j1,multi_dot([j71,j72,j80]),j1,multi_dot([j65,j16,j70]),j1,multi_dot([j78,j72,j80]),j73,(multi_dot([j71,j72,j75]) + multi_dot([j77,j68])),-1.0*j73,-1.0*multi_dot([j71,j72,j81]),j79,(multi_dot([j78,j72,j75]) + multi_dot([j77,j70])),-1.0*j79,-1.0*multi_dot([j78,j72,j81]),j2,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j3.T,2.0*j6.T,2.0*j8.T,2.0*j13.T,2.0*j5.T,2.0*j7.T,2.0*j48.T,2.0*j66.T,2.0*j10.T,2.0*j15.T,2.0*j18.T,2.0*j32.T,2.0*j21.T,2.0*j35.T]
  
