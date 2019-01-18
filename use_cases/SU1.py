
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcl_hub_bearing = lambda t : 0
        self.ax1_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcr_hub_bearing = lambda t : 0
        self.ax1_jcl_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.SU1.P_rbr_lower_strut).T
        c1 = self.pt1_jcr_strut
        c2 = -1.0*multi_dot([c0,self.SU1.R_rbr_lower_strut])
        c3 = A(self.SU1.P_rbr_upper_strut).T
        c4 = Triad(self.ax1_jcr_strut,)
        c5 = self.pt1_jcr_strut_lca
        c6 = A(self.SU1.P_rbr_lca).T
        c7 = -1.0*multi_dot([c6,self.SU1.R_rbr_lca])
        c8 = self.ax1_jcr_strut_lca
        c9 = A(self.SU1.P_rbl_lca).T
        c10 = self.pt1_jcl_strut_lca
        c11 = -1.0*multi_dot([c9,self.SU1.R_rbl_lca])
        c12 = A(self.SU1.P_rbl_lower_strut).T
        c13 = -1.0*multi_dot([c12,self.SU1.R_rbl_lower_strut])
        c14 = self.ax1_jcl_strut_lca
        c15 = self.pt1_jcl_lca_upright
        c16 = A(self.SU1.P_rbl_upright).T
        c17 = -1.0*multi_dot([c16,self.SU1.R_rbl_upright])
        c18 = Triad(self.ax1_jcl_lca_upright,)
        c19 = A(self.SU1.P_rbl_hub).T
        c20 = self.pt1_jcl_hub_bearing
        c21 = Triad(self.ax1_jcl_hub_bearing,)
        c22 = A(self.SU1.P_rbr_upright).T
        c23 = self.pt1_jcr_tie_upright
        c24 = -1.0*multi_dot([c22,self.SU1.R_rbr_upright])
        c25 = A(self.SU1.P_rbr_tie_rod).T
        c26 = Triad(self.ax1_jcr_tie_upright,)
        c27 = self.pt1_jcr_uca_upright
        c28 = A(self.SU1.P_rbr_uca).T
        c29 = Triad(self.ax1_jcr_uca_upright,)
        c30 = self.pt1_jcr_lca_upright
        c31 = Triad(self.ax1_jcr_lca_upright,)
        c32 = self.pt1_jcr_hub_bearing
        c33 = A(self.SU1.P_rbr_hub).T
        c34 = Triad(self.ax1_jcr_hub_bearing,)
        c35 = self.pt1_jcl_strut
        c36 = A(self.SU1.P_rbl_upper_strut).T
        c37 = Triad(self.ax1_jcl_strut,)
        c38 = self.pt1_jcl_tie_upright
        c39 = A(self.SU1.P_rbl_tie_rod).T
        c40 = Triad(self.ax1_jcl_tie_upright,)
        c41 = self.pt1_jcl_uca_upright
        c42 = A(self.SU1.P_rbl_uca).T
        c43 = Triad(self.ax1_jcl_uca_upright,)

        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c0,c1]) + c2)
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c3,c1]) + -1.0*multi_dot([c3,self.SU1.R_rbr_upper_strut]))
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c0,c4])
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c3,c4])
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c0,c5]) + c2)
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c6,c5]) + c7)
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c0,Triad(c8,)])
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c6,Triad(c8,self.ax2_jcr_strut_lca)])
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c9,c10]) + c11)
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c12,c10]) + c13)
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c9,Triad(c14,)])
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c12,Triad(c14,self.ax2_jcl_strut_lca)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c9,c15]) + c11)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c16,c15]) + c17)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c9,c18])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c16,c18])
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c19,c20]) + -1.0*multi_dot([c19,self.SU1.R_rbl_hub]))
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c16,c20]) + c17)
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c19,c21])
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c16,c21])
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c22,c23]) + c24)
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c25,c23]) + -1.0*multi_dot([c25,self.SU1.R_rbr_tie_rod]))
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c22,c26])
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c25,c26])
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c22,c27]) + c24)
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c28,c27]) + -1.0*multi_dot([c28,self.SU1.R_rbr_uca]))
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c22,c29])
        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c28,c29])
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c22,c30]) + c24)
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c6,c30]) + c7)
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c22,c31])
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c6,c31])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c22,c32]) + c24)
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c33,c32]) + -1.0*multi_dot([c33,self.SU1.R_rbr_hub]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c22,c34])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c33,c34])
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c12,c35]) + c13)
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c36,c35]) + -1.0*multi_dot([c36,self.SU1.R_rbl_upper_strut]))
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c12,c37])
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c36,c37])
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c16,c38]) + c17)
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c39,c38]) + -1.0*multi_dot([c39,self.SU1.R_rbl_tie_rod]))
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c16,c40])
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c39,c40])
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c16,c41]) + c17)
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c42,c41]) + -1.0*multi_dot([c42,self.SU1.R_rbl_uca]))
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c16,c43])
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c42,c43])

    @property
    def q_initial(self):
        q = np.concatenate([self.SU1.R_rbr_lower_strut,self.SU1.P_rbr_lower_strut,self.SU1.R_rbl_lca,self.SU1.P_rbl_lca,self.SU1.R_rbr_upper_strut,self.SU1.P_rbr_upper_strut,self.SU1.R_rbl_hub,self.SU1.P_rbl_hub,self.SU1.R_rbr_upright,self.SU1.P_rbr_upright,self.SU1.R_rbr_tie_rod,self.SU1.P_rbr_tie_rod,self.SU1.R_rbr_uca,self.SU1.P_rbr_uca,self.SU1.R_rbl_lower_strut,self.SU1.P_rbl_lower_strut,self.SU1.R_rbr_lca,self.SU1.P_rbr_lca,self.SU1.R_rbl_upright,self.SU1.P_rbl_upright,self.SU1.R_rbr_hub,self.SU1.P_rbr_hub,self.SU1.R_rbl_tie_rod,self.SU1.P_rbl_tie_rod,self.SU1.R_rbl_upper_strut,self.SU1.P_rbl_upper_strut,self.SU1.R_rbl_uca,self.SU1.P_rbl_uca])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.SU1.Rd_rbr_lower_strut,self.SU1.Pd_rbr_lower_strut,self.SU1.Rd_rbl_lca,self.SU1.Pd_rbl_lca,self.SU1.Rd_rbr_upper_strut,self.SU1.Pd_rbr_upper_strut,self.SU1.Rd_rbl_hub,self.SU1.Pd_rbl_hub,self.SU1.Rd_rbr_upright,self.SU1.Pd_rbr_upright,self.SU1.Rd_rbr_tie_rod,self.SU1.Pd_rbr_tie_rod,self.SU1.Rd_rbr_uca,self.SU1.Pd_rbr_uca,self.SU1.Rd_rbl_lower_strut,self.SU1.Pd_rbl_lower_strut,self.SU1.Rd_rbr_lca,self.SU1.Pd_rbr_lca,self.SU1.Rd_rbl_upright,self.SU1.Pd_rbl_upright,self.SU1.Rd_rbr_hub,self.SU1.Pd_rbr_hub,self.SU1.Rd_rbl_tie_rod,self.SU1.Pd_rbl_tie_rod,self.SU1.Rd_rbl_upper_strut,self.SU1.Pd_rbl_upper_strut,self.SU1.Rd_rbl_uca,self.SU1.Pd_rbl_uca])
        return qd



class numerical_assembly(object):

    def __init__(self,config):
        self.t = 0.0
        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
        self.jac_cols = np.array([2,3,8,9,2,3,8,9,2,3,8,9,2,3,8,9,2,3,22,23,2,3,22,23,6,7,20,21,6,7,20,21,6,7,24,25,10,11,24,25,10,11,24,25,10,11,24,25,10,11,24,25,14,15,16,17,14,15,18,19,14,15,22,23,14,15,26,27,14,15,26,27,14,15,26,27,14,15,26,27,20,21,30,31,20,21,30,31,20,21,30,31,20,21,30,31,24,25,28,29,24,25,32,33,3,7,9,11,15,17,19,21,23,25,27,29,31,33])

        self.nrows = max(self.pos_rows)
        self.ncols = max(self.jac_cols)

    
    def set_q(self,q):
        self.SU1.R_rbr_lower_strut = q[0:3,0:1]
        self.SU1.P_rbr_lower_strut = q[3:7,0:1]
        self.SU1.R_rbl_lca = q[7:10,0:1]
        self.SU1.P_rbl_lca = q[10:14,0:1]
        self.SU1.R_rbr_upper_strut = q[14:17,0:1]
        self.SU1.P_rbr_upper_strut = q[17:21,0:1]
        self.SU1.R_rbl_hub = q[21:24,0:1]
        self.SU1.P_rbl_hub = q[24:28,0:1]
        self.SU1.R_rbr_upright = q[28:31,0:1]
        self.SU1.P_rbr_upright = q[31:35,0:1]
        self.SU1.R_rbr_tie_rod = q[35:38,0:1]
        self.SU1.P_rbr_tie_rod = q[38:42,0:1]
        self.SU1.R_rbr_uca = q[42:45,0:1]
        self.SU1.P_rbr_uca = q[45:49,0:1]
        self.SU1.R_rbl_lower_strut = q[49:52,0:1]
        self.SU1.P_rbl_lower_strut = q[52:56,0:1]
        self.SU1.R_rbr_lca = q[56:59,0:1]
        self.SU1.P_rbr_lca = q[59:63,0:1]
        self.SU1.R_rbl_upright = q[63:66,0:1]
        self.SU1.P_rbl_upright = q[66:70,0:1]
        self.SU1.R_rbr_hub = q[70:73,0:1]
        self.SU1.P_rbr_hub = q[73:77,0:1]
        self.SU1.R_rbl_tie_rod = q[77:80,0:1]
        self.SU1.P_rbl_tie_rod = q[80:84,0:1]
        self.SU1.R_rbl_upper_strut = q[84:87,0:1]
        self.SU1.P_rbl_upper_strut = q[87:91,0:1]
        self.SU1.R_rbl_uca = q[91:94,0:1]
        self.SU1.P_rbl_uca = q[94:98,0:1]

    
    def set_qd(self,qd):
        self.SU1.Rd_rbr_lower_strut = qd[0:3,0:1]
        self.SU1.Pd_rbr_lower_strut = qd[3:7,0:1]
        self.SU1.Rd_rbl_lca = qd[7:10,0:1]
        self.SU1.Pd_rbl_lca = qd[10:14,0:1]
        self.SU1.Rd_rbr_upper_strut = qd[14:17,0:1]
        self.SU1.Pd_rbr_upper_strut = qd[17:21,0:1]
        self.SU1.Rd_rbl_hub = qd[21:24,0:1]
        self.SU1.Pd_rbl_hub = qd[24:28,0:1]
        self.SU1.Rd_rbr_upright = qd[28:31,0:1]
        self.SU1.Pd_rbr_upright = qd[31:35,0:1]
        self.SU1.Rd_rbr_tie_rod = qd[35:38,0:1]
        self.SU1.Pd_rbr_tie_rod = qd[38:42,0:1]
        self.SU1.Rd_rbr_uca = qd[42:45,0:1]
        self.SU1.Pd_rbr_uca = qd[45:49,0:1]
        self.SU1.Rd_rbl_lower_strut = qd[49:52,0:1]
        self.SU1.Pd_rbl_lower_strut = qd[52:56,0:1]
        self.SU1.Rd_rbr_lca = qd[56:59,0:1]
        self.SU1.Pd_rbr_lca = qd[59:63,0:1]
        self.SU1.Rd_rbl_upright = qd[63:66,0:1]
        self.SU1.Pd_rbl_upright = qd[66:70,0:1]
        self.SU1.Rd_rbr_hub = qd[70:73,0:1]
        self.SU1.Pd_rbr_hub = qd[73:77,0:1]
        self.SU1.Rd_rbl_tie_rod = qd[77:80,0:1]
        self.SU1.Pd_rbl_tie_rod = qd[80:84,0:1]
        self.SU1.Rd_rbl_upper_strut = qd[84:87,0:1]
        self.SU1.Pd_rbl_upper_strut = qd[87:91,0:1]
        self.SU1.Rd_rbl_uca = qd[91:94,0:1]
        self.SU1.Pd_rbl_uca = qd[94:98,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = config.Mbar_rbr_lower_strut_jcr_strut[:,0:1].T
        x1 = self.SU1.P_rbr_lower_strut
        x2 = A(x1)
        x3 = x2.T
        x4 = self.SU1.P_rbr_upper_strut
        x5 = A(x4)
        x6 = config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]
        x7 = config.Mbar_rbr_lower_strut_jcr_strut[:,1:2].T
        x8 = self.SU1.R_rbr_lower_strut
        x9 = (x8 + -1.0*self.SU1.R_rbr_upper_strut + multi_dot([x2,config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]) + multi_dot([x2,config.ubar_rbr_lower_strut_jcr_strut]) + -1.0*multi_dot([x5,config.ubar_rbr_upper_strut_jcr_strut]))
        x10 = -1.0*self.SU1.R_rbr_lca
        x11 = self.SU1.P_rbr_lca
        x12 = A(x11)
        x13 = self.SU1.R_rbl_lca
        x14 = self.SU1.R_rbl_lower_strut
        x15 = self.SU1.P_rbl_lca
        x16 = A(x15)
        x17 = self.SU1.P_rbl_lower_strut
        x18 = A(x17)
        x19 = self.SU1.R_rbl_upright
        x20 = -1.0*x19
        x21 = self.SU1.P_rbl_upright
        x22 = A(x21)
        x23 = self.SU1.P_rbl_hub
        x24 = A(x23)
        x25 = x24.T
        x26 = config.Mbar_rbl_upright_jcl_hub_bearing[:,2:3]
        x27 = config.F_jcl_hub_bearing(t,)
        x28 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        x29 = self.SU1.R_rbr_upright
        x30 = self.SU1.P_rbr_upright
        x31 = A(x30)
        x32 = self.SU1.P_rbr_tie_rod
        x33 = self.SU1.P_rbr_uca
        x34 = self.SU1.P_rbr_hub
        x35 = A(x34)
        x36 = x31.T
        x37 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x38 = config.F_jcr_hub_bearing(t,)
        x39 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        x40 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1].T
        x41 = x18.T
        x42 = self.SU1.P_rbl_upper_strut
        x43 = A(x42)
        x44 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        x45 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2].T
        x46 = (x14 + -1.0*self.SU1.R_rbl_upper_strut + multi_dot([x18,config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]) + multi_dot([x18,config.ubar_rbl_lower_strut_jcl_strut]) + -1.0*multi_dot([x43,config.ubar_rbl_upper_strut_jcl_strut]))
        x47 = self.SU1.P_rbl_tie_rod
        x48 = self.SU1.P_rbl_uca
        x49 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [multi_dot([x0,x3,x5,x6]),multi_dot([x7,x3,x5,x6]),multi_dot([x0,x3,x9]),multi_dot([x7,x3,x9]),(x8 + x10 + multi_dot([x2,config.ubar_rbr_lower_strut_jcr_strut_lca]) + -1.0*multi_dot([x12,config.ubar_rbr_lca_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x3,x12,config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),(x13 + -1.0*x14 + multi_dot([x16,config.ubar_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x18,config.ubar_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lca_jcl_strut_lca[:,0:1].T,x16.T,x18,config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x13 + x20 + multi_dot([x16,config.ubar_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x22,config.ubar_rbl_upright_jcl_lca_upright])),(self.SU1.R_rbl_hub + x20 + multi_dot([x24,config.ubar_rbl_hub_jcl_hub_bearing]) + -1.0*multi_dot([x22,config.ubar_rbl_upright_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1].T,x25,x22,x26]),multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2].T,x25,x22,x26]),(cos(x27)*multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2].T,x25,x22,x28]) + sin(x27)*-1.0*multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1].T,x25,x22,x28])),(x29 + -1.0*self.SU1.R_rbr_tie_rod + multi_dot([x31,config.ubar_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([A(x32),config.ubar_rbr_tie_rod_jcr_tie_upright])),(x29 + -1.0*self.SU1.R_rbr_uca + multi_dot([x31,config.ubar_rbr_upright_jcr_uca_upright]) + -1.0*multi_dot([A(x33),config.ubar_rbr_uca_jcr_uca_upright])),(x29 + x10 + multi_dot([x31,config.ubar_rbr_upright_jcr_lca_upright]) + -1.0*multi_dot([x12,config.ubar_rbr_lca_jcr_lca_upright])),(x29 + -1.0*self.SU1.R_rbr_hub + multi_dot([x31,config.ubar_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x35,config.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x36,x35,x37]),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x36,x35,x37]),(cos(x38)*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x36,x35,x39]) + sin(x38)*-1.0*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x36,x35,x39])),multi_dot([x40,x41,x43,x44]),multi_dot([x45,x41,x43,x44]),multi_dot([x40,x41,x46]),multi_dot([x45,x41,x46]),(x19 + -1.0*self.SU1.R_rbl_tie_rod + multi_dot([x22,config.ubar_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([A(x47),config.ubar_rbl_tie_rod_jcl_tie_upright])),(x19 + -1.0*self.SU1.R_rbl_uca + multi_dot([x22,config.ubar_rbl_upright_jcl_uca_upright]) + -1.0*multi_dot([A(x48),config.ubar_rbl_uca_jcl_uca_upright])),(x49 + (multi_dot([x1.T,x1]))**(1.0/2.0)),(x49 + (multi_dot([x15.T,x15]))**(1.0/2.0)),(x49 + (multi_dot([x4.T,x4]))**(1.0/2.0)),(x49 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x49 + (multi_dot([x30.T,x30]))**(1.0/2.0)),(x49 + (multi_dot([x32.T,x32]))**(1.0/2.0)),(x49 + (multi_dot([x33.T,x33]))**(1.0/2.0)),(x49 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x49 + (multi_dot([x11.T,x11]))**(1.0/2.0)),(x49 + (multi_dot([x21.T,x21]))**(1.0/2.0)),(x49 + (multi_dot([x34.T,x34]))**(1.0/2.0)),(x49 + (multi_dot([x47.T,x47]))**(1.0/2.0)),(x49 + (multi_dot([x42.T,x42]))**(1.0/2.0)),(x49 + (multi_dot([x48.T,x48]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)
        v1 = np.zeros((3,1),dtype=np.float64)
        v2 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [v0,v0,v0,v0,v1,v0,v1,v0,v1,v1,v0,v0,(v0 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v2),v1,v1,v1,v1,v0,v0,(v0 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v2),v0,v0,v0,v0,v1,v1,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]
        a1 = a0.T
        a2 = self.SU1.P_rbr_upper_strut
        a3 = A(a2).T
        a4 = self.SU1.Pd_rbr_lower_strut
        a5 = config.Mbar_rbr_lower_strut_jcr_strut[:,0:1]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.SU1.P_rbr_lower_strut
        a9 = A(a8).T
        a10 = self.SU1.Pd_rbr_upper_strut
        a11 = B(a10,a0)
        a12 = a4.T
        a13 = B(a8,a5).T
        a14 = B(a2,a0)
        a15 = config.Mbar_rbr_lower_strut_jcr_strut[:,1:2]
        a16 = B(a4,a15)
        a17 = a15.T
        a18 = B(a8,a15).T
        a19 = config.ubar_rbr_lower_strut_jcr_strut
        a20 = config.ubar_rbr_upper_strut_jcr_strut
        a21 = (multi_dot([B(a4,a19),a4]) + -1.0*multi_dot([B(a10,a20),a10]))
        a22 = (self.SU1.Rd_rbr_lower_strut + -1.0*self.SU1.Rd_rbr_upper_strut + multi_dot([B(a8,a19),a4]) + multi_dot([B(a2,a20),a10]))
        a23 = (self.SU1.R_rbr_lower_strut.T + -1.0*self.SU1.R_rbr_upper_strut.T + multi_dot([config.Mbar_rbr_lower_strut_jcr_strut[:,2:3].T,a9]) + multi_dot([a19.T,a9]) + -1.0*multi_dot([a20.T,a3]))
        a24 = self.SU1.Pd_rbr_lca
        a25 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a26 = self.SU1.P_rbr_lca
        a27 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a28 = self.SU1.Pd_rbl_lca
        a29 = self.SU1.Pd_rbl_lower_strut
        a30 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a31 = self.SU1.P_rbl_lca
        a32 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a33 = self.SU1.P_rbl_lower_strut
        a34 = A(a33).T
        a35 = a28.T
        a36 = self.SU1.Pd_rbl_upright
        a37 = self.SU1.Pd_rbl_hub
        a38 = config.Mbar_rbl_upright_jcl_hub_bearing[:,2:3]
        a39 = a38.T
        a40 = self.SU1.P_rbl_upright
        a41 = A(a40).T
        a42 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a43 = self.SU1.P_rbl_hub
        a44 = A(a43).T
        a45 = B(a36,a38)
        a46 = a37.T
        a47 = B(a40,a38)
        a48 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        a49 = config.F_jcl_hub_bearing(t,)
        a50 = np.eye(1,dtype=np.float64)
        a51 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a52 = cos(a49)
        a53 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        a54 = sin(a49)
        a55 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a56 = self.SU1.Pd_rbr_upright
        a57 = self.SU1.Pd_rbr_tie_rod
        a58 = self.SU1.Pd_rbr_uca
        a59 = self.SU1.Pd_rbr_hub
        a60 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a61 = self.SU1.P_rbr_upright
        a62 = A(a61).T
        a63 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a64 = B(a59,a63)
        a65 = a63.T
        a66 = self.SU1.P_rbr_hub
        a67 = A(a66).T
        a68 = a56.T
        a69 = B(a66,a63)
        a70 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a71 = config.F_jcr_hub_bearing(t,)
        a72 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a73 = cos(a71)
        a74 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a75 = sin(a71)
        a76 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a77 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1]
        a78 = a77.T
        a79 = self.SU1.Pd_rbl_upper_strut
        a80 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        a81 = B(a79,a80)
        a82 = a80.T
        a83 = self.SU1.P_rbl_upper_strut
        a84 = A(a83).T
        a85 = B(a29,a77)
        a86 = a29.T
        a87 = B(a33,a77).T
        a88 = B(a83,a80)
        a89 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2]
        a90 = a89.T
        a91 = B(a29,a89)
        a92 = B(a33,a89).T
        a93 = config.ubar_rbl_lower_strut_jcl_strut
        a94 = config.ubar_rbl_upper_strut_jcl_strut
        a95 = (multi_dot([B(a29,a93),a29]) + -1.0*multi_dot([B(a79,a94),a79]))
        a96 = (self.SU1.Rd_rbl_lower_strut + -1.0*self.SU1.Rd_rbl_upper_strut + multi_dot([B(a33,a93),a29]) + multi_dot([B(a83,a94),a79]))
        a97 = (self.SU1.R_rbl_lower_strut.T + -1.0*self.SU1.R_rbl_upper_strut.T + multi_dot([config.Mbar_rbl_lower_strut_jcl_strut[:,2:3].T,a34]) + multi_dot([a93.T,a34]) + -1.0*multi_dot([a94.T,a84]))
        a98 = self.SU1.Pd_rbl_tie_rod
        a99 = self.SU1.Pd_rbl_uca

        self.acc_eq_blocks = [(multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + 2.0*multi_dot([a12,a13,a14,a10])),(multi_dot([a1,a3,a16,a4]) + multi_dot([a17,a9,a11,a10]) + 2.0*multi_dot([a12,a18,a14,a10])),(multi_dot([a7,a9,a21]) + 2.0*multi_dot([a12,a13,a22]) + multi_dot([a23,a6,a4])),(multi_dot([a17,a9,a21]) + 2.0*multi_dot([a12,a18,a22]) + multi_dot([a23,a16,a4])),(multi_dot([B(a4,config.ubar_rbr_lower_strut_jcr_strut_lca),a4]) + -1.0*multi_dot([B(a24,config.ubar_rbr_lca_jcr_strut_lca),a24])),(multi_dot([a25.T,A(a26).T,B(a4,a27),a4]) + multi_dot([a27.T,a9,B(a24,a25),a24]) + 2.0*multi_dot([a12,B(a8,a27).T,B(a26,a25),a24])),(multi_dot([B(a28,config.ubar_rbl_lca_jcl_strut_lca),a28]) + -1.0*multi_dot([B(a29,config.ubar_rbl_lower_strut_jcl_strut_lca),a29])),(multi_dot([a30.T,A(a31).T,B(a29,a32),a29]) + multi_dot([a32.T,a34,B(a28,a30),a28]) + 2.0*multi_dot([a35,B(a31,a30).T,B(a33,a32),a29])),(multi_dot([B(a28,config.ubar_rbl_lca_jcl_lca_upright),a28]) + -1.0*multi_dot([B(a36,config.ubar_rbl_upright_jcl_lca_upright),a36])),(multi_dot([B(a37,config.ubar_rbl_hub_jcl_hub_bearing),a37]) + -1.0*multi_dot([B(a36,config.ubar_rbl_upright_jcl_hub_bearing),a36])),(multi_dot([a39,a41,B(a37,a42),a37]) + multi_dot([a42.T,a44,a45,a36]) + 2.0*multi_dot([a46,B(a43,a42).T,a47,a36])),(multi_dot([a39,a41,B(a37,a48),a37]) + multi_dot([a48.T,a44,a45,a36]) + 2.0*multi_dot([a46,B(a43,a48).T,a47,a36])),(derivative(a49,t,0.1,2)*-1.0*a50 + multi_dot([a51.T,a41,(a52*B(a37,a53) + a54*-1.0*B(a37,a55)),a37]) + multi_dot([(a52*multi_dot([a53.T,a44]) + a54*-1.0*multi_dot([a55.T,a44])),B(a36,a51),a36]) + 2.0*multi_dot([((a52*multi_dot([B(a43,a53),a37])).T + a54*-1.0*multi_dot([a46,B(a43,a55).T])),B(a40,a51),a36])),(multi_dot([B(a56,config.ubar_rbr_upright_jcr_tie_upright),a56]) + -1.0*multi_dot([B(a57,config.ubar_rbr_tie_rod_jcr_tie_upright),a57])),(multi_dot([B(a56,config.ubar_rbr_upright_jcr_uca_upright),a56]) + -1.0*multi_dot([B(a58,config.ubar_rbr_uca_jcr_uca_upright),a58])),(multi_dot([B(a56,config.ubar_rbr_upright_jcr_lca_upright),a56]) + -1.0*multi_dot([B(a24,config.ubar_rbr_lca_jcr_lca_upright),a24])),(multi_dot([B(a56,config.ubar_rbr_upright_jcr_hub_bearing),a56]) + -1.0*multi_dot([B(a59,config.ubar_rbr_hub_jcr_hub_bearing),a59])),(multi_dot([a60.T,a62,a64,a59]) + multi_dot([a65,a67,B(a56,a60),a56]) + 2.0*multi_dot([a68,B(a61,a60).T,a69,a59])),(multi_dot([a70.T,a62,a64,a59]) + multi_dot([a65,a67,B(a56,a70),a56]) + 2.0*multi_dot([a68,B(a61,a70).T,a69,a59])),(derivative(a71,t,0.1,2)*-1.0*a50 + multi_dot([a72.T,a67,(a73*B(a56,a74) + a75*-1.0*B(a56,a76)),a56]) + multi_dot([(a73*multi_dot([a74.T,a62]) + a75*-1.0*multi_dot([a76.T,a62])),B(a59,a72),a59]) + 2.0*multi_dot([((a73*multi_dot([B(a61,a74),a56])).T + a75*-1.0*multi_dot([a68,B(a61,a76).T])),B(a66,a72),a59])),(multi_dot([a78,a34,a81,a79]) + multi_dot([a82,a84,a85,a29]) + 2.0*multi_dot([a86,a87,a88,a79])),(multi_dot([a90,a34,a81,a79]) + multi_dot([a82,a84,a91,a29]) + 2.0*multi_dot([a86,a92,a88,a79])),(multi_dot([a78,a34,a95]) + 2.0*multi_dot([a86,a87,a96]) + multi_dot([a97,a85,a29])),(multi_dot([a90,a34,a95]) + 2.0*multi_dot([a86,a92,a96]) + multi_dot([a97,a91,a29])),(multi_dot([B(a36,config.ubar_rbl_upright_jcl_tie_upright),a36]) + -1.0*multi_dot([B(a98,config.ubar_rbl_tie_rod_jcl_tie_upright),a98])),(multi_dot([B(a36,config.ubar_rbl_upright_jcl_uca_upright),a36]) + -1.0*multi_dot([B(a99,config.ubar_rbl_uca_jcl_uca_upright),a99])),2.0*(multi_dot([a12,a4]))**(1.0/2.0),2.0*(multi_dot([a35,a28]))**(1.0/2.0),2.0*(multi_dot([a10.T,a10]))**(1.0/2.0),2.0*(multi_dot([a46,a37]))**(1.0/2.0),2.0*(multi_dot([a68,a56]))**(1.0/2.0),2.0*(multi_dot([a57.T,a57]))**(1.0/2.0),2.0*(multi_dot([a58.T,a58]))**(1.0/2.0),2.0*(multi_dot([a86,a29]))**(1.0/2.0),2.0*(multi_dot([a24.T,a24]))**(1.0/2.0),2.0*(multi_dot([a36.T,a36]))**(1.0/2.0),2.0*(multi_dot([a59.T,a59]))**(1.0/2.0),2.0*(multi_dot([a98.T,a98]))**(1.0/2.0),2.0*(multi_dot([a79.T,a79]))**(1.0/2.0),2.0*(multi_dot([a99.T,a99]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]
        j2 = j1.T
        j3 = self.SU1.P_rbr_upper_strut
        j4 = A(j3).T
        j5 = self.SU1.P_rbr_lower_strut
        j6 = config.Mbar_rbr_lower_strut_jcr_strut[:,0:1]
        j7 = B(j5,j6)
        j8 = config.Mbar_rbr_lower_strut_jcr_strut[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = config.ubar_rbr_lower_strut_jcr_strut
        j14 = B(j5,j13)
        j15 = config.ubar_rbr_upper_strut_jcr_strut
        j16 = (self.SU1.R_rbr_lower_strut.T + -1.0*self.SU1.R_rbr_upper_strut.T + multi_dot([config.Mbar_rbr_lower_strut_jcr_strut[:,2:3].T,j11]) + multi_dot([j13.T,j11]) + -1.0*multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = B(j3,j1)
        j20 = B(j3,j15)
        j21 = np.eye(3,dtype=np.float64)
        j22 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j23 = self.SU1.P_rbr_lca
        j24 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j25 = -1.0*j21
        j26 = self.SU1.P_rbl_lca
        j27 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j28 = self.SU1.P_rbl_lower_strut
        j29 = A(j28).T
        j30 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j31 = self.SU1.P_rbl_upright
        j32 = self.SU1.P_rbl_hub
        j33 = config.Mbar_rbl_upright_jcl_hub_bearing[:,2:3]
        j34 = j33.T
        j35 = A(j31).T
        j36 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j37 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        j38 = A(j32).T
        j39 = B(j31,j33)
        j40 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j41 = config.F_jcl_hub_bearing(t,)
        j42 = cos(j41)
        j43 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        j44 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j45 = self.SU1.P_rbr_upright
        j46 = self.SU1.P_rbr_tie_rod
        j47 = self.SU1.P_rbr_uca
        j48 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j49 = j48.T
        j50 = self.SU1.P_rbr_hub
        j51 = A(j50).T
        j52 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j53 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j54 = A(j45).T
        j55 = B(j50,j48)
        j56 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j57 = config.F_jcr_hub_bearing(t,)
        j58 = cos(j57)
        j59 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j60 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j61 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        j62 = j61.T
        j63 = self.SU1.P_rbl_upper_strut
        j64 = A(j63).T
        j65 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1]
        j66 = B(j28,j65)
        j67 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2]
        j68 = B(j28,j67)
        j69 = j65.T
        j70 = multi_dot([j69,j29])
        j71 = config.ubar_rbl_lower_strut_jcl_strut
        j72 = B(j28,j71)
        j73 = config.ubar_rbl_upper_strut_jcl_strut
        j74 = (self.SU1.R_rbl_lower_strut.T + -1.0*self.SU1.R_rbl_upper_strut.T + multi_dot([config.Mbar_rbl_lower_strut_jcl_strut[:,2:3].T,j29]) + multi_dot([j71.T,j29]) + -1.0*multi_dot([j73.T,j64]))
        j75 = j67.T
        j76 = multi_dot([j75,j29])
        j77 = B(j63,j61)
        j78 = B(j63,j73)
        j79 = self.SU1.P_rbl_tie_rod
        j80 = self.SU1.P_rbl_uca

        self.jac_eq_blocks = [j0,multi_dot([j2,j4,j7]),j0,multi_dot([j10,j11,j19]),j0,multi_dot([j2,j4,j9]),j0,multi_dot([j17,j11,j19]),j12,(multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),-1.0*j12,-1.0*multi_dot([j10,j11,j20]),j18,(multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),-1.0*j18,-1.0*multi_dot([j17,j11,j20]),j21,B(j5,config.ubar_rbr_lower_strut_jcr_strut_lca),j25,-1.0*B(j23,config.ubar_rbr_lca_jcr_strut_lca),j0,multi_dot([j22.T,A(j23).T,B(j5,j24)]),j0,multi_dot([j24.T,j11,B(j23,j22)]),j21,B(j26,config.ubar_rbl_lca_jcl_strut_lca),j25,-1.0*B(j28,config.ubar_rbl_lower_strut_jcl_strut_lca),j0,multi_dot([j27.T,j29,B(j26,j30)]),j0,multi_dot([j30.T,A(j26).T,B(j28,j27)]),j21,B(j26,config.ubar_rbl_lca_jcl_lca_upright),j25,-1.0*B(j31,config.ubar_rbl_upright_jcl_lca_upright),j21,B(j32,config.ubar_rbl_hub_jcl_hub_bearing),j25,-1.0*B(j31,config.ubar_rbl_upright_jcl_hub_bearing),j0,multi_dot([j34,j35,B(j32,j36)]),j0,multi_dot([j36.T,j38,j39]),j0,multi_dot([j34,j35,B(j32,j37)]),j0,multi_dot([j37.T,j38,j39]),j0,multi_dot([j40.T,j35,(j42*B(j32,j43) + sin(j41)*-1.0*B(j32,j44))]),j0,multi_dot([(j42*multi_dot([j43.T,j38]) + sin(j41)*-1.0*multi_dot([j44.T,j38])),B(j31,j40)]),j21,B(j45,config.ubar_rbr_upright_jcr_tie_upright),j25,-1.0*B(j46,config.ubar_rbr_tie_rod_jcr_tie_upright),j21,B(j45,config.ubar_rbr_upright_jcr_uca_upright),j25,-1.0*B(j47,config.ubar_rbr_uca_jcr_uca_upright),j21,B(j45,config.ubar_rbr_upright_jcr_lca_upright),j25,-1.0*B(j23,config.ubar_rbr_lca_jcr_lca_upright),j21,B(j45,config.ubar_rbr_upright_jcr_hub_bearing),j25,-1.0*B(j50,config.ubar_rbr_hub_jcr_hub_bearing),j0,multi_dot([j49,j51,B(j45,j52)]),j0,multi_dot([j52.T,j54,j55]),j0,multi_dot([j49,j51,B(j45,j53)]),j0,multi_dot([j53.T,j54,j55]),j0,multi_dot([j56.T,j51,(j58*B(j45,j59) + sin(j57)*-1.0*B(j45,j60))]),j0,multi_dot([(j58*multi_dot([j59.T,j54]) + sin(j57)*-1.0*multi_dot([j60.T,j54])),B(j50,j56)]),j0,multi_dot([j62,j64,j66]),j0,multi_dot([j69,j29,j77]),j0,multi_dot([j62,j64,j68]),j0,multi_dot([j75,j29,j77]),j70,(multi_dot([j69,j29,j72]) + multi_dot([j74,j66])),-1.0*j70,-1.0*multi_dot([j69,j29,j78]),j76,(multi_dot([j75,j29,j72]) + multi_dot([j74,j68])),-1.0*j76,-1.0*multi_dot([j75,j29,j78]),j21,B(j31,config.ubar_rbl_upright_jcl_tie_upright),j25,-1.0*B(j79,config.ubar_rbl_tie_rod_jcl_tie_upright),j21,B(j31,config.ubar_rbl_upright_jcl_uca_upright),j25,-1.0*B(j80,config.ubar_rbl_uca_jcl_uca_upright),2.0*j5.T,2.0*j26.T,2.0*j3.T,2.0*j32.T,2.0*j45.T,2.0*j46.T,2.0*j47.T,2.0*j28.T,2.0*j23.T,2.0*j31.T,2.0*j50.T,2.0*j79.T,2.0*j63.T,2.0*j80.T]
  
