
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcr_hub_bearing = lambda t : 0
        self.ax1_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcl_hub_bearing = lambda t : 0
        self.SU1.R_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.SU1.R_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.P_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.SU1.Rd_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.SU1.Pd_rbl_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.SU1.P_rbl_tie_rod).T
        c1 = self.pt1_jcl_tie_upright
        c2 = A(self.SU1.P_rbl_upright).T
        c3 = -1.0*multi_dot([c2,self.SU1.R_rbl_upright])
        c4 = Triad(self.ax1_jcl_tie_upright,)
        c5 = A(self.SU1.P_rbl_lower_strut).T
        c6 = self.pt1_jcl_strut
        c7 = -1.0*multi_dot([c5,self.SU1.R_rbl_lower_strut])
        c8 = A(self.SU1.P_rbl_upper_strut).T
        c9 = Triad(self.ax1_jcl_strut,)
        c10 = self.pt1_jcl_strut_lca
        c11 = A(self.SU1.P_rbl_lca).T
        c12 = -1.0*multi_dot([c11,self.SU1.R_rbl_lca])
        c13 = self.ax1_jcl_strut_lca
        c14 = A(self.SU1.P_rbr_hub).T
        c15 = self.pt1_jcr_hub_bearing
        c16 = A(self.SU1.P_rbr_upright).T
        c17 = -1.0*multi_dot([c16,self.SU1.R_rbr_upright])
        c18 = Triad(self.ax1_jcr_hub_bearing,)
        c19 = A(self.SU1.P_rbr_upper_strut).T
        c20 = self.pt1_jcr_strut
        c21 = A(self.SU1.P_rbr_lower_strut).T
        c22 = -1.0*multi_dot([c21,self.SU1.R_rbr_lower_strut])
        c23 = Triad(self.ax1_jcr_strut,)
        c24 = self.pt1_jcr_lca_upright
        c25 = A(self.SU1.P_rbr_lca).T
        c26 = -1.0*multi_dot([c25,self.SU1.R_rbr_lca])
        c27 = Triad(self.ax1_jcr_lca_upright,)
        c28 = self.pt1_jcr_tie_upright
        c29 = A(self.SU1.P_rbr_tie_rod).T
        c30 = Triad(self.ax1_jcr_tie_upright,)
        c31 = self.pt1_jcr_uca_upright
        c32 = A(self.SU1.P_rbr_uca).T
        c33 = Triad(self.ax1_jcr_uca_upright,)
        c34 = self.pt1_jcr_strut_lca
        c35 = self.ax1_jcr_strut_lca
        c36 = A(self.SU1.P_rbl_uca).T
        c37 = self.pt1_jcl_uca_upright
        c38 = Triad(self.ax1_jcl_uca_upright,)
        c39 = self.pt1_jcl_lca_upright
        c40 = Triad(self.ax1_jcl_lca_upright,)
        c41 = self.pt1_jcl_hub_bearing
        c42 = A(self.SU1.P_rbl_hub).T
        c43 = Triad(self.ax1_jcl_hub_bearing,)

        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,self.SU1.R_rbl_tie_rod]))
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c2,c1]) + c3)
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c0,c4])
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c2,c4])
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c5,c6]) + c7)
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c8,c6]) + -1.0*multi_dot([c8,self.SU1.R_rbl_upper_strut]))
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c5,c9])
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c8,c9])
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c5,c10]) + c7)
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c11,c10]) + c12)
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c5,Triad(c13,)])
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c11,Triad(c13,self.ax2_jcl_strut_lca)])
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c14,c15]) + -1.0*multi_dot([c14,self.SU1.R_rbr_hub]))
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c16,c15]) + c17)
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c14,c18])
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c16,c18])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c19,c20]) + -1.0*multi_dot([c19,self.SU1.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c21,c20]) + c22)
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c19,c23])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c21,c23])
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c16,c24]) + c17)
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c25,c24]) + c26)
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c16,c27])
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c25,c27])
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c16,c28]) + c17)
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c29,c28]) + -1.0*multi_dot([c29,self.SU1.R_rbr_tie_rod]))
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c16,c30])
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c29,c30])
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c16,c31]) + c17)
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c32,c31]) + -1.0*multi_dot([c32,self.SU1.R_rbr_uca]))
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c16,c33])
        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c32,c33])
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c21,c34]) + c22)
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c25,c34]) + c26)
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c21,Triad(c35,)])
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c25,Triad(c35,self.ax2_jcr_strut_lca)])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c36,c37]) + -1.0*multi_dot([c36,self.SU1.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c2,c37]) + c3)
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c36,c38])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c2,c38])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c11,c39]) + c12)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c2,c39]) + c3)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c11,c40])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c2,c40])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c2,c41]) + c3)
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c42,c41]) + -1.0*multi_dot([c42,self.SU1.R_rbl_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c2,c43])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c42,c43])

    @property
    def q_initial(self):
        q = np.concatenate([self.SU1.R_rbl_tie_rod,self.SU1.P_rbl_tie_rod,self.SU1.R_rbl_lower_strut,self.SU1.P_rbl_lower_strut,self.SU1.R_rbr_hub,self.SU1.P_rbr_hub,self.SU1.R_rbl_upper_strut,self.SU1.P_rbl_upper_strut,self.SU1.R_rbr_upper_strut,self.SU1.P_rbr_upper_strut,self.SU1.R_rbr_upright,self.SU1.P_rbr_upright,self.SU1.R_rbr_lower_strut,self.SU1.P_rbr_lower_strut,self.SU1.R_rbr_lca,self.SU1.P_rbr_lca,self.SU1.R_rbr_tie_rod,self.SU1.P_rbr_tie_rod,self.SU1.R_rbr_uca,self.SU1.P_rbr_uca,self.SU1.R_rbl_uca,self.SU1.P_rbl_uca,self.SU1.R_rbl_lca,self.SU1.P_rbl_lca,self.SU1.R_rbl_upright,self.SU1.P_rbl_upright,self.SU1.R_rbl_hub,self.SU1.P_rbl_hub])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.SU1.Rd_rbl_tie_rod,self.SU1.Pd_rbl_tie_rod,self.SU1.Rd_rbl_lower_strut,self.SU1.Pd_rbl_lower_strut,self.SU1.Rd_rbr_hub,self.SU1.Pd_rbr_hub,self.SU1.Rd_rbl_upper_strut,self.SU1.Pd_rbl_upper_strut,self.SU1.Rd_rbr_upper_strut,self.SU1.Pd_rbr_upper_strut,self.SU1.Rd_rbr_upright,self.SU1.Pd_rbr_upright,self.SU1.Rd_rbr_lower_strut,self.SU1.Pd_rbr_lower_strut,self.SU1.Rd_rbr_lca,self.SU1.Pd_rbr_lca,self.SU1.Rd_rbr_tie_rod,self.SU1.Pd_rbr_tie_rod,self.SU1.Rd_rbr_uca,self.SU1.Pd_rbr_uca,self.SU1.Rd_rbl_uca,self.SU1.Pd_rbl_uca,self.SU1.Rd_rbl_lca,self.SU1.Pd_rbl_lca,self.SU1.Rd_rbl_upright,self.SU1.Pd_rbl_upright,self.SU1.Rd_rbl_hub,self.SU1.Pd_rbl_hub])
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
        self.jac_cols = np.array([0,1,32,33,2,3,12,13,2,3,12,13,2,3,12,13,2,3,12,13,2,3,30,31,2,3,30,31,4,5,16,17,4,5,16,17,4,5,16,17,4,5,16,17,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,22,23,16,17,26,27,18,19,20,21,18,19,20,21,28,29,32,33,30,31,32,33,32,33,34,35,32,33,34,35,32,33,34,35,32,33,34,35,1,3,5,13,15,17,19,21,23,27,29,31,33,35])

        self.nrows = max(self.pos_rows)
        self.ncols = max(self.jac_cols)

    
    def set_q(self,q):
        self.SU1.R_rbl_tie_rod = q[0:3,0:1]
        self.SU1.P_rbl_tie_rod = q[3:7,0:1]
        self.SU1.R_rbl_lower_strut = q[7:10,0:1]
        self.SU1.P_rbl_lower_strut = q[10:14,0:1]
        self.SU1.R_rbr_hub = q[14:17,0:1]
        self.SU1.P_rbr_hub = q[17:21,0:1]
        self.SU1.R_rbl_upper_strut = q[21:24,0:1]
        self.SU1.P_rbl_upper_strut = q[24:28,0:1]
        self.SU1.R_rbr_upper_strut = q[28:31,0:1]
        self.SU1.P_rbr_upper_strut = q[31:35,0:1]
        self.SU1.R_rbr_upright = q[35:38,0:1]
        self.SU1.P_rbr_upright = q[38:42,0:1]
        self.SU1.R_rbr_lower_strut = q[42:45,0:1]
        self.SU1.P_rbr_lower_strut = q[45:49,0:1]
        self.SU1.R_rbr_lca = q[49:52,0:1]
        self.SU1.P_rbr_lca = q[52:56,0:1]
        self.SU1.R_rbr_tie_rod = q[56:59,0:1]
        self.SU1.P_rbr_tie_rod = q[59:63,0:1]
        self.SU1.R_rbr_uca = q[63:66,0:1]
        self.SU1.P_rbr_uca = q[66:70,0:1]
        self.SU1.R_rbl_uca = q[70:73,0:1]
        self.SU1.P_rbl_uca = q[73:77,0:1]
        self.SU1.R_rbl_lca = q[77:80,0:1]
        self.SU1.P_rbl_lca = q[80:84,0:1]
        self.SU1.R_rbl_upright = q[84:87,0:1]
        self.SU1.P_rbl_upright = q[87:91,0:1]
        self.SU1.R_rbl_hub = q[91:94,0:1]
        self.SU1.P_rbl_hub = q[94:98,0:1]

    
    def set_qd(self,qd):
        self.SU1.Rd_rbl_tie_rod = qd[0:3,0:1]
        self.SU1.Pd_rbl_tie_rod = qd[3:7,0:1]
        self.SU1.Rd_rbl_lower_strut = qd[7:10,0:1]
        self.SU1.Pd_rbl_lower_strut = qd[10:14,0:1]
        self.SU1.Rd_rbr_hub = qd[14:17,0:1]
        self.SU1.Pd_rbr_hub = qd[17:21,0:1]
        self.SU1.Rd_rbl_upper_strut = qd[21:24,0:1]
        self.SU1.Pd_rbl_upper_strut = qd[24:28,0:1]
        self.SU1.Rd_rbr_upper_strut = qd[28:31,0:1]
        self.SU1.Pd_rbr_upper_strut = qd[31:35,0:1]
        self.SU1.Rd_rbr_upright = qd[35:38,0:1]
        self.SU1.Pd_rbr_upright = qd[38:42,0:1]
        self.SU1.Rd_rbr_lower_strut = qd[42:45,0:1]
        self.SU1.Pd_rbr_lower_strut = qd[45:49,0:1]
        self.SU1.Rd_rbr_lca = qd[49:52,0:1]
        self.SU1.Pd_rbr_lca = qd[52:56,0:1]
        self.SU1.Rd_rbr_tie_rod = qd[56:59,0:1]
        self.SU1.Pd_rbr_tie_rod = qd[59:63,0:1]
        self.SU1.Rd_rbr_uca = qd[63:66,0:1]
        self.SU1.Pd_rbr_uca = qd[66:70,0:1]
        self.SU1.Rd_rbl_uca = qd[70:73,0:1]
        self.SU1.Pd_rbl_uca = qd[73:77,0:1]
        self.SU1.Rd_rbl_lca = qd[77:80,0:1]
        self.SU1.Pd_rbl_lca = qd[80:84,0:1]
        self.SU1.Rd_rbl_upright = qd[84:87,0:1]
        self.SU1.Pd_rbl_upright = qd[87:91,0:1]
        self.SU1.Rd_rbl_hub = qd[91:94,0:1]
        self.SU1.Pd_rbl_hub = qd[94:98,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.SU1.R_rbl_upright
        x1 = -1.0*x0
        x2 = self.SU1.P_rbl_tie_rod
        x3 = self.SU1.P_rbl_upright
        x4 = A(x3)
        x5 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1].T
        x6 = self.SU1.P_rbl_lower_strut
        x7 = A(x6)
        x8 = x7.T
        x9 = self.SU1.P_rbl_upper_strut
        x10 = A(x9)
        x11 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        x12 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2].T
        x13 = self.SU1.R_rbl_lower_strut
        x14 = (x13 + -1.0*self.SU1.R_rbl_upper_strut + multi_dot([x7,config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]) + multi_dot([x7,config.ubar_rbl_lower_strut_jcl_strut]) + -1.0*multi_dot([x10,config.ubar_rbl_upper_strut_jcl_strut]))
        x15 = self.SU1.R_rbl_lca
        x16 = self.SU1.P_rbl_lca
        x17 = A(x16)
        x18 = self.SU1.R_rbr_upright
        x19 = self.SU1.P_rbr_hub
        x20 = A(x19)
        x21 = self.SU1.P_rbr_upright
        x22 = A(x21)
        x23 = x20.T
        x24 = config.Mbar_rbr_upright_jcr_hub_bearing[:,2:3]
        x25 = config.F_jcr_hub_bearing(t,)
        x26 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        x27 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x28 = self.SU1.P_rbr_upper_strut
        x29 = A(x28)
        x30 = x29.T
        x31 = self.SU1.P_rbr_lower_strut
        x32 = A(x31)
        x33 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x34 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x35 = self.SU1.R_rbr_lower_strut
        x36 = (self.SU1.R_rbr_upper_strut + -1.0*x35 + multi_dot([x29,config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]]) + multi_dot([x29,config.ubar_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x32,config.ubar_rbr_lower_strut_jcr_strut]))
        x37 = -1.0*self.SU1.R_rbr_lca
        x38 = self.SU1.P_rbr_lca
        x39 = A(x38)
        x40 = self.SU1.P_rbr_tie_rod
        x41 = self.SU1.P_rbr_uca
        x42 = self.SU1.P_rbl_uca
        x43 = self.SU1.P_rbl_hub
        x44 = A(x43)
        x45 = x4.T
        x46 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x47 = config.F_jcl_hub_bearing(t,)
        x48 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        x49 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(self.SU1.R_rbl_tie_rod + x1 + multi_dot([A(x2),config.ubar_rbl_tie_rod_jcl_tie_upright]) + -1.0*multi_dot([x4,config.ubar_rbl_upright_jcl_tie_upright])),multi_dot([x5,x8,x10,x11]),multi_dot([x12,x8,x10,x11]),multi_dot([x5,x8,x14]),multi_dot([x12,x8,x14]),(x13 + -1.0*x15 + multi_dot([x7,config.ubar_rbl_lower_strut_jcl_strut_lca]) + -1.0*multi_dot([x17,config.ubar_rbl_lca_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1].T,x8,x17,config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]]),(self.SU1.R_rbr_hub + -1.0*x18 + multi_dot([x20,config.ubar_rbr_hub_jcr_hub_bearing]) + -1.0*multi_dot([x22,config.ubar_rbr_upright_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1].T,x23,x22,x24]),multi_dot([config.Mbar_rbr_hub_jcr_hub_bearing[:,1:2].T,x23,x22,x24]),(cos(x25)*multi_dot([config.Mbar_rbr_hub_jcr_hub_bearing[:,1:2].T,x23,x22,x26]) + sin(x25)*-1.0*multi_dot([config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1].T,x23,x22,x26])),multi_dot([x27,x30,x32,x33]),multi_dot([x34,x30,x32,x33]),multi_dot([x27,x30,x36]),multi_dot([x34,x30,x36]),(x18 + x37 + multi_dot([x22,config.ubar_rbr_upright_jcr_lca_upright]) + -1.0*multi_dot([x39,config.ubar_rbr_lca_jcr_lca_upright])),(x18 + -1.0*self.SU1.R_rbr_tie_rod + multi_dot([x22,config.ubar_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([A(x40),config.ubar_rbr_tie_rod_jcr_tie_upright])),(x18 + -1.0*self.SU1.R_rbr_uca + multi_dot([x22,config.ubar_rbr_upright_jcr_uca_upright]) + -1.0*multi_dot([A(x41),config.ubar_rbr_uca_jcr_uca_upright])),(x35 + x37 + multi_dot([x32,config.ubar_rbr_lower_strut_jcr_strut_lca]) + -1.0*multi_dot([x39,config.ubar_rbr_lca_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x32.T,x39,config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),(self.SU1.R_rbl_uca + x1 + multi_dot([A(x42),config.ubar_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x4,config.ubar_rbl_upright_jcl_uca_upright])),(x15 + x1 + multi_dot([x17,config.ubar_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x4,config.ubar_rbl_upright_jcl_lca_upright])),(x0 + -1.0*self.SU1.R_rbl_hub + multi_dot([x4,config.ubar_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x44,config.ubar_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x45,x44,x46]),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x45,x44,x46]),(cos(x47)*multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x45,x44,x48]) + sin(x47)*-1.0*multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x45,x44,x48])),(x49 + (multi_dot([x2.T,x2]))**(1.0/2.0)),(x49 + (multi_dot([x6.T,x6]))**(1.0/2.0)),(x49 + (multi_dot([x19.T,x19]))**(1.0/2.0)),(x49 + (multi_dot([x9.T,x9]))**(1.0/2.0)),(x49 + (multi_dot([x28.T,x28]))**(1.0/2.0)),(x49 + (multi_dot([x21.T,x21]))**(1.0/2.0)),(x49 + (multi_dot([x31.T,x31]))**(1.0/2.0)),(x49 + (multi_dot([x38.T,x38]))**(1.0/2.0)),(x49 + (multi_dot([x40.T,x40]))**(1.0/2.0)),(x49 + (multi_dot([x41.T,x41]))**(1.0/2.0)),(x49 + (multi_dot([x42.T,x42]))**(1.0/2.0)),(x49 + (multi_dot([x16.T,x16]))**(1.0/2.0)),(x49 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x49 + (multi_dot([x43.T,x43]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)
        v2 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v1,v1,v0,v1,v0,v1,v1,(v1 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v2),v1,v1,v1,v1,v0,v0,v0,v0,v1,v0,v0,v0,v1,v1,(v1 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v2),v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.SU1.Pd_rbl_tie_rod
        a1 = self.SU1.Pd_rbl_upright
        a2 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1]
        a3 = a2.T
        a4 = self.SU1.P_rbl_lower_strut
        a5 = A(a4).T
        a6 = self.SU1.Pd_rbl_upper_strut
        a7 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        a8 = B(a6,a7)
        a9 = a7.T
        a10 = self.SU1.P_rbl_upper_strut
        a11 = A(a10).T
        a12 = self.SU1.Pd_rbl_lower_strut
        a13 = B(a12,a2)
        a14 = a12.T
        a15 = B(a4,a2).T
        a16 = B(a10,a7)
        a17 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2]
        a18 = a17.T
        a19 = B(a12,a17)
        a20 = B(a4,a17).T
        a21 = config.ubar_rbl_lower_strut_jcl_strut
        a22 = config.ubar_rbl_upper_strut_jcl_strut
        a23 = (multi_dot([B(a12,a21),a12]) + -1.0*multi_dot([B(a6,a22),a6]))
        a24 = (self.SU1.Rd_rbl_lower_strut + -1.0*self.SU1.Rd_rbl_upper_strut + multi_dot([B(a4,a21),a12]) + multi_dot([B(a10,a22),a6]))
        a25 = (self.SU1.R_rbl_lower_strut.T + -1.0*self.SU1.R_rbl_upper_strut.T + multi_dot([config.Mbar_rbl_lower_strut_jcl_strut[:,2:3].T,a5]) + multi_dot([a21.T,a5]) + -1.0*multi_dot([a22.T,a11]))
        a26 = self.SU1.Pd_rbl_lca
        a27 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a28 = self.SU1.P_rbl_lca
        a29 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a30 = self.SU1.Pd_rbr_hub
        a31 = self.SU1.Pd_rbr_upright
        a32 = config.Mbar_rbr_upright_jcr_hub_bearing[:,2:3]
        a33 = a32.T
        a34 = self.SU1.P_rbr_upright
        a35 = A(a34).T
        a36 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a37 = self.SU1.P_rbr_hub
        a38 = A(a37).T
        a39 = B(a31,a32)
        a40 = a30.T
        a41 = B(a34,a32)
        a42 = config.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]
        a43 = config.F_jcr_hub_bearing(t,)
        a44 = np.eye(1,dtype=np.float64)
        a45 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a46 = cos(a43)
        a47 = config.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]
        a48 = sin(a43)
        a49 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a50 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a51 = a50.T
        a52 = self.SU1.P_rbr_lower_strut
        a53 = A(a52).T
        a54 = self.SU1.Pd_rbr_upper_strut
        a55 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a56 = B(a54,a55)
        a57 = a55.T
        a58 = self.SU1.P_rbr_upper_strut
        a59 = A(a58).T
        a60 = self.SU1.Pd_rbr_lower_strut
        a61 = B(a60,a50)
        a62 = a54.T
        a63 = B(a58,a55).T
        a64 = B(a52,a50)
        a65 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a66 = B(a54,a65)
        a67 = a65.T
        a68 = B(a58,a65).T
        a69 = config.ubar_rbr_upper_strut_jcr_strut
        a70 = config.ubar_rbr_lower_strut_jcr_strut
        a71 = (multi_dot([B(a54,a69),a54]) + -1.0*multi_dot([B(a60,a70),a60]))
        a72 = (self.SU1.Rd_rbr_upper_strut + -1.0*self.SU1.Rd_rbr_lower_strut + multi_dot([B(a52,a70),a60]) + multi_dot([B(a58,a69),a54]))
        a73 = (self.SU1.R_rbr_upper_strut.T + -1.0*self.SU1.R_rbr_lower_strut.T + multi_dot([config.Mbar_rbr_upper_strut_jcr_strut[:,2:3].T,a59]) + multi_dot([a69.T,a59]) + -1.0*multi_dot([a70.T,a53]))
        a74 = self.SU1.Pd_rbr_lca
        a75 = self.SU1.Pd_rbr_tie_rod
        a76 = self.SU1.Pd_rbr_uca
        a77 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a78 = self.SU1.P_rbr_lca
        a79 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a80 = a60.T
        a81 = self.SU1.Pd_rbl_uca
        a82 = self.SU1.Pd_rbl_hub
        a83 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a84 = a83.T
        a85 = self.SU1.P_rbl_hub
        a86 = A(a85).T
        a87 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a88 = self.SU1.P_rbl_upright
        a89 = A(a88).T
        a90 = B(a82,a83)
        a91 = a1.T
        a92 = B(a85,a83)
        a93 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a94 = config.F_jcl_hub_bearing(t,)
        a95 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a96 = cos(a94)
        a97 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a98 = sin(a94)
        a99 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_rbl_tie_rod_jcl_tie_upright),a0]) + -1.0*multi_dot([B(a1,config.ubar_rbl_upright_jcl_tie_upright),a1])),(multi_dot([a3,a5,a8,a6]) + multi_dot([a9,a11,a13,a12]) + 2.0*multi_dot([a14,a15,a16,a6])),(multi_dot([a18,a5,a8,a6]) + multi_dot([a9,a11,a19,a12]) + 2.0*multi_dot([a14,a20,a16,a6])),(multi_dot([a3,a5,a23]) + 2.0*multi_dot([a14,a15,a24]) + multi_dot([a25,a13,a12])),(multi_dot([a18,a5,a23]) + 2.0*multi_dot([a14,a20,a24]) + multi_dot([a25,a19,a12])),(multi_dot([B(a12,config.ubar_rbl_lower_strut_jcl_strut_lca),a12]) + -1.0*multi_dot([B(a26,config.ubar_rbl_lca_jcl_strut_lca),a26])),(multi_dot([a27.T,A(a28).T,B(a12,a29),a12]) + multi_dot([a29.T,a5,B(a26,a27),a26]) + 2.0*multi_dot([a14,B(a4,a29).T,B(a28,a27),a26])),(multi_dot([B(a30,config.ubar_rbr_hub_jcr_hub_bearing),a30]) + -1.0*multi_dot([B(a31,config.ubar_rbr_upright_jcr_hub_bearing),a31])),(multi_dot([a33,a35,B(a30,a36),a30]) + multi_dot([a36.T,a38,a39,a31]) + 2.0*multi_dot([a40,B(a37,a36).T,a41,a31])),(multi_dot([a33,a35,B(a30,a42),a30]) + multi_dot([a42.T,a38,a39,a31]) + 2.0*multi_dot([a40,B(a37,a42).T,a41,a31])),(derivative(a43,t,0.1,2)*-1.0*a44 + multi_dot([a45.T,a35,(a46*B(a30,a47) + a48*-1.0*B(a30,a49)),a30]) + multi_dot([(a46*multi_dot([a47.T,a38]) + a48*-1.0*multi_dot([a49.T,a38])),B(a31,a45),a31]) + 2.0*multi_dot([((a46*multi_dot([B(a37,a47),a30])).T + a48*-1.0*multi_dot([a40,B(a37,a49).T])),B(a34,a45),a31])),(multi_dot([a51,a53,a56,a54]) + multi_dot([a57,a59,a61,a60]) + 2.0*multi_dot([a62,a63,a64,a60])),(multi_dot([a51,a53,a66,a54]) + multi_dot([a67,a59,a61,a60]) + 2.0*multi_dot([a62,a68,a64,a60])),(multi_dot([a57,a59,a71]) + 2.0*multi_dot([a62,a63,a72]) + multi_dot([a73,a56,a54])),(multi_dot([a67,a59,a71]) + 2.0*multi_dot([a62,a68,a72]) + multi_dot([a73,a66,a54])),(multi_dot([B(a31,config.ubar_rbr_upright_jcr_lca_upright),a31]) + -1.0*multi_dot([B(a74,config.ubar_rbr_lca_jcr_lca_upright),a74])),(multi_dot([B(a31,config.ubar_rbr_upright_jcr_tie_upright),a31]) + -1.0*multi_dot([B(a75,config.ubar_rbr_tie_rod_jcr_tie_upright),a75])),(multi_dot([B(a31,config.ubar_rbr_upright_jcr_uca_upright),a31]) + -1.0*multi_dot([B(a76,config.ubar_rbr_uca_jcr_uca_upright),a76])),(multi_dot([B(a60,config.ubar_rbr_lower_strut_jcr_strut_lca),a60]) + -1.0*multi_dot([B(a74,config.ubar_rbr_lca_jcr_strut_lca),a74])),(multi_dot([a77.T,A(a78).T,B(a60,a79),a60]) + multi_dot([a79.T,a53,B(a74,a77),a74]) + 2.0*multi_dot([a80,B(a52,a79).T,B(a78,a77),a74])),(multi_dot([B(a81,config.ubar_rbl_uca_jcl_uca_upright),a81]) + -1.0*multi_dot([B(a1,config.ubar_rbl_upright_jcl_uca_upright),a1])),(multi_dot([B(a26,config.ubar_rbl_lca_jcl_lca_upright),a26]) + -1.0*multi_dot([B(a1,config.ubar_rbl_upright_jcl_lca_upright),a1])),(multi_dot([B(a1,config.ubar_rbl_upright_jcl_hub_bearing),a1]) + -1.0*multi_dot([B(a82,config.ubar_rbl_hub_jcl_hub_bearing),a82])),(multi_dot([a84,a86,B(a1,a87),a1]) + multi_dot([a87.T,a89,a90,a82]) + 2.0*multi_dot([a91,B(a88,a87).T,a92,a82])),(multi_dot([a84,a86,B(a1,a93),a1]) + multi_dot([a93.T,a89,a90,a82]) + 2.0*multi_dot([a91,B(a88,a93).T,a92,a82])),(derivative(a94,t,0.1,2)*-1.0*a44 + multi_dot([a95.T,a86,(a96*B(a1,a97) + a98*-1.0*B(a1,a99)),a1]) + multi_dot([(a96*multi_dot([a97.T,a89]) + a98*-1.0*multi_dot([a99.T,a89])),B(a82,a95),a82]) + 2.0*multi_dot([((a96*multi_dot([B(a88,a97),a1])).T + a98*-1.0*multi_dot([a91,B(a88,a99).T])),B(a85,a95),a82])),2.0*(multi_dot([a0.T,a0]))**(1.0/2.0),2.0*(multi_dot([a14,a12]))**(1.0/2.0),2.0*(multi_dot([a40,a30]))**(1.0/2.0),2.0*(multi_dot([a6.T,a6]))**(1.0/2.0),2.0*(multi_dot([a62,a54]))**(1.0/2.0),2.0*(multi_dot([a31.T,a31]))**(1.0/2.0),2.0*(multi_dot([a80,a60]))**(1.0/2.0),2.0*(multi_dot([a74.T,a74]))**(1.0/2.0),2.0*(multi_dot([a75.T,a75]))**(1.0/2.0),2.0*(multi_dot([a76.T,a76]))**(1.0/2.0),2.0*(multi_dot([a81.T,a81]))**(1.0/2.0),2.0*(multi_dot([a26.T,a26]))**(1.0/2.0),2.0*(multi_dot([a91,a1]))**(1.0/2.0),2.0*(multi_dot([a82.T,a82]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.SU1.P_rbl_tie_rod
        j2 = -1.0*j0
        j3 = self.SU1.P_rbl_upright
        j4 = np.zeros((1,3),dtype=np.float64)
        j5 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        j6 = j5.T
        j7 = self.SU1.P_rbl_upper_strut
        j8 = A(j7).T
        j9 = self.SU1.P_rbl_lower_strut
        j10 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1]
        j11 = B(j9,j10)
        j12 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2]
        j13 = B(j9,j12)
        j14 = j10.T
        j15 = A(j9).T
        j16 = multi_dot([j14,j15])
        j17 = config.ubar_rbl_lower_strut_jcl_strut
        j18 = B(j9,j17)
        j19 = config.ubar_rbl_upper_strut_jcl_strut
        j20 = (self.SU1.R_rbl_lower_strut.T + -1.0*self.SU1.R_rbl_upper_strut.T + multi_dot([config.Mbar_rbl_lower_strut_jcl_strut[:,2:3].T,j15]) + multi_dot([j17.T,j15]) + -1.0*multi_dot([j19.T,j8]))
        j21 = j12.T
        j22 = multi_dot([j21,j15])
        j23 = B(j7,j5)
        j24 = B(j7,j19)
        j25 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j26 = self.SU1.P_rbl_lca
        j27 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j28 = self.SU1.P_rbr_hub
        j29 = config.Mbar_rbr_upright_jcr_hub_bearing[:,2:3]
        j30 = j29.T
        j31 = self.SU1.P_rbr_upright
        j32 = A(j31).T
        j33 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j34 = config.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]
        j35 = A(j28).T
        j36 = B(j31,j29)
        j37 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j38 = config.F_jcr_hub_bearing(t,)
        j39 = cos(j38)
        j40 = config.Mbar_rbr_hub_jcr_hub_bearing[:,1:2]
        j41 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j42 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j43 = j42.T
        j44 = self.SU1.P_rbr_lower_strut
        j45 = A(j44).T
        j46 = self.SU1.P_rbr_upper_strut
        j47 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j48 = B(j46,j47)
        j49 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j50 = B(j46,j49)
        j51 = j47.T
        j52 = A(j46).T
        j53 = multi_dot([j51,j52])
        j54 = config.ubar_rbr_upper_strut_jcr_strut
        j55 = B(j46,j54)
        j56 = config.ubar_rbr_lower_strut_jcr_strut
        j57 = (self.SU1.R_rbr_upper_strut.T + -1.0*self.SU1.R_rbr_lower_strut.T + multi_dot([config.Mbar_rbr_upper_strut_jcr_strut[:,2:3].T,j52]) + multi_dot([j54.T,j52]) + -1.0*multi_dot([j56.T,j45]))
        j58 = j49.T
        j59 = multi_dot([j58,j52])
        j60 = B(j44,j42)
        j61 = B(j44,j56)
        j62 = self.SU1.P_rbr_lca
        j63 = self.SU1.P_rbr_tie_rod
        j64 = self.SU1.P_rbr_uca
        j65 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j66 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j67 = self.SU1.P_rbl_uca
        j68 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j69 = j68.T
        j70 = self.SU1.P_rbl_hub
        j71 = A(j70).T
        j72 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j73 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j74 = A(j3).T
        j75 = B(j70,j68)
        j76 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j77 = config.F_jcl_hub_bearing(t,)
        j78 = cos(j77)
        j79 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j80 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]

        self.jac_eq_blocks = [j0,B(j1,config.ubar_rbl_tie_rod_jcl_tie_upright),j2,-1.0*B(j3,config.ubar_rbl_upright_jcl_tie_upright),j4,multi_dot([j6,j8,j11]),j4,multi_dot([j14,j15,j23]),j4,multi_dot([j6,j8,j13]),j4,multi_dot([j21,j15,j23]),j16,(multi_dot([j14,j15,j18]) + multi_dot([j20,j11])),-1.0*j16,-1.0*multi_dot([j14,j15,j24]),j22,(multi_dot([j21,j15,j18]) + multi_dot([j20,j13])),-1.0*j22,-1.0*multi_dot([j21,j15,j24]),j0,B(j9,config.ubar_rbl_lower_strut_jcl_strut_lca),j2,-1.0*B(j26,config.ubar_rbl_lca_jcl_strut_lca),j4,multi_dot([j25.T,A(j26).T,B(j9,j27)]),j4,multi_dot([j27.T,j15,B(j26,j25)]),j0,B(j28,config.ubar_rbr_hub_jcr_hub_bearing),j2,-1.0*B(j31,config.ubar_rbr_upright_jcr_hub_bearing),j4,multi_dot([j30,j32,B(j28,j33)]),j4,multi_dot([j33.T,j35,j36]),j4,multi_dot([j30,j32,B(j28,j34)]),j4,multi_dot([j34.T,j35,j36]),j4,multi_dot([j37.T,j32,(j39*B(j28,j40) + sin(j38)*-1.0*B(j28,j41))]),j4,multi_dot([(j39*multi_dot([j40.T,j35]) + sin(j38)*-1.0*multi_dot([j41.T,j35])),B(j31,j37)]),j4,multi_dot([j43,j45,j48]),j4,multi_dot([j51,j52,j60]),j4,multi_dot([j43,j45,j50]),j4,multi_dot([j58,j52,j60]),j53,(multi_dot([j51,j52,j55]) + multi_dot([j57,j48])),-1.0*j53,-1.0*multi_dot([j51,j52,j61]),j59,(multi_dot([j58,j52,j55]) + multi_dot([j57,j50])),-1.0*j59,-1.0*multi_dot([j58,j52,j61]),j0,B(j31,config.ubar_rbr_upright_jcr_lca_upright),j2,-1.0*B(j62,config.ubar_rbr_lca_jcr_lca_upright),j0,B(j31,config.ubar_rbr_upright_jcr_tie_upright),j2,-1.0*B(j63,config.ubar_rbr_tie_rod_jcr_tie_upright),j0,B(j31,config.ubar_rbr_upright_jcr_uca_upright),j2,-1.0*B(j64,config.ubar_rbr_uca_jcr_uca_upright),j0,B(j44,config.ubar_rbr_lower_strut_jcr_strut_lca),j2,-1.0*B(j62,config.ubar_rbr_lca_jcr_strut_lca),j4,multi_dot([j65.T,A(j62).T,B(j44,j66)]),j4,multi_dot([j66.T,j45,B(j62,j65)]),j0,B(j67,config.ubar_rbl_uca_jcl_uca_upright),j2,-1.0*B(j3,config.ubar_rbl_upright_jcl_uca_upright),j0,B(j26,config.ubar_rbl_lca_jcl_lca_upright),j2,-1.0*B(j3,config.ubar_rbl_upright_jcl_lca_upright),j0,B(j3,config.ubar_rbl_upright_jcl_hub_bearing),j2,-1.0*B(j70,config.ubar_rbl_hub_jcl_hub_bearing),j4,multi_dot([j69,j71,B(j3,j72)]),j4,multi_dot([j72.T,j74,j75]),j4,multi_dot([j69,j71,B(j3,j73)]),j4,multi_dot([j73.T,j74,j75]),j4,multi_dot([j76.T,j71,(j78*B(j3,j79) + sin(j77)*-1.0*B(j3,j80))]),j4,multi_dot([(j78*multi_dot([j79.T,j74]) + sin(j77)*-1.0*multi_dot([j80.T,j74])),B(j70,j76)]),2.0*j1.T,2.0*j9.T,2.0*j28.T,2.0*j7.T,2.0*j46.T,2.0*j31.T,2.0*j44.T,2.0*j62.T,2.0*j63.T,2.0*j64.T,2.0*j67.T,2.0*j26.T,2.0*j3.T,2.0*j70.T]
  
