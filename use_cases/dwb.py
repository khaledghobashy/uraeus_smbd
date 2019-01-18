
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcr_hub_bearing = lambda t : 0
        self.ax1_jcl_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcl_hub_bearing = lambda t : 0
        self.ax1_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_uca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_lca = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_upright = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_upper_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_lower_strut = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_tie_rod = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbl_hub = np.array([[1], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbl_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbl_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)

    def eval_constants(self):

        c0 = A(self.P_rbr_uca).T
        c1 = self.pt1_jcr_uca_upright
        c2 = A(self.P_rbr_upright).T
        c3 = -1.0*multi_dot([c2,self.R_rbr_upright])
        c4 = Triad(self.ax1_jcr_uca_upright,)
        c5 = A(self.P_rbl_uca).T
        c6 = self.pt1_jcl_uca_upright
        c7 = A(self.P_rbl_upright).T
        c8 = -1.0*multi_dot([c7,self.R_rbl_upright])
        c9 = Triad(self.ax1_jcl_uca_upright,)
        c10 = A(self.P_rbr_lca).T
        c11 = self.pt1_jcr_lca_upright
        c12 = -1.0*multi_dot([c10,self.R_rbr_lca])
        c13 = Triad(self.ax1_jcr_lca_upright,)
        c14 = self.pt1_jcr_strut_lca
        c15 = A(self.P_rbr_lower_strut).T
        c16 = -1.0*multi_dot([c15,self.R_rbr_lower_strut])
        c17 = self.ax1_jcr_strut_lca
        c18 = A(self.P_rbl_lca).T
        c19 = self.pt1_jcl_lca_upright
        c20 = -1.0*multi_dot([c18,self.R_rbl_lca])
        c21 = Triad(self.ax1_jcl_lca_upright,)
        c22 = self.pt1_jcl_strut_lca
        c23 = A(self.P_rbl_lower_strut).T
        c24 = -1.0*multi_dot([c23,self.R_rbl_lower_strut])
        c25 = self.ax1_jcl_strut_lca
        c26 = self.pt1_jcr_tie_upright
        c27 = A(self.P_rbr_tie_rod).T
        c28 = Triad(self.ax1_jcr_tie_upright,)
        c29 = self.pt1_jcr_hub_bearing
        c30 = A(self.P_rbr_hub).T
        c31 = Triad(self.ax1_jcr_hub_bearing,)
        c32 = self.pt1_jcl_tie_upright
        c33 = A(self.P_rbl_tie_rod).T
        c34 = Triad(self.ax1_jcl_tie_upright,)
        c35 = self.pt1_jcl_hub_bearing
        c36 = A(self.P_rbl_hub).T
        c37 = Triad(self.ax1_jcl_hub_bearing,)
        c38 = A(self.P_rbr_upper_strut).T
        c39 = self.pt1_jcr_strut
        c40 = Triad(self.ax1_jcr_strut,)
        c41 = A(self.P_rbl_upper_strut).T
        c42 = self.pt1_jcl_strut
        c43 = Triad(self.ax1_jcl_strut,)

        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,self.R_rbr_uca]))
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c2,c1]) + c3)
        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c0,c4])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c2,c4])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c5,c6]) + -1.0*multi_dot([c5,self.R_rbl_uca]))
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c7,c6]) + c8)
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c5,c9])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c7,c9])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c10,c11]) + c12)
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c2,c11]) + c3)
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c10,c13])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c2,c13])
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c10,c14]) + c12)
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c15,c14]) + c16)
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c10,Triad(c17,)])
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c15,Triad(c17,self.ax2_jcr_strut_lca)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c18,c19]) + c20)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c7,c19]) + c8)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c18,c21])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c7,c21])
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c18,c22]) + c20)
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c23,c22]) + c24)
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c18,Triad(c25,)])
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c23,Triad(c25,self.ax2_jcl_strut_lca)])
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c2,c26]) + c3)
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c27,c26]) + -1.0*multi_dot([c27,self.R_rbr_tie_rod]))
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c2,c28])
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c27,c28])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c2,c29]) + c3)
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c30,c29]) + -1.0*multi_dot([c30,self.R_rbr_hub]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c2,c31])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c30,c31])
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c7,c32]) + c8)
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c33,c32]) + -1.0*multi_dot([c33,self.R_rbl_tie_rod]))
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c7,c34])
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c33,c34])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c7,c35]) + c8)
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c36,c35]) + -1.0*multi_dot([c36,self.R_rbl_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c7,c37])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c36,c37])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c38,c39]) + -1.0*multi_dot([c38,self.R_rbr_upper_strut]))
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c15,c39]) + c16)
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c38,c40])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c15,c40])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c41,c42]) + -1.0*multi_dot([c41,self.R_rbl_upper_strut]))
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c23,c42]) + c24)
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c41,c43])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c23,c43])

    @property
    def q_initial(self):
        q = np.concatenate([self.R_rbr_uca,self.P_rbr_uca,self.R_rbl_uca,self.P_rbl_uca,self.R_rbr_lca,self.P_rbr_lca,self.R_rbl_lca,self.P_rbl_lca,self.R_rbr_upright,self.P_rbr_upright,self.R_rbl_upright,self.P_rbl_upright,self.R_rbr_upper_strut,self.P_rbr_upper_strut,self.R_rbl_upper_strut,self.P_rbl_upper_strut,self.R_rbr_lower_strut,self.P_rbr_lower_strut,self.R_rbl_lower_strut,self.P_rbl_lower_strut,self.R_rbr_tie_rod,self.P_rbr_tie_rod,self.R_rbl_tie_rod,self.P_rbl_tie_rod,self.R_rbr_hub,self.P_rbr_hub,self.R_rbl_hub,self.P_rbl_hub])
        return q

    @property
    def qd_initial(self):
        qd = np.concatenate([self.Rd_rbr_uca,self.Pd_rbr_uca,self.Rd_rbl_uca,self.Pd_rbl_uca,self.Rd_rbr_lca,self.Pd_rbr_lca,self.Rd_rbl_lca,self.Pd_rbl_lca,self.Rd_rbr_upright,self.Pd_rbr_upright,self.Rd_rbl_upright,self.Pd_rbl_upright,self.Rd_rbr_upper_strut,self.Pd_rbr_upper_strut,self.Rd_rbl_upper_strut,self.Pd_rbl_upper_strut,self.Rd_rbr_lower_strut,self.Pd_rbr_lower_strut,self.Rd_rbl_lower_strut,self.Pd_rbl_lower_strut,self.Rd_rbr_tie_rod,self.Pd_rbr_tie_rod,self.Rd_rbl_tie_rod,self.Pd_rbl_tie_rod,self.Rd_rbr_hub,self.Pd_rbr_hub,self.Rd_rbl_hub,self.Pd_rbl_hub])
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
        self.jac_cols = np.array([2,3,10,11,4,5,12,13,6,7,10,11,6,7,18,19,6,7,18,19,8,9,12,13,8,9,20,21,8,9,20,21,10,11,22,23,10,11,26,27,10,11,26,27,10,11,26,27,10,11,26,27,12,13,24,25,12,13,28,29,12,13,28,29,12,13,28,29,12,13,28,29,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,20,21,16,17,20,21,16,17,20,21,3,5,7,9,11,13,15,17,19,21,23,25,27,29])

        self.nrows = max(self.pos_rows)
        self.ncols = max(self.jac_cols)

    
    def set_q(self,q):
        self.R_rbr_uca = q[0:3,0:1]
        self.P_rbr_uca = q[3:7,0:1]
        self.R_rbl_uca = q[7:10,0:1]
        self.P_rbl_uca = q[10:14,0:1]
        self.R_rbr_lca = q[14:17,0:1]
        self.P_rbr_lca = q[17:21,0:1]
        self.R_rbl_lca = q[21:24,0:1]
        self.P_rbl_lca = q[24:28,0:1]
        self.R_rbr_upright = q[28:31,0:1]
        self.P_rbr_upright = q[31:35,0:1]
        self.R_rbl_upright = q[35:38,0:1]
        self.P_rbl_upright = q[38:42,0:1]
        self.R_rbr_upper_strut = q[42:45,0:1]
        self.P_rbr_upper_strut = q[45:49,0:1]
        self.R_rbl_upper_strut = q[49:52,0:1]
        self.P_rbl_upper_strut = q[52:56,0:1]
        self.R_rbr_lower_strut = q[56:59,0:1]
        self.P_rbr_lower_strut = q[59:63,0:1]
        self.R_rbl_lower_strut = q[63:66,0:1]
        self.P_rbl_lower_strut = q[66:70,0:1]
        self.R_rbr_tie_rod = q[70:73,0:1]
        self.P_rbr_tie_rod = q[73:77,0:1]
        self.R_rbl_tie_rod = q[77:80,0:1]
        self.P_rbl_tie_rod = q[80:84,0:1]
        self.R_rbr_hub = q[84:87,0:1]
        self.P_rbr_hub = q[87:91,0:1]
        self.R_rbl_hub = q[91:94,0:1]
        self.P_rbl_hub = q[94:98,0:1]

    
    def set_qd(self,qd):
        self.Rd_rbr_uca = qd[0:3,0:1]
        self.Pd_rbr_uca = qd[3:7,0:1]
        self.Rd_rbl_uca = qd[7:10,0:1]
        self.Pd_rbl_uca = qd[10:14,0:1]
        self.Rd_rbr_lca = qd[14:17,0:1]
        self.Pd_rbr_lca = qd[17:21,0:1]
        self.Rd_rbl_lca = qd[21:24,0:1]
        self.Pd_rbl_lca = qd[24:28,0:1]
        self.Rd_rbr_upright = qd[28:31,0:1]
        self.Pd_rbr_upright = qd[31:35,0:1]
        self.Rd_rbl_upright = qd[35:38,0:1]
        self.Pd_rbl_upright = qd[38:42,0:1]
        self.Rd_rbr_upper_strut = qd[42:45,0:1]
        self.Pd_rbr_upper_strut = qd[45:49,0:1]
        self.Rd_rbl_upper_strut = qd[49:52,0:1]
        self.Pd_rbl_upper_strut = qd[52:56,0:1]
        self.Rd_rbr_lower_strut = qd[56:59,0:1]
        self.Pd_rbr_lower_strut = qd[59:63,0:1]
        self.Rd_rbl_lower_strut = qd[63:66,0:1]
        self.Pd_rbl_lower_strut = qd[66:70,0:1]
        self.Rd_rbr_tie_rod = qd[70:73,0:1]
        self.Pd_rbr_tie_rod = qd[73:77,0:1]
        self.Rd_rbl_tie_rod = qd[77:80,0:1]
        self.Pd_rbl_tie_rod = qd[80:84,0:1]
        self.Rd_rbr_hub = qd[84:87,0:1]
        self.Pd_rbr_hub = qd[87:91,0:1]
        self.Rd_rbl_hub = qd[91:94,0:1]
        self.Pd_rbl_hub = qd[94:98,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_upright
        x1 = -1.0*x0
        x2 = self.P_rbr_uca
        x3 = self.P_rbr_upright
        x4 = A(x3)
        x5 = self.R_rbl_upright
        x6 = -1.0*x5
        x7 = self.P_rbl_uca
        x8 = self.P_rbl_upright
        x9 = A(x8)
        x10 = self.R_rbr_lca
        x11 = self.P_rbr_lca
        x12 = A(x11)
        x13 = -1.0*self.R_rbr_lower_strut
        x14 = self.P_rbr_lower_strut
        x15 = A(x14)
        x16 = self.R_rbl_lca
        x17 = self.P_rbl_lca
        x18 = A(x17)
        x19 = -1.0*self.R_rbl_lower_strut
        x20 = self.P_rbl_lower_strut
        x21 = A(x20)
        x22 = self.P_rbr_tie_rod
        x23 = self.P_rbr_hub
        x24 = A(x23)
        x25 = x4.T
        x26 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x27 = config.F_jcr_hub_bearing(t,)
        x28 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        x29 = self.P_rbl_tie_rod
        x30 = self.P_rbl_hub
        x31 = A(x30)
        x32 = x9.T
        x33 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x34 = config.F_jcl_hub_bearing(t,)
        x35 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        x36 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x37 = self.P_rbr_upper_strut
        x38 = A(x37)
        x39 = x38.T
        x40 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x41 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x42 = (self.R_rbr_upper_strut + x13 + multi_dot([x38,config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]]) + multi_dot([x38,config.ubar_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x15,config.ubar_rbr_lower_strut_jcr_strut]))
        x43 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x44 = self.P_rbl_upper_strut
        x45 = A(x44)
        x46 = x45.T
        x47 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x48 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x49 = (self.R_rbl_upper_strut + x19 + multi_dot([x45,config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]]) + multi_dot([x45,config.ubar_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x21,config.ubar_rbl_lower_strut_jcl_strut]))
        x50 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(self.R_rbr_uca + x1 + multi_dot([A(x2),config.ubar_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x4,config.ubar_rbr_upright_jcr_uca_upright])),(self.R_rbl_uca + x6 + multi_dot([A(x7),config.ubar_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x9,config.ubar_rbl_upright_jcl_uca_upright])),(x10 + x1 + multi_dot([x12,config.ubar_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x4,config.ubar_rbr_upright_jcr_lca_upright])),(x10 + x13 + multi_dot([x12,config.ubar_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x15,config.ubar_rbr_lower_strut_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lca_jcr_strut_lca[:,0:1].T,x12.T,x15,config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x16 + x6 + multi_dot([x18,config.ubar_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x9,config.ubar_rbl_upright_jcl_lca_upright])),(x16 + x19 + multi_dot([x18,config.ubar_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x21,config.ubar_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lca_jcl_strut_lca[:,0:1].T,x18.T,x21,config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x0 + -1.0*self.R_rbr_tie_rod + multi_dot([x4,config.ubar_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([A(x22),config.ubar_rbr_tie_rod_jcr_tie_upright])),(x0 + -1.0*self.R_rbr_hub + multi_dot([x4,config.ubar_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x24,config.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x25,x24,x26]),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x25,x24,x26]),(cos(x27)*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x25,x24,x28]) + sin(x27)*-1.0*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x25,x24,x28])),(x5 + -1.0*self.R_rbl_tie_rod + multi_dot([x9,config.ubar_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([A(x29),config.ubar_rbl_tie_rod_jcl_tie_upright])),(x5 + -1.0*self.R_rbl_hub + multi_dot([x9,config.ubar_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x31,config.ubar_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x32,x31,x33]),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x32,x31,x33]),(cos(x34)*multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x32,x31,x35]) + sin(x34)*-1.0*multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x32,x31,x35])),multi_dot([x36,x39,x15,x40]),multi_dot([x41,x39,x15,x40]),multi_dot([x36,x39,x42]),multi_dot([x41,x39,x42]),multi_dot([x43,x46,x21,x47]),multi_dot([x48,x46,x21,x47]),multi_dot([x43,x46,x49]),multi_dot([x48,x46,x49]),(x50 + (multi_dot([x2.T,x2]))**(1.0/2.0)),(x50 + (multi_dot([x7.T,x7]))**(1.0/2.0)),(x50 + (multi_dot([x11.T,x11]))**(1.0/2.0)),(x50 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x50 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x50 + (multi_dot([x8.T,x8]))**(1.0/2.0)),(x50 + (multi_dot([x37.T,x37]))**(1.0/2.0)),(x50 + (multi_dot([x44.T,x44]))**(1.0/2.0)),(x50 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x50 + (multi_dot([x20.T,x20]))**(1.0/2.0)),(x50 + (multi_dot([x22.T,x22]))**(1.0/2.0)),(x50 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x50 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x50 + (multi_dot([x30.T,x30]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)
        v2 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [v0,v0,v0,v0,v1,v0,v0,v1,v0,v0,v1,v1,(v1 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v2),v0,v0,v1,v1,(v1 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v2),v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_rbr_upright
        a2 = self.Pd_rbl_uca
        a3 = self.Pd_rbl_upright
        a4 = self.Pd_rbr_lca
        a5 = self.Pd_rbr_lower_strut
        a6 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a7 = self.P_rbr_lca
        a8 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a9 = self.P_rbr_lower_strut
        a10 = A(a9).T
        a11 = a4.T
        a12 = self.Pd_rbl_lca
        a13 = self.Pd_rbl_lower_strut
        a14 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a15 = self.P_rbl_lca
        a16 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a17 = self.P_rbl_lower_strut
        a18 = A(a17).T
        a19 = a12.T
        a20 = self.Pd_rbr_tie_rod
        a21 = self.Pd_rbr_hub
        a22 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a23 = a22.T
        a24 = self.P_rbr_hub
        a25 = A(a24).T
        a26 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a27 = self.P_rbr_upright
        a28 = A(a27).T
        a29 = B(a21,a22)
        a30 = a1.T
        a31 = B(a24,a22)
        a32 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a33 = config.F_jcr_hub_bearing(t,)
        a34 = np.eye(1,dtype=np.float64)
        a35 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a36 = cos(a33)
        a37 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a38 = sin(a33)
        a39 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a40 = self.Pd_rbl_tie_rod
        a41 = self.Pd_rbl_hub
        a42 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a43 = a42.T
        a44 = self.P_rbl_hub
        a45 = A(a44).T
        a46 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a47 = self.P_rbl_upright
        a48 = A(a47).T
        a49 = B(a41,a42)
        a50 = a3.T
        a51 = B(a44,a42)
        a52 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a53 = config.F_jcl_hub_bearing(t,)
        a54 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a55 = cos(a53)
        a56 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a57 = sin(a53)
        a58 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a59 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a60 = a59.T
        a61 = self.Pd_rbr_upper_strut
        a62 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a63 = B(a61,a62)
        a64 = a62.T
        a65 = self.P_rbr_upper_strut
        a66 = A(a65).T
        a67 = B(a5,a59)
        a68 = a61.T
        a69 = B(a65,a62).T
        a70 = B(a9,a59)
        a71 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a72 = B(a61,a71)
        a73 = a71.T
        a74 = B(a65,a71).T
        a75 = config.ubar_rbr_upper_strut_jcr_strut
        a76 = config.ubar_rbr_lower_strut_jcr_strut
        a77 = (multi_dot([B(a61,a75),a61]) + -1.0*multi_dot([B(a5,a76),a5]))
        a78 = (self.Rd_rbr_upper_strut + -1.0*self.Rd_rbr_lower_strut + multi_dot([B(a9,a76),a5]) + multi_dot([B(a65,a75),a61]))
        a79 = (self.R_rbr_upper_strut.T + -1.0*self.R_rbr_lower_strut.T + multi_dot([config.Mbar_rbr_upper_strut_jcr_strut[:,2:3].T,a66]) + multi_dot([a75.T,a66]) + -1.0*multi_dot([a76.T,a10]))
        a80 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a81 = a80.T
        a82 = self.P_rbl_upper_strut
        a83 = A(a82).T
        a84 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a85 = B(a13,a84)
        a86 = a84.T
        a87 = self.Pd_rbl_upper_strut
        a88 = B(a87,a80)
        a89 = a87.T
        a90 = B(a82,a80).T
        a91 = B(a17,a84)
        a92 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a93 = a92.T
        a94 = B(a87,a92)
        a95 = B(a82,a92).T
        a96 = config.ubar_rbl_upper_strut_jcl_strut
        a97 = config.ubar_rbl_lower_strut_jcl_strut
        a98 = (multi_dot([B(a87,a96),a87]) + -1.0*multi_dot([B(a13,a97),a13]))
        a99 = (self.Rd_rbl_upper_strut + -1.0*self.Rd_rbl_lower_strut + multi_dot([B(a17,a97),a13]) + multi_dot([B(a82,a96),a87]))
        a100 = (self.R_rbl_upper_strut.T + -1.0*self.R_rbl_lower_strut.T + multi_dot([config.Mbar_rbl_upper_strut_jcl_strut[:,2:3].T,a83]) + multi_dot([a96.T,a83]) + -1.0*multi_dot([a97.T,a18]))

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_rbr_uca_jcr_uca_upright),a0]) + -1.0*multi_dot([B(a1,config.ubar_rbr_upright_jcr_uca_upright),a1])),(multi_dot([B(a2,config.ubar_rbl_uca_jcl_uca_upright),a2]) + -1.0*multi_dot([B(a3,config.ubar_rbl_upright_jcl_uca_upright),a3])),(multi_dot([B(a4,config.ubar_rbr_lca_jcr_lca_upright),a4]) + -1.0*multi_dot([B(a1,config.ubar_rbr_upright_jcr_lca_upright),a1])),(multi_dot([B(a4,config.ubar_rbr_lca_jcr_strut_lca),a4]) + -1.0*multi_dot([B(a5,config.ubar_rbr_lower_strut_jcr_strut_lca),a5])),(multi_dot([a6.T,A(a7).T,B(a5,a8),a5]) + multi_dot([a8.T,a10,B(a4,a6),a4]) + 2.0*multi_dot([a11,B(a7,a6).T,B(a9,a8),a5])),(multi_dot([B(a12,config.ubar_rbl_lca_jcl_lca_upright),a12]) + -1.0*multi_dot([B(a3,config.ubar_rbl_upright_jcl_lca_upright),a3])),(multi_dot([B(a12,config.ubar_rbl_lca_jcl_strut_lca),a12]) + -1.0*multi_dot([B(a13,config.ubar_rbl_lower_strut_jcl_strut_lca),a13])),(multi_dot([a14.T,A(a15).T,B(a13,a16),a13]) + multi_dot([a16.T,a18,B(a12,a14),a12]) + 2.0*multi_dot([a19,B(a15,a14).T,B(a17,a16),a13])),(multi_dot([B(a1,config.ubar_rbr_upright_jcr_tie_upright),a1]) + -1.0*multi_dot([B(a20,config.ubar_rbr_tie_rod_jcr_tie_upright),a20])),(multi_dot([B(a1,config.ubar_rbr_upright_jcr_hub_bearing),a1]) + -1.0*multi_dot([B(a21,config.ubar_rbr_hub_jcr_hub_bearing),a21])),(multi_dot([a23,a25,B(a1,a26),a1]) + multi_dot([a26.T,a28,a29,a21]) + 2.0*multi_dot([a30,B(a27,a26).T,a31,a21])),(multi_dot([a23,a25,B(a1,a32),a1]) + multi_dot([a32.T,a28,a29,a21]) + 2.0*multi_dot([a30,B(a27,a32).T,a31,a21])),(derivative(a33,t,0.1,2)*-1.0*a34 + multi_dot([a35.T,a25,(a36*B(a1,a37) + a38*-1.0*B(a1,a39)),a1]) + multi_dot([(a36*multi_dot([a37.T,a28]) + a38*-1.0*multi_dot([a39.T,a28])),B(a21,a35),a21]) + 2.0*multi_dot([((a36*multi_dot([B(a27,a37),a1])).T + a38*-1.0*multi_dot([a30,B(a27,a39).T])),B(a24,a35),a21])),(multi_dot([B(a3,config.ubar_rbl_upright_jcl_tie_upright),a3]) + -1.0*multi_dot([B(a40,config.ubar_rbl_tie_rod_jcl_tie_upright),a40])),(multi_dot([B(a3,config.ubar_rbl_upright_jcl_hub_bearing),a3]) + -1.0*multi_dot([B(a41,config.ubar_rbl_hub_jcl_hub_bearing),a41])),(multi_dot([a43,a45,B(a3,a46),a3]) + multi_dot([a46.T,a48,a49,a41]) + 2.0*multi_dot([a50,B(a47,a46).T,a51,a41])),(multi_dot([a43,a45,B(a3,a52),a3]) + multi_dot([a52.T,a48,a49,a41]) + 2.0*multi_dot([a50,B(a47,a52).T,a51,a41])),(derivative(a53,t,0.1,2)*-1.0*a34 + multi_dot([a54.T,a45,(a55*B(a3,a56) + a57*-1.0*B(a3,a58)),a3]) + multi_dot([(a55*multi_dot([a56.T,a48]) + a57*-1.0*multi_dot([a58.T,a48])),B(a41,a54),a41]) + 2.0*multi_dot([((a55*multi_dot([B(a47,a56),a3])).T + a57*-1.0*multi_dot([a50,B(a47,a58).T])),B(a44,a54),a41])),(multi_dot([a60,a10,a63,a61]) + multi_dot([a64,a66,a67,a5]) + 2.0*multi_dot([a68,a69,a70,a5])),(multi_dot([a60,a10,a72,a61]) + multi_dot([a73,a66,a67,a5]) + 2.0*multi_dot([a68,a74,a70,a5])),(multi_dot([a64,a66,a77]) + 2.0*multi_dot([a68,a69,a78]) + multi_dot([a79,a63,a61])),(multi_dot([a73,a66,a77]) + 2.0*multi_dot([a68,a74,a78]) + multi_dot([a79,a72,a61])),(multi_dot([a81,a83,a85,a13]) + multi_dot([a86,a18,a88,a87]) + 2.0*multi_dot([a89,a90,a91,a13])),(multi_dot([a93,a83,a85,a13]) + multi_dot([a86,a18,a94,a87]) + 2.0*multi_dot([a89,a95,a91,a13])),(multi_dot([a81,a83,a98]) + 2.0*multi_dot([a89,a90,a99]) + multi_dot([a100,a88,a87])),(multi_dot([a93,a83,a98]) + 2.0*multi_dot([a89,a95,a99]) + multi_dot([a100,a94,a87])),2.0*(multi_dot([a0.T,a0]))**(1.0/2.0),2.0*(multi_dot([a2.T,a2]))**(1.0/2.0),2.0*(multi_dot([a11,a4]))**(1.0/2.0),2.0*(multi_dot([a19,a12]))**(1.0/2.0),2.0*(multi_dot([a30,a1]))**(1.0/2.0),2.0*(multi_dot([a50,a3]))**(1.0/2.0),2.0*(multi_dot([a68,a61]))**(1.0/2.0),2.0*(multi_dot([a89,a87]))**(1.0/2.0),2.0*(multi_dot([a5.T,a5]))**(1.0/2.0),2.0*(multi_dot([a13.T,a13]))**(1.0/2.0),2.0*(multi_dot([a20.T,a20]))**(1.0/2.0),2.0*(multi_dot([a40.T,a40]))**(1.0/2.0),2.0*(multi_dot([a21.T,a21]))**(1.0/2.0),2.0*(multi_dot([a41.T,a41]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbr_uca
        j2 = -1.0*j0
        j3 = self.P_rbr_upright
        j4 = self.P_rbl_uca
        j5 = self.P_rbl_upright
        j6 = self.P_rbr_lca
        j7 = np.zeros((1,3),dtype=np.float64)
        j8 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j9 = self.P_rbr_lower_strut
        j10 = A(j9).T
        j11 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j12 = self.P_rbl_lca
        j13 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j14 = self.P_rbl_lower_strut
        j15 = A(j14).T
        j16 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j17 = self.P_rbr_tie_rod
        j18 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j19 = j18.T
        j20 = self.P_rbr_hub
        j21 = A(j20).T
        j22 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j23 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j24 = A(j3).T
        j25 = B(j20,j18)
        j26 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j27 = config.F_jcr_hub_bearing(t,)
        j28 = cos(j27)
        j29 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j30 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j31 = self.P_rbl_tie_rod
        j32 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j33 = j32.T
        j34 = self.P_rbl_hub
        j35 = A(j34).T
        j36 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j37 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j38 = A(j5).T
        j39 = B(j34,j32)
        j40 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j41 = config.F_jcl_hub_bearing(t,)
        j42 = cos(j41)
        j43 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j44 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j45 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j46 = j45.T
        j47 = self.P_rbr_upper_strut
        j48 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j49 = B(j47,j48)
        j50 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j51 = B(j47,j50)
        j52 = j48.T
        j53 = A(j47).T
        j54 = multi_dot([j52,j53])
        j55 = config.ubar_rbr_upper_strut_jcr_strut
        j56 = B(j47,j55)
        j57 = config.ubar_rbr_lower_strut_jcr_strut
        j58 = (self.R_rbr_upper_strut.T + -1.0*self.R_rbr_lower_strut.T + multi_dot([config.Mbar_rbr_upper_strut_jcr_strut[:,2:3].T,j53]) + multi_dot([j55.T,j53]) + -1.0*multi_dot([j57.T,j10]))
        j59 = j50.T
        j60 = multi_dot([j59,j53])
        j61 = B(j9,j45)
        j62 = B(j9,j57)
        j63 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j64 = j63.T
        j65 = self.P_rbl_upper_strut
        j66 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j67 = B(j65,j66)
        j68 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j69 = B(j65,j68)
        j70 = j66.T
        j71 = A(j65).T
        j72 = multi_dot([j70,j71])
        j73 = config.ubar_rbl_upper_strut_jcl_strut
        j74 = B(j65,j73)
        j75 = config.ubar_rbl_lower_strut_jcl_strut
        j76 = (self.R_rbl_upper_strut.T + -1.0*self.R_rbl_lower_strut.T + multi_dot([config.Mbar_rbl_upper_strut_jcl_strut[:,2:3].T,j71]) + multi_dot([j73.T,j71]) + -1.0*multi_dot([j75.T,j15]))
        j77 = j68.T
        j78 = multi_dot([j77,j71])
        j79 = B(j14,j63)
        j80 = B(j14,j75)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_rbr_uca_jcr_uca_upright),j2,-1.0*B(j3,config.ubar_rbr_upright_jcr_uca_upright),j0,B(j4,config.ubar_rbl_uca_jcl_uca_upright),j2,-1.0*B(j5,config.ubar_rbl_upright_jcl_uca_upright),j0,B(j6,config.ubar_rbr_lca_jcr_lca_upright),j2,-1.0*B(j3,config.ubar_rbr_upright_jcr_lca_upright),j0,B(j6,config.ubar_rbr_lca_jcr_strut_lca),j2,-1.0*B(j9,config.ubar_rbr_lower_strut_jcr_strut_lca),j7,multi_dot([j8.T,j10,B(j6,j11)]),j7,multi_dot([j11.T,A(j6).T,B(j9,j8)]),j0,B(j12,config.ubar_rbl_lca_jcl_lca_upright),j2,-1.0*B(j5,config.ubar_rbl_upright_jcl_lca_upright),j0,B(j12,config.ubar_rbl_lca_jcl_strut_lca),j2,-1.0*B(j14,config.ubar_rbl_lower_strut_jcl_strut_lca),j7,multi_dot([j13.T,j15,B(j12,j16)]),j7,multi_dot([j16.T,A(j12).T,B(j14,j13)]),j0,B(j3,config.ubar_rbr_upright_jcr_tie_upright),j2,-1.0*B(j17,config.ubar_rbr_tie_rod_jcr_tie_upright),j0,B(j3,config.ubar_rbr_upright_jcr_hub_bearing),j2,-1.0*B(j20,config.ubar_rbr_hub_jcr_hub_bearing),j7,multi_dot([j19,j21,B(j3,j22)]),j7,multi_dot([j22.T,j24,j25]),j7,multi_dot([j19,j21,B(j3,j23)]),j7,multi_dot([j23.T,j24,j25]),j7,multi_dot([j26.T,j21,(j28*B(j3,j29) + sin(j27)*-1.0*B(j3,j30))]),j7,multi_dot([(j28*multi_dot([j29.T,j24]) + sin(j27)*-1.0*multi_dot([j30.T,j24])),B(j20,j26)]),j0,B(j5,config.ubar_rbl_upright_jcl_tie_upright),j2,-1.0*B(j31,config.ubar_rbl_tie_rod_jcl_tie_upright),j0,B(j5,config.ubar_rbl_upright_jcl_hub_bearing),j2,-1.0*B(j34,config.ubar_rbl_hub_jcl_hub_bearing),j7,multi_dot([j33,j35,B(j5,j36)]),j7,multi_dot([j36.T,j38,j39]),j7,multi_dot([j33,j35,B(j5,j37)]),j7,multi_dot([j37.T,j38,j39]),j7,multi_dot([j40.T,j35,(j42*B(j5,j43) + sin(j41)*-1.0*B(j5,j44))]),j7,multi_dot([(j42*multi_dot([j43.T,j38]) + sin(j41)*-1.0*multi_dot([j44.T,j38])),B(j34,j40)]),j7,multi_dot([j46,j10,j49]),j7,multi_dot([j52,j53,j61]),j7,multi_dot([j46,j10,j51]),j7,multi_dot([j59,j53,j61]),j54,(multi_dot([j52,j53,j56]) + multi_dot([j58,j49])),-1.0*j54,-1.0*multi_dot([j52,j53,j62]),j60,(multi_dot([j59,j53,j56]) + multi_dot([j58,j51])),-1.0*j60,-1.0*multi_dot([j59,j53,j62]),j7,multi_dot([j64,j15,j67]),j7,multi_dot([j70,j71,j79]),j7,multi_dot([j64,j15,j69]),j7,multi_dot([j77,j71,j79]),j72,(multi_dot([j70,j71,j74]) + multi_dot([j76,j67])),-1.0*j72,-1.0*multi_dot([j70,j71,j80]),j78,(multi_dot([j77,j71,j74]) + multi_dot([j76,j69])),-1.0*j78,-1.0*multi_dot([j77,j71,j80]),2.0*j1.T,2.0*j4.T,2.0*j6.T,2.0*j12.T,2.0*j3.T,2.0*j5.T,2.0*j47.T,2.0*j65.T,2.0*j9.T,2.0*j14.T,2.0*j17.T,2.0*j31.T,2.0*j20.T,2.0*j34.T]
  
