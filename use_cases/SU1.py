
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin


class inputs(object):

    def __init__(self):
        self.ax1_jcl_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_mcl_zact = lambda t : 0
        self.J_mcl_zact = np.array([[0, 0, 1]],dtype=np.float64)
        self.F_mcr_zact = lambda t : 0
        self.J_mcr_zact = np.array([[0, 0, 1]],dtype=np.float64)
        self.ax1_jcl_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_lca = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcl_hub_bearing = lambda t : 0
        self.ax1_jcr_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcl_strut_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcl_uca_chassis = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcl_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcr_hub_bearing = lambda t : 0
        self.ax1_jcr_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [1]],dtype=np.float64)
        self.pt1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
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

        c0 = A('SU1.P_vbl_steer').T
        c1 = self.pt1_jcl_tie_steering
        c2 = A(self.SU1.P_rbl_tie_rod).T
        c3 = -1.0*multi_dot([c2,self.SU1.R_rbl_tie_rod])
        c4 = self.ax1_jcl_tie_steering
        c5 = A(self.SU1.P_rbr_lower_strut).T
        c6 = self.pt1_jcr_strut
        c7 = -1.0*multi_dot([c5,self.SU1.R_rbr_lower_strut])
        c8 = A(self.SU1.P_rbr_upper_strut).T
        c9 = -1.0*multi_dot([c8,self.SU1.R_rbr_upper_strut])
        c10 = Triad(self.ax1_jcr_strut,)
        c11 = self.pt1_jcr_strut_lca
        c12 = A(self.SU1.P_rbr_lca).T
        c13 = -1.0*multi_dot([c12,self.SU1.R_rbr_lca])
        c14 = self.ax1_jcr_strut_lca
        c15 = A(self.SU1.P_rbl_lca).T
        c16 = self.pt1_jcl_lca_chassis
        c17 = -1.0*multi_dot([c15,self.SU1.R_rbl_lca])
        c18 = A('SU1.P_vbs_chassis').T
        c19 = -1.0*multi_dot([c18,'SU1.R_vbs_chassis'])
        c20 = Triad(self.ax1_jcl_lca_chassis,)
        c21 = self.pt1_jcl_strut_lca
        c22 = A(self.SU1.P_rbl_lower_strut).T
        c23 = -1.0*multi_dot([c22,self.SU1.R_rbl_lower_strut])
        c24 = self.ax1_jcl_strut_lca
        c25 = self.pt1_jcl_lca_upright
        c26 = A(self.SU1.P_rbl_upright).T
        c27 = -1.0*multi_dot([c26,self.SU1.R_rbl_upright])
        c28 = Triad(self.ax1_jcl_lca_upright,)
        c29 = self.pt1_jcr_strut_chassis
        c30 = self.ax1_jcr_strut_chassis
        c31 = A(self.SU1.P_rbl_hub).T
        c32 = self.pt1_jcl_hub_bearing
        c33 = Triad(self.ax1_jcl_hub_bearing,)
        c34 = self.pt1_jcr_uca_chassis
        c35 = A(self.SU1.P_rbr_uca).T
        c36 = -1.0*multi_dot([c35,self.SU1.R_rbr_uca])
        c37 = Triad(self.ax1_jcr_uca_chassis,)
        c38 = self.pt1_jcr_lca_chassis
        c39 = Triad(self.ax1_jcr_lca_chassis,)
        c40 = self.pt1_jcl_strut_chassis
        c41 = A(self.SU1.P_rbl_upper_strut).T
        c42 = -1.0*multi_dot([c41,self.SU1.R_rbl_upper_strut])
        c43 = self.ax1_jcl_strut_chassis
        c44 = self.pt1_jcl_uca_chassis
        c45 = A(self.SU1.P_rbl_uca).T
        c46 = -1.0*multi_dot([c45,self.SU1.R_rbl_uca])
        c47 = Triad(self.ax1_jcl_uca_chassis,)
        c48 = A(self.SU1.P_rbr_upright).T
        c49 = self.pt1_jcr_tie_upright
        c50 = -1.0*multi_dot([c48,self.SU1.R_rbr_upright])
        c51 = A(self.SU1.P_rbr_tie_rod).T
        c52 = -1.0*multi_dot([c51,self.SU1.R_rbr_tie_rod])
        c53 = Triad(self.ax1_jcr_tie_upright,)
        c54 = self.pt1_jcr_uca_upright
        c55 = Triad(self.ax1_jcr_uca_upright,)
        c56 = self.pt1_jcr_lca_upright
        c57 = Triad(self.ax1_jcr_lca_upright,)
        c58 = self.pt1_jcr_hub_bearing
        c59 = A(self.SU1.P_rbr_hub).T
        c60 = Triad(self.ax1_jcr_hub_bearing,)
        c61 = self.pt1_jcr_tie_steering
        c62 = A('SU1.P_vbr_steer').T
        c63 = self.ax1_jcr_tie_steering
        c64 = self.pt1_jcl_strut
        c65 = Triad(self.ax1_jcl_strut,)
        c66 = self.pt1_jcl_tie_upright
        c67 = Triad(self.ax1_jcl_tie_upright,)
        c68 = self.pt1_jcl_uca_upright
        c69 = Triad(self.ax1_jcl_uca_upright,)

        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,'SU1.R_vbl_steer']))
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([c2,c1]) + c3)
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([c0,Triad(c4,)])
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([c2,Triad(c4,self.ax2_jcl_tie_steering)])
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c5,c6]) + c7)
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c8,c6]) + c9)
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c5,c10])
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c8,c10])
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c5,c11]) + c7)
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c12,c11]) + c13)
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c5,Triad(c14,)])
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c12,Triad(c14,self.ax2_jcr_strut_lca)])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([c15,c16]) + c17)
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([c18,c16]) + c19)
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([c15,c20])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([c18,c20])
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c15,c21]) + c17)
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c22,c21]) + c23)
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c15,Triad(c24,)])
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c22,Triad(c24,self.ax2_jcl_strut_lca)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c15,c25]) + c17)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c26,c25]) + c27)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c15,c28])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c26,c28])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c8,c29]) + c9)
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([c18,c29]) + c19)
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([c8,Triad(c30,)])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([c18,Triad(c30,self.ax2_jcr_strut_chassis)])
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c31,c32]) + -1.0*multi_dot([c31,self.SU1.R_rbl_hub]))
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c26,c32]) + c27)
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c31,c33])
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c26,c33])
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([c18,c34]) + c19)
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([c35,c34]) + c36)
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([c18,c37])
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([c35,c37])
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([c18,c38]) + c19)
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([c12,c38]) + c13)
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([c18,c39])
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([c12,c39])
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([c18,c40]) + c19)
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c41,c40]) + c42)
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([c18,Triad(c43,)])
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([c41,Triad(c43,self.ax2_jcl_strut_chassis)])
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([c18,c44]) + c19)
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([c45,c44]) + c46)
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([c18,c47])
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([c45,c47])
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c48,c49]) + c50)
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c51,c49]) + c52)
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c48,c53])
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c51,c53])
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c48,c54]) + c50)
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c35,c54]) + c36)
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c48,c55])
        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c35,c55])
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c48,c56]) + c50)
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c12,c56]) + c13)
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c48,c57])
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c12,c57])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c48,c58]) + c50)
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c59,c58]) + -1.0*multi_dot([c59,self.SU1.R_rbr_hub]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c48,c60])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c59,c60])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([c51,c61]) + c52)
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([c62,c61]) + -1.0*multi_dot([c62,'SU1.R_vbr_steer']))
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([c51,Triad(c63,)])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([c62,Triad(c63,self.ax2_jcr_tie_steering)])
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c22,c64]) + c23)
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c41,c64]) + c42)
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c22,c65])
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c41,c65])
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c26,c66]) + c27)
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c2,c66]) + c3)
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c26,c67])
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c2,c67])
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c26,c68]) + c27)
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c45,c68]) + c46)
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c26,c69])
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c45,c69])

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

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])
        self.pos_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])
        self.vel_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])
        self.acc_cols = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])
        self.jac_cols = np.array([0,1,28,29,0,1,28,29,2,3,8,9,2,3,8,9,2,3,8,9,2,3,8,9,2,3,22,23,2,3,22,23,4,5,10,11,4,5,26,27,6,7,12,13,6,7,12,13,6,7,12,13,6,7,20,21,6,7,20,21,6,7,24,25,8,9,12,13,8,9,12,13,10,11,24,25,10,11,24,25,10,11,24,25,10,11,24,25,12,13,18,19,12,13,18,19,12,13,18,19,12,13,22,23,12,13,22,23,12,13,22,23,12,13,30,31,12,13,30,31,12,13,32,33,12,13,32,33,12,13,32,33,14,15,16,17,14,15,18,19,14,15,22,23,14,15,26,27,14,15,26,27,14,15,26,27,14,15,26,27,16,17,34,35,16,17,34,35,20,21,30,31,20,21,30,31,20,21,30,31,20,21,30,31,24,25,28,29,24,25,32,33,3,7,9,11,15,17,19,21,23,25,27,29,31,33])

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

        x0 = -1.0*self.SU1.R_rbl_tie_rod
        x1 = A('SU1.P_vbl_steer')
        x2 = self.SU1.P_rbl_tie_rod
        x3 = A(x2)
        x4 = config.Mbar_rbr_lower_strut_jcr_strut[:,0:1].T
        x5 = self.SU1.P_rbr_lower_strut
        x6 = A(x5)
        x7 = x6.T
        x8 = self.SU1.P_rbr_upper_strut
        x9 = A(x8)
        x10 = config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]
        x11 = config.Mbar_rbr_lower_strut_jcr_strut[:,1:2].T
        x12 = self.SU1.R_rbr_lower_strut
        x13 = self.SU1.R_rbr_upper_strut
        x14 = (x12 + -1.0*x13 + multi_dot([x6,config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]]) + multi_dot([x6,config.ubar_rbr_lower_strut_jcr_strut]) + -1.0*multi_dot([x9,config.ubar_rbr_upper_strut_jcr_strut]))
        x15 = -1.0*self.SU1.R_rbr_lca
        x16 = self.SU1.P_rbr_lca
        x17 = A(x16)
        x18 = 'SU1.R_vbs_ground'[2]
        x19 = np.eye(1,dtype=np.float64)
        x20 = self.SU1.R_rbl_lca
        x21 = 'SU1.R_vbs_chassis'
        x22 = -1.0*x21
        x23 = self.SU1.P_rbl_lca
        x24 = A(x23)
        x25 = A('SU1.P_vbs_chassis')
        x26 = x24.T
        x27 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x28 = self.SU1.R_rbl_lower_strut
        x29 = self.SU1.P_rbl_lower_strut
        x30 = A(x29)
        x31 = self.SU1.R_rbl_upright
        x32 = -1.0*x31
        x33 = self.SU1.P_rbl_upright
        x34 = A(x33)
        x35 = self.SU1.P_rbl_hub
        x36 = A(x35)
        x37 = x36.T
        x38 = config.Mbar_rbl_upright_jcl_hub_bearing[:,2:3]
        x39 = config.F_jcl_hub_bearing(t,)
        x40 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        x41 = -1.0*self.SU1.R_rbr_uca
        x42 = self.SU1.P_rbr_uca
        x43 = A(x42)
        x44 = x25.T
        x45 = config.Mbar_rbr_uca_jcr_uca_chassis[:,2:3]
        x46 = config.Mbar_rbr_lca_jcr_lca_chassis[:,2:3]
        x47 = -1.0*self.SU1.R_rbl_upper_strut
        x48 = self.SU1.P_rbl_upper_strut
        x49 = A(x48)
        x50 = -1.0*self.SU1.R_rbl_uca
        x51 = self.SU1.P_rbl_uca
        x52 = A(x51)
        x53 = config.Mbar_rbl_uca_jcl_uca_chassis[:,2:3]
        x54 = self.SU1.R_rbr_upright
        x55 = self.SU1.R_rbr_tie_rod
        x56 = self.SU1.P_rbr_upright
        x57 = A(x56)
        x58 = self.SU1.P_rbr_tie_rod
        x59 = A(x58)
        x60 = self.SU1.P_rbr_hub
        x61 = A(x60)
        x62 = x57.T
        x63 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x64 = config.F_jcr_hub_bearing(t,)
        x65 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        x66 = A('SU1.P_vbr_steer')
        x67 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1].T
        x68 = x30.T
        x69 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        x70 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2].T
        x71 = (x28 + x47 + multi_dot([x30,config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]]) + multi_dot([x30,config.ubar_rbl_lower_strut_jcl_strut]) + -1.0*multi_dot([x49,config.ubar_rbl_upper_strut_jcl_strut]))
        x72 = -1.0*x19

        self.pos_eq_blocks = [('SU1.R_vbl_steer' + x0 + multi_dot([x1,config.ubar_vbl_steer_jcl_tie_steering]) + -1.0*multi_dot([x3,config.ubar_rbl_tie_rod_jcl_tie_steering])),multi_dot([config.Mbar_vbl_steer_jcl_tie_steering[:,0:1].T,x1.T,x3,config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]]),multi_dot([x4,x7,x9,x10]),multi_dot([x11,x7,x9,x10]),multi_dot([x4,x7,x14]),multi_dot([x11,x7,x14]),(x12 + x15 + multi_dot([x6,config.ubar_rbr_lower_strut_jcr_strut_lca]) + -1.0*multi_dot([x17,config.ubar_rbr_lca_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1].T,x7,x17,config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]]),x18 - 1*config.F_mcl_zact(t,)*x19,x18 - 1*config.F_mcr_zact(t,)*x19,(x20 + x22 + multi_dot([x24,config.ubar_rbl_lca_jcl_lca_chassis]) + -1.0*multi_dot([x25,config.ubar_vbs_chassis_jcl_lca_chassis])),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x26,x25,x27]),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x26,x25,x27]),(x20 + -1.0*x28 + multi_dot([x24,config.ubar_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x30,config.ubar_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lca_jcl_strut_lca[:,0:1].T,x26,x30,config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x20 + x32 + multi_dot([x24,config.ubar_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x34,config.ubar_rbl_upright_jcl_lca_upright])),(x13 + x22 + multi_dot([x9,config.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1.0*multi_dot([x25,config.ubar_vbs_chassis_jcr_strut_chassis])),multi_dot([config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x9.T,x25,config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),(self.SU1.R_rbl_hub + x32 + multi_dot([x36,config.ubar_rbl_hub_jcl_hub_bearing]) + -1.0*multi_dot([x34,config.ubar_rbl_upright_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1].T,x37,x34,x38]),multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2].T,x37,x34,x38]),(cos(x39)*multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2].T,x37,x34,x40]) + sin(x39)*-1.0*multi_dot([config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1].T,x37,x34,x40])),(x21 + x41 + multi_dot([x25,config.ubar_vbs_chassis_jcr_uca_chassis]) + -1.0*multi_dot([x43,config.ubar_rbr_uca_jcr_uca_chassis])),multi_dot([config.Mbar_vbs_chassis_jcr_uca_chassis[:,0:1].T,x44,x43,x45]),multi_dot([config.Mbar_vbs_chassis_jcr_uca_chassis[:,1:2].T,x44,x43,x45]),(x21 + x15 + multi_dot([x25,config.ubar_vbs_chassis_jcr_lca_chassis]) + -1.0*multi_dot([x17,config.ubar_rbr_lca_jcr_lca_chassis])),multi_dot([config.Mbar_vbs_chassis_jcr_lca_chassis[:,0:1].T,x44,x17,x46]),multi_dot([config.Mbar_vbs_chassis_jcr_lca_chassis[:,1:2].T,x44,x17,x46]),(x21 + x47 + multi_dot([x25,config.ubar_vbs_chassis_jcl_strut_chassis]) + -1.0*multi_dot([x49,config.ubar_rbl_upper_strut_jcl_strut_chassis])),multi_dot([config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1].T,x44,x49,config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]]),(x21 + x50 + multi_dot([x25,config.ubar_vbs_chassis_jcl_uca_chassis]) + -1.0*multi_dot([x52,config.ubar_rbl_uca_jcl_uca_chassis])),multi_dot([config.Mbar_vbs_chassis_jcl_uca_chassis[:,0:1].T,x44,x52,x53]),multi_dot([config.Mbar_vbs_chassis_jcl_uca_chassis[:,1:2].T,x44,x52,x53]),(x54 + -1.0*x55 + multi_dot([x57,config.ubar_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x59,config.ubar_rbr_tie_rod_jcr_tie_upright])),(x54 + x41 + multi_dot([x57,config.ubar_rbr_upright_jcr_uca_upright]) + -1.0*multi_dot([x43,config.ubar_rbr_uca_jcr_uca_upright])),(x54 + x15 + multi_dot([x57,config.ubar_rbr_upright_jcr_lca_upright]) + -1.0*multi_dot([x17,config.ubar_rbr_lca_jcr_lca_upright])),(x54 + -1.0*self.SU1.R_rbr_hub + multi_dot([x57,config.ubar_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x61,config.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x62,x61,x63]),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x62,x61,x63]),(cos(x64)*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x62,x61,x65]) + sin(x64)*-1.0*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x62,x61,x65])),(x55 + -1.0*'SU1.R_vbr_steer' + multi_dot([x59,config.ubar_rbr_tie_rod_jcr_tie_steering]) + -1.0*multi_dot([x66,config.ubar_vbr_steer_jcr_tie_steering])),multi_dot([config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x59.T,x66,config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),multi_dot([x67,x68,x49,x69]),multi_dot([x70,x68,x49,x69]),multi_dot([x67,x68,x71]),multi_dot([x70,x68,x71]),(x31 + x0 + multi_dot([x34,config.ubar_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x3,config.ubar_rbl_tie_rod_jcl_tie_upright])),(x31 + x50 + multi_dot([x34,config.ubar_rbl_upright_jcl_uca_upright]) + -1.0*multi_dot([x52,config.ubar_rbl_uca_jcl_uca_upright])),(x72 + (multi_dot([x5.T,x5]))**(1.0/2.0)),(x72 + (multi_dot([x23.T,x23]))**(1.0/2.0)),(x72 + (multi_dot([x8.T,x8]))**(1.0/2.0)),(x72 + (multi_dot([x35.T,x35]))**(1.0/2.0)),(x72 + (multi_dot([x56.T,x56]))**(1.0/2.0)),(x72 + (multi_dot([x58.T,x58]))**(1.0/2.0)),(x72 + (multi_dot([x42.T,x42]))**(1.0/2.0)),(x72 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x72 + (multi_dot([x16.T,x16]))**(1.0/2.0)),(x72 + (multi_dot([x33.T,x33]))**(1.0/2.0)),(x72 + (multi_dot([x60.T,x60]))**(1.0/2.0)),(x72 + (multi_dot([x2.T,x2]))**(1.0/2.0)),(x72 + (multi_dot([x48.T,x48]))**(1.0/2.0)),(x72 + (multi_dot([x51.T,x51]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)
        v2 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v1,v1,v1,v0,v1,(v1 + derivative(config.F_mcl_zact,t,0.1,1)*-1.0*v2),(v1 + derivative(config.F_mcr_zact,t,0.1,1)*-1.0*v2),v0,v1,v1,v0,v1,v0,v0,v1,v0,v1,v1,(v1 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v2),v0,v1,v1,v0,v1,v1,v0,v1,v0,v1,v1,v0,v0,v0,v0,v1,v1,(v1 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v2),v0,v1,v1,v1,v1,v1,v0,v0,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = 'SU1.Pd_vbl_steer'
        a1 = self.SU1.Pd_rbl_tie_rod
        a2 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a3 = self.SU1.P_rbl_tie_rod
        a4 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a5 = 'SU1.P_vbl_steer'
        a6 = config.Mbar_rbr_lower_strut_jcr_strut[:,0:1]
        a7 = a6.T
        a8 = self.SU1.P_rbr_lower_strut
        a9 = A(a8).T
        a10 = self.SU1.Pd_rbr_upper_strut
        a11 = config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]
        a12 = B(a10,a11)
        a13 = a11.T
        a14 = self.SU1.P_rbr_upper_strut
        a15 = A(a14).T
        a16 = self.SU1.Pd_rbr_lower_strut
        a17 = B(a16,a6)
        a18 = a16.T
        a19 = B(a8,a6).T
        a20 = B(a14,a11)
        a21 = config.Mbar_rbr_lower_strut_jcr_strut[:,1:2]
        a22 = a21.T
        a23 = B(a16,a21)
        a24 = B(a8,a21).T
        a25 = config.ubar_rbr_lower_strut_jcr_strut
        a26 = config.ubar_rbr_upper_strut_jcr_strut
        a27 = (multi_dot([B(a16,a25),a16]) + -1.0*multi_dot([B(a10,a26),a10]))
        a28 = (self.SU1.Rd_rbr_lower_strut + -1.0*self.SU1.Rd_rbr_upper_strut + multi_dot([B(a8,a25),a16]) + multi_dot([B(a14,a26),a10]))
        a29 = (self.SU1.R_rbr_lower_strut.T + -1.0*self.SU1.R_rbr_upper_strut.T + multi_dot([config.Mbar_rbr_lower_strut_jcr_strut[:,2:3].T,a9]) + multi_dot([a25.T,a9]) + -1.0*multi_dot([a26.T,a15]))
        a30 = self.SU1.Pd_rbr_lca
        a31 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a32 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a33 = self.SU1.P_rbr_lca
        a34 = A(a33).T
        a35 = np.zeros((1,1),dtype=np.float64)
        a36 = np.eye(1,dtype=np.float64)
        a37 = self.SU1.Pd_rbl_lca
        a38 = 'SU1.Pd_vbs_chassis'
        a39 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a40 = a39.T
        a41 = 'SU1.P_vbs_chassis'
        a42 = A(a41).T
        a43 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a44 = self.SU1.P_rbl_lca
        a45 = A(a44).T
        a46 = B(a38,a39)
        a47 = a37.T
        a48 = B(a41,a39)
        a49 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a50 = self.SU1.Pd_rbl_lower_strut
        a51 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a52 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a53 = self.SU1.P_rbl_lower_strut
        a54 = A(a53).T
        a55 = self.SU1.Pd_rbl_upright
        a56 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a57 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a58 = a10.T
        a59 = self.SU1.Pd_rbl_hub
        a60 = config.Mbar_rbl_upright_jcl_hub_bearing[:,2:3]
        a61 = a60.T
        a62 = self.SU1.P_rbl_upright
        a63 = A(a62).T
        a64 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a65 = self.SU1.P_rbl_hub
        a66 = A(a65).T
        a67 = B(a55,a60)
        a68 = a59.T
        a69 = B(a62,a60)
        a70 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        a71 = config.F_jcl_hub_bearing(t,)
        a72 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a73 = cos(a71)
        a74 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        a75 = sin(a71)
        a76 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a77 = self.SU1.Pd_rbr_uca
        a78 = config.Mbar_rbr_uca_jcr_uca_chassis[:,2:3]
        a79 = a78.T
        a80 = self.SU1.P_rbr_uca
        a81 = A(a80).T
        a82 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,0:1]
        a83 = B(a77,a78)
        a84 = a38.T
        a85 = B(a80,a78)
        a86 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,1:2]
        a87 = config.Mbar_rbr_lca_jcr_lca_chassis[:,2:3]
        a88 = a87.T
        a89 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,0:1]
        a90 = B(a30,a87)
        a91 = B(a33,a87)
        a92 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,1:2]
        a93 = self.SU1.Pd_rbl_upper_strut
        a94 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a95 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a96 = self.SU1.P_rbl_upper_strut
        a97 = A(a96).T
        a98 = self.SU1.Pd_rbl_uca
        a99 = config.Mbar_rbl_uca_jcl_uca_chassis[:,2:3]
        a100 = a99.T
        a101 = self.SU1.P_rbl_uca
        a102 = A(a101).T
        a103 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,0:1]
        a104 = B(a98,a99)
        a105 = B(a101,a99)
        a106 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,1:2]
        a107 = self.SU1.Pd_rbr_upright
        a108 = self.SU1.Pd_rbr_tie_rod
        a109 = self.SU1.Pd_rbr_hub
        a110 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a111 = a110.T
        a112 = self.SU1.P_rbr_hub
        a113 = A(a112).T
        a114 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a115 = self.SU1.P_rbr_upright
        a116 = A(a115).T
        a117 = B(a109,a110)
        a118 = a107.T
        a119 = B(a112,a110)
        a120 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a121 = config.F_jcr_hub_bearing(t,)
        a122 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a123 = cos(a121)
        a124 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a125 = sin(a121)
        a126 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a127 = 'SU1.Pd_vbr_steer'
        a128 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a129 = 'SU1.P_vbr_steer'
        a130 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a131 = self.SU1.P_rbr_tie_rod
        a132 = a108.T
        a133 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1]
        a134 = a133.T
        a135 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        a136 = B(a93,a135)
        a137 = a135.T
        a138 = B(a50,a133)
        a139 = a50.T
        a140 = B(a53,a133).T
        a141 = B(a96,a135)
        a142 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2]
        a143 = a142.T
        a144 = B(a50,a142)
        a145 = B(a53,a142).T
        a146 = config.ubar_rbl_lower_strut_jcl_strut
        a147 = config.ubar_rbl_upper_strut_jcl_strut
        a148 = (multi_dot([B(a50,a146),a50]) + -1.0*multi_dot([B(a93,a147),a93]))
        a149 = (self.SU1.Rd_rbl_lower_strut + -1.0*self.SU1.Rd_rbl_upper_strut + multi_dot([B(a53,a146),a50]) + multi_dot([B(a96,a147),a93]))
        a150 = (self.SU1.R_rbl_lower_strut.T + -1.0*self.SU1.R_rbl_upper_strut.T + multi_dot([config.Mbar_rbl_lower_strut_jcl_strut[:,2:3].T,a54]) + multi_dot([a146.T,a54]) + -1.0*multi_dot([a147.T,a97]))

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_vbl_steer_jcl_tie_steering),a0]) + -1.0*multi_dot([B(a1,config.ubar_rbl_tie_rod_jcl_tie_steering),a1])),(multi_dot([a2.T,A(a3).T,B(a0,a4),a0]) + multi_dot([a4.T,A(a5).T,B(a1,a2),a1]) + 2.0*multi_dot([a0.T,B(a5,a4).T,B(a3,a2),a1])),(multi_dot([a7,a9,a12,a10]) + multi_dot([a13,a15,a17,a16]) + 2.0*multi_dot([a18,a19,a20,a10])),(multi_dot([a22,a9,a12,a10]) + multi_dot([a13,a15,a23,a16]) + 2.0*multi_dot([a18,a24,a20,a10])),(multi_dot([a7,a9,a27]) + 2.0*multi_dot([a18,a19,a28]) + multi_dot([a29,a17,a16])),(multi_dot([a22,a9,a27]) + 2.0*multi_dot([a18,a24,a28]) + multi_dot([a29,a23,a16])),(multi_dot([B(a16,config.ubar_rbr_lower_strut_jcr_strut_lca),a16]) + -1.0*multi_dot([B(a30,config.ubar_rbr_lca_jcr_strut_lca),a30])),(multi_dot([a31.T,a9,B(a30,a32),a30]) + multi_dot([a32.T,a34,B(a16,a31),a16]) + 2.0*multi_dot([a18,B(a8,a31).T,B(a33,a32),a30])),(a35 + derivative(config.F_mcl_zact,t,0.1,2)*-1.0*a36),(a35 + derivative(config.F_mcr_zact,t,0.1,2)*-1.0*a36),(multi_dot([B(a37,config.ubar_rbl_lca_jcl_lca_chassis),a37]) + -1.0*multi_dot([B(a38,config.ubar_vbs_chassis_jcl_lca_chassis),a38])),(multi_dot([a40,a42,B(a37,a43),a37]) + multi_dot([a43.T,a45,a46,a38]) + 2.0*multi_dot([a47,B(a44,a43).T,a48,a38])),(multi_dot([a40,a42,B(a37,a49),a37]) + multi_dot([a49.T,a45,a46,a38]) + 2.0*multi_dot([a47,B(a44,a49).T,a48,a38])),(multi_dot([B(a37,config.ubar_rbl_lca_jcl_strut_lca),a37]) + -1.0*multi_dot([B(a50,config.ubar_rbl_lower_strut_jcl_strut_lca),a50])),(multi_dot([a51.T,a45,B(a50,a52),a50]) + multi_dot([a52.T,a54,B(a37,a51),a37]) + 2.0*multi_dot([a47,B(a44,a51).T,B(a53,a52),a50])),(multi_dot([B(a37,config.ubar_rbl_lca_jcl_lca_upright),a37]) + -1.0*multi_dot([B(a55,config.ubar_rbl_upright_jcl_lca_upright),a55])),(multi_dot([B(a10,config.ubar_rbr_upper_strut_jcr_strut_chassis),a10]) + -1.0*multi_dot([B(a38,config.ubar_vbs_chassis_jcr_strut_chassis),a38])),(multi_dot([a56.T,a15,B(a38,a57),a38]) + multi_dot([a57.T,a42,B(a10,a56),a10]) + 2.0*multi_dot([a58,B(a14,a56).T,B(a41,a57),a38])),(multi_dot([B(a59,config.ubar_rbl_hub_jcl_hub_bearing),a59]) + -1.0*multi_dot([B(a55,config.ubar_rbl_upright_jcl_hub_bearing),a55])),(multi_dot([a61,a63,B(a59,a64),a59]) + multi_dot([a64.T,a66,a67,a55]) + 2.0*multi_dot([a68,B(a65,a64).T,a69,a55])),(multi_dot([a61,a63,B(a59,a70),a59]) + multi_dot([a70.T,a66,a67,a55]) + 2.0*multi_dot([a68,B(a65,a70).T,a69,a55])),(derivative(a71,t,0.1,2)*-1.0*a36 + multi_dot([a72.T,a63,(a73*B(a59,a74) + a75*-1.0*B(a59,a76)),a59]) + multi_dot([(a73*multi_dot([a74.T,a66]) + a75*-1.0*multi_dot([a76.T,a66])),B(a55,a72),a55]) + 2.0*multi_dot([((a73*multi_dot([B(a65,a74),a59])).T + a75*-1.0*multi_dot([a68,B(a65,a76).T])),B(a62,a72),a55])),(multi_dot([B(a38,config.ubar_vbs_chassis_jcr_uca_chassis),a38]) + -1.0*multi_dot([B(a77,config.ubar_rbr_uca_jcr_uca_chassis),a77])),(multi_dot([a79,a81,B(a38,a82),a38]) + multi_dot([a82.T,a42,a83,a77]) + 2.0*multi_dot([a84,B(a41,a82).T,a85,a77])),(multi_dot([a79,a81,B(a38,a86),a38]) + multi_dot([a86.T,a42,a83,a77]) + 2.0*multi_dot([a84,B(a41,a86).T,a85,a77])),(multi_dot([B(a38,config.ubar_vbs_chassis_jcr_lca_chassis),a38]) + -1.0*multi_dot([B(a30,config.ubar_rbr_lca_jcr_lca_chassis),a30])),(multi_dot([a88,a34,B(a38,a89),a38]) + multi_dot([a89.T,a42,a90,a30]) + 2.0*multi_dot([a84,B(a41,a89).T,a91,a30])),(multi_dot([a88,a34,B(a38,a92),a38]) + multi_dot([a92.T,a42,a90,a30]) + 2.0*multi_dot([a84,B(a41,a92).T,a91,a30])),(multi_dot([B(a38,config.ubar_vbs_chassis_jcl_strut_chassis),a38]) + -1.0*multi_dot([B(a93,config.ubar_rbl_upper_strut_jcl_strut_chassis),a93])),(multi_dot([a94.T,a42,B(a93,a95),a93]) + multi_dot([a95.T,a97,B(a38,a94),a38]) + 2.0*multi_dot([a84,B(a41,a94).T,B(a96,a95),a93])),(multi_dot([B(a38,config.ubar_vbs_chassis_jcl_uca_chassis),a38]) + -1.0*multi_dot([B(a98,config.ubar_rbl_uca_jcl_uca_chassis),a98])),(multi_dot([a100,a102,B(a38,a103),a38]) + multi_dot([a103.T,a42,a104,a98]) + 2.0*multi_dot([a84,B(a41,a103).T,a105,a98])),(multi_dot([a100,a102,B(a38,a106),a38]) + multi_dot([a106.T,a42,a104,a98]) + 2.0*multi_dot([a84,B(a41,a106).T,a105,a98])),(multi_dot([B(a107,config.ubar_rbr_upright_jcr_tie_upright),a107]) + -1.0*multi_dot([B(a108,config.ubar_rbr_tie_rod_jcr_tie_upright),a108])),(multi_dot([B(a107,config.ubar_rbr_upright_jcr_uca_upright),a107]) + -1.0*multi_dot([B(a77,config.ubar_rbr_uca_jcr_uca_upright),a77])),(multi_dot([B(a107,config.ubar_rbr_upright_jcr_lca_upright),a107]) + -1.0*multi_dot([B(a30,config.ubar_rbr_lca_jcr_lca_upright),a30])),(multi_dot([B(a107,config.ubar_rbr_upright_jcr_hub_bearing),a107]) + -1.0*multi_dot([B(a109,config.ubar_rbr_hub_jcr_hub_bearing),a109])),(multi_dot([a111,a113,B(a107,a114),a107]) + multi_dot([a114.T,a116,a117,a109]) + 2.0*multi_dot([a118,B(a115,a114).T,a119,a109])),(multi_dot([a111,a113,B(a107,a120),a107]) + multi_dot([a120.T,a116,a117,a109]) + 2.0*multi_dot([a118,B(a115,a120).T,a119,a109])),(derivative(a121,t,0.1,2)*-1.0*a36 + multi_dot([a122.T,a113,(a123*B(a107,a124) + a125*-1.0*B(a107,a126)),a107]) + multi_dot([(a123*multi_dot([a124.T,a116]) + a125*-1.0*multi_dot([a126.T,a116])),B(a109,a122),a109]) + 2.0*multi_dot([((a123*multi_dot([B(a115,a124),a107])).T + a125*-1.0*multi_dot([a118,B(a115,a126).T])),B(a112,a122),a109])),(multi_dot([B(a108,config.ubar_rbr_tie_rod_jcr_tie_steering),a108]) + -1.0*multi_dot([B(a127,config.ubar_vbr_steer_jcr_tie_steering),a127])),(multi_dot([a128.T,A(a129).T,B(a108,a130),a108]) + multi_dot([a130.T,A(a131).T,B(a127,a128),a127]) + 2.0*multi_dot([a132,B(a131,a130).T,B(a129,a128),a127])),(multi_dot([a134,a54,a136,a93]) + multi_dot([a137,a97,a138,a50]) + 2.0*multi_dot([a139,a140,a141,a93])),(multi_dot([a143,a54,a136,a93]) + multi_dot([a137,a97,a144,a50]) + 2.0*multi_dot([a139,a145,a141,a93])),(multi_dot([a134,a54,a148]) + 2.0*multi_dot([a139,a140,a149]) + multi_dot([a150,a138,a50])),(multi_dot([a143,a54,a148]) + 2.0*multi_dot([a139,a145,a149]) + multi_dot([a150,a144,a50])),(multi_dot([B(a55,config.ubar_rbl_upright_jcl_tie_upright),a55]) + -1.0*multi_dot([B(a1,config.ubar_rbl_tie_rod_jcl_tie_upright),a1])),(multi_dot([B(a55,config.ubar_rbl_upright_jcl_uca_upright),a55]) + -1.0*multi_dot([B(a98,config.ubar_rbl_uca_jcl_uca_upright),a98])),2.0*(multi_dot([a18,a16]))**(1.0/2.0),2.0*(multi_dot([a47,a37]))**(1.0/2.0),2.0*(multi_dot([a58,a10]))**(1.0/2.0),2.0*(multi_dot([a68,a59]))**(1.0/2.0),2.0*(multi_dot([a118,a107]))**(1.0/2.0),2.0*(multi_dot([a132,a108]))**(1.0/2.0),2.0*(multi_dot([a77.T,a77]))**(1.0/2.0),2.0*(multi_dot([a139,a50]))**(1.0/2.0),2.0*(multi_dot([a30.T,a30]))**(1.0/2.0),2.0*(multi_dot([a55.T,a55]))**(1.0/2.0),2.0*(multi_dot([a109.T,a109]))**(1.0/2.0),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a93.T,a93]))**(1.0/2.0),2.0*(multi_dot([a98.T,a98]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = 'SU1.P_vbl_steer'
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        j4 = self.SU1.P_rbl_tie_rod
        j5 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j6 = -1.0*j0
        j7 = config.Mbar_rbr_upper_strut_jcr_strut[:,2:3]
        j8 = j7.T
        j9 = self.SU1.P_rbr_upper_strut
        j10 = A(j9).T
        j11 = self.SU1.P_rbr_lower_strut
        j12 = config.Mbar_rbr_lower_strut_jcr_strut[:,0:1]
        j13 = B(j11,j12)
        j14 = config.Mbar_rbr_lower_strut_jcr_strut[:,1:2]
        j15 = B(j11,j14)
        j16 = j12.T
        j17 = A(j11).T
        j18 = multi_dot([j16,j17])
        j19 = config.ubar_rbr_lower_strut_jcr_strut
        j20 = B(j11,j19)
        j21 = config.ubar_rbr_upper_strut_jcr_strut
        j22 = (self.SU1.R_rbr_lower_strut.T + -1.0*self.SU1.R_rbr_upper_strut.T + multi_dot([config.Mbar_rbr_lower_strut_jcr_strut[:,2:3].T,j17]) + multi_dot([j19.T,j17]) + -1.0*multi_dot([j21.T,j10]))
        j23 = j14.T
        j24 = multi_dot([j23,j17])
        j25 = B(j9,j7)
        j26 = B(j9,j21)
        j27 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j28 = self.SU1.P_rbr_lca
        j29 = A(j28).T
        j30 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j31 = np.zeros((1,4),dtype=np.float64)
        j32 = self.SU1.P_rbl_lca
        j33 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j34 = j33.T
        j35 = 'SU1.P_vbs_chassis'
        j36 = A(j35).T
        j37 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j38 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j39 = A(j32).T
        j40 = B(j35,j33)
        j41 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j42 = self.SU1.P_rbl_lower_strut
        j43 = A(j42).T
        j44 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j45 = self.SU1.P_rbl_upright
        j46 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j47 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j48 = self.SU1.P_rbl_hub
        j49 = config.Mbar_rbl_upright_jcl_hub_bearing[:,2:3]
        j50 = j49.T
        j51 = A(j45).T
        j52 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j53 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        j54 = A(j48).T
        j55 = B(j45,j49)
        j56 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j57 = config.F_jcl_hub_bearing(t,)
        j58 = cos(j57)
        j59 = config.Mbar_rbl_hub_jcl_hub_bearing[:,1:2]
        j60 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j61 = config.Mbar_rbr_uca_jcr_uca_chassis[:,2:3]
        j62 = j61.T
        j63 = self.SU1.P_rbr_uca
        j64 = A(j63).T
        j65 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,0:1]
        j66 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,1:2]
        j67 = B(j63,j61)
        j68 = config.Mbar_rbr_lca_jcr_lca_chassis[:,2:3]
        j69 = j68.T
        j70 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,0:1]
        j71 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,1:2]
        j72 = B(j28,j68)
        j73 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j74 = self.SU1.P_rbl_upper_strut
        j75 = A(j74).T
        j76 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j77 = config.Mbar_rbl_uca_jcl_uca_chassis[:,2:3]
        j78 = j77.T
        j79 = self.SU1.P_rbl_uca
        j80 = A(j79).T
        j81 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,0:1]
        j82 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,1:2]
        j83 = B(j79,j77)
        j84 = self.SU1.P_rbr_upright
        j85 = self.SU1.P_rbr_tie_rod
        j86 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j87 = j86.T
        j88 = self.SU1.P_rbr_hub
        j89 = A(j88).T
        j90 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j91 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j92 = A(j84).T
        j93 = B(j88,j86)
        j94 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j95 = config.F_jcr_hub_bearing(t,)
        j96 = cos(j95)
        j97 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j98 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j99 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j100 = 'SU1.P_vbr_steer'
        j101 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j102 = config.Mbar_rbl_upper_strut_jcl_strut[:,2:3]
        j103 = j102.T
        j104 = config.Mbar_rbl_lower_strut_jcl_strut[:,0:1]
        j105 = B(j42,j104)
        j106 = config.Mbar_rbl_lower_strut_jcl_strut[:,1:2]
        j107 = B(j42,j106)
        j108 = j104.T
        j109 = multi_dot([j108,j43])
        j110 = config.ubar_rbl_lower_strut_jcl_strut
        j111 = B(j42,j110)
        j112 = config.ubar_rbl_upper_strut_jcl_strut
        j113 = (self.SU1.R_rbl_lower_strut.T + -1.0*self.SU1.R_rbl_upper_strut.T + multi_dot([config.Mbar_rbl_lower_strut_jcl_strut[:,2:3].T,j43]) + multi_dot([j110.T,j43]) + -1.0*multi_dot([j112.T,j75]))
        j114 = j106.T
        j115 = multi_dot([j114,j43])
        j116 = B(j74,j102)
        j117 = B(j74,j112)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_vbl_steer_jcl_tie_steering),j6,-1.0*B(j4,config.ubar_rbl_tie_rod_jcl_tie_steering),j2,multi_dot([j3.T,A(j4).T,B(j1,j5)]),j2,multi_dot([j5.T,A(j1).T,B(j4,j3)]),j2,multi_dot([j8,j10,j13]),j2,multi_dot([j16,j17,j25]),j2,multi_dot([j8,j10,j15]),j2,multi_dot([j23,j17,j25]),j18,(multi_dot([j16,j17,j20]) + multi_dot([j22,j13])),-1.0*j18,-1.0*multi_dot([j16,j17,j26]),j24,(multi_dot([j23,j17,j20]) + multi_dot([j22,j15])),-1.0*j24,-1.0*multi_dot([j23,j17,j26]),j0,B(j11,config.ubar_rbr_lower_strut_jcr_strut_lca),j6,-1.0*B(j28,config.ubar_rbr_lca_jcr_strut_lca),j2,multi_dot([j27.T,j29,B(j11,j30)]),j2,multi_dot([j30.T,j17,B(j28,j27)]),config.J_mcl_zact,j31,j2,j31,config.J_mcr_zact,j31,j2,j31,j0,B(j32,config.ubar_rbl_lca_jcl_lca_chassis),j6,-1.0*B(j35,config.ubar_vbs_chassis_jcl_lca_chassis),j2,multi_dot([j34,j36,B(j32,j37)]),j2,multi_dot([j37.T,j39,j40]),j2,multi_dot([j34,j36,B(j32,j38)]),j2,multi_dot([j38.T,j39,j40]),j0,B(j32,config.ubar_rbl_lca_jcl_strut_lca),j6,-1.0*B(j42,config.ubar_rbl_lower_strut_jcl_strut_lca),j2,multi_dot([j41.T,j43,B(j32,j44)]),j2,multi_dot([j44.T,j39,B(j42,j41)]),j0,B(j32,config.ubar_rbl_lca_jcl_lca_upright),j6,-1.0*B(j45,config.ubar_rbl_upright_jcl_lca_upright),j0,B(j9,config.ubar_rbr_upper_strut_jcr_strut_chassis),j6,-1.0*B(j35,config.ubar_vbs_chassis_jcr_strut_chassis),j2,multi_dot([j46.T,j36,B(j9,j47)]),j2,multi_dot([j47.T,j10,B(j35,j46)]),j0,B(j48,config.ubar_rbl_hub_jcl_hub_bearing),j6,-1.0*B(j45,config.ubar_rbl_upright_jcl_hub_bearing),j2,multi_dot([j50,j51,B(j48,j52)]),j2,multi_dot([j52.T,j54,j55]),j2,multi_dot([j50,j51,B(j48,j53)]),j2,multi_dot([j53.T,j54,j55]),j2,multi_dot([j56.T,j51,(j58*B(j48,j59) + sin(j57)*-1.0*B(j48,j60))]),j2,multi_dot([(j58*multi_dot([j59.T,j54]) + sin(j57)*-1.0*multi_dot([j60.T,j54])),B(j45,j56)]),j0,B(j35,config.ubar_vbs_chassis_jcr_uca_chassis),j6,-1.0*B(j63,config.ubar_rbr_uca_jcr_uca_chassis),j2,multi_dot([j62,j64,B(j35,j65)]),j2,multi_dot([j65.T,j36,j67]),j2,multi_dot([j62,j64,B(j35,j66)]),j2,multi_dot([j66.T,j36,j67]),j0,B(j35,config.ubar_vbs_chassis_jcr_lca_chassis),j6,-1.0*B(j28,config.ubar_rbr_lca_jcr_lca_chassis),j2,multi_dot([j69,j29,B(j35,j70)]),j2,multi_dot([j70.T,j36,j72]),j2,multi_dot([j69,j29,B(j35,j71)]),j2,multi_dot([j71.T,j36,j72]),j0,B(j35,config.ubar_vbs_chassis_jcl_strut_chassis),j6,-1.0*B(j74,config.ubar_rbl_upper_strut_jcl_strut_chassis),j2,multi_dot([j73.T,j75,B(j35,j76)]),j2,multi_dot([j76.T,j36,B(j74,j73)]),j0,B(j35,config.ubar_vbs_chassis_jcl_uca_chassis),j6,-1.0*B(j79,config.ubar_rbl_uca_jcl_uca_chassis),j2,multi_dot([j78,j80,B(j35,j81)]),j2,multi_dot([j81.T,j36,j83]),j2,multi_dot([j78,j80,B(j35,j82)]),j2,multi_dot([j82.T,j36,j83]),j0,B(j84,config.ubar_rbr_upright_jcr_tie_upright),j6,-1.0*B(j85,config.ubar_rbr_tie_rod_jcr_tie_upright),j0,B(j84,config.ubar_rbr_upright_jcr_uca_upright),j6,-1.0*B(j63,config.ubar_rbr_uca_jcr_uca_upright),j0,B(j84,config.ubar_rbr_upright_jcr_lca_upright),j6,-1.0*B(j28,config.ubar_rbr_lca_jcr_lca_upright),j0,B(j84,config.ubar_rbr_upright_jcr_hub_bearing),j6,-1.0*B(j88,config.ubar_rbr_hub_jcr_hub_bearing),j2,multi_dot([j87,j89,B(j84,j90)]),j2,multi_dot([j90.T,j92,j93]),j2,multi_dot([j87,j89,B(j84,j91)]),j2,multi_dot([j91.T,j92,j93]),j2,multi_dot([j94.T,j89,(j96*B(j84,j97) + sin(j95)*-1.0*B(j84,j98))]),j2,multi_dot([(j96*multi_dot([j97.T,j92]) + sin(j95)*-1.0*multi_dot([j98.T,j92])),B(j88,j94)]),j0,B(j85,config.ubar_rbr_tie_rod_jcr_tie_steering),j6,-1.0*B(j100,config.ubar_vbr_steer_jcr_tie_steering),j2,multi_dot([j99.T,A(j100).T,B(j85,j101)]),j2,multi_dot([j101.T,A(j85).T,B(j100,j99)]),j2,multi_dot([j103,j75,j105]),j2,multi_dot([j108,j43,j116]),j2,multi_dot([j103,j75,j107]),j2,multi_dot([j114,j43,j116]),j109,(multi_dot([j108,j43,j111]) + multi_dot([j113,j105])),-1.0*j109,-1.0*multi_dot([j108,j43,j117]),j115,(multi_dot([j114,j43,j111]) + multi_dot([j113,j107])),-1.0*j115,-1.0*multi_dot([j114,j43,j117]),j0,B(j45,config.ubar_rbl_upright_jcl_tie_upright),j6,-1.0*B(j4,config.ubar_rbl_tie_rod_jcl_tie_upright),j0,B(j45,config.ubar_rbl_upright_jcl_uca_upright),j6,-1.0*B(j79,config.ubar_rbl_uca_jcl_uca_upright),2.0*j11.T,2.0*j32.T,2.0*j9.T,2.0*j48.T,2.0*j84.T,2.0*j85.T,2.0*j63.T,2.0*j42.T,2.0*j28.T,2.0*j45.T,2.0*j88.T,2.0*j4.T,2.0*j74.T,2.0*j79.T]
  
