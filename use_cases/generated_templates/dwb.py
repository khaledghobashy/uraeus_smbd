
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin
import pandas as pd

def Mirror(v):
    if v.shape in ((1,3),(4,1)):
        return v
    else:
        m = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=np.float64)
        return m.dot(v)


class configuration(object):

    def __init__(self):
        self.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_uca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lca = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upright = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_upper_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_lower_strut = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_tie_rod = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.P_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.Rd_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.Pd_rbr_hub = np.array([[0], [0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_uca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_uca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_lca_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_lca_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_hub_bearing = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)

        self._set_arguments()

    def _set_arguments(self):
        self.R_rbl_uca = Mirror(self.R_rbr_uca)
        self.P_rbl_uca = Mirror(self.P_rbr_uca)
        self.Rd_rbl_uca = Mirror(self.Rd_rbr_uca)
        self.Pd_rbl_uca = Mirror(self.Pd_rbr_uca)
        self.R_rbl_lca = Mirror(self.R_rbr_lca)
        self.P_rbl_lca = Mirror(self.P_rbr_lca)
        self.Rd_rbl_lca = Mirror(self.Rd_rbr_lca)
        self.Pd_rbl_lca = Mirror(self.Pd_rbr_lca)
        self.R_rbl_upright = Mirror(self.R_rbr_upright)
        self.P_rbl_upright = Mirror(self.P_rbr_upright)
        self.Rd_rbl_upright = Mirror(self.Rd_rbr_upright)
        self.Pd_rbl_upright = Mirror(self.Pd_rbr_upright)
        self.R_rbl_upper_strut = Mirror(self.R_rbr_upper_strut)
        self.P_rbl_upper_strut = Mirror(self.P_rbr_upper_strut)
        self.Rd_rbl_upper_strut = Mirror(self.Rd_rbr_upper_strut)
        self.Pd_rbl_upper_strut = Mirror(self.Pd_rbr_upper_strut)
        self.R_rbl_lower_strut = Mirror(self.R_rbr_lower_strut)
        self.P_rbl_lower_strut = Mirror(self.P_rbr_lower_strut)
        self.Rd_rbl_lower_strut = Mirror(self.Rd_rbr_lower_strut)
        self.Pd_rbl_lower_strut = Mirror(self.Pd_rbr_lower_strut)
        self.R_rbl_tie_rod = Mirror(self.R_rbr_tie_rod)
        self.P_rbl_tie_rod = Mirror(self.P_rbr_tie_rod)
        self.Rd_rbl_tie_rod = Mirror(self.Rd_rbr_tie_rod)
        self.Pd_rbl_tie_rod = Mirror(self.Pd_rbr_tie_rod)
        self.R_rbl_hub = Mirror(self.R_rbr_hub)
        self.P_rbl_hub = Mirror(self.P_rbr_hub)
        self.Rd_rbl_hub = Mirror(self.Rd_rbr_hub)
        self.Pd_rbl_hub = Mirror(self.Pd_rbr_hub)
        self.ax1_jcl_uca_upright = Mirror(self.ax1_jcr_uca_upright)
        self.pt1_jcl_uca_upright = Mirror(self.pt1_jcr_uca_upright)
        self.ax1_jcl_uca_chassis = Mirror(self.ax1_jcr_uca_chassis)
        self.pt1_jcl_uca_chassis = Mirror(self.pt1_jcr_uca_chassis)
        self.ax1_jcl_lca_upright = Mirror(self.ax1_jcr_lca_upright)
        self.pt1_jcl_lca_upright = Mirror(self.pt1_jcr_lca_upright)
        self.ax1_jcl_lca_chassis = Mirror(self.ax1_jcr_lca_chassis)
        self.pt1_jcl_lca_chassis = Mirror(self.pt1_jcr_lca_chassis)
        self.ax1_jcl_strut_lca = Mirror(self.ax1_jcr_strut_lca)
        self.ax2_jcl_strut_lca = Mirror(self.ax2_jcr_strut_lca)
        self.pt1_jcl_strut_lca = Mirror(self.pt1_jcr_strut_lca)
        self.ax1_jcl_tie_upright = Mirror(self.ax1_jcr_tie_upright)
        self.pt1_jcl_tie_upright = Mirror(self.pt1_jcr_tie_upright)
        self.ax1_jcl_hub_bearing = Mirror(self.ax1_jcr_hub_bearing)
        self.pt1_jcl_hub_bearing = Mirror(self.pt1_jcr_hub_bearing)
        self.ax1_jcl_strut_chassis = Mirror(self.ax1_jcr_strut_chassis)
        self.ax2_jcl_strut_chassis = Mirror(self.ax2_jcr_strut_chassis)
        self.pt1_jcl_strut_chassis = Mirror(self.pt1_jcr_strut_chassis)
        self.ax1_jcl_strut = Mirror(self.ax1_jcr_strut)
        self.pt1_jcl_strut = Mirror(self.pt1_jcr_strut)
        self.ax1_jcl_tie_steering = Mirror(self.ax1_jcr_tie_steering)
        self.ax2_jcl_tie_steering = Mirror(self.ax2_jcr_tie_steering)
        self.pt1_jcl_tie_steering = Mirror(self.pt1_jcr_tie_steering)

    def load_from_csv(self,csv_file):
        dataframe = pd.read_csv(csv_file,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def eval_constants(self):

        c0 = A(self.P_rbr_uca).T
        c1 = Triad(self.ax1_jcr_uca_upright,)
        c2 = A(self.P_rbr_upright).T
        c3 = self.pt1_jcr_uca_upright
        c4 = -1.0*multi_dot([c0,self.R_rbr_uca])
        c5 = -1.0*multi_dot([c2,self.R_rbr_upright])
        c6 = Triad(self.ax1_jcr_uca_chassis,)
        c7 = A(self.P_vbs_chassis).T
        c8 = self.pt1_jcr_uca_chassis
        c9 = -1.0*multi_dot([c7,self.R_vbs_chassis])
        c10 = A(self.P_rbl_uca).T
        c11 = Triad(self.ax1_jcl_uca_upright,)
        c12 = A(self.P_rbl_upright).T
        c13 = self.pt1_jcl_uca_upright
        c14 = -1.0*multi_dot([c10,self.R_rbl_uca])
        c15 = -1.0*multi_dot([c12,self.R_rbl_upright])
        c16 = Triad(self.ax1_jcl_uca_chassis,)
        c17 = self.pt1_jcl_uca_chassis
        c18 = A(self.P_rbr_lca).T
        c19 = Triad(self.ax1_jcr_lca_upright,)
        c20 = self.pt1_jcr_lca_upright
        c21 = -1.0*multi_dot([c18,self.R_rbr_lca])
        c22 = Triad(self.ax1_jcr_lca_chassis,)
        c23 = self.pt1_jcr_lca_chassis
        c24 = self.ax1_jcr_strut_lca
        c25 = A(self.P_rbr_lower_strut).T
        c26 = self.pt1_jcr_strut_lca
        c27 = -1.0*multi_dot([c25,self.R_rbr_lower_strut])
        c28 = A(self.P_rbl_lca).T
        c29 = Triad(self.ax1_jcl_lca_upright,)
        c30 = self.pt1_jcl_lca_upright
        c31 = -1.0*multi_dot([c28,self.R_rbl_lca])
        c32 = Triad(self.ax1_jcl_lca_chassis,)
        c33 = self.pt1_jcl_lca_chassis
        c34 = self.ax1_jcl_strut_lca
        c35 = A(self.P_rbl_lower_strut).T
        c36 = self.pt1_jcl_strut_lca
        c37 = -1.0*multi_dot([c35,self.R_rbl_lower_strut])
        c38 = Triad(self.ax1_jcr_tie_upright,)
        c39 = A(self.P_rbr_tie_rod).T
        c40 = self.pt1_jcr_tie_upright
        c41 = -1.0*multi_dot([c39,self.R_rbr_tie_rod])
        c42 = Triad(self.ax1_jcr_hub_bearing,)
        c43 = A(self.P_rbr_hub).T
        c44 = self.pt1_jcr_hub_bearing
        c45 = Triad(self.ax1_jcl_tie_upright,)
        c46 = A(self.P_rbl_tie_rod).T
        c47 = self.pt1_jcl_tie_upright
        c48 = -1.0*multi_dot([c46,self.R_rbl_tie_rod])
        c49 = Triad(self.ax1_jcl_hub_bearing,)
        c50 = A(self.P_rbl_hub).T
        c51 = self.pt1_jcl_hub_bearing
        c52 = A(self.P_rbr_upper_strut).T
        c53 = self.ax1_jcr_strut_chassis
        c54 = self.pt1_jcr_strut_chassis
        c55 = -1.0*multi_dot([c52,self.R_rbr_upper_strut])
        c56 = Triad(self.ax1_jcr_strut,)
        c57 = self.pt1_jcr_strut
        c58 = A(self.P_rbl_upper_strut).T
        c59 = self.ax1_jcl_strut_chassis
        c60 = self.pt1_jcl_strut_chassis
        c61 = -1.0*multi_dot([c58,self.R_rbl_upper_strut])
        c62 = Triad(self.ax1_jcl_strut,)
        c63 = self.pt1_jcl_strut
        c64 = self.ax1_jcr_tie_steering
        c65 = A(self.P_vbr_steer).T
        c66 = self.pt1_jcr_tie_steering
        c67 = self.ax1_jcl_tie_steering
        c68 = A(self.P_vbl_steer).T
        c69 = self.pt1_jcl_tie_steering

        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c0,c1])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c2,c1])
        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c0,c3]) + c4)
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c2,c3]) + c5)
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([c0,c6])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([c7,c6])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([c0,c8]) + c4)
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([c7,c8]) + c9)
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c10,c11])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c12,c11])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c10,c13]) + c14)
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c12,c13]) + c15)
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([c10,c16])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([c7,c16])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([c10,c17]) + c14)
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([c7,c17]) + c9)
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c18,c19])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c2,c19])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c18,c20]) + c21)
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c2,c20]) + c5)
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([c18,c22])
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([c7,c22])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([c18,c23]) + c21)
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([c7,c23]) + c9)
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c18,Triad(c24,)])
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c25,Triad(c24,self.ax2_jcr_strut_lca)])
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c18,c26]) + c21)
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c25,c26]) + c27)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c28,c29])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c12,c29])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c28,c30]) + c31)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c12,c30]) + c15)
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([c28,c32])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([c7,c32])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([c28,c33]) + c31)
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([c7,c33]) + c9)
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c28,Triad(c34,)])
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c35,Triad(c34,self.ax2_jcl_strut_lca)])
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c28,c36]) + c31)
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c35,c36]) + c37)
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c2,c38])
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c39,c38])
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c2,c40]) + c5)
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c39,c40]) + c41)
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c2,c42])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c43,c42])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c2,c44]) + c5)
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c43,c44]) + -1.0*multi_dot([c43,self.R_rbr_hub]))
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c12,c45])
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c46,c45])
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c12,c47]) + c15)
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c46,c47]) + c48)
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c12,c49])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c50,c49])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c12,c51]) + c15)
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c50,c51]) + -1.0*multi_dot([c50,self.R_rbl_hub]))
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([c52,Triad(c53,)])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([c7,Triad(c53,self.ax2_jcr_strut_chassis)])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c52,c54]) + c55)
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([c7,c54]) + c9)
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c52,c56])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c25,c56])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c52,c57]) + c55)
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c25,c57]) + c27)
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([c58,Triad(c59,)])
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([c7,Triad(c59,self.ax2_jcl_strut_chassis)])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c58,c60]) + c61)
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([c7,c60]) + c9)
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c58,c62])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c35,c62])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c58,c63]) + c61)
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c35,c63]) + c37)
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([c39,Triad(c64,)])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([c65,Triad(c64,self.ax2_jcr_tie_steering)])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([c39,c66]) + c41)
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([c65,c66]) + -1.0*multi_dot([c65,self.R_vbr_steer]))
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([c46,Triad(c67,)])
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([c68,Triad(c67,self.ax2_jcl_tie_steering)])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([c46,c69]) + c48)
        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([c68,c69]) + -1.0*multi_dot([c68,self.R_vbl_steer]))

    @property
    def q(self):
        q = np.concatenate([self.R_rbr_uca,self.P_rbr_uca,self.R_rbl_uca,self.P_rbl_uca,self.R_rbr_lca,self.P_rbr_lca,self.R_rbl_lca,self.P_rbl_lca,self.R_rbr_upright,self.P_rbr_upright,self.R_rbl_upright,self.P_rbl_upright,self.R_rbr_upper_strut,self.P_rbr_upper_strut,self.R_rbl_upper_strut,self.P_rbl_upper_strut,self.R_rbr_lower_strut,self.P_rbr_lower_strut,self.R_rbl_lower_strut,self.P_rbl_lower_strut,self.R_rbr_tie_rod,self.P_rbr_tie_rod,self.R_rbl_tie_rod,self.P_rbl_tie_rod,self.R_rbr_hub,self.P_rbr_hub,self.R_rbl_hub,self.P_rbl_hub])
        return q

    @property
    def qd(self):
        qd = np.concatenate([self.Rd_rbr_uca,self.Pd_rbr_uca,self.Rd_rbl_uca,self.Pd_rbl_uca,self.Rd_rbr_lca,self.Pd_rbr_lca,self.Rd_rbl_lca,self.Pd_rbl_lca,self.Rd_rbr_upright,self.Pd_rbr_upright,self.Rd_rbl_upright,self.Pd_rbl_upright,self.Rd_rbr_upper_strut,self.Pd_rbr_upper_strut,self.Rd_rbl_upper_strut,self.Pd_rbl_upper_strut,self.Rd_rbr_lower_strut,self.Pd_rbr_lower_strut,self.Rd_rbl_lower_strut,self.Pd_rbl_lower_strut,self.Rd_rbr_tie_rod,self.Pd_rbr_tie_rod,self.Rd_rbl_tie_rod,self.Pd_rbl_tie_rod,self.Rd_rbr_hub,self.Pd_rbr_hub,self.Rd_rbl_hub,self.Pd_rbl_hub])
        return qd



class topology(object):

    def __init__(self,config,prefix=''):
        self.t = 0.0
        self.config = config
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 98
        self.nrows = 58
        self.ncols = 2*18
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
        self.rbr_uca = indicies_map[p+'rbr_uca']
        self.rbl_uca = indicies_map[p+'rbl_uca']
        self.rbr_lca = indicies_map[p+'rbr_lca']
        self.rbl_lca = indicies_map[p+'rbl_lca']
        self.rbr_upright = indicies_map[p+'rbr_upright']
        self.rbl_upright = indicies_map[p+'rbl_upright']
        self.rbr_upper_strut = indicies_map[p+'rbr_upper_strut']
        self.rbl_upper_strut = indicies_map[p+'rbl_upper_strut']
        self.rbr_lower_strut = indicies_map[p+'rbr_lower_strut']
        self.rbl_lower_strut = indicies_map[p+'rbl_lower_strut']
        self.rbr_tie_rod = indicies_map[p+'rbr_tie_rod']
        self.rbl_tie_rod = indicies_map[p+'rbl_tie_rod']
        self.rbr_hub = indicies_map[p+'rbr_hub']
        self.rbl_hub = indicies_map[p+'rbl_hub']
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbl_steer = indicies_map[interface_map[p+'vbl_steer']]
        self.vbr_steer = indicies_map[interface_map[p+'vbr_steer']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.rbr_uca*2,self.rbr_uca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbr_uca*2+1,self.rbl_uca*2+1,self.rbr_lca*2+1,self.rbl_lca*2+1,self.rbr_upright*2+1,self.rbl_upright*2+1,self.rbr_upper_strut*2+1,self.rbl_upper_strut*2+1,self.rbr_lower_strut*2+1,self.rbl_lower_strut*2+1,self.rbr_tie_rod*2+1,self.rbl_tie_rod*2+1,self.rbr_hub*2+1,self.rbl_hub*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def set_gen_coordinates(self,q):
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

    
    def set_gen_velocities(self,qd):
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

        x0 = self.R_rbr_uca
        x1 = self.R_rbr_upright
        x2 = -1.0*x1
        x3 = self.P_rbr_uca
        x4 = A(x3)
        x5 = self.P_rbr_upright
        x6 = A(x5)
        x7 = -1.0*self.R_vbs_chassis
        x8 = A(self.P_vbs_chassis)
        x9 = x4.T
        x10 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        x11 = self.R_rbl_uca
        x12 = self.R_rbl_upright
        x13 = -1.0*x12
        x14 = self.P_rbl_uca
        x15 = A(x14)
        x16 = self.P_rbl_upright
        x17 = A(x16)
        x18 = x15.T
        x19 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x20 = self.R_rbr_lca
        x21 = self.P_rbr_lca
        x22 = A(x21)
        x23 = x22.T
        x24 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x25 = -1.0*self.R_rbr_lower_strut
        x26 = self.P_rbr_lower_strut
        x27 = A(x26)
        x28 = self.R_rbl_lca
        x29 = self.P_rbl_lca
        x30 = A(x29)
        x31 = x30.T
        x32 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x33 = -1.0*self.R_rbl_lower_strut
        x34 = self.P_rbl_lower_strut
        x35 = A(x34)
        x36 = self.R_rbr_tie_rod
        x37 = self.P_rbr_tie_rod
        x38 = A(x37)
        x39 = self.P_rbr_hub
        x40 = A(x39)
        x41 = x6.T
        x42 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x43 = self.R_rbl_tie_rod
        x44 = self.P_rbl_tie_rod
        x45 = A(x44)
        x46 = self.P_rbl_hub
        x47 = A(x46)
        x48 = x17.T
        x49 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x50 = self.R_rbr_upper_strut
        x51 = self.P_rbr_upper_strut
        x52 = A(x51)
        x53 = x52.T
        x54 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x55 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x56 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x57 = (x50 + x25 + multi_dot([x52,config.ubar_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x27,config.ubar_rbr_lower_strut_jcr_strut]))
        x58 = self.R_rbl_upper_strut
        x59 = self.P_rbl_upper_strut
        x60 = A(x59)
        x61 = x60.T
        x62 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x63 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x64 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x65 = (x58 + x33 + multi_dot([x60,config.ubar_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x35,config.ubar_rbl_lower_strut_jcl_strut]))
        x66 = A(self.P_vbr_steer)
        x67 = A(self.P_vbl_steer)
        x68 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x2 + multi_dot([x4,config.ubar_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x6,config.ubar_rbr_upright_jcr_uca_upright])),(x0 + x7 + multi_dot([x4,config.ubar_rbr_uca_jcr_uca_chassis]) + -1.0*multi_dot([x8,config.ubar_vbs_chassis_jcr_uca_chassis])),multi_dot([config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x9,x8,x10]),multi_dot([config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x9,x8,x10]),(x11 + x13 + multi_dot([x15,config.ubar_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x17,config.ubar_rbl_upright_jcl_uca_upright])),(x11 + x7 + multi_dot([x15,config.ubar_rbl_uca_jcl_uca_chassis]) + -1.0*multi_dot([x8,config.ubar_vbs_chassis_jcl_uca_chassis])),multi_dot([config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x18,x8,x19]),multi_dot([config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x18,x8,x19]),(x20 + x2 + multi_dot([x22,config.ubar_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x6,config.ubar_rbr_upright_jcr_lca_upright])),(x20 + x7 + multi_dot([x22,config.ubar_rbr_lca_jcr_lca_chassis]) + -1.0*multi_dot([x8,config.ubar_vbs_chassis_jcr_lca_chassis])),multi_dot([config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x23,x8,x24]),multi_dot([config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x23,x8,x24]),(x20 + x25 + multi_dot([x22,config.ubar_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x27,config.ubar_rbr_lower_strut_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lca_jcr_strut_lca[:,0:1].T,x23,x27,config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x28 + x13 + multi_dot([x30,config.ubar_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x17,config.ubar_rbl_upright_jcl_lca_upright])),(x28 + x7 + multi_dot([x30,config.ubar_rbl_lca_jcl_lca_chassis]) + -1.0*multi_dot([x8,config.ubar_vbs_chassis_jcl_lca_chassis])),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x31,x8,x32]),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x31,x8,x32]),(x28 + x33 + multi_dot([x30,config.ubar_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x35,config.ubar_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lca_jcl_strut_lca[:,0:1].T,x31,x35,config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x1 + -1.0*x36 + multi_dot([x6,config.ubar_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x38,config.ubar_rbr_tie_rod_jcr_tie_upright])),(x1 + -1.0*self.R_rbr_hub + multi_dot([x6,config.ubar_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x40,config.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x41,x40,x42]),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x41,x40,x42]),(x12 + -1.0*x43 + multi_dot([x17,config.ubar_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x45,config.ubar_rbl_tie_rod_jcl_tie_upright])),(x12 + -1.0*self.R_rbl_hub + multi_dot([x17,config.ubar_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x47,config.ubar_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x48,x47,x49]),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x48,x47,x49]),(x50 + x7 + multi_dot([x52,config.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1.0*multi_dot([x8,config.ubar_vbs_chassis_jcr_strut_chassis])),multi_dot([config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x53,x8,config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),multi_dot([x54,x53,x27,x55]),multi_dot([x56,x53,x27,x55]),multi_dot([x54,x53,x57]),multi_dot([x56,x53,x57]),(x58 + x7 + multi_dot([x60,config.ubar_rbl_upper_strut_jcl_strut_chassis]) + -1.0*multi_dot([x8,config.ubar_vbs_chassis_jcl_strut_chassis])),multi_dot([config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x61,x8,config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),multi_dot([x62,x61,x35,x63]),multi_dot([x64,x61,x35,x63]),multi_dot([x62,x61,x65]),multi_dot([x64,x61,x65]),(x36 + -1.0*self.R_vbr_steer + multi_dot([x38,config.ubar_rbr_tie_rod_jcr_tie_steering]) + -1.0*multi_dot([x66,config.ubar_vbr_steer_jcr_tie_steering])),multi_dot([config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x38.T,x66,config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),(x43 + -1.0*self.R_vbl_steer + multi_dot([x45,config.ubar_rbl_tie_rod_jcl_tie_steering]) + -1.0*multi_dot([x67,config.ubar_vbl_steer_jcl_tie_steering])),multi_dot([config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x45.T,x67,config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),(x68 + (multi_dot([x3.T,x3]))**(1.0/2.0)),(x68 + (multi_dot([x14.T,x14]))**(1.0/2.0)),(x68 + (multi_dot([x21.T,x21]))**(1.0/2.0)),(x68 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x68 + (multi_dot([x5.T,x5]))**(1.0/2.0)),(x68 + (multi_dot([x16.T,x16]))**(1.0/2.0)),(x68 + (multi_dot([x51.T,x51]))**(1.0/2.0)),(x68 + (multi_dot([x59.T,x59]))**(1.0/2.0)),(x68 + (multi_dot([x26.T,x26]))**(1.0/2.0)),(x68 + (multi_dot([x34.T,x34]))**(1.0/2.0)),(x68 + (multi_dot([x37.T,x37]))**(1.0/2.0)),(x68 + (multi_dot([x44.T,x44]))**(1.0/2.0)),(x68 + (multi_dot([x39.T,x39]))**(1.0/2.0)),(x68 + (multi_dot([x46.T,x46]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v0,v1,v1,v0,v0,v1,v1,v0,v0,v1,v1,v0,v1,v0,v0,v1,v1,v0,v1,v0,v0,v1,v1,v0,v0,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v1,v1,v1,v1,v0,v1,v0,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_uca
        a1 = self.Pd_rbr_upright
        a2 = self.Pd_vbs_chassis
        a3 = config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a4 = self.P_rbr_uca
        a5 = A(a4).T
        a6 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        a7 = B(a2,a6)
        a8 = a6.T
        a9 = self.P_vbs_chassis
        a10 = A(a9).T
        a11 = a0.T
        a12 = B(a9,a6)
        a13 = config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a14 = self.Pd_rbl_uca
        a15 = self.Pd_rbl_upright
        a16 = config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a17 = self.P_rbl_uca
        a18 = A(a17).T
        a19 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a20 = B(a2,a19)
        a21 = a19.T
        a22 = a14.T
        a23 = B(a9,a19)
        a24 = config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a25 = self.Pd_rbr_lca
        a26 = config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a27 = self.P_rbr_lca
        a28 = A(a27).T
        a29 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a30 = B(a2,a29)
        a31 = a29.T
        a32 = a25.T
        a33 = B(a9,a29)
        a34 = config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a35 = self.Pd_rbr_lower_strut
        a36 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a37 = self.P_rbr_lower_strut
        a38 = A(a37).T
        a39 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a40 = self.Pd_rbl_lca
        a41 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a42 = self.P_rbl_lca
        a43 = A(a42).T
        a44 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a45 = B(a2,a44)
        a46 = a44.T
        a47 = a40.T
        a48 = B(a9,a44)
        a49 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a50 = self.Pd_rbl_lower_strut
        a51 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a52 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a53 = self.P_rbl_lower_strut
        a54 = A(a53).T
        a55 = self.Pd_rbr_tie_rod
        a56 = self.Pd_rbr_hub
        a57 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a58 = self.P_rbr_upright
        a59 = A(a58).T
        a60 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a61 = B(a56,a60)
        a62 = a60.T
        a63 = self.P_rbr_hub
        a64 = A(a63).T
        a65 = a1.T
        a66 = B(a63,a60)
        a67 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a68 = self.Pd_rbl_tie_rod
        a69 = self.Pd_rbl_hub
        a70 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a71 = self.P_rbl_upright
        a72 = A(a71).T
        a73 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a74 = B(a69,a73)
        a75 = a73.T
        a76 = self.P_rbl_hub
        a77 = A(a76).T
        a78 = a15.T
        a79 = B(a76,a73)
        a80 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a81 = self.Pd_rbr_upper_strut
        a82 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a83 = self.P_rbr_upper_strut
        a84 = A(a83).T
        a85 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a86 = a81.T
        a87 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a88 = a87.T
        a89 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a90 = B(a35,a89)
        a91 = a89.T
        a92 = B(a81,a87)
        a93 = B(a83,a87).T
        a94 = B(a37,a89)
        a95 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a96 = a95.T
        a97 = B(a81,a95)
        a98 = B(a83,a95).T
        a99 = config.ubar_rbr_upper_strut_jcr_strut
        a100 = config.ubar_rbr_lower_strut_jcr_strut
        a101 = (multi_dot([B(a81,a99),a81]) + -1.0*multi_dot([B(a35,a100),a35]))
        a102 = (self.Rd_rbr_upper_strut + -1.0*self.Rd_rbr_lower_strut + multi_dot([B(a37,a100),a35]) + multi_dot([B(a83,a99),a81]))
        a103 = (self.R_rbr_upper_strut.T + -1.0*self.R_rbr_lower_strut.T + multi_dot([a99.T,a84]) + -1.0*multi_dot([a100.T,a38]))
        a104 = self.Pd_rbl_upper_strut
        a105 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a106 = self.P_rbl_upper_strut
        a107 = A(a106).T
        a108 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a109 = a104.T
        a110 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a111 = a110.T
        a112 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a113 = B(a50,a112)
        a114 = a112.T
        a115 = B(a104,a110)
        a116 = B(a106,a110).T
        a117 = B(a53,a112)
        a118 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a119 = a118.T
        a120 = B(a104,a118)
        a121 = B(a106,a118).T
        a122 = config.ubar_rbl_upper_strut_jcl_strut
        a123 = config.ubar_rbl_lower_strut_jcl_strut
        a124 = (multi_dot([B(a104,a122),a104]) + -1.0*multi_dot([B(a50,a123),a50]))
        a125 = (self.Rd_rbl_upper_strut + -1.0*self.Rd_rbl_lower_strut + multi_dot([B(a53,a123),a50]) + multi_dot([B(a106,a122),a104]))
        a126 = (self.R_rbl_upper_strut.T + -1.0*self.R_rbl_lower_strut.T + multi_dot([a122.T,a107]) + -1.0*multi_dot([a123.T,a54]))
        a127 = self.Pd_vbr_steer
        a128 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a129 = self.P_rbr_tie_rod
        a130 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a131 = self.P_vbr_steer
        a132 = a55.T
        a133 = self.Pd_vbl_steer
        a134 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a135 = self.P_vbl_steer
        a136 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a137 = self.P_rbl_tie_rod
        a138 = a68.T

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_rbr_uca_jcr_uca_upright),a0]) + -1.0*multi_dot([B(a1,config.ubar_rbr_upright_jcr_uca_upright),a1])),(multi_dot([B(a0,config.ubar_rbr_uca_jcr_uca_chassis),a0]) + -1.0*multi_dot([B(a2,config.ubar_vbs_chassis_jcr_uca_chassis),a2])),(multi_dot([a3.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a3),a0]) + 2.0*multi_dot([a11,B(a4,a3).T,a12,a2])),(multi_dot([a13.T,a5,a7,a2]) + multi_dot([a8,a10,B(a0,a13),a0]) + 2.0*multi_dot([a11,B(a4,a13).T,a12,a2])),(multi_dot([B(a14,config.ubar_rbl_uca_jcl_uca_upright),a14]) + -1.0*multi_dot([B(a15,config.ubar_rbl_upright_jcl_uca_upright),a15])),(multi_dot([B(a14,config.ubar_rbl_uca_jcl_uca_chassis),a14]) + -1.0*multi_dot([B(a2,config.ubar_vbs_chassis_jcl_uca_chassis),a2])),(multi_dot([a16.T,a18,a20,a2]) + multi_dot([a21,a10,B(a14,a16),a14]) + 2.0*multi_dot([a22,B(a17,a16).T,a23,a2])),(multi_dot([a24.T,a18,a20,a2]) + multi_dot([a21,a10,B(a14,a24),a14]) + 2.0*multi_dot([a22,B(a17,a24).T,a23,a2])),(multi_dot([B(a25,config.ubar_rbr_lca_jcr_lca_upright),a25]) + -1.0*multi_dot([B(a1,config.ubar_rbr_upright_jcr_lca_upright),a1])),(multi_dot([B(a25,config.ubar_rbr_lca_jcr_lca_chassis),a25]) + -1.0*multi_dot([B(a2,config.ubar_vbs_chassis_jcr_lca_chassis),a2])),(multi_dot([a26.T,a28,a30,a2]) + multi_dot([a31,a10,B(a25,a26),a25]) + 2.0*multi_dot([a32,B(a27,a26).T,a33,a2])),(multi_dot([a34.T,a28,a30,a2]) + multi_dot([a31,a10,B(a25,a34),a25]) + 2.0*multi_dot([a32,B(a27,a34).T,a33,a2])),(multi_dot([B(a25,config.ubar_rbr_lca_jcr_strut_lca),a25]) + -1.0*multi_dot([B(a35,config.ubar_rbr_lower_strut_jcr_strut_lca),a35])),(multi_dot([a36.T,a38,B(a25,a39),a25]) + multi_dot([a39.T,a28,B(a35,a36),a35]) + 2.0*multi_dot([a32,B(a27,a39).T,B(a37,a36),a35])),(multi_dot([B(a40,config.ubar_rbl_lca_jcl_lca_upright),a40]) + -1.0*multi_dot([B(a15,config.ubar_rbl_upright_jcl_lca_upright),a15])),(multi_dot([B(a40,config.ubar_rbl_lca_jcl_lca_chassis),a40]) + -1.0*multi_dot([B(a2,config.ubar_vbs_chassis_jcl_lca_chassis),a2])),(multi_dot([a41.T,a43,a45,a2]) + multi_dot([a46,a10,B(a40,a41),a40]) + 2.0*multi_dot([a47,B(a42,a41).T,a48,a2])),(multi_dot([a49.T,a43,a45,a2]) + multi_dot([a46,a10,B(a40,a49),a40]) + 2.0*multi_dot([a47,B(a42,a49).T,a48,a2])),(multi_dot([B(a40,config.ubar_rbl_lca_jcl_strut_lca),a40]) + -1.0*multi_dot([B(a50,config.ubar_rbl_lower_strut_jcl_strut_lca),a50])),(multi_dot([a51.T,a43,B(a50,a52),a50]) + multi_dot([a52.T,a54,B(a40,a51),a40]) + 2.0*multi_dot([a47,B(a42,a51).T,B(a53,a52),a50])),(multi_dot([B(a1,config.ubar_rbr_upright_jcr_tie_upright),a1]) + -1.0*multi_dot([B(a55,config.ubar_rbr_tie_rod_jcr_tie_upright),a55])),(multi_dot([B(a1,config.ubar_rbr_upright_jcr_hub_bearing),a1]) + -1.0*multi_dot([B(a56,config.ubar_rbr_hub_jcr_hub_bearing),a56])),(multi_dot([a57.T,a59,a61,a56]) + multi_dot([a62,a64,B(a1,a57),a1]) + 2.0*multi_dot([a65,B(a58,a57).T,a66,a56])),(multi_dot([a67.T,a59,a61,a56]) + multi_dot([a62,a64,B(a1,a67),a1]) + 2.0*multi_dot([a65,B(a58,a67).T,a66,a56])),(multi_dot([B(a15,config.ubar_rbl_upright_jcl_tie_upright),a15]) + -1.0*multi_dot([B(a68,config.ubar_rbl_tie_rod_jcl_tie_upright),a68])),(multi_dot([B(a15,config.ubar_rbl_upright_jcl_hub_bearing),a15]) + -1.0*multi_dot([B(a69,config.ubar_rbl_hub_jcl_hub_bearing),a69])),(multi_dot([a70.T,a72,a74,a69]) + multi_dot([a75,a77,B(a15,a70),a15]) + 2.0*multi_dot([a78,B(a71,a70).T,a79,a69])),(multi_dot([a80.T,a72,a74,a69]) + multi_dot([a75,a77,B(a15,a80),a15]) + 2.0*multi_dot([a78,B(a71,a80).T,a79,a69])),(multi_dot([B(a81,config.ubar_rbr_upper_strut_jcr_strut_chassis),a81]) + -1.0*multi_dot([B(a2,config.ubar_vbs_chassis_jcr_strut_chassis),a2])),(multi_dot([a82.T,a84,B(a2,a85),a2]) + multi_dot([a85.T,a10,B(a81,a82),a81]) + 2.0*multi_dot([a86,B(a83,a82).T,B(a9,a85),a2])),(multi_dot([a88,a84,a90,a35]) + multi_dot([a91,a38,a92,a81]) + 2.0*multi_dot([a86,a93,a94,a35])),(multi_dot([a96,a84,a90,a35]) + multi_dot([a91,a38,a97,a81]) + 2.0*multi_dot([a86,a98,a94,a35])),(multi_dot([a88,a84,a101]) + 2.0*multi_dot([a86,a93,a102]) + multi_dot([a103,a92,a81])),(multi_dot([a96,a84,a101]) + 2.0*multi_dot([a86,a98,a102]) + multi_dot([a103,a97,a81])),(multi_dot([B(a104,config.ubar_rbl_upper_strut_jcl_strut_chassis),a104]) + -1.0*multi_dot([B(a2,config.ubar_vbs_chassis_jcl_strut_chassis),a2])),(multi_dot([a105.T,a107,B(a2,a108),a2]) + multi_dot([a108.T,a10,B(a104,a105),a104]) + 2.0*multi_dot([a109,B(a106,a105).T,B(a9,a108),a2])),(multi_dot([a111,a107,a113,a50]) + multi_dot([a114,a54,a115,a104]) + 2.0*multi_dot([a109,a116,a117,a50])),(multi_dot([a119,a107,a113,a50]) + multi_dot([a114,a54,a120,a104]) + 2.0*multi_dot([a109,a121,a117,a50])),(multi_dot([a111,a107,a124]) + 2.0*multi_dot([a109,a116,a125]) + multi_dot([a126,a115,a104])),(multi_dot([a119,a107,a124]) + 2.0*multi_dot([a109,a121,a125]) + multi_dot([a126,a120,a104])),(multi_dot([B(a55,config.ubar_rbr_tie_rod_jcr_tie_steering),a55]) + -1.0*multi_dot([B(a127,config.ubar_vbr_steer_jcr_tie_steering),a127])),(multi_dot([a128.T,A(a129).T,B(a127,a130),a127]) + multi_dot([a130.T,A(a131).T,B(a55,a128),a55]) + 2.0*multi_dot([a132,B(a129,a128).T,B(a131,a130),a127])),(multi_dot([B(a68,config.ubar_rbl_tie_rod_jcl_tie_steering),a68]) + -1.0*multi_dot([B(a133,config.ubar_vbl_steer_jcl_tie_steering),a133])),(multi_dot([a134.T,A(a135).T,B(a68,a136),a68]) + multi_dot([a136.T,A(a137).T,B(a133,a134),a133]) + 2.0*multi_dot([a138,B(a137,a136).T,B(a135,a134),a133])),2.0*(multi_dot([a11,a0]))**(1.0/2.0),2.0*(multi_dot([a22,a14]))**(1.0/2.0),2.0*(multi_dot([a32,a25]))**(1.0/2.0),2.0*(multi_dot([a47,a40]))**(1.0/2.0),2.0*(multi_dot([a65,a1]))**(1.0/2.0),2.0*(multi_dot([a78,a15]))**(1.0/2.0),2.0*(multi_dot([a86,a81]))**(1.0/2.0),2.0*(multi_dot([a109,a104]))**(1.0/2.0),2.0*(multi_dot([a35.T,a35]))**(1.0/2.0),2.0*(multi_dot([a50.T,a50]))**(1.0/2.0),2.0*(multi_dot([a132,a55]))**(1.0/2.0),2.0*(multi_dot([a138,a68]))**(1.0/2.0),2.0*(multi_dot([a56.T,a56]))**(1.0/2.0),2.0*(multi_dot([a69.T,a69]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbr_uca
        j2 = -1.0*j0
        j3 = self.P_rbr_upright
        j4 = np.zeros((1,3),dtype=np.float64)
        j5 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        j6 = j5.T
        j7 = self.P_vbs_chassis
        j8 = A(j7).T
        j9 = config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j10 = config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j11 = A(j1).T
        j12 = B(j7,j5)
        j13 = self.P_rbl_uca
        j14 = self.P_rbl_upright
        j15 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j16 = j15.T
        j17 = config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j18 = config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j19 = A(j13).T
        j20 = B(j7,j15)
        j21 = self.P_rbr_lca
        j22 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j23 = j22.T
        j24 = config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j25 = config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j26 = A(j21).T
        j27 = B(j7,j22)
        j28 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j29 = self.P_rbr_lower_strut
        j30 = A(j29).T
        j31 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j32 = self.P_rbl_lca
        j33 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j34 = j33.T
        j35 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j36 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j37 = A(j32).T
        j38 = B(j7,j33)
        j39 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j40 = self.P_rbl_lower_strut
        j41 = A(j40).T
        j42 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j43 = self.P_rbr_tie_rod
        j44 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j45 = j44.T
        j46 = self.P_rbr_hub
        j47 = A(j46).T
        j48 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j49 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j50 = A(j3).T
        j51 = B(j46,j44)
        j52 = self.P_rbl_tie_rod
        j53 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j54 = j53.T
        j55 = self.P_rbl_hub
        j56 = A(j55).T
        j57 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j58 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j59 = A(j14).T
        j60 = B(j55,j53)
        j61 = self.P_rbr_upper_strut
        j62 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j63 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j64 = A(j61).T
        j65 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j66 = j65.T
        j67 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j68 = B(j61,j67)
        j69 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j70 = B(j61,j69)
        j71 = j67.T
        j72 = multi_dot([j71,j64])
        j73 = config.ubar_rbr_upper_strut_jcr_strut
        j74 = B(j61,j73)
        j75 = config.ubar_rbr_lower_strut_jcr_strut
        j76 = (self.R_rbr_upper_strut.T + -1.0*self.R_rbr_lower_strut.T + multi_dot([j73.T,j64]) + -1.0*multi_dot([j75.T,j30]))
        j77 = j69.T
        j78 = multi_dot([j77,j64])
        j79 = B(j29,j65)
        j80 = B(j29,j75)
        j81 = self.P_rbl_upper_strut
        j82 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j83 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j84 = A(j81).T
        j85 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j86 = j85.T
        j87 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j88 = B(j81,j87)
        j89 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j90 = B(j81,j89)
        j91 = j87.T
        j92 = multi_dot([j91,j84])
        j93 = config.ubar_rbl_upper_strut_jcl_strut
        j94 = B(j81,j93)
        j95 = config.ubar_rbl_lower_strut_jcl_strut
        j96 = (self.R_rbl_upper_strut.T + -1.0*self.R_rbl_lower_strut.T + multi_dot([j93.T,j84]) + -1.0*multi_dot([j95.T,j41]))
        j97 = j89.T
        j98 = multi_dot([j97,j84])
        j99 = B(j40,j85)
        j100 = B(j40,j95)
        j101 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j102 = self.P_vbr_steer
        j103 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j104 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j105 = self.P_vbl_steer
        j106 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = [j0,B(j1,config.ubar_rbr_uca_jcr_uca_upright),j2,-1.0*B(j3,config.ubar_rbr_upright_jcr_uca_upright),j0,B(j1,config.ubar_rbr_uca_jcr_uca_chassis),j2,-1.0*B(j7,config.ubar_vbs_chassis_jcr_uca_chassis),j4,multi_dot([j6,j8,B(j1,j9)]),j4,multi_dot([j9.T,j11,j12]),j4,multi_dot([j6,j8,B(j1,j10)]),j4,multi_dot([j10.T,j11,j12]),j0,B(j13,config.ubar_rbl_uca_jcl_uca_upright),j2,-1.0*B(j14,config.ubar_rbl_upright_jcl_uca_upright),j0,B(j13,config.ubar_rbl_uca_jcl_uca_chassis),j2,-1.0*B(j7,config.ubar_vbs_chassis_jcl_uca_chassis),j4,multi_dot([j16,j8,B(j13,j17)]),j4,multi_dot([j17.T,j19,j20]),j4,multi_dot([j16,j8,B(j13,j18)]),j4,multi_dot([j18.T,j19,j20]),j0,B(j21,config.ubar_rbr_lca_jcr_lca_upright),j2,-1.0*B(j3,config.ubar_rbr_upright_jcr_lca_upright),j0,B(j21,config.ubar_rbr_lca_jcr_lca_chassis),j2,-1.0*B(j7,config.ubar_vbs_chassis_jcr_lca_chassis),j4,multi_dot([j23,j8,B(j21,j24)]),j4,multi_dot([j24.T,j26,j27]),j4,multi_dot([j23,j8,B(j21,j25)]),j4,multi_dot([j25.T,j26,j27]),j0,B(j21,config.ubar_rbr_lca_jcr_strut_lca),j2,-1.0*B(j29,config.ubar_rbr_lower_strut_jcr_strut_lca),j4,multi_dot([j28.T,j30,B(j21,j31)]),j4,multi_dot([j31.T,j26,B(j29,j28)]),j0,B(j32,config.ubar_rbl_lca_jcl_lca_upright),j2,-1.0*B(j14,config.ubar_rbl_upright_jcl_lca_upright),j0,B(j32,config.ubar_rbl_lca_jcl_lca_chassis),j2,-1.0*B(j7,config.ubar_vbs_chassis_jcl_lca_chassis),j4,multi_dot([j34,j8,B(j32,j35)]),j4,multi_dot([j35.T,j37,j38]),j4,multi_dot([j34,j8,B(j32,j36)]),j4,multi_dot([j36.T,j37,j38]),j0,B(j32,config.ubar_rbl_lca_jcl_strut_lca),j2,-1.0*B(j40,config.ubar_rbl_lower_strut_jcl_strut_lca),j4,multi_dot([j39.T,j41,B(j32,j42)]),j4,multi_dot([j42.T,j37,B(j40,j39)]),j0,B(j3,config.ubar_rbr_upright_jcr_tie_upright),j2,-1.0*B(j43,config.ubar_rbr_tie_rod_jcr_tie_upright),j0,B(j3,config.ubar_rbr_upright_jcr_hub_bearing),j2,-1.0*B(j46,config.ubar_rbr_hub_jcr_hub_bearing),j4,multi_dot([j45,j47,B(j3,j48)]),j4,multi_dot([j48.T,j50,j51]),j4,multi_dot([j45,j47,B(j3,j49)]),j4,multi_dot([j49.T,j50,j51]),j0,B(j14,config.ubar_rbl_upright_jcl_tie_upright),j2,-1.0*B(j52,config.ubar_rbl_tie_rod_jcl_tie_upright),j0,B(j14,config.ubar_rbl_upright_jcl_hub_bearing),j2,-1.0*B(j55,config.ubar_rbl_hub_jcl_hub_bearing),j4,multi_dot([j54,j56,B(j14,j57)]),j4,multi_dot([j57.T,j59,j60]),j4,multi_dot([j54,j56,B(j14,j58)]),j4,multi_dot([j58.T,j59,j60]),j0,B(j61,config.ubar_rbr_upper_strut_jcr_strut_chassis),j2,-1.0*B(j7,config.ubar_vbs_chassis_jcr_strut_chassis),j4,multi_dot([j62.T,j8,B(j61,j63)]),j4,multi_dot([j63.T,j64,B(j7,j62)]),j4,multi_dot([j66,j30,j68]),j4,multi_dot([j71,j64,j79]),j4,multi_dot([j66,j30,j70]),j4,multi_dot([j77,j64,j79]),j72,(multi_dot([j71,j64,j74]) + multi_dot([j76,j68])),-1.0*j72,-1.0*multi_dot([j71,j64,j80]),j78,(multi_dot([j77,j64,j74]) + multi_dot([j76,j70])),-1.0*j78,-1.0*multi_dot([j77,j64,j80]),j0,B(j81,config.ubar_rbl_upper_strut_jcl_strut_chassis),j2,-1.0*B(j7,config.ubar_vbs_chassis_jcl_strut_chassis),j4,multi_dot([j82.T,j8,B(j81,j83)]),j4,multi_dot([j83.T,j84,B(j7,j82)]),j4,multi_dot([j86,j41,j88]),j4,multi_dot([j91,j84,j99]),j4,multi_dot([j86,j41,j90]),j4,multi_dot([j97,j84,j99]),j92,(multi_dot([j91,j84,j94]) + multi_dot([j96,j88])),-1.0*j92,-1.0*multi_dot([j91,j84,j100]),j98,(multi_dot([j97,j84,j94]) + multi_dot([j96,j90])),-1.0*j98,-1.0*multi_dot([j97,j84,j100]),j0,B(j43,config.ubar_rbr_tie_rod_jcr_tie_steering),j2,-1.0*B(j102,config.ubar_vbr_steer_jcr_tie_steering),j4,multi_dot([j101.T,A(j102).T,B(j43,j103)]),j4,multi_dot([j103.T,A(j43).T,B(j102,j101)]),j0,B(j52,config.ubar_rbl_tie_rod_jcl_tie_steering),j2,-1.0*B(j105,config.ubar_vbl_steer_jcl_tie_steering),j4,multi_dot([j104.T,A(j105).T,B(j52,j106)]),j4,multi_dot([j106.T,A(j52).T,B(j105,j104)]),2.0*j1.T,2.0*j13.T,2.0*j21.T,2.0*j32.T,2.0*j3.T,2.0*j14.T,2.0*j61.T,2.0*j81.T,2.0*j29.T,2.0*j40.T,2.0*j43.T,2.0*j52.T,2.0*j46.T,2.0*j55.T]
  
