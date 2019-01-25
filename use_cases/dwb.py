
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin

def Mirror(v):
    m = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=np.float64)
    return m.dot(v)


class configuration(object):

    def __init__(self):
        self.R_rbr_uca = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_lca = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_upright = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_upper_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_lower_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_tie_rod = np.array([[0], [0], [0]],dtype=np.float64)
        self.R_rbr_hub = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_mcr_zact = lambda t : 0
        self.J_mcr_zact = np.array([[0, 0, 0]],dtype=np.float64)
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
        self.F_jcr_hub_bearing = lambda t : 0
        self.ax1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_strut = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.ax2_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_tie_steering = np.array([[0], [0], [0]],dtype=np.float64)

    def _set_arguments(self):
        self.R_rbl_uca = Mirror(self.R_rbr_uca)
        self.R_rbl_lca = Mirror(self.R_rbr_lca)
        self.R_rbl_upright = Mirror(self.R_rbr_upright)
        self.R_rbl_upper_strut = Mirror(self.R_rbr_upper_strut)
        self.R_rbl_lower_strut = Mirror(self.R_rbr_lower_strut)
        self.R_rbl_tie_rod = Mirror(self.R_rbr_tie_rod)
        self.R_rbl_hub = Mirror(self.R_rbr_hub)
        self.F_mcl_zact = self.F_mcr_zact
        self.J_mcl_zact = Mirror(self.J_mcr_zact)
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
        self.F_jcl_hub_bearing = self.F_jcr_hub_bearing
        self.ax1_jcl_strut_chassis = Mirror(self.ax1_jcr_strut_chassis)
        self.ax2_jcl_strut_chassis = Mirror(self.ax2_jcr_strut_chassis)
        self.pt1_jcl_strut_chassis = Mirror(self.pt1_jcr_strut_chassis)
        self.ax1_jcl_strut = Mirror(self.ax1_jcr_strut)
        self.pt1_jcl_strut = Mirror(self.pt1_jcr_strut)
        self.ax1_jcl_tie_steering = Mirror(self.ax1_jcr_tie_steering)
        self.ax2_jcl_tie_steering = Mirror(self.ax2_jcr_tie_steering)
        self.pt1_jcl_tie_steering = Mirror(self.pt1_jcr_tie_steering)

    def eval_constants(self):
        self._set_arguments()

        c0 = A(self.P_rbr_uca).T
        c1 = self.pt1_jcr_uca_upright
        c2 = -1.0*multi_dot([c0,self.R_rbr_uca])
        c3 = A(self.P_rbr_upright).T
        c4 = -1.0*multi_dot([c3,self.R_rbr_upright])
        c5 = Triad(self.ax1_jcr_uca_upright,)
        c6 = self.pt1_jcr_uca_chassis
        c7 = A(self.P_vbs_chassis).T
        c8 = -1.0*multi_dot([c7,self.R_vbs_chassis])
        c9 = Triad(self.ax1_jcr_uca_chassis,)
        c10 = A(self.P_rbl_uca).T
        c11 = self.pt1_jcl_uca_upright
        c12 = -1.0*multi_dot([c10,self.R_rbl_uca])
        c13 = A(self.P_rbl_upright).T
        c14 = -1.0*multi_dot([c13,self.R_rbl_upright])
        c15 = Triad(self.ax1_jcl_uca_upright,)
        c16 = self.pt1_jcl_uca_chassis
        c17 = Triad(self.ax1_jcl_uca_chassis,)
        c18 = A(self.P_rbr_lca).T
        c19 = self.pt1_jcr_lca_upright
        c20 = -1.0*multi_dot([c18,self.R_rbr_lca])
        c21 = Triad(self.ax1_jcr_lca_upright,)
        c22 = self.pt1_jcr_lca_chassis
        c23 = Triad(self.ax1_jcr_lca_chassis,)
        c24 = self.pt1_jcr_strut_lca
        c25 = A(self.P_rbr_lower_strut).T
        c26 = -1.0*multi_dot([c25,self.R_rbr_lower_strut])
        c27 = self.ax1_jcr_strut_lca
        c28 = A(self.P_rbl_lca).T
        c29 = self.pt1_jcl_lca_upright
        c30 = -1.0*multi_dot([c28,self.R_rbl_lca])
        c31 = Triad(self.ax1_jcl_lca_upright,)
        c32 = self.pt1_jcl_lca_chassis
        c33 = Triad(self.ax1_jcl_lca_chassis,)
        c34 = self.pt1_jcl_strut_lca
        c35 = A(self.P_rbl_lower_strut).T
        c36 = -1.0*multi_dot([c35,self.R_rbl_lower_strut])
        c37 = self.ax1_jcl_strut_lca
        c38 = self.pt1_jcr_tie_upright
        c39 = A(self.P_rbr_tie_rod).T
        c40 = -1.0*multi_dot([c39,self.R_rbr_tie_rod])
        c41 = Triad(self.ax1_jcr_tie_upright,)
        c42 = self.pt1_jcr_hub_bearing
        c43 = A(self.P_rbr_hub).T
        c44 = Triad(self.ax1_jcr_hub_bearing,)
        c45 = self.pt1_jcl_tie_upright
        c46 = A(self.P_rbl_tie_rod).T
        c47 = -1.0*multi_dot([c46,self.R_rbl_tie_rod])
        c48 = Triad(self.ax1_jcl_tie_upright,)
        c49 = self.pt1_jcl_hub_bearing
        c50 = A(self.P_rbl_hub).T
        c51 = Triad(self.ax1_jcl_hub_bearing,)
        c52 = A(self.P_rbr_upper_strut).T
        c53 = self.pt1_jcr_strut_chassis
        c54 = -1.0*multi_dot([c52,self.R_rbr_upper_strut])
        c55 = self.ax1_jcr_strut_chassis
        c56 = self.pt1_jcr_strut
        c57 = Triad(self.ax1_jcr_strut,)
        c58 = A(self.P_rbl_upper_strut).T
        c59 = self.pt1_jcl_strut_chassis
        c60 = -1.0*multi_dot([c58,self.R_rbl_upper_strut])
        c61 = self.ax1_jcl_strut_chassis
        c62 = self.pt1_jcl_strut
        c63 = Triad(self.ax1_jcl_strut,)
        c64 = self.pt1_jcr_tie_steering
        c65 = A(self.P_vbr_steer).T
        c66 = self.ax1_jcr_tie_steering
        c67 = self.pt1_jcl_tie_steering
        c68 = A(self.P_vbl_steer).T
        c69 = self.ax1_jcl_tie_steering

        self.ubar_rbr_uca_jcr_uca_upright = (multi_dot([c0,c1]) + c2)
        self.ubar_rbr_upright_jcr_uca_upright = (multi_dot([c3,c1]) + c4)
        self.Mbar_rbr_uca_jcr_uca_upright = multi_dot([c0,c5])
        self.Mbar_rbr_upright_jcr_uca_upright = multi_dot([c3,c5])
        self.ubar_rbr_uca_jcr_uca_chassis = (multi_dot([c0,c6]) + c2)
        self.ubar_vbs_chassis_jcr_uca_chassis = (multi_dot([c7,c6]) + c8)
        self.Mbar_rbr_uca_jcr_uca_chassis = multi_dot([c0,c9])
        self.Mbar_vbs_chassis_jcr_uca_chassis = multi_dot([c7,c9])
        self.ubar_rbl_uca_jcl_uca_upright = (multi_dot([c10,c11]) + c12)
        self.ubar_rbl_upright_jcl_uca_upright = (multi_dot([c13,c11]) + c14)
        self.Mbar_rbl_uca_jcl_uca_upright = multi_dot([c10,c15])
        self.Mbar_rbl_upright_jcl_uca_upright = multi_dot([c13,c15])
        self.ubar_rbl_uca_jcl_uca_chassis = (multi_dot([c10,c16]) + c12)
        self.ubar_vbs_chassis_jcl_uca_chassis = (multi_dot([c7,c16]) + c8)
        self.Mbar_rbl_uca_jcl_uca_chassis = multi_dot([c10,c17])
        self.Mbar_vbs_chassis_jcl_uca_chassis = multi_dot([c7,c17])
        self.ubar_rbr_lca_jcr_lca_upright = (multi_dot([c18,c19]) + c20)
        self.ubar_rbr_upright_jcr_lca_upright = (multi_dot([c3,c19]) + c4)
        self.Mbar_rbr_lca_jcr_lca_upright = multi_dot([c18,c21])
        self.Mbar_rbr_upright_jcr_lca_upright = multi_dot([c3,c21])
        self.ubar_rbr_lca_jcr_lca_chassis = (multi_dot([c18,c22]) + c20)
        self.ubar_vbs_chassis_jcr_lca_chassis = (multi_dot([c7,c22]) + c8)
        self.Mbar_rbr_lca_jcr_lca_chassis = multi_dot([c18,c23])
        self.Mbar_vbs_chassis_jcr_lca_chassis = multi_dot([c7,c23])
        self.ubar_rbr_lca_jcr_strut_lca = (multi_dot([c18,c24]) + c20)
        self.ubar_rbr_lower_strut_jcr_strut_lca = (multi_dot([c25,c24]) + c26)
        self.Mbar_rbr_lca_jcr_strut_lca = multi_dot([c18,Triad(c27,)])
        self.Mbar_rbr_lower_strut_jcr_strut_lca = multi_dot([c25,Triad(c27,self.ax2_jcr_strut_lca)])
        self.ubar_rbl_lca_jcl_lca_upright = (multi_dot([c28,c29]) + c30)
        self.ubar_rbl_upright_jcl_lca_upright = (multi_dot([c13,c29]) + c14)
        self.Mbar_rbl_lca_jcl_lca_upright = multi_dot([c28,c31])
        self.Mbar_rbl_upright_jcl_lca_upright = multi_dot([c13,c31])
        self.ubar_rbl_lca_jcl_lca_chassis = (multi_dot([c28,c32]) + c30)
        self.ubar_vbs_chassis_jcl_lca_chassis = (multi_dot([c7,c32]) + c8)
        self.Mbar_rbl_lca_jcl_lca_chassis = multi_dot([c28,c33])
        self.Mbar_vbs_chassis_jcl_lca_chassis = multi_dot([c7,c33])
        self.ubar_rbl_lca_jcl_strut_lca = (multi_dot([c28,c34]) + c30)
        self.ubar_rbl_lower_strut_jcl_strut_lca = (multi_dot([c35,c34]) + c36)
        self.Mbar_rbl_lca_jcl_strut_lca = multi_dot([c28,Triad(c37,)])
        self.Mbar_rbl_lower_strut_jcl_strut_lca = multi_dot([c35,Triad(c37,self.ax2_jcl_strut_lca)])
        self.ubar_rbr_upright_jcr_tie_upright = (multi_dot([c3,c38]) + c4)
        self.ubar_rbr_tie_rod_jcr_tie_upright = (multi_dot([c39,c38]) + c40)
        self.Mbar_rbr_upright_jcr_tie_upright = multi_dot([c3,c41])
        self.Mbar_rbr_tie_rod_jcr_tie_upright = multi_dot([c39,c41])
        self.ubar_rbr_upright_jcr_hub_bearing = (multi_dot([c3,c42]) + c4)
        self.ubar_rbr_hub_jcr_hub_bearing = (multi_dot([c43,c42]) + -1.0*multi_dot([c43,self.R_rbr_hub]))
        self.Mbar_rbr_upright_jcr_hub_bearing = multi_dot([c3,c44])
        self.Mbar_rbr_hub_jcr_hub_bearing = multi_dot([c43,c44])
        self.ubar_rbl_upright_jcl_tie_upright = (multi_dot([c13,c45]) + c14)
        self.ubar_rbl_tie_rod_jcl_tie_upright = (multi_dot([c46,c45]) + c47)
        self.Mbar_rbl_upright_jcl_tie_upright = multi_dot([c13,c48])
        self.Mbar_rbl_tie_rod_jcl_tie_upright = multi_dot([c46,c48])
        self.ubar_rbl_upright_jcl_hub_bearing = (multi_dot([c13,c49]) + c14)
        self.ubar_rbl_hub_jcl_hub_bearing = (multi_dot([c50,c49]) + -1.0*multi_dot([c50,self.R_rbl_hub]))
        self.Mbar_rbl_upright_jcl_hub_bearing = multi_dot([c13,c51])
        self.Mbar_rbl_hub_jcl_hub_bearing = multi_dot([c50,c51])
        self.ubar_rbr_upper_strut_jcr_strut_chassis = (multi_dot([c52,c53]) + c54)
        self.ubar_vbs_chassis_jcr_strut_chassis = (multi_dot([c7,c53]) + c8)
        self.Mbar_rbr_upper_strut_jcr_strut_chassis = multi_dot([c52,Triad(c55,)])
        self.Mbar_vbs_chassis_jcr_strut_chassis = multi_dot([c7,Triad(c55,self.ax2_jcr_strut_chassis)])
        self.ubar_rbr_upper_strut_jcr_strut = (multi_dot([c52,c56]) + c54)
        self.ubar_rbr_lower_strut_jcr_strut = (multi_dot([c25,c56]) + c26)
        self.Mbar_rbr_upper_strut_jcr_strut = multi_dot([c52,c57])
        self.Mbar_rbr_lower_strut_jcr_strut = multi_dot([c25,c57])
        self.ubar_rbl_upper_strut_jcl_strut_chassis = (multi_dot([c58,c59]) + c60)
        self.ubar_vbs_chassis_jcl_strut_chassis = (multi_dot([c7,c59]) + c8)
        self.Mbar_rbl_upper_strut_jcl_strut_chassis = multi_dot([c58,Triad(c61,)])
        self.Mbar_vbs_chassis_jcl_strut_chassis = multi_dot([c7,Triad(c61,self.ax2_jcl_strut_chassis)])
        self.ubar_rbl_upper_strut_jcl_strut = (multi_dot([c58,c62]) + c60)
        self.ubar_rbl_lower_strut_jcl_strut = (multi_dot([c35,c62]) + c36)
        self.Mbar_rbl_upper_strut_jcl_strut = multi_dot([c58,c63])
        self.Mbar_rbl_lower_strut_jcl_strut = multi_dot([c35,c63])
        self.ubar_rbr_tie_rod_jcr_tie_steering = (multi_dot([c39,c64]) + c40)
        self.ubar_vbr_steer_jcr_tie_steering = (multi_dot([c65,c64]) + -1.0*multi_dot([c65,self.R_vbr_steer]))
        self.Mbar_rbr_tie_rod_jcr_tie_steering = multi_dot([c39,Triad(c66,)])
        self.Mbar_vbr_steer_jcr_tie_steering = multi_dot([c65,Triad(c66,self.ax2_jcr_tie_steering)])
        self.ubar_rbl_tie_rod_jcl_tie_steering = (multi_dot([c46,c67]) + c47)
        self.ubar_vbl_steer_jcl_tie_steering = (multi_dot([c68,c67]) + -1.0*multi_dot([c68,self.R_vbl_steer]))
        self.Mbar_rbl_tie_rod_jcl_tie_steering = multi_dot([c46,Triad(c69,)])
        self.Mbar_vbl_steer_jcl_tie_steering = multi_dot([c68,Triad(c69,self.ax2_jcl_tie_steering)])



class topology(object):

    def __init__(self,config,prefix=''):
        self.t = 0.0
        self.config = config
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.n = 98
        self.nrows = 62
        self.ncols = 2*18
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])                        

    
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
        self.vbr_steer = indicies_map[interface_map[p+'vbr_steer']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbl_steer = indicies_map[interface_map[p+'vbl_steer']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_uca*2,self.rbr_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_uca*2,self.rbl_uca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_lca*2,self.rbr_lca*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_lca*2,self.rbl_lca*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbr_upright*2,self.rbr_upright*2+1,self.rbr_hub*2,self.rbr_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbl_upright*2,self.rbl_upright*2+1,self.rbl_hub*2,self.rbl_hub*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbr_upper_strut*2,self.rbr_upper_strut*2+1,self.rbr_lower_strut*2,self.rbr_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbl_upper_strut*2,self.rbl_upper_strut*2+1,self.rbl_lower_strut*2,self.rbl_lower_strut*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbr_tie_rod*2,self.rbr_tie_rod*2+1,self.vbr_steer*2,self.vbr_steer*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbl_tie_rod*2,self.rbl_tie_rod*2+1,self.vbl_steer*2,self.vbl_steer*2+1,self.rbr_uca*2+1,self.rbl_uca*2+1,self.rbr_lca*2+1,self.rbl_lca*2+1,self.rbr_upright*2+1,self.rbl_upright*2+1,self.rbr_upper_strut*2+1,self.rbl_upper_strut*2+1,self.rbr_lower_strut*2+1,self.rbl_lower_strut*2+1,self.rbr_tie_rod*2+1,self.rbl_tie_rod*2+1,self.rbr_hub*2+1,self.rbl_hub*2+1])

    
    def set_gen_coordinates(self,q):
        self.R_rbr_lower_strut = q[0:3,0:1]
        self.P_rbr_lower_strut = q[3:7,0:1]
        self.R_rbl_lower_strut = q[7:10,0:1]
        self.P_rbl_lower_strut = q[10:14,0:1]
        self.R_rbr_uca = q[14:17,0:1]
        self.P_rbr_uca = q[17:21,0:1]
        self.R_rbl_lca = q[21:24,0:1]
        self.P_rbl_lca = q[24:28,0:1]
        self.R_rbr_lca = q[28:31,0:1]
        self.P_rbr_lca = q[31:35,0:1]
        self.R_rbl_hub = q[35:38,0:1]
        self.P_rbl_hub = q[38:42,0:1]
        self.R_rbr_tie_rod = q[42:45,0:1]
        self.P_rbr_tie_rod = q[45:49,0:1]
        self.R_rbl_upright = q[49:52,0:1]
        self.P_rbl_upright = q[52:56,0:1]
        self.R_rbl_upper_strut = q[56:59,0:1]
        self.P_rbl_upper_strut = q[59:63,0:1]
        self.R_rbr_hub = q[63:66,0:1]
        self.P_rbr_hub = q[66:70,0:1]
        self.R_rbl_uca = q[70:73,0:1]
        self.P_rbl_uca = q[73:77,0:1]
        self.R_rbr_upper_strut = q[77:80,0:1]
        self.P_rbr_upper_strut = q[80:84,0:1]
        self.R_rbl_tie_rod = q[84:87,0:1]
        self.P_rbl_tie_rod = q[87:91,0:1]
        self.R_rbr_upright = q[91:94,0:1]
        self.P_rbr_upright = q[94:98,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbr_lower_strut = qd[0:3,0:1]
        self.Pd_rbr_lower_strut = qd[3:7,0:1]
        self.Rd_rbl_lower_strut = qd[7:10,0:1]
        self.Pd_rbl_lower_strut = qd[10:14,0:1]
        self.Rd_rbr_uca = qd[14:17,0:1]
        self.Pd_rbr_uca = qd[17:21,0:1]
        self.Rd_rbl_lca = qd[21:24,0:1]
        self.Pd_rbl_lca = qd[24:28,0:1]
        self.Rd_rbr_lca = qd[28:31,0:1]
        self.Pd_rbr_lca = qd[31:35,0:1]
        self.Rd_rbl_hub = qd[35:38,0:1]
        self.Pd_rbl_hub = qd[38:42,0:1]
        self.Rd_rbr_tie_rod = qd[42:45,0:1]
        self.Pd_rbr_tie_rod = qd[45:49,0:1]
        self.Rd_rbl_upright = qd[49:52,0:1]
        self.Pd_rbl_upright = qd[52:56,0:1]
        self.Rd_rbl_upper_strut = qd[56:59,0:1]
        self.Pd_rbl_upper_strut = qd[59:63,0:1]
        self.Rd_rbr_hub = qd[63:66,0:1]
        self.Pd_rbr_hub = qd[66:70,0:1]
        self.Rd_rbl_uca = qd[70:73,0:1]
        self.Pd_rbl_uca = qd[73:77,0:1]
        self.Rd_rbr_upper_strut = qd[77:80,0:1]
        self.Pd_rbr_upper_strut = qd[80:84,0:1]
        self.Rd_rbl_tie_rod = qd[84:87,0:1]
        self.Pd_rbl_tie_rod = qd[87:91,0:1]
        self.Rd_rbr_upright = qd[91:94,0:1]
        self.Pd_rbr_upright = qd[94:98,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_hub
        x1 = np.eye(1,dtype=np.float64)
        x2 = self.R_rbl_hub
        x3 = self.R_rbr_uca
        x4 = self.R_rbr_upright
        x5 = -1.0*x4
        x6 = self.P_rbr_uca
        x7 = A(x6)
        x8 = self.P_rbr_upright
        x9 = A(x8)
        x10 = -1.0*self.R_vbs_chassis
        x11 = A(self.P_vbs_chassis)
        x12 = x7.T
        x13 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        x14 = self.R_rbl_uca
        x15 = self.R_rbl_upright
        x16 = -1.0*x15
        x17 = self.P_rbl_uca
        x18 = A(x17)
        x19 = self.P_rbl_upright
        x20 = A(x19)
        x21 = x18.T
        x22 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        x23 = self.R_rbr_lca
        x24 = self.P_rbr_lca
        x25 = A(x24)
        x26 = x25.T
        x27 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        x28 = -1.0*self.R_rbr_lower_strut
        x29 = self.P_rbr_lower_strut
        x30 = A(x29)
        x31 = self.R_rbl_lca
        x32 = self.P_rbl_lca
        x33 = A(x32)
        x34 = x33.T
        x35 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        x36 = -1.0*self.R_rbl_lower_strut
        x37 = self.P_rbl_lower_strut
        x38 = A(x37)
        x39 = self.R_rbr_tie_rod
        x40 = self.P_rbr_tie_rod
        x41 = A(x40)
        x42 = self.P_rbr_hub
        x43 = A(x42)
        x44 = x9.T
        x45 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        x46 = config.F_jcr_hub_bearing(t,)
        x47 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        x48 = self.R_rbl_tie_rod
        x49 = self.P_rbl_tie_rod
        x50 = A(x49)
        x51 = self.P_rbl_hub
        x52 = A(x51)
        x53 = x20.T
        x54 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        x55 = config.F_jcl_hub_bearing(t,)
        x56 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        x57 = self.R_rbr_upper_strut
        x58 = self.P_rbr_upper_strut
        x59 = A(x58)
        x60 = x59.T
        x61 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1].T
        x62 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        x63 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2].T
        x64 = (x57 + x28 + multi_dot([x59,config.ubar_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x30,config.ubar_rbr_lower_strut_jcr_strut]))
        x65 = self.R_rbl_upper_strut
        x66 = self.P_rbl_upper_strut
        x67 = A(x66)
        x68 = x67.T
        x69 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1].T
        x70 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        x71 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2].T
        x72 = (x65 + x36 + multi_dot([x67,config.ubar_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x38,config.ubar_rbl_lower_strut_jcl_strut]))
        x73 = A(self.P_vbr_steer)
        x74 = A(self.P_vbl_steer)
        x75 = -1.0*x1

        self.pos_eq_blocks = [-1*config.F_mcr_zact(t,) + x0[2]*x1,-1*config.F_mcl_zact(t,) + x2[2]*x1,(x3 + x5 + multi_dot([x7,config.ubar_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x9,config.ubar_rbr_upright_jcr_uca_upright])),(x3 + x10 + multi_dot([x7,config.ubar_rbr_uca_jcr_uca_chassis]) + -1.0*multi_dot([x11,config.ubar_vbs_chassis_jcr_uca_chassis])),multi_dot([config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1].T,x12,x11,x13]),multi_dot([config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2].T,x12,x11,x13]),(x14 + x16 + multi_dot([x18,config.ubar_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x20,config.ubar_rbl_upright_jcl_uca_upright])),(x14 + x10 + multi_dot([x18,config.ubar_rbl_uca_jcl_uca_chassis]) + -1.0*multi_dot([x11,config.ubar_vbs_chassis_jcl_uca_chassis])),multi_dot([config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1].T,x21,x11,x22]),multi_dot([config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2].T,x21,x11,x22]),(x23 + x5 + multi_dot([x25,config.ubar_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x9,config.ubar_rbr_upright_jcr_lca_upright])),(x23 + x10 + multi_dot([x25,config.ubar_rbr_lca_jcr_lca_chassis]) + -1.0*multi_dot([x11,config.ubar_vbs_chassis_jcr_lca_chassis])),multi_dot([config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1].T,x26,x11,x27]),multi_dot([config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2].T,x26,x11,x27]),(x23 + x28 + multi_dot([x25,config.ubar_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x30,config.ubar_rbr_lower_strut_jcr_strut_lca])),multi_dot([config.Mbar_rbr_lca_jcr_strut_lca[:,0:1].T,x26,x30,config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x31 + x16 + multi_dot([x33,config.ubar_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x20,config.ubar_rbl_upright_jcl_lca_upright])),(x31 + x10 + multi_dot([x33,config.ubar_rbl_lca_jcl_lca_chassis]) + -1.0*multi_dot([x11,config.ubar_vbs_chassis_jcl_lca_chassis])),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1].T,x34,x11,x35]),multi_dot([config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2].T,x34,x11,x35]),(x31 + x36 + multi_dot([x33,config.ubar_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x38,config.ubar_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_rbl_lca_jcl_strut_lca[:,0:1].T,x34,x38,config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x4 + -1.0*x39 + multi_dot([x9,config.ubar_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x41,config.ubar_rbr_tie_rod_jcr_tie_upright])),(x4 + -1.0*x0 + multi_dot([x9,config.ubar_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x43,config.ubar_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x44,x43,x45]),multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x44,x43,x45]),(cos(x46)*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2].T,x44,x43,x47]) + sin(x46)*-1.0*multi_dot([config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1].T,x44,x43,x47])),(x15 + -1.0*x48 + multi_dot([x20,config.ubar_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x50,config.ubar_rbl_tie_rod_jcl_tie_upright])),(x15 + -1.0*x2 + multi_dot([x20,config.ubar_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x52,config.ubar_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x53,x52,x54]),multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x53,x52,x54]),(cos(x55)*multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2].T,x53,x52,x56]) + sin(x55)*-1.0*multi_dot([config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1].T,x53,x52,x56])),(x57 + x10 + multi_dot([x59,config.ubar_rbr_upper_strut_jcr_strut_chassis]) + -1.0*multi_dot([x11,config.ubar_vbs_chassis_jcr_strut_chassis])),multi_dot([config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1].T,x60,x11,config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]]),multi_dot([x61,x60,x30,x62]),multi_dot([x63,x60,x30,x62]),multi_dot([x61,x60,x64]),multi_dot([x63,x60,x64]),(x65 + x10 + multi_dot([x67,config.ubar_rbl_upper_strut_jcl_strut_chassis]) + -1.0*multi_dot([x11,config.ubar_vbs_chassis_jcl_strut_chassis])),multi_dot([config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1].T,x68,x11,config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]]),multi_dot([x69,x68,x38,x70]),multi_dot([x71,x68,x38,x70]),multi_dot([x69,x68,x72]),multi_dot([x71,x68,x72]),(x39 + -1.0*self.R_vbr_steer + multi_dot([x41,config.ubar_rbr_tie_rod_jcr_tie_steering]) + -1.0*multi_dot([x73,config.ubar_vbr_steer_jcr_tie_steering])),multi_dot([config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x41.T,x73,config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]]),(x48 + -1.0*self.R_vbl_steer + multi_dot([x50,config.ubar_rbl_tie_rod_jcl_tie_steering]) + -1.0*multi_dot([x74,config.ubar_vbl_steer_jcl_tie_steering])),multi_dot([config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x50.T,x74,config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]]),(x75 + (multi_dot([x6.T,x6]))**(1.0/2.0)),(x75 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x75 + (multi_dot([x24.T,x24]))**(1.0/2.0)),(x75 + (multi_dot([x32.T,x32]))**(1.0/2.0)),(x75 + (multi_dot([x8.T,x8]))**(1.0/2.0)),(x75 + (multi_dot([x19.T,x19]))**(1.0/2.0)),(x75 + (multi_dot([x58.T,x58]))**(1.0/2.0)),(x75 + (multi_dot([x66.T,x66]))**(1.0/2.0)),(x75 + (multi_dot([x29.T,x29]))**(1.0/2.0)),(x75 + (multi_dot([x37.T,x37]))**(1.0/2.0)),(x75 + (multi_dot([x40.T,x40]))**(1.0/2.0)),(x75 + (multi_dot([x49.T,x49]))**(1.0/2.0)),(x75 + (multi_dot([x42.T,x42]))**(1.0/2.0)),(x75 + (multi_dot([x51.T,x51]))**(1.0/2.0))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)
        v1 = np.eye(1,dtype=np.float64)
        v2 = np.zeros((3,1),dtype=np.float64)

        self.vel_eq_blocks = [(v0 + derivative(config.F_mcr_zact,t,0.1,1)*-1.0*v1),(v0 + derivative(config.F_mcl_zact,t,0.1,1)*-1.0*v1),v2,v2,v0,v0,v2,v2,v0,v0,v2,v2,v0,v0,v2,v0,v2,v2,v0,v0,v2,v0,v2,v2,v0,v0,(v0 + derivative(config.F_jcr_hub_bearing,t,0.1,1)*-1.0*v1),v2,v2,v0,v0,(v0 + derivative(config.F_jcl_hub_bearing,t,0.1,1)*-1.0*v1),v2,v0,v0,v0,v0,v0,v2,v0,v0,v0,v0,v0,v2,v0,v2,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0,v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.zeros((1,1),dtype=np.float64)
        a1 = np.eye(1,dtype=np.float64)
        a2 = self.Pd_rbr_uca
        a3 = self.Pd_rbr_upright
        a4 = self.Pd_vbs_chassis
        a5 = config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        a6 = self.P_rbr_uca
        a7 = A(a6).T
        a8 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        a9 = B(a4,a8)
        a10 = a8.T
        a11 = self.P_vbs_chassis
        a12 = A(a11).T
        a13 = a2.T
        a14 = B(a11,a8)
        a15 = config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        a16 = self.Pd_rbl_uca
        a17 = self.Pd_rbl_upright
        a18 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        a19 = a18.T
        a20 = config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        a21 = self.P_rbl_uca
        a22 = A(a21).T
        a23 = B(a4,a18)
        a24 = a16.T
        a25 = B(a11,a18)
        a26 = config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        a27 = self.Pd_rbr_lca
        a28 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        a29 = a28.T
        a30 = config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        a31 = self.P_rbr_lca
        a32 = A(a31).T
        a33 = B(a4,a28)
        a34 = a27.T
        a35 = B(a11,a28)
        a36 = config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        a37 = self.Pd_rbr_lower_strut
        a38 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a39 = self.P_rbr_lower_strut
        a40 = A(a39).T
        a41 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        a42 = self.Pd_rbl_lca
        a43 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        a44 = a43.T
        a45 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        a46 = self.P_rbl_lca
        a47 = A(a46).T
        a48 = B(a4,a43)
        a49 = a42.T
        a50 = B(a11,a43)
        a51 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        a52 = self.Pd_rbl_lower_strut
        a53 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a54 = self.P_rbl_lower_strut
        a55 = A(a54).T
        a56 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        a57 = self.Pd_rbr_tie_rod
        a58 = self.Pd_rbr_hub
        a59 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a60 = self.P_rbr_upright
        a61 = A(a60).T
        a62 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        a63 = B(a58,a62)
        a64 = a62.T
        a65 = self.P_rbr_hub
        a66 = A(a65).T
        a67 = a3.T
        a68 = B(a65,a62)
        a69 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a70 = config.F_jcr_hub_bearing(t,)
        a71 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        a72 = cos(a70)
        a73 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        a74 = sin(a70)
        a75 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        a76 = self.Pd_rbl_tie_rod
        a77 = self.Pd_rbl_hub
        a78 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a79 = self.P_rbl_upright
        a80 = A(a79).T
        a81 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        a82 = B(a77,a81)
        a83 = a81.T
        a84 = self.P_rbl_hub
        a85 = A(a84).T
        a86 = a17.T
        a87 = B(a84,a81)
        a88 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a89 = config.F_jcl_hub_bearing(t,)
        a90 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        a91 = cos(a89)
        a92 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        a93 = sin(a89)
        a94 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        a95 = self.Pd_rbr_upper_strut
        a96 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        a97 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a98 = self.P_rbr_upper_strut
        a99 = A(a98).T
        a100 = a95.T
        a101 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        a102 = a101.T
        a103 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        a104 = B(a37,a103)
        a105 = a103.T
        a106 = B(a95,a101)
        a107 = B(a98,a101).T
        a108 = B(a39,a103)
        a109 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        a110 = a109.T
        a111 = B(a95,a109)
        a112 = B(a98,a109).T
        a113 = config.ubar_rbr_upper_strut_jcr_strut
        a114 = config.ubar_rbr_lower_strut_jcr_strut
        a115 = (multi_dot([B(a95,a113),a95]) + -1.0*multi_dot([B(a37,a114),a37]))
        a116 = (self.Rd_rbr_upper_strut + -1.0*self.Rd_rbr_lower_strut + multi_dot([B(a39,a114),a37]) + multi_dot([B(a98,a113),a95]))
        a117 = (self.R_rbr_upper_strut.T + -1.0*self.R_rbr_lower_strut.T + multi_dot([a113.T,a99]) + -1.0*multi_dot([a114.T,a40]))
        a118 = self.Pd_rbl_upper_strut
        a119 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a120 = self.P_rbl_upper_strut
        a121 = A(a120).T
        a122 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        a123 = a118.T
        a124 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        a125 = a124.T
        a126 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        a127 = B(a118,a126)
        a128 = a126.T
        a129 = B(a52,a124)
        a130 = B(a120,a126).T
        a131 = B(a54,a124)
        a132 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        a133 = B(a118,a132)
        a134 = a132.T
        a135 = B(a120,a132).T
        a136 = config.ubar_rbl_upper_strut_jcl_strut
        a137 = config.ubar_rbl_lower_strut_jcl_strut
        a138 = (multi_dot([B(a118,a136),a118]) + -1.0*multi_dot([B(a52,a137),a52]))
        a139 = (self.Rd_rbl_upper_strut + -1.0*self.Rd_rbl_lower_strut + multi_dot([B(a54,a137),a52]) + multi_dot([B(a120,a136),a118]))
        a140 = (self.R_rbl_upper_strut.T + -1.0*self.R_rbl_lower_strut.T + multi_dot([a136.T,a121]) + -1.0*multi_dot([a137.T,a55]))
        a141 = self.Pd_vbr_steer
        a142 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a143 = self.P_rbr_tie_rod
        a144 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        a145 = self.P_vbr_steer
        a146 = a57.T
        a147 = self.Pd_vbl_steer
        a148 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        a149 = self.P_vbl_steer
        a150 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a151 = self.P_rbl_tie_rod
        a152 = a76.T

        self.acc_eq_blocks = [(a0 + derivative(config.F_mcr_zact,t,0.1,2)*-1.0*a1),(a0 + derivative(config.F_mcl_zact,t,0.1,2)*-1.0*a1),(multi_dot([B(a2,config.ubar_rbr_uca_jcr_uca_upright),a2]) + -1.0*multi_dot([B(a3,config.ubar_rbr_upright_jcr_uca_upright),a3])),(multi_dot([B(a2,config.ubar_rbr_uca_jcr_uca_chassis),a2]) + -1.0*multi_dot([B(a4,config.ubar_vbs_chassis_jcr_uca_chassis),a4])),(multi_dot([a5.T,a7,a9,a4]) + multi_dot([a10,a12,B(a2,a5),a2]) + 2.0*multi_dot([a13,B(a6,a5).T,a14,a4])),(multi_dot([a15.T,a7,a9,a4]) + multi_dot([a10,a12,B(a2,a15),a2]) + 2.0*multi_dot([a13,B(a6,a15).T,a14,a4])),(multi_dot([B(a16,config.ubar_rbl_uca_jcl_uca_upright),a16]) + -1.0*multi_dot([B(a17,config.ubar_rbl_upright_jcl_uca_upright),a17])),(multi_dot([B(a16,config.ubar_rbl_uca_jcl_uca_chassis),a16]) + -1.0*multi_dot([B(a4,config.ubar_vbs_chassis_jcl_uca_chassis),a4])),(multi_dot([a19,a12,B(a16,a20),a16]) + multi_dot([a20.T,a22,a23,a4]) + 2.0*multi_dot([a24,B(a21,a20).T,a25,a4])),(multi_dot([a19,a12,B(a16,a26),a16]) + multi_dot([a26.T,a22,a23,a4]) + 2.0*multi_dot([a24,B(a21,a26).T,a25,a4])),(multi_dot([B(a27,config.ubar_rbr_lca_jcr_lca_upright),a27]) + -1.0*multi_dot([B(a3,config.ubar_rbr_upright_jcr_lca_upright),a3])),(multi_dot([B(a27,config.ubar_rbr_lca_jcr_lca_chassis),a27]) + -1.0*multi_dot([B(a4,config.ubar_vbs_chassis_jcr_lca_chassis),a4])),(multi_dot([a29,a12,B(a27,a30),a27]) + multi_dot([a30.T,a32,a33,a4]) + 2.0*multi_dot([a34,B(a31,a30).T,a35,a4])),(multi_dot([a29,a12,B(a27,a36),a27]) + multi_dot([a36.T,a32,a33,a4]) + 2.0*multi_dot([a34,B(a31,a36).T,a35,a4])),(multi_dot([B(a27,config.ubar_rbr_lca_jcr_strut_lca),a27]) + -1.0*multi_dot([B(a37,config.ubar_rbr_lower_strut_jcr_strut_lca),a37])),(multi_dot([a38.T,a40,B(a27,a41),a27]) + multi_dot([a41.T,a32,B(a37,a38),a37]) + 2.0*multi_dot([a34,B(a31,a41).T,B(a39,a38),a37])),(multi_dot([B(a42,config.ubar_rbl_lca_jcl_lca_upright),a42]) + -1.0*multi_dot([B(a17,config.ubar_rbl_upright_jcl_lca_upright),a17])),(multi_dot([B(a42,config.ubar_rbl_lca_jcl_lca_chassis),a42]) + -1.0*multi_dot([B(a4,config.ubar_vbs_chassis_jcl_lca_chassis),a4])),(multi_dot([a44,a12,B(a42,a45),a42]) + multi_dot([a45.T,a47,a48,a4]) + 2.0*multi_dot([a49,B(a46,a45).T,a50,a4])),(multi_dot([a44,a12,B(a42,a51),a42]) + multi_dot([a51.T,a47,a48,a4]) + 2.0*multi_dot([a49,B(a46,a51).T,a50,a4])),(multi_dot([B(a42,config.ubar_rbl_lca_jcl_strut_lca),a42]) + -1.0*multi_dot([B(a52,config.ubar_rbl_lower_strut_jcl_strut_lca),a52])),(multi_dot([a53.T,a55,B(a42,a56),a42]) + multi_dot([a56.T,a47,B(a52,a53),a52]) + 2.0*multi_dot([a49,B(a46,a56).T,B(a54,a53),a52])),(multi_dot([B(a3,config.ubar_rbr_upright_jcr_tie_upright),a3]) + -1.0*multi_dot([B(a57,config.ubar_rbr_tie_rod_jcr_tie_upright),a57])),(multi_dot([B(a3,config.ubar_rbr_upright_jcr_hub_bearing),a3]) + -1.0*multi_dot([B(a58,config.ubar_rbr_hub_jcr_hub_bearing),a58])),(multi_dot([a59.T,a61,a63,a58]) + multi_dot([a64,a66,B(a3,a59),a3]) + 2.0*multi_dot([a67,B(a60,a59).T,a68,a58])),(multi_dot([a69.T,a61,a63,a58]) + multi_dot([a64,a66,B(a3,a69),a3]) + 2.0*multi_dot([a67,B(a60,a69).T,a68,a58])),(derivative(a70,t,0.1,2)*-1.0*a1 + multi_dot([a71.T,a66,(a72*B(a3,a73) + a74*-1.0*B(a3,a75)),a3]) + multi_dot([(a72*multi_dot([a73.T,a61]) + a74*-1.0*multi_dot([a75.T,a61])),B(a58,a71),a58]) + 2.0*multi_dot([((a72*multi_dot([B(a60,a73),a3])).T + a74*-1.0*multi_dot([a67,B(a60,a75).T])),B(a65,a71),a58])),(multi_dot([B(a17,config.ubar_rbl_upright_jcl_tie_upright),a17]) + -1.0*multi_dot([B(a76,config.ubar_rbl_tie_rod_jcl_tie_upright),a76])),(multi_dot([B(a17,config.ubar_rbl_upright_jcl_hub_bearing),a17]) + -1.0*multi_dot([B(a77,config.ubar_rbl_hub_jcl_hub_bearing),a77])),(multi_dot([a78.T,a80,a82,a77]) + multi_dot([a83,a85,B(a17,a78),a17]) + 2.0*multi_dot([a86,B(a79,a78).T,a87,a77])),(multi_dot([a88.T,a80,a82,a77]) + multi_dot([a83,a85,B(a17,a88),a17]) + 2.0*multi_dot([a86,B(a79,a88).T,a87,a77])),(derivative(a89,t,0.1,2)*-1.0*a1 + multi_dot([a90.T,a85,(a91*B(a17,a92) + a93*-1.0*B(a17,a94)),a17]) + multi_dot([(a91*multi_dot([a92.T,a80]) + a93*-1.0*multi_dot([a94.T,a80])),B(a77,a90),a77]) + 2.0*multi_dot([((a91*multi_dot([B(a79,a92),a17])).T + a93*-1.0*multi_dot([a86,B(a79,a94).T])),B(a84,a90),a77])),(multi_dot([B(a95,config.ubar_rbr_upper_strut_jcr_strut_chassis),a95]) + -1.0*multi_dot([B(a4,config.ubar_vbs_chassis_jcr_strut_chassis),a4])),(multi_dot([a96.T,a12,B(a95,a97),a95]) + multi_dot([a97.T,a99,B(a4,a96),a4]) + 2.0*multi_dot([a100,B(a98,a97).T,B(a11,a96),a4])),(multi_dot([a102,a99,a104,a37]) + multi_dot([a105,a40,a106,a95]) + 2.0*multi_dot([a100,a107,a108,a37])),(multi_dot([a110,a99,a104,a37]) + multi_dot([a105,a40,a111,a95]) + 2.0*multi_dot([a100,a112,a108,a37])),(multi_dot([a102,a99,a115]) + 2.0*multi_dot([a100,a107,a116]) + multi_dot([a117,a106,a95])),(multi_dot([a110,a99,a115]) + 2.0*multi_dot([a100,a112,a116]) + multi_dot([a117,a111,a95])),(multi_dot([B(a118,config.ubar_rbl_upper_strut_jcl_strut_chassis),a118]) + -1.0*multi_dot([B(a4,config.ubar_vbs_chassis_jcl_strut_chassis),a4])),(multi_dot([a119.T,a121,B(a4,a122),a4]) + multi_dot([a122.T,a12,B(a118,a119),a118]) + 2.0*multi_dot([a123,B(a120,a119).T,B(a11,a122),a4])),(multi_dot([a125,a55,a127,a118]) + multi_dot([a128,a121,a129,a52]) + 2.0*multi_dot([a123,a130,a131,a52])),(multi_dot([a125,a55,a133,a118]) + multi_dot([a134,a121,a129,a52]) + 2.0*multi_dot([a123,a135,a131,a52])),(multi_dot([a128,a121,a138]) + 2.0*multi_dot([a123,a130,a139]) + multi_dot([a140,a127,a118])),(multi_dot([a134,a121,a138]) + 2.0*multi_dot([a123,a135,a139]) + multi_dot([a140,a133,a118])),(multi_dot([B(a57,config.ubar_rbr_tie_rod_jcr_tie_steering),a57]) + -1.0*multi_dot([B(a141,config.ubar_vbr_steer_jcr_tie_steering),a141])),(multi_dot([a142.T,A(a143).T,B(a141,a144),a141]) + multi_dot([a144.T,A(a145).T,B(a57,a142),a57]) + 2.0*multi_dot([a146,B(a143,a142).T,B(a145,a144),a141])),(multi_dot([B(a76,config.ubar_rbl_tie_rod_jcl_tie_steering),a76]) + -1.0*multi_dot([B(a147,config.ubar_vbl_steer_jcl_tie_steering),a147])),(multi_dot([a148.T,A(a149).T,B(a76,a150),a76]) + multi_dot([a150.T,A(a151).T,B(a147,a148),a147]) + 2.0*multi_dot([a152,B(a151,a150).T,B(a149,a148),a147])),2.0*(multi_dot([a13,a2]))**(1.0/2.0),2.0*(multi_dot([a24,a16]))**(1.0/2.0),2.0*(multi_dot([a34,a27]))**(1.0/2.0),2.0*(multi_dot([a49,a42]))**(1.0/2.0),2.0*(multi_dot([a67,a3]))**(1.0/2.0),2.0*(multi_dot([a86,a17]))**(1.0/2.0),2.0*(multi_dot([a100,a95]))**(1.0/2.0),2.0*(multi_dot([a123,a118]))**(1.0/2.0),2.0*(multi_dot([a37.T,a37]))**(1.0/2.0),2.0*(multi_dot([a52.T,a52]))**(1.0/2.0),2.0*(multi_dot([a146,a57]))**(1.0/2.0),2.0*(multi_dot([a152,a76]))**(1.0/2.0),2.0*(multi_dot([a58.T,a58]))**(1.0/2.0),2.0*(multi_dot([a77.T,a77]))**(1.0/2.0)]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,4),dtype=np.float64)
        j1 = np.zeros((1,3),dtype=np.float64)
        j2 = np.eye(3,dtype=np.float64)
        j3 = self.P_rbr_uca
        j4 = -1.0*j2
        j5 = self.P_rbr_upright
        j6 = config.Mbar_vbs_chassis_jcr_uca_chassis[:,2:3]
        j7 = j6.T
        j8 = self.P_vbs_chassis
        j9 = A(j8).T
        j10 = config.Mbar_rbr_uca_jcr_uca_chassis[:,0:1]
        j11 = config.Mbar_rbr_uca_jcr_uca_chassis[:,1:2]
        j12 = A(j3).T
        j13 = B(j8,j6)
        j14 = self.P_rbl_uca
        j15 = self.P_rbl_upright
        j16 = config.Mbar_vbs_chassis_jcl_uca_chassis[:,2:3]
        j17 = j16.T
        j18 = config.Mbar_rbl_uca_jcl_uca_chassis[:,0:1]
        j19 = config.Mbar_rbl_uca_jcl_uca_chassis[:,1:2]
        j20 = A(j14).T
        j21 = B(j8,j16)
        j22 = self.P_rbr_lca
        j23 = config.Mbar_vbs_chassis_jcr_lca_chassis[:,2:3]
        j24 = j23.T
        j25 = config.Mbar_rbr_lca_jcr_lca_chassis[:,0:1]
        j26 = config.Mbar_rbr_lca_jcr_lca_chassis[:,1:2]
        j27 = A(j22).T
        j28 = B(j8,j23)
        j29 = config.Mbar_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j30 = self.P_rbr_lower_strut
        j31 = A(j30).T
        j32 = config.Mbar_rbr_lca_jcr_strut_lca[:,0:1]
        j33 = self.P_rbl_lca
        j34 = config.Mbar_vbs_chassis_jcl_lca_chassis[:,2:3]
        j35 = j34.T
        j36 = config.Mbar_rbl_lca_jcl_lca_chassis[:,0:1]
        j37 = config.Mbar_rbl_lca_jcl_lca_chassis[:,1:2]
        j38 = A(j33).T
        j39 = B(j8,j34)
        j40 = config.Mbar_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j41 = self.P_rbl_lower_strut
        j42 = A(j41).T
        j43 = config.Mbar_rbl_lca_jcl_strut_lca[:,0:1]
        j44 = self.P_rbr_tie_rod
        j45 = config.Mbar_rbr_hub_jcr_hub_bearing[:,2:3]
        j46 = j45.T
        j47 = self.P_rbr_hub
        j48 = A(j47).T
        j49 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j50 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j51 = A(j5).T
        j52 = B(j47,j45)
        j53 = config.Mbar_rbr_hub_jcr_hub_bearing[:,0:1]
        j54 = config.F_jcr_hub_bearing(t,)
        j55 = cos(j54)
        j56 = config.Mbar_rbr_upright_jcr_hub_bearing[:,1:2]
        j57 = config.Mbar_rbr_upright_jcr_hub_bearing[:,0:1]
        j58 = self.P_rbl_tie_rod
        j59 = config.Mbar_rbl_hub_jcl_hub_bearing[:,2:3]
        j60 = j59.T
        j61 = self.P_rbl_hub
        j62 = A(j61).T
        j63 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j64 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j65 = A(j15).T
        j66 = B(j61,j59)
        j67 = config.Mbar_rbl_hub_jcl_hub_bearing[:,0:1]
        j68 = config.F_jcl_hub_bearing(t,)
        j69 = cos(j68)
        j70 = config.Mbar_rbl_upright_jcl_hub_bearing[:,1:2]
        j71 = config.Mbar_rbl_upright_jcl_hub_bearing[:,0:1]
        j72 = self.P_rbr_upper_strut
        j73 = config.Mbar_vbs_chassis_jcr_strut_chassis[:,0:1]
        j74 = config.Mbar_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j75 = A(j72).T
        j76 = config.Mbar_rbr_lower_strut_jcr_strut[:,2:3]
        j77 = j76.T
        j78 = config.Mbar_rbr_upper_strut_jcr_strut[:,0:1]
        j79 = B(j72,j78)
        j80 = config.Mbar_rbr_upper_strut_jcr_strut[:,1:2]
        j81 = B(j72,j80)
        j82 = j78.T
        j83 = multi_dot([j82,j75])
        j84 = config.ubar_rbr_upper_strut_jcr_strut
        j85 = B(j72,j84)
        j86 = config.ubar_rbr_lower_strut_jcr_strut
        j87 = (self.R_rbr_upper_strut.T + -1.0*self.R_rbr_lower_strut.T + multi_dot([j84.T,j75]) + -1.0*multi_dot([j86.T,j31]))
        j88 = j80.T
        j89 = multi_dot([j88,j75])
        j90 = B(j30,j76)
        j91 = B(j30,j86)
        j92 = self.P_rbl_upper_strut
        j93 = config.Mbar_vbs_chassis_jcl_strut_chassis[:,0:1]
        j94 = config.Mbar_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j95 = A(j92).T
        j96 = config.Mbar_rbl_lower_strut_jcl_strut[:,2:3]
        j97 = j96.T
        j98 = config.Mbar_rbl_upper_strut_jcl_strut[:,0:1]
        j99 = B(j92,j98)
        j100 = config.Mbar_rbl_upper_strut_jcl_strut[:,1:2]
        j101 = B(j92,j100)
        j102 = j98.T
        j103 = multi_dot([j102,j95])
        j104 = config.ubar_rbl_upper_strut_jcl_strut
        j105 = B(j92,j104)
        j106 = config.ubar_rbl_lower_strut_jcl_strut
        j107 = (self.R_rbl_upper_strut.T + -1.0*self.R_rbl_lower_strut.T + multi_dot([j104.T,j95]) + -1.0*multi_dot([j106.T,j42]))
        j108 = j100.T
        j109 = multi_dot([j108,j95])
        j110 = B(j41,j96)
        j111 = B(j41,j106)
        j112 = config.Mbar_vbr_steer_jcr_tie_steering[:,0:1]
        j113 = self.P_vbr_steer
        j114 = config.Mbar_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j115 = config.Mbar_vbl_steer_jcl_tie_steering[:,0:1]
        j116 = self.P_vbl_steer
        j117 = config.Mbar_rbl_tie_rod_jcl_tie_steering[:,0:1]

        self.jac_eq_blocks = [config.J_mcr_zact,j0,j1,j0,config.J_mcl_zact,j0,j1,j0,j2,B(j3,config.ubar_rbr_uca_jcr_uca_upright),j4,-1.0*B(j5,config.ubar_rbr_upright_jcr_uca_upright),j2,B(j3,config.ubar_rbr_uca_jcr_uca_chassis),j4,-1.0*B(j8,config.ubar_vbs_chassis_jcr_uca_chassis),j1,multi_dot([j7,j9,B(j3,j10)]),j1,multi_dot([j10.T,j12,j13]),j1,multi_dot([j7,j9,B(j3,j11)]),j1,multi_dot([j11.T,j12,j13]),j2,B(j14,config.ubar_rbl_uca_jcl_uca_upright),j4,-1.0*B(j15,config.ubar_rbl_upright_jcl_uca_upright),j2,B(j14,config.ubar_rbl_uca_jcl_uca_chassis),j4,-1.0*B(j8,config.ubar_vbs_chassis_jcl_uca_chassis),j1,multi_dot([j17,j9,B(j14,j18)]),j1,multi_dot([j18.T,j20,j21]),j1,multi_dot([j17,j9,B(j14,j19)]),j1,multi_dot([j19.T,j20,j21]),j2,B(j22,config.ubar_rbr_lca_jcr_lca_upright),j4,-1.0*B(j5,config.ubar_rbr_upright_jcr_lca_upright),j2,B(j22,config.ubar_rbr_lca_jcr_lca_chassis),j4,-1.0*B(j8,config.ubar_vbs_chassis_jcr_lca_chassis),j1,multi_dot([j24,j9,B(j22,j25)]),j1,multi_dot([j25.T,j27,j28]),j1,multi_dot([j24,j9,B(j22,j26)]),j1,multi_dot([j26.T,j27,j28]),j2,B(j22,config.ubar_rbr_lca_jcr_strut_lca),j4,-1.0*B(j30,config.ubar_rbr_lower_strut_jcr_strut_lca),j1,multi_dot([j29.T,j31,B(j22,j32)]),j1,multi_dot([j32.T,j27,B(j30,j29)]),j2,B(j33,config.ubar_rbl_lca_jcl_lca_upright),j4,-1.0*B(j15,config.ubar_rbl_upright_jcl_lca_upright),j2,B(j33,config.ubar_rbl_lca_jcl_lca_chassis),j4,-1.0*B(j8,config.ubar_vbs_chassis_jcl_lca_chassis),j1,multi_dot([j35,j9,B(j33,j36)]),j1,multi_dot([j36.T,j38,j39]),j1,multi_dot([j35,j9,B(j33,j37)]),j1,multi_dot([j37.T,j38,j39]),j2,B(j33,config.ubar_rbl_lca_jcl_strut_lca),j4,-1.0*B(j41,config.ubar_rbl_lower_strut_jcl_strut_lca),j1,multi_dot([j40.T,j42,B(j33,j43)]),j1,multi_dot([j43.T,j38,B(j41,j40)]),j2,B(j5,config.ubar_rbr_upright_jcr_tie_upright),j4,-1.0*B(j44,config.ubar_rbr_tie_rod_jcr_tie_upright),j2,B(j5,config.ubar_rbr_upright_jcr_hub_bearing),j4,-1.0*B(j47,config.ubar_rbr_hub_jcr_hub_bearing),j1,multi_dot([j46,j48,B(j5,j49)]),j1,multi_dot([j49.T,j51,j52]),j1,multi_dot([j46,j48,B(j5,j50)]),j1,multi_dot([j50.T,j51,j52]),j1,multi_dot([j53.T,j48,(j55*B(j5,j56) + sin(j54)*-1.0*B(j5,j57))]),j1,multi_dot([(j55*multi_dot([j56.T,j51]) + sin(j54)*-1.0*multi_dot([j57.T,j51])),B(j47,j53)]),j2,B(j15,config.ubar_rbl_upright_jcl_tie_upright),j4,-1.0*B(j58,config.ubar_rbl_tie_rod_jcl_tie_upright),j2,B(j15,config.ubar_rbl_upright_jcl_hub_bearing),j4,-1.0*B(j61,config.ubar_rbl_hub_jcl_hub_bearing),j1,multi_dot([j60,j62,B(j15,j63)]),j1,multi_dot([j63.T,j65,j66]),j1,multi_dot([j60,j62,B(j15,j64)]),j1,multi_dot([j64.T,j65,j66]),j1,multi_dot([j67.T,j62,(j69*B(j15,j70) + sin(j68)*-1.0*B(j15,j71))]),j1,multi_dot([(j69*multi_dot([j70.T,j65]) + sin(j68)*-1.0*multi_dot([j71.T,j65])),B(j61,j67)]),j2,B(j72,config.ubar_rbr_upper_strut_jcr_strut_chassis),j4,-1.0*B(j8,config.ubar_vbs_chassis_jcr_strut_chassis),j1,multi_dot([j73.T,j9,B(j72,j74)]),j1,multi_dot([j74.T,j75,B(j8,j73)]),j1,multi_dot([j77,j31,j79]),j1,multi_dot([j82,j75,j90]),j1,multi_dot([j77,j31,j81]),j1,multi_dot([j88,j75,j90]),j83,(multi_dot([j82,j75,j85]) + multi_dot([j87,j79])),-1.0*j83,-1.0*multi_dot([j82,j75,j91]),j89,(multi_dot([j88,j75,j85]) + multi_dot([j87,j81])),-1.0*j89,-1.0*multi_dot([j88,j75,j91]),j2,B(j92,config.ubar_rbl_upper_strut_jcl_strut_chassis),j4,-1.0*B(j8,config.ubar_vbs_chassis_jcl_strut_chassis),j1,multi_dot([j93.T,j9,B(j92,j94)]),j1,multi_dot([j94.T,j95,B(j8,j93)]),j1,multi_dot([j97,j42,j99]),j1,multi_dot([j102,j95,j110]),j1,multi_dot([j97,j42,j101]),j1,multi_dot([j108,j95,j110]),j103,(multi_dot([j102,j95,j105]) + multi_dot([j107,j99])),-1.0*j103,-1.0*multi_dot([j102,j95,j111]),j109,(multi_dot([j108,j95,j105]) + multi_dot([j107,j101])),-1.0*j109,-1.0*multi_dot([j108,j95,j111]),j2,B(j44,config.ubar_rbr_tie_rod_jcr_tie_steering),j4,-1.0*B(j113,config.ubar_vbr_steer_jcr_tie_steering),j1,multi_dot([j112.T,A(j113).T,B(j44,j114)]),j1,multi_dot([j114.T,A(j44).T,B(j113,j112)]),j2,B(j58,config.ubar_rbl_tie_rod_jcl_tie_steering),j4,-1.0*B(j116,config.ubar_vbl_steer_jcl_tie_steering),j1,multi_dot([j115.T,A(j116).T,B(j58,j117)]),j1,multi_dot([j117.T,A(j58).T,B(j116,j115)]),2.0*j3.T,2.0*j14.T,2.0*j22.T,2.0*j33.T,2.0*j5.T,2.0*j15.T,2.0*j72.T,2.0*j92.T,2.0*j30.T,2.0*j41.T,2.0*j44.T,2.0*j58.T,2.0*j47.T,2.0*j61.T]
  
