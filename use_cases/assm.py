
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin
from source.solvers.python_solver import solver



class numerical_assembly(object):

    def __init__(self,config):                    
        self.config = config
        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.pos_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72])
        self.pos_rows = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.vel_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72])
        self.vel_rows = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.acc_rows = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72])
        self.acc_rows = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12,13,13,13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,31,31,32,32,32,32,33,33,33,33,34,34,34,34,35,35,35,35,36,36,36,36,37,37,37,37,38,38,38,38,39,39,39,39,40,40,40,40,41,41,41,41,42,42,42,42,43,43,43,43,44,44,44,44,45,45,45,45,46,46,46,46,47,47,47,47,48,48,48,48,49,49,49,49,50,50,50,50,51,51,51,51,52,52,52,52,53,53,53,53,54,54,55,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72])
        self.jac_rows = np.array([0,1,2,3,0,1,2,3,0,1,2,3,0,1,4,5,0,1,4,5,0,1,4,5,0,1,6,7,0,1,6,7,0,1,6,7,0,1,8,9,0,1,8,9,0,1,8,9,0,1,14,15,0,1,14,15,0,1,16,17,0,1,16,17,0,1,32,33,0,1,32,33,0,1,32,33,0,1,34,35,0,1,34,35,0,1,34,35,2,3,10,11,4,5,12,13,6,7,10,11,6,7,18,19,6,7,18,19,8,9,12,13,8,9,20,21,8,9,20,21,10,11,22,23,10,11,26,27,10,11,26,27,10,11,26,27,12,13,24,25,12,13,28,29,12,13,28,29,12,13,28,29,14,15,18,19,14,15,18,19,14,15,18,19,14,15,18,19,16,17,20,21,16,17,20,21,16,17,20,21,16,17,20,21,22,23,32,33,22,23,32,33,24,25,34,35,24,25,34,35,30,31,32,33,30,31,34,35,30,31,34,35,30,31,34,35,0,1,0,1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35])

    
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

        x0 = self.R_ground
        x1 = self.R_SU1_rbr_uca
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_SU1_rbr_uca
        x5 = A(x4)
        x6 = x3.T
        x7 = config.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
        x8 = self.R_SU1_rbl_uca
        x9 = self.P_SU1_rbl_uca
        x10 = A(x9)
        x11 = config.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
        x12 = self.R_SU1_rbr_lca
        x13 = self.P_SU1_rbr_lca
        x14 = A(x13)
        x15 = config.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
        x16 = self.R_SU1_rbl_lca
        x17 = self.P_SU1_rbl_lca
        x18 = A(x17)
        x19 = config.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
        x20 = self.R_SU1_rbr_upper_strut
        x21 = self.P_SU1_rbr_upper_strut
        x22 = A(x21)
        x23 = self.R_SU1_rbl_upper_strut
        x24 = self.P_SU1_rbl_upper_strut
        x25 = A(x24)
        x26 = -1.0*self.R_ST_rbr_rocker
        x27 = self.P_ST_rbr_rocker
        x28 = A(x27)
        x29 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        x30 = -1.0*self.R_ST_rbl_rocker
        x31 = self.P_ST_rbl_rocker
        x32 = A(x31)
        x33 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        x34 = self.R_SU1_rbr_upright
        x35 = -1.0*x34
        x36 = self.P_SU1_rbr_upright
        x37 = A(x36)
        x38 = self.R_SU1_rbl_upright
        x39 = -1.0*x38
        x40 = self.P_SU1_rbl_upright
        x41 = A(x40)
        x42 = -1.0*self.R_SU1_rbr_lower_strut
        x43 = self.P_SU1_rbr_lower_strut
        x44 = A(x43)
        x45 = -1.0*self.R_SU1_rbl_lower_strut
        x46 = self.P_SU1_rbl_lower_strut
        x47 = A(x46)
        x48 = self.R_SU1_rbr_tie_rod
        x49 = self.P_SU1_rbr_tie_rod
        x50 = A(x49)
        x51 = self.P_SU1_rbr_hub
        x52 = A(x51)
        x53 = x37.T
        x54 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        x55 = self.R_SU1_rbl_tie_rod
        x56 = self.P_SU1_rbl_tie_rod
        x57 = A(x56)
        x58 = self.P_SU1_rbl_hub
        x59 = A(x58)
        x60 = x41.T
        x61 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        x62 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1].T
        x63 = x22.T
        x64 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        x65 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2].T
        x66 = (x20 + x42 + multi_dot([x22,config.ubar_SU1_rbr_upper_strut_jcr_strut]) + -1.0*multi_dot([x44,config.ubar_SU1_rbr_lower_strut_jcr_strut]))
        x67 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1].T
        x68 = x25.T
        x69 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        x70 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2].T
        x71 = (x23 + x45 + multi_dot([x25,config.ubar_SU1_rbl_upper_strut_jcl_strut]) + -1.0*multi_dot([x47,config.ubar_SU1_rbl_lower_strut_jcl_strut]))
        x72 = self.R_ST_rbs_coupler
        x73 = self.P_ST_rbs_coupler
        x74 = A(x73)
        x75 = x74.T
        x76 = config.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
        x77 = -1.0*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1.0*x1 + multi_dot([x3,config.ubar_ground_jcr_uca_chassis]) + -1.0*multi_dot([x5,config.ubar_SU1_rbr_uca_jcr_uca_chassis])),multi_dot([config.Mbar_ground_jcr_uca_chassis[:,0:1].T,x6,x5,x7]),multi_dot([config.Mbar_ground_jcr_uca_chassis[:,1:2].T,x6,x5,x7]),(x0 + -1.0*x8 + multi_dot([x3,config.ubar_ground_jcl_uca_chassis]) + -1.0*multi_dot([x10,config.ubar_SU1_rbl_uca_jcl_uca_chassis])),multi_dot([config.Mbar_ground_jcl_uca_chassis[:,0:1].T,x6,x10,x11]),multi_dot([config.Mbar_ground_jcl_uca_chassis[:,1:2].T,x6,x10,x11]),(x0 + -1.0*x12 + multi_dot([x3,config.ubar_ground_jcr_lca_chassis]) + -1.0*multi_dot([x14,config.ubar_SU1_rbr_lca_jcr_lca_chassis])),multi_dot([config.Mbar_ground_jcr_lca_chassis[:,0:1].T,x6,x14,x15]),multi_dot([config.Mbar_ground_jcr_lca_chassis[:,1:2].T,x6,x14,x15]),(x0 + -1.0*x16 + multi_dot([x3,config.ubar_ground_jcl_lca_chassis]) + -1.0*multi_dot([x18,config.ubar_SU1_rbl_lca_jcl_lca_chassis])),multi_dot([config.Mbar_ground_jcl_lca_chassis[:,0:1].T,x6,x18,x19]),multi_dot([config.Mbar_ground_jcl_lca_chassis[:,1:2].T,x6,x18,x19]),(x0 + -1.0*x20 + multi_dot([x3,config.ubar_ground_jcr_strut_chassis]) + -1.0*multi_dot([x22,config.ubar_SU1_rbr_upper_strut_jcr_strut_chassis])),multi_dot([config.Mbar_ground_jcr_strut_chassis[:,0:1].T,x6,x22,config.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]]),(x0 + -1.0*x23 + multi_dot([x3,config.ubar_ground_jcl_strut_chassis]) + -1.0*multi_dot([x25,config.ubar_SU1_rbl_upper_strut_jcl_strut_chassis])),multi_dot([config.Mbar_ground_jcl_strut_chassis[:,0:1].T,x6,x25,config.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]]),(x0 + x26 + multi_dot([x3,config.ubar_ground_jcr_rocker_ch]) + -1.0*multi_dot([x28,config.ubar_ST_rbr_rocker_jcr_rocker_ch])),multi_dot([config.Mbar_ground_jcr_rocker_ch[:,0:1].T,x6,x28,x29]),multi_dot([config.Mbar_ground_jcr_rocker_ch[:,1:2].T,x6,x28,x29]),(x0 + x30 + multi_dot([x3,config.ubar_ground_jcl_rocker_ch]) + -1.0*multi_dot([x32,config.ubar_ST_rbl_rocker_jcl_rocker_ch])),multi_dot([config.Mbar_ground_jcl_rocker_ch[:,0:1].T,x6,x32,x33]),multi_dot([config.Mbar_ground_jcl_rocker_ch[:,1:2].T,x6,x32,x33]),(x1 + x35 + multi_dot([x5,config.ubar_SU1_rbr_uca_jcr_uca_upright]) + -1.0*multi_dot([x37,config.ubar_SU1_rbr_upright_jcr_uca_upright])),(x8 + x39 + multi_dot([x10,config.ubar_SU1_rbl_uca_jcl_uca_upright]) + -1.0*multi_dot([x41,config.ubar_SU1_rbl_upright_jcl_uca_upright])),(x12 + x35 + multi_dot([x14,config.ubar_SU1_rbr_lca_jcr_lca_upright]) + -1.0*multi_dot([x37,config.ubar_SU1_rbr_upright_jcr_lca_upright])),(x12 + x42 + multi_dot([x14,config.ubar_SU1_rbr_lca_jcr_strut_lca]) + -1.0*multi_dot([x44,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca])),multi_dot([config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1].T,x14.T,x44,config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]]),(x16 + x39 + multi_dot([x18,config.ubar_SU1_rbl_lca_jcl_lca_upright]) + -1.0*multi_dot([x41,config.ubar_SU1_rbl_upright_jcl_lca_upright])),(x16 + x45 + multi_dot([x18,config.ubar_SU1_rbl_lca_jcl_strut_lca]) + -1.0*multi_dot([x47,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca])),multi_dot([config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1].T,x18.T,x47,config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]]),(x34 + -1.0*x48 + multi_dot([x37,config.ubar_SU1_rbr_upright_jcr_tie_upright]) + -1.0*multi_dot([x50,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright])),(x34 + -1.0*self.R_SU1_rbr_hub + multi_dot([x37,config.ubar_SU1_rbr_upright_jcr_hub_bearing]) + -1.0*multi_dot([x52,config.ubar_SU1_rbr_hub_jcr_hub_bearing])),multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1].T,x53,x52,x54]),multi_dot([config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2].T,x53,x52,x54]),(x38 + -1.0*x55 + multi_dot([x41,config.ubar_SU1_rbl_upright_jcl_tie_upright]) + -1.0*multi_dot([x57,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright])),(x38 + -1.0*self.R_SU1_rbl_hub + multi_dot([x41,config.ubar_SU1_rbl_upright_jcl_hub_bearing]) + -1.0*multi_dot([x59,config.ubar_SU1_rbl_hub_jcl_hub_bearing])),multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1].T,x60,x59,x61]),multi_dot([config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2].T,x60,x59,x61]),multi_dot([x62,x63,x44,x64]),multi_dot([x65,x63,x44,x64]),multi_dot([x62,x63,x66]),multi_dot([x65,x63,x66]),multi_dot([x67,x68,x47,x69]),multi_dot([x70,x68,x47,x69]),multi_dot([x67,x68,x71]),multi_dot([x70,x68,x71]),(x48 + x26 + multi_dot([x50,config.ubar_SU1_rbr_tie_rod_jcr_tie_steering]) + -1.0*multi_dot([x28,config.ubar_ST_rbr_rocker_jcr_tie_steering])),multi_dot([config.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1].T,x50.T,x28,config.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]]),(x55 + x30 + multi_dot([x57,config.ubar_SU1_rbl_tie_rod_jcl_tie_steering]) + -1.0*multi_dot([x32,config.ubar_ST_rbl_rocker_jcl_tie_steering])),multi_dot([config.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1].T,x57.T,x32,config.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]]),(x72 + x26 + multi_dot([x74,config.ubar_ST_rbs_coupler_jcs_rc_sph]) + -1.0*multi_dot([x28,config.ubar_ST_rbr_rocker_jcs_rc_sph])),(x72 + x30 + multi_dot([x74,config.ubar_ST_rbs_coupler_jcs_rc_uni]) + -1.0*multi_dot([x32,config.ubar_ST_rbl_rocker_jcs_rc_uni])),multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1].T,x75,x32,x76]),multi_dot([config.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2].T,x75,x32,x76]),x0,(x2 + -1.0*'Pg_ground'),(x77 + (multi_dot([x4.T,x4]))**(1.0/2.0)),(x77 + (multi_dot([x9.T,x9]))**(1.0/2.0)),(x77 + (multi_dot([x13.T,x13]))**(1.0/2.0)),(x77 + (multi_dot([x17.T,x17]))**(1.0/2.0)),(x77 + (multi_dot([x36.T,x36]))**(1.0/2.0)),(x77 + (multi_dot([x40.T,x40]))**(1.0/2.0)),(x77 + (multi_dot([x21.T,x21]))**(1.0/2.0)),(x77 + (multi_dot([x24.T,x24]))**(1.0/2.0)),(x77 + (multi_dot([x43.T,x43]))**(1.0/2.0)),(x77 + (multi_dot([x46.T,x46]))**(1.0/2.0)),(x77 + (multi_dot([x49.T,x49]))**(1.0/2.0)),(x77 + (multi_dot([x56.T,x56]))**(1.0/2.0)),(x77 + (multi_dot([x51.T,x51]))**(1.0/2.0)),(x77 + (multi_dot([x58.T,x58]))**(1.0/2.0)),(x77 + (multi_dot([x73.T,x73]))**(1.0/2.0)),(x77 + (multi_dot([x27.T,x27]))**(1.0/2.0)),(x77 + (multi_dot([x31.T,x31]))**(1.0/2.0))]

    
    def eval_vel_eq(self):

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,v0,v1,v1,v0,v1,v1,v0,v1,v1,v0,v1,v0,v1,v0,v1,v1,v0,v1,v1,v0,v0,v0,v0,v1,v0,v0,v1,v0,v0,v1,v1,v0,v0,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v0,v1,v0,v1,v0,v0,v1,v1,v0,np.zeros((4,1),dtype=np.float64),v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1,v1]

    
    def eval_acc_eq(self):

        a0 = self.Pd_ground
        a1 = self.Pd_SU1_rbr_uca
        a2 = config.Mbar_ground_jcr_uca_chassis[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = config.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_SU1_rbr_uca
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = config.Mbar_ground_jcr_uca_chassis[:,1:2]
        a13 = self.Pd_SU1_rbl_uca
        a14 = config.Mbar_ground_jcl_uca_chassis[:,0:1]
        a15 = config.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
        a16 = B(a13,a15)
        a17 = a15.T
        a18 = self.P_SU1_rbl_uca
        a19 = A(a18).T
        a20 = B(a18,a15)
        a21 = config.Mbar_ground_jcl_uca_chassis[:,1:2]
        a22 = self.Pd_SU1_rbr_lca
        a23 = config.Mbar_ground_jcr_lca_chassis[:,0:1]
        a24 = config.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
        a25 = B(a22,a24)
        a26 = a24.T
        a27 = self.P_SU1_rbr_lca
        a28 = A(a27).T
        a29 = B(a27,a24)
        a30 = config.Mbar_ground_jcr_lca_chassis[:,1:2]
        a31 = self.Pd_SU1_rbl_lca
        a32 = config.Mbar_ground_jcl_lca_chassis[:,0:1]
        a33 = config.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
        a34 = B(a31,a33)
        a35 = a33.T
        a36 = self.P_SU1_rbl_lca
        a37 = A(a36).T
        a38 = B(a36,a33)
        a39 = config.Mbar_ground_jcl_lca_chassis[:,1:2]
        a40 = self.Pd_SU1_rbr_upper_strut
        a41 = config.Mbar_ground_jcr_strut_chassis[:,0:1]
        a42 = config.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        a43 = self.P_SU1_rbr_upper_strut
        a44 = A(a43).T
        a45 = self.Pd_SU1_rbl_upper_strut
        a46 = config.Mbar_ground_jcl_strut_chassis[:,0:1]
        a47 = config.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        a48 = self.P_SU1_rbl_upper_strut
        a49 = A(a48).T
        a50 = self.Pd_ST_rbr_rocker
        a51 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        a52 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        a53 = B(a50,a52)
        a54 = a52.T
        a55 = self.P_ST_rbr_rocker
        a56 = A(a55).T
        a57 = B(a55,a52)
        a58 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        a59 = self.Pd_ST_rbl_rocker
        a60 = config.Mbar_ground_jcl_rocker_ch[:,0:1]
        a61 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        a62 = B(a59,a61)
        a63 = a61.T
        a64 = self.P_ST_rbl_rocker
        a65 = A(a64).T
        a66 = B(a64,a61)
        a67 = config.Mbar_ground_jcl_rocker_ch[:,1:2]
        a68 = self.Pd_SU1_rbr_upright
        a69 = self.Pd_SU1_rbl_upright
        a70 = self.Pd_SU1_rbr_lower_strut
        a71 = config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
        a72 = config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
        a73 = self.P_SU1_rbr_lower_strut
        a74 = A(a73).T
        a75 = a22.T
        a76 = self.Pd_SU1_rbl_lower_strut
        a77 = config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
        a78 = self.P_SU1_rbl_lower_strut
        a79 = A(a78).T
        a80 = config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
        a81 = a31.T
        a82 = self.Pd_SU1_rbr_tie_rod
        a83 = self.Pd_SU1_rbr_hub
        a84 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        a85 = a84.T
        a86 = self.P_SU1_rbr_hub
        a87 = A(a86).T
        a88 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        a89 = self.P_SU1_rbr_upright
        a90 = A(a89).T
        a91 = B(a83,a84)
        a92 = a68.T
        a93 = B(a86,a84)
        a94 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        a95 = self.Pd_SU1_rbl_tie_rod
        a96 = self.Pd_SU1_rbl_hub
        a97 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        a98 = a97.T
        a99 = self.P_SU1_rbl_hub
        a100 = A(a99).T
        a101 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        a102 = self.P_SU1_rbl_upright
        a103 = A(a102).T
        a104 = B(a96,a97)
        a105 = a69.T
        a106 = B(a99,a97)
        a107 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        a108 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
        a109 = a108.T
        a110 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        a111 = B(a70,a110)
        a112 = a110.T
        a113 = B(a40,a108)
        a114 = a40.T
        a115 = B(a43,a108).T
        a116 = B(a73,a110)
        a117 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
        a118 = a117.T
        a119 = B(a40,a117)
        a120 = B(a43,a117).T
        a121 = config.ubar_SU1_rbr_lower_strut_jcr_strut
        a122 = config.ubar_SU1_rbr_upper_strut_jcr_strut
        a123 = (multi_dot([B(a70,a121),a70]) + -1.0*multi_dot([B(a40,a122),a40]))
        a124 = (self.Rd_SU1_rbr_upper_strut + -1.0*self.Rd_SU1_rbr_lower_strut + multi_dot([B(a73,a121),a70]) + multi_dot([B(a43,a122),a40]))
        a125 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([a122.T,a44]) + -1.0*multi_dot([a121.T,a74]))
        a126 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
        a127 = a126.T
        a128 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        a129 = B(a76,a128)
        a130 = a128.T
        a131 = B(a45,a126)
        a132 = a45.T
        a133 = B(a48,a126).T
        a134 = B(a78,a128)
        a135 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
        a136 = a135.T
        a137 = B(a45,a135)
        a138 = B(a48,a135).T
        a139 = config.ubar_SU1_rbl_lower_strut_jcl_strut
        a140 = config.ubar_SU1_rbl_upper_strut_jcl_strut
        a141 = (multi_dot([B(a76,a139),a76]) + -1.0*multi_dot([B(a45,a140),a45]))
        a142 = (self.Rd_SU1_rbl_upper_strut + -1.0*self.Rd_SU1_rbl_lower_strut + multi_dot([B(a78,a139),a76]) + multi_dot([B(a48,a140),a45]))
        a143 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([a140.T,a49]) + -1.0*multi_dot([a139.T,a79]))
        a144 = config.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]
        a145 = self.P_SU1_rbr_tie_rod
        a146 = config.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]
        a147 = a82.T
        a148 = config.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]
        a149 = self.P_SU1_rbl_tie_rod
        a150 = config.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]
        a151 = a95.T
        a152 = self.Pd_ST_rbs_coupler
        a153 = config.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
        a154 = a153.T
        a155 = config.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1]
        a156 = self.P_ST_rbs_coupler
        a157 = A(a156).T
        a158 = B(a59,a153)
        a159 = a152.T
        a160 = B(a64,a153)
        a161 = config.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2]

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_ground_jcr_uca_chassis),a0]) + -1.0*multi_dot([B(a1,config.ubar_SU1_rbr_uca_jcr_uca_chassis),a1])),(multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + 2.0*multi_dot([a10,B(a3,a2).T,a11,a1])),(multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + 2.0*multi_dot([a10,B(a3,a12).T,a11,a1])),(multi_dot([B(a0,config.ubar_ground_jcl_uca_chassis),a0]) + -1.0*multi_dot([B(a13,config.ubar_SU1_rbl_uca_jcl_uca_chassis),a13])),(multi_dot([a14.T,a4,a16,a13]) + multi_dot([a17,a19,B(a0,a14),a0]) + 2.0*multi_dot([a10,B(a3,a14).T,a20,a13])),(multi_dot([a21.T,a4,a16,a13]) + multi_dot([a17,a19,B(a0,a21),a0]) + 2.0*multi_dot([a10,B(a3,a21).T,a20,a13])),(multi_dot([B(a0,config.ubar_ground_jcr_lca_chassis),a0]) + -1.0*multi_dot([B(a22,config.ubar_SU1_rbr_lca_jcr_lca_chassis),a22])),(multi_dot([a23.T,a4,a25,a22]) + multi_dot([a26,a28,B(a0,a23),a0]) + 2.0*multi_dot([a10,B(a3,a23).T,a29,a22])),(multi_dot([a30.T,a4,a25,a22]) + multi_dot([a26,a28,B(a0,a30),a0]) + 2.0*multi_dot([a10,B(a3,a30).T,a29,a22])),(multi_dot([B(a0,config.ubar_ground_jcl_lca_chassis),a0]) + -1.0*multi_dot([B(a31,config.ubar_SU1_rbl_lca_jcl_lca_chassis),a31])),(multi_dot([a32.T,a4,a34,a31]) + multi_dot([a35,a37,B(a0,a32),a0]) + 2.0*multi_dot([a10,B(a3,a32).T,a38,a31])),(multi_dot([a39.T,a4,a34,a31]) + multi_dot([a35,a37,B(a0,a39),a0]) + 2.0*multi_dot([a10,B(a3,a39).T,a38,a31])),(multi_dot([B(a0,config.ubar_ground_jcr_strut_chassis),a0]) + -1.0*multi_dot([B(a40,config.ubar_SU1_rbr_upper_strut_jcr_strut_chassis),a40])),(multi_dot([a41.T,a4,B(a40,a42),a40]) + multi_dot([a42.T,a44,B(a0,a41),a0]) + 2.0*multi_dot([a10,B(a3,a41).T,B(a43,a42),a40])),(multi_dot([B(a0,config.ubar_ground_jcl_strut_chassis),a0]) + -1.0*multi_dot([B(a45,config.ubar_SU1_rbl_upper_strut_jcl_strut_chassis),a45])),(multi_dot([a46.T,a4,B(a45,a47),a45]) + multi_dot([a47.T,a49,B(a0,a46),a0]) + 2.0*multi_dot([a10,B(a3,a46).T,B(a48,a47),a45])),(multi_dot([B(a0,config.ubar_ground_jcr_rocker_ch),a0]) + -1.0*multi_dot([B(a50,config.ubar_ST_rbr_rocker_jcr_rocker_ch),a50])),(multi_dot([a51.T,a4,a53,a50]) + multi_dot([a54,a56,B(a0,a51),a0]) + 2.0*multi_dot([a10,B(a3,a51).T,a57,a50])),(multi_dot([a58.T,a4,a53,a50]) + multi_dot([a54,a56,B(a0,a58),a0]) + 2.0*multi_dot([a10,B(a3,a58).T,a57,a50])),(multi_dot([B(a0,config.ubar_ground_jcl_rocker_ch),a0]) + -1.0*multi_dot([B(a59,config.ubar_ST_rbl_rocker_jcl_rocker_ch),a59])),(multi_dot([a60.T,a4,a62,a59]) + multi_dot([a63,a65,B(a0,a60),a0]) + 2.0*multi_dot([a10,B(a3,a60).T,a66,a59])),(multi_dot([a67.T,a4,a62,a59]) + multi_dot([a63,a65,B(a0,a67),a0]) + 2.0*multi_dot([a10,B(a3,a67).T,a66,a59])),(multi_dot([B(a1,config.ubar_SU1_rbr_uca_jcr_uca_upright),a1]) + -1.0*multi_dot([B(a68,config.ubar_SU1_rbr_upright_jcr_uca_upright),a68])),(multi_dot([B(a13,config.ubar_SU1_rbl_uca_jcl_uca_upright),a13]) + -1.0*multi_dot([B(a69,config.ubar_SU1_rbl_upright_jcl_uca_upright),a69])),(multi_dot([B(a22,config.ubar_SU1_rbr_lca_jcr_lca_upright),a22]) + -1.0*multi_dot([B(a68,config.ubar_SU1_rbr_upright_jcr_lca_upright),a68])),(multi_dot([B(a22,config.ubar_SU1_rbr_lca_jcr_strut_lca),a22]) + -1.0*multi_dot([B(a70,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca),a70])),(multi_dot([a71.T,a28,B(a70,a72),a70]) + multi_dot([a72.T,a74,B(a22,a71),a22]) + 2.0*multi_dot([a75,B(a27,a71).T,B(a73,a72),a70])),(multi_dot([B(a31,config.ubar_SU1_rbl_lca_jcl_lca_upright),a31]) + -1.0*multi_dot([B(a69,config.ubar_SU1_rbl_upright_jcl_lca_upright),a69])),(multi_dot([B(a31,config.ubar_SU1_rbl_lca_jcl_strut_lca),a31]) + -1.0*multi_dot([B(a76,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca),a76])),(multi_dot([a77.T,a79,B(a31,a80),a31]) + multi_dot([a80.T,a37,B(a76,a77),a76]) + 2.0*multi_dot([a81,B(a36,a80).T,B(a78,a77),a76])),(multi_dot([B(a68,config.ubar_SU1_rbr_upright_jcr_tie_upright),a68]) + -1.0*multi_dot([B(a82,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright),a82])),(multi_dot([B(a68,config.ubar_SU1_rbr_upright_jcr_hub_bearing),a68]) + -1.0*multi_dot([B(a83,config.ubar_SU1_rbr_hub_jcr_hub_bearing),a83])),(multi_dot([a85,a87,B(a68,a88),a68]) + multi_dot([a88.T,a90,a91,a83]) + 2.0*multi_dot([a92,B(a89,a88).T,a93,a83])),(multi_dot([a85,a87,B(a68,a94),a68]) + multi_dot([a94.T,a90,a91,a83]) + 2.0*multi_dot([a92,B(a89,a94).T,a93,a83])),(multi_dot([B(a69,config.ubar_SU1_rbl_upright_jcl_tie_upright),a69]) + -1.0*multi_dot([B(a95,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright),a95])),(multi_dot([B(a69,config.ubar_SU1_rbl_upright_jcl_hub_bearing),a69]) + -1.0*multi_dot([B(a96,config.ubar_SU1_rbl_hub_jcl_hub_bearing),a96])),(multi_dot([a98,a100,B(a69,a101),a69]) + multi_dot([a101.T,a103,a104,a96]) + 2.0*multi_dot([a105,B(a102,a101).T,a106,a96])),(multi_dot([a98,a100,B(a69,a107),a69]) + multi_dot([a107.T,a103,a104,a96]) + 2.0*multi_dot([a105,B(a102,a107).T,a106,a96])),(multi_dot([a109,a44,a111,a70]) + multi_dot([a112,a74,a113,a40]) + 2.0*multi_dot([a114,a115,a116,a70])),(multi_dot([a118,a44,a111,a70]) + multi_dot([a112,a74,a119,a40]) + 2.0*multi_dot([a114,a120,a116,a70])),(multi_dot([a109,a44,a123]) + 2.0*multi_dot([a114,a115,a124]) + multi_dot([a125,a113,a40])),(multi_dot([a118,a44,a123]) + 2.0*multi_dot([a114,a120,a124]) + multi_dot([a125,a119,a40])),(multi_dot([a127,a49,a129,a76]) + multi_dot([a130,a79,a131,a45]) + 2.0*multi_dot([a132,a133,a134,a76])),(multi_dot([a136,a49,a129,a76]) + multi_dot([a130,a79,a137,a45]) + 2.0*multi_dot([a132,a138,a134,a76])),(multi_dot([a127,a49,a141]) + 2.0*multi_dot([a132,a133,a142]) + multi_dot([a143,a131,a45])),(multi_dot([a136,a49,a141]) + 2.0*multi_dot([a132,a138,a142]) + multi_dot([a143,a137,a45])),(multi_dot([B(a82,config.ubar_SU1_rbr_tie_rod_jcr_tie_steering),a82]) + -1.0*multi_dot([B(a50,config.ubar_ST_rbr_rocker_jcr_tie_steering),a50])),(multi_dot([a144.T,A(a145).T,B(a50,a146),a50]) + multi_dot([a146.T,a56,B(a82,a144),a82]) + 2.0*multi_dot([a147,B(a145,a144).T,B(a55,a146),a50])),(multi_dot([B(a95,config.ubar_SU1_rbl_tie_rod_jcl_tie_steering),a95]) + -1.0*multi_dot([B(a59,config.ubar_ST_rbl_rocker_jcl_tie_steering),a59])),(multi_dot([a148.T,A(a149).T,B(a59,a150),a59]) + multi_dot([a150.T,a65,B(a95,a148),a95]) + 2.0*multi_dot([a151,B(a149,a148).T,B(a64,a150),a59])),(multi_dot([B(a152,config.ubar_ST_rbs_coupler_jcs_rc_sph),a152]) + -1.0*multi_dot([B(a50,config.ubar_ST_rbr_rocker_jcs_rc_sph),a50])),(multi_dot([B(a152,config.ubar_ST_rbs_coupler_jcs_rc_uni),a152]) + -1.0*multi_dot([B(a59,config.ubar_ST_rbl_rocker_jcs_rc_uni),a59])),(multi_dot([a154,a65,B(a152,a155),a152]) + multi_dot([a155.T,a157,a158,a59]) + 2.0*multi_dot([a159,B(a156,a155).T,a160,a59])),(multi_dot([a154,a65,B(a152,a161),a152]) + multi_dot([a161.T,a157,a158,a59]) + 2.0*multi_dot([a159,B(a156,a161).T,a160,a59])),np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64),2.0*(multi_dot([a1.T,a1]))**(1.0/2.0),2.0*(multi_dot([a13.T,a13]))**(1.0/2.0),2.0*(multi_dot([a75,a22]))**(1.0/2.0),2.0*(multi_dot([a81,a31]))**(1.0/2.0),2.0*(multi_dot([a92,a68]))**(1.0/2.0),2.0*(multi_dot([a105,a69]))**(1.0/2.0),2.0*(multi_dot([a114,a40]))**(1.0/2.0),2.0*(multi_dot([a132,a45]))**(1.0/2.0),2.0*(multi_dot([a70.T,a70]))**(1.0/2.0),2.0*(multi_dot([a76.T,a76]))**(1.0/2.0),2.0*(multi_dot([a147,a82]))**(1.0/2.0),2.0*(multi_dot([a151,a95]))**(1.0/2.0),2.0*(multi_dot([a83.T,a83]))**(1.0/2.0),2.0*(multi_dot([a96.T,a96]))**(1.0/2.0),2.0*(multi_dot([a159,a152]))**(1.0/2.0),2.0*(multi_dot([a50.T,a50]))**(1.0/2.0),2.0*(multi_dot([a59.T,a59]))**(1.0/2.0)]

    
    def eval_jac_eq(self):

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = config.Mbar_SU1_rbr_uca_jcr_uca_chassis[:,2:3]
        j4 = j3.T
        j5 = self.P_SU1_rbr_uca
        j6 = A(j5).T
        j7 = config.Mbar_ground_jcr_uca_chassis[:,0:1]
        j8 = config.Mbar_ground_jcr_uca_chassis[:,1:2]
        j9 = -1.0*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = config.Mbar_SU1_rbl_uca_jcl_uca_chassis[:,2:3]
        j13 = j12.T
        j14 = self.P_SU1_rbl_uca
        j15 = A(j14).T
        j16 = config.Mbar_ground_jcl_uca_chassis[:,0:1]
        j17 = config.Mbar_ground_jcl_uca_chassis[:,1:2]
        j18 = B(j14,j12)
        j19 = config.Mbar_SU1_rbr_lca_jcr_lca_chassis[:,2:3]
        j20 = j19.T
        j21 = self.P_SU1_rbr_lca
        j22 = A(j21).T
        j23 = config.Mbar_ground_jcr_lca_chassis[:,0:1]
        j24 = config.Mbar_ground_jcr_lca_chassis[:,1:2]
        j25 = B(j21,j19)
        j26 = config.Mbar_SU1_rbl_lca_jcl_lca_chassis[:,2:3]
        j27 = j26.T
        j28 = self.P_SU1_rbl_lca
        j29 = A(j28).T
        j30 = config.Mbar_ground_jcl_lca_chassis[:,0:1]
        j31 = config.Mbar_ground_jcl_lca_chassis[:,1:2]
        j32 = B(j28,j26)
        j33 = config.Mbar_SU1_rbr_upper_strut_jcr_strut_chassis[:,0:1]
        j34 = self.P_SU1_rbr_upper_strut
        j35 = A(j34).T
        j36 = config.Mbar_ground_jcr_strut_chassis[:,0:1]
        j37 = config.Mbar_SU1_rbl_upper_strut_jcl_strut_chassis[:,0:1]
        j38 = self.P_SU1_rbl_upper_strut
        j39 = A(j38).T
        j40 = config.Mbar_ground_jcl_strut_chassis[:,0:1]
        j41 = config.Mbar_ST_rbr_rocker_jcr_rocker_ch[:,2:3]
        j42 = j41.T
        j43 = self.P_ST_rbr_rocker
        j44 = A(j43).T
        j45 = config.Mbar_ground_jcr_rocker_ch[:,0:1]
        j46 = config.Mbar_ground_jcr_rocker_ch[:,1:2]
        j47 = B(j43,j41)
        j48 = config.Mbar_ST_rbl_rocker_jcl_rocker_ch[:,2:3]
        j49 = j48.T
        j50 = self.P_ST_rbl_rocker
        j51 = A(j50).T
        j52 = config.Mbar_ground_jcl_rocker_ch[:,0:1]
        j53 = config.Mbar_ground_jcl_rocker_ch[:,1:2]
        j54 = B(j50,j48)
        j55 = self.P_SU1_rbr_upright
        j56 = self.P_SU1_rbl_upright
        j57 = config.Mbar_SU1_rbr_lower_strut_jcr_strut_lca[:,0:1]
        j58 = self.P_SU1_rbr_lower_strut
        j59 = A(j58).T
        j60 = config.Mbar_SU1_rbr_lca_jcr_strut_lca[:,0:1]
        j61 = config.Mbar_SU1_rbl_lower_strut_jcl_strut_lca[:,0:1]
        j62 = self.P_SU1_rbl_lower_strut
        j63 = A(j62).T
        j64 = config.Mbar_SU1_rbl_lca_jcl_strut_lca[:,0:1]
        j65 = self.P_SU1_rbr_tie_rod
        j66 = config.Mbar_SU1_rbr_hub_jcr_hub_bearing[:,2:3]
        j67 = j66.T
        j68 = self.P_SU1_rbr_hub
        j69 = A(j68).T
        j70 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,0:1]
        j71 = config.Mbar_SU1_rbr_upright_jcr_hub_bearing[:,1:2]
        j72 = A(j55).T
        j73 = B(j68,j66)
        j74 = self.P_SU1_rbl_tie_rod
        j75 = config.Mbar_SU1_rbl_hub_jcl_hub_bearing[:,2:3]
        j76 = j75.T
        j77 = self.P_SU1_rbl_hub
        j78 = A(j77).T
        j79 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,0:1]
        j80 = config.Mbar_SU1_rbl_upright_jcl_hub_bearing[:,1:2]
        j81 = A(j56).T
        j82 = B(j77,j75)
        j83 = config.Mbar_SU1_rbr_lower_strut_jcr_strut[:,2:3]
        j84 = j83.T
        j85 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,0:1]
        j86 = B(j34,j85)
        j87 = config.Mbar_SU1_rbr_upper_strut_jcr_strut[:,1:2]
        j88 = B(j34,j87)
        j89 = j85.T
        j90 = multi_dot([j89,j35])
        j91 = config.ubar_SU1_rbr_upper_strut_jcr_strut
        j92 = B(j34,j91)
        j93 = config.ubar_SU1_rbr_lower_strut_jcr_strut
        j94 = (self.R_SU1_rbr_upper_strut.T + -1.0*self.R_SU1_rbr_lower_strut.T + multi_dot([j91.T,j35]) + -1.0*multi_dot([j93.T,j59]))
        j95 = j87.T
        j96 = multi_dot([j95,j35])
        j97 = B(j58,j83)
        j98 = B(j58,j93)
        j99 = config.Mbar_SU1_rbl_lower_strut_jcl_strut[:,2:3]
        j100 = j99.T
        j101 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,0:1]
        j102 = B(j38,j101)
        j103 = config.Mbar_SU1_rbl_upper_strut_jcl_strut[:,1:2]
        j104 = B(j38,j103)
        j105 = j101.T
        j106 = multi_dot([j105,j39])
        j107 = config.ubar_SU1_rbl_upper_strut_jcl_strut
        j108 = B(j38,j107)
        j109 = config.ubar_SU1_rbl_lower_strut_jcl_strut
        j110 = (self.R_SU1_rbl_upper_strut.T + -1.0*self.R_SU1_rbl_lower_strut.T + multi_dot([j107.T,j39]) + -1.0*multi_dot([j109.T,j63]))
        j111 = j103.T
        j112 = multi_dot([j111,j39])
        j113 = B(j62,j99)
        j114 = B(j62,j109)
        j115 = config.Mbar_ST_rbr_rocker_jcr_tie_steering[:,0:1]
        j116 = config.Mbar_SU1_rbr_tie_rod_jcr_tie_steering[:,0:1]
        j117 = config.Mbar_ST_rbl_rocker_jcl_tie_steering[:,0:1]
        j118 = config.Mbar_SU1_rbl_tie_rod_jcl_tie_steering[:,0:1]
        j119 = self.P_ST_rbs_coupler
        j120 = config.Mbar_ST_rbl_rocker_jcs_rc_uni[:,2:3]
        j121 = j120.T
        j122 = config.Mbar_ST_rbs_coupler_jcs_rc_uni[:,0:1]
        j123 = config.Mbar_ST_rbs_coupler_jcs_rc_uni[:,1:2]
        j124 = A(j119).T
        j125 = B(j50,j120)

        self.jac_eq_blocks = [j0,B(j1,config.ubar_ground_jcr_uca_chassis),j9,-1.0*B(j5,config.ubar_SU1_rbr_uca_jcr_uca_chassis),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j0,B(j1,config.ubar_ground_jcl_uca_chassis),j9,-1.0*B(j14,config.ubar_SU1_rbl_uca_jcl_uca_chassis),j2,multi_dot([j13,j15,B(j1,j16)]),j2,multi_dot([j16.T,j10,j18]),j2,multi_dot([j13,j15,B(j1,j17)]),j2,multi_dot([j17.T,j10,j18]),j0,B(j1,config.ubar_ground_jcr_lca_chassis),j9,-1.0*B(j21,config.ubar_SU1_rbr_lca_jcr_lca_chassis),j2,multi_dot([j20,j22,B(j1,j23)]),j2,multi_dot([j23.T,j10,j25]),j2,multi_dot([j20,j22,B(j1,j24)]),j2,multi_dot([j24.T,j10,j25]),j0,B(j1,config.ubar_ground_jcl_lca_chassis),j9,-1.0*B(j28,config.ubar_SU1_rbl_lca_jcl_lca_chassis),j2,multi_dot([j27,j29,B(j1,j30)]),j2,multi_dot([j30.T,j10,j32]),j2,multi_dot([j27,j29,B(j1,j31)]),j2,multi_dot([j31.T,j10,j32]),j0,B(j1,config.ubar_ground_jcr_strut_chassis),j9,-1.0*B(j34,config.ubar_SU1_rbr_upper_strut_jcr_strut_chassis),j2,multi_dot([j33.T,j35,B(j1,j36)]),j2,multi_dot([j36.T,j10,B(j34,j33)]),j0,B(j1,config.ubar_ground_jcl_strut_chassis),j9,-1.0*B(j38,config.ubar_SU1_rbl_upper_strut_jcl_strut_chassis),j2,multi_dot([j37.T,j39,B(j1,j40)]),j2,multi_dot([j40.T,j10,B(j38,j37)]),j0,B(j1,config.ubar_ground_jcr_rocker_ch),j9,-1.0*B(j43,config.ubar_ST_rbr_rocker_jcr_rocker_ch),j2,multi_dot([j42,j44,B(j1,j45)]),j2,multi_dot([j45.T,j10,j47]),j2,multi_dot([j42,j44,B(j1,j46)]),j2,multi_dot([j46.T,j10,j47]),j0,B(j1,config.ubar_ground_jcl_rocker_ch),j9,-1.0*B(j50,config.ubar_ST_rbl_rocker_jcl_rocker_ch),j2,multi_dot([j49,j51,B(j1,j52)]),j2,multi_dot([j52.T,j10,j54]),j2,multi_dot([j49,j51,B(j1,j53)]),j2,multi_dot([j53.T,j10,j54]),j0,B(j5,config.ubar_SU1_rbr_uca_jcr_uca_upright),j9,-1.0*B(j55,config.ubar_SU1_rbr_upright_jcr_uca_upright),j0,B(j14,config.ubar_SU1_rbl_uca_jcl_uca_upright),j9,-1.0*B(j56,config.ubar_SU1_rbl_upright_jcl_uca_upright),j0,B(j21,config.ubar_SU1_rbr_lca_jcr_lca_upright),j9,-1.0*B(j55,config.ubar_SU1_rbr_upright_jcr_lca_upright),j0,B(j21,config.ubar_SU1_rbr_lca_jcr_strut_lca),j9,-1.0*B(j58,config.ubar_SU1_rbr_lower_strut_jcr_strut_lca),j2,multi_dot([j57.T,j59,B(j21,j60)]),j2,multi_dot([j60.T,j22,B(j58,j57)]),j0,B(j28,config.ubar_SU1_rbl_lca_jcl_lca_upright),j9,-1.0*B(j56,config.ubar_SU1_rbl_upright_jcl_lca_upright),j0,B(j28,config.ubar_SU1_rbl_lca_jcl_strut_lca),j9,-1.0*B(j62,config.ubar_SU1_rbl_lower_strut_jcl_strut_lca),j2,multi_dot([j61.T,j63,B(j28,j64)]),j2,multi_dot([j64.T,j29,B(j62,j61)]),j0,B(j55,config.ubar_SU1_rbr_upright_jcr_tie_upright),j9,-1.0*B(j65,config.ubar_SU1_rbr_tie_rod_jcr_tie_upright),j0,B(j55,config.ubar_SU1_rbr_upright_jcr_hub_bearing),j9,-1.0*B(j68,config.ubar_SU1_rbr_hub_jcr_hub_bearing),j2,multi_dot([j67,j69,B(j55,j70)]),j2,multi_dot([j70.T,j72,j73]),j2,multi_dot([j67,j69,B(j55,j71)]),j2,multi_dot([j71.T,j72,j73]),j0,B(j56,config.ubar_SU1_rbl_upright_jcl_tie_upright),j9,-1.0*B(j74,config.ubar_SU1_rbl_tie_rod_jcl_tie_upright),j0,B(j56,config.ubar_SU1_rbl_upright_jcl_hub_bearing),j9,-1.0*B(j77,config.ubar_SU1_rbl_hub_jcl_hub_bearing),j2,multi_dot([j76,j78,B(j56,j79)]),j2,multi_dot([j79.T,j81,j82]),j2,multi_dot([j76,j78,B(j56,j80)]),j2,multi_dot([j80.T,j81,j82]),j2,multi_dot([j84,j59,j86]),j2,multi_dot([j89,j35,j97]),j2,multi_dot([j84,j59,j88]),j2,multi_dot([j95,j35,j97]),-1.0*j90,(-1.0*multi_dot([j89,j35,j92]) + multi_dot([j94,j86])),j90,multi_dot([j89,j35,j98]),-1.0*j96,(-1.0*multi_dot([j95,j35,j92]) + multi_dot([j94,j88])),j96,multi_dot([j95,j35,j98]),j2,multi_dot([j100,j63,j102]),j2,multi_dot([j105,j39,j113]),j2,multi_dot([j100,j63,j104]),j2,multi_dot([j111,j39,j113]),-1.0*j106,(-1.0*multi_dot([j105,j39,j108]) + multi_dot([j110,j102])),j106,multi_dot([j105,j39,j114]),-1.0*j112,(-1.0*multi_dot([j111,j39,j108]) + multi_dot([j110,j104])),j112,multi_dot([j111,j39,j114]),j0,B(j65,config.ubar_SU1_rbr_tie_rod_jcr_tie_steering),j9,-1.0*B(j43,config.ubar_ST_rbr_rocker_jcr_tie_steering),j2,multi_dot([j115.T,j44,B(j65,j116)]),j2,multi_dot([j116.T,A(j65).T,B(j43,j115)]),j0,B(j74,config.ubar_SU1_rbl_tie_rod_jcl_tie_steering),j9,-1.0*B(j50,config.ubar_ST_rbl_rocker_jcl_tie_steering),j2,multi_dot([j117.T,j51,B(j74,j118)]),j2,multi_dot([j118.T,A(j74).T,B(j50,j117)]),j0,B(j119,config.ubar_ST_rbs_coupler_jcs_rc_sph),j9,-1.0*B(j43,config.ubar_ST_rbr_rocker_jcs_rc_sph),j0,B(j119,config.ubar_ST_rbs_coupler_jcs_rc_uni),j9,-1.0*B(j50,config.ubar_ST_rbl_rocker_jcs_rc_uni),j2,multi_dot([j121,j51,B(j119,j122)]),j2,multi_dot([j122.T,j124,j125]),j2,multi_dot([j121,j51,B(j119,j123)]),j2,multi_dot([j123.T,j124,j125]),j0,np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64),2.0*j5.T,2.0*j14.T,2.0*j21.T,2.0*j28.T,2.0*j55.T,2.0*j56.T,2.0*j34.T,2.0*j38.T,2.0*j58.T,2.0*j62.T,2.0*j65.T,2.0*j74.T,2.0*j68.T,2.0*j77.T,2.0*j119.T,2.0*j43.T,2.0*j50.T]
  
