
import numpy as np
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, G, E, triad, skew_matrix as skew



class topology(object):

    def __init__(self,prefix=''):
        self.t = 0.0
        self.prefix = (prefix if prefix=='' else prefix+'.')
        self.config = None

        self.n  = 0
        self.nc = 3
        self.nrows = 3
        self.ncols = 2*7
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_vbr_hub_mcr_ver_act', 'T_vbr_hub_mcr_ver_act', 'F_vbl_hub_mcl_ver_act', 'T_vbl_hub_mcl_ver_act', 'F_vbs_steer_gear_jcs_steer_gear', 'T_vbs_steer_gear_jcs_steer_gear']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
    
        self.vbr_upright = indicies_map[interface_map[p+'vbr_upright']]
        self.vbl_hub = indicies_map[interface_map[p+'vbl_hub']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbs_steer_gear = indicies_map[interface_map[p+'vbs_steer_gear']]
        self.vbl_upright = indicies_map[interface_map[p+'vbl_upright']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbr_hub = indicies_map[interface_map[p+'vbr_hub']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.vbs_ground*2, self.vbs_ground*2+1, self.vbr_hub*2, self.vbr_hub*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.vbl_hub*2, self.vbl_hub*2+1, self.vbs_steer_gear*2, self.vbs_steer_gear*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

    

        self.ubar_vbr_hub_mcr_ver_act = (multi_dot([A(config.P_vbr_hub).T,config.pt1_mcr_ver_act]) + -1*multi_dot([A(config.P_vbr_hub).T,config.R_vbr_hub]))
        self.ubar_vbs_ground_mcr_ver_act = (multi_dot([A(config.P_vbs_ground).T,config.pt1_mcr_ver_act]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_vbl_hub_mcl_ver_act = (multi_dot([A(config.P_vbl_hub).T,config.pt1_mcl_ver_act]) + -1*multi_dot([A(config.P_vbl_hub).T,config.R_vbl_hub]))
        self.ubar_vbs_ground_mcl_ver_act = (multi_dot([A(config.P_vbs_ground).T,config.pt1_mcl_ver_act]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.Mbar_vbs_steer_gear_jcs_steer_gear = multi_dot([A(config.P_vbs_steer_gear).T,triad(config.ax1_jcs_steer_gear)])
        self.Mbar_vbs_chassis_jcs_steer_gear = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcs_steer_gear)])

    
    def set_gen_coordinates(self,q):
        pass

    
    def set_gen_velocities(self,qd):
        pass

    
    def set_gen_accelerations(self,qdd):
        pass

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_mcr_ver_act = Lambda[0:1,0:1]
        self.L_mcl_ver_act = Lambda[1:2,0:1]
        self.L_jcs_steer_gear = Lambda[2:3,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = np.eye(1,dtype=np.float64)
        x1 = A(self.P_vbs_steer_gear).T
        x2 = A(self.P_vbs_chassis)
        x3 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]

        self.pos_eq_blocks = [(config.AF_mcr_ver_act(t)*-1*x0 + (self.R_vbr_hub + -1*config.pt1_mcr_ver_act + multi_dot([A(self.P_vbr_hub),self.ubar_vbr_hub_mcr_ver_act]))[2:3,0:1]),
        (config.AF_mcl_ver_act(t)*-1*x0 + (self.R_vbl_hub + -1*config.pt1_mcl_ver_act + multi_dot([A(self.P_vbl_hub),self.ubar_vbl_hub_mcl_ver_act]))[2:3,0:1]),
        (cos(config.AF_jcs_steer_gear(t))*multi_dot([self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2].T,x1,x2,x3]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1].T,x1,x2,x3]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [derivative(config.AF_mcr_ver_act,t,0.0634665182543,1)*-1*v0,
        derivative(config.AF_mcl_ver_act,t,0.0634665182543,1)*-1*v0,
        derivative(config.AF_jcs_steer_gear,t,0.0634665182543,1)*-1*v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1,dtype=np.float64)
        a1 = self.Pd_vbr_hub
        a2 = self.Pd_vbl_hub
        a3 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        a4 = self.P_vbs_chassis
        a5 = self.Pd_vbs_steer_gear
        a6 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        a7 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        a8 = self.P_vbs_steer_gear
        a9 = A(a8).T
        a10 = self.Pd_vbs_chassis

        self.acc_eq_blocks = [(derivative(config.AF_mcr_ver_act,t,0.0634665182543,2)*-1*a0 + multi_dot([B(a1,self.ubar_vbr_hub_mcr_ver_act),a1])[2:3,0:1]),
        (derivative(config.AF_mcl_ver_act,t,0.0634665182543,2)*-1*a0 + multi_dot([B(a2,self.ubar_vbl_hub_mcl_ver_act),a2])[2:3,0:1]),
        (derivative(config.AF_jcs_steer_gear,t,0.0634665182543,2)*-1*a0 + multi_dot([a3.T,A(a4).T,(cos(config.AF_jcs_steer_gear(t))*B(a5,a6) + sin(config.AF_jcs_steer_gear(t))*-1*B(a5,a7)),a5]) + multi_dot([(cos(config.AF_jcs_steer_gear(t))*multi_dot([a6.T,a9]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([a7.T,a9])),B(a10,a3),a10]) + 2*multi_dot([((cos(config.AF_jcs_steer_gear(t))*multi_dot([B(a8,a6),a5])).T + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([a5.T,B(a8,a7).T])),B(a4,a3),a10]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)[2:3,0:3]
        j1 = np.zeros((1,3),dtype=np.float64)
        j2 = np.zeros((1,4),dtype=np.float64)
        j3 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        j4 = self.P_vbs_chassis
        j5 = self.P_vbs_steer_gear
        j6 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        j7 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        j8 = A(j5).T

        self.jac_eq_blocks = [j1,
        j2,
        j0,
        B(self.P_vbr_hub,self.ubar_vbr_hub_mcr_ver_act)[2:3,0:4],
        j1,
        j2,
        j0,
        B(self.P_vbl_hub,self.ubar_vbl_hub_mcl_ver_act)[2:3,0:4],
        j1,
        multi_dot([j3.T,A(j4).T,(cos(config.AF_jcs_steer_gear(t))*B(j5,j6) + sin(config.AF_jcs_steer_gear(t))*-1*B(j5,j7))]),
        j1,
        multi_dot([(cos(config.AF_jcs_steer_gear(t))*multi_dot([j6.T,j8]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([j7.T,j8])),B(j4,j3)])]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

    

        self.mass_eq_blocks = []

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

    

        self.frc_eq_blocks = []

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_vbr_hub_mcr_ver_act = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)[2:3,0:3].T],[B(self.P_vbr_hub,self.ubar_vbr_hub_mcr_ver_act)[2:3,0:4].T]]),self.L_mcr_ver_act])
        self.F_vbr_hub_mcr_ver_act = Q_vbr_hub_mcr_ver_act[0:3,0:1]
        Te_vbr_hub_mcr_ver_act = Q_vbr_hub_mcr_ver_act[3:7,0:1]
        self.T_vbr_hub_mcr_ver_act = (-1*multi_dot([skew(multi_dot([A(self.P_vbr_hub),self.ubar_vbr_hub_mcr_ver_act])),self.F_vbr_hub_mcr_ver_act]) + 0.5*multi_dot([E(self.P_vbr_hub),Te_vbr_hub_mcr_ver_act]))
        Q_vbl_hub_mcl_ver_act = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)[2:3,0:3].T],[B(self.P_vbl_hub,self.ubar_vbl_hub_mcl_ver_act)[2:3,0:4].T]]),self.L_mcl_ver_act])
        self.F_vbl_hub_mcl_ver_act = Q_vbl_hub_mcl_ver_act[0:3,0:1]
        Te_vbl_hub_mcl_ver_act = Q_vbl_hub_mcl_ver_act[3:7,0:1]
        self.T_vbl_hub_mcl_ver_act = (-1*multi_dot([skew(multi_dot([A(self.P_vbl_hub),self.ubar_vbl_hub_mcl_ver_act])),self.F_vbl_hub_mcl_ver_act]) + 0.5*multi_dot([E(self.P_vbl_hub),Te_vbl_hub_mcl_ver_act]))
        Q_vbs_steer_gear_jcs_steer_gear = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcs_steer_gear(t))*B(self.P_vbs_steer_gear,self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]).T + (cos(config.AF_jcs_steer_gear(t))*B(self.P_vbs_steer_gear,self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2])).T),A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]])]]),self.L_jcs_steer_gear])
        self.F_vbs_steer_gear_jcs_steer_gear = Q_vbs_steer_gear_jcs_steer_gear[0:3,0:1]
        Te_vbs_steer_gear_jcs_steer_gear = Q_vbs_steer_gear_jcs_steer_gear[3:7,0:1]
        self.T_vbs_steer_gear_jcs_steer_gear = 0.5*multi_dot([E(self.P_vbs_steer_gear),Te_vbs_steer_gear_jcs_steer_gear])

        self.reactions = {'F_vbr_hub_mcr_ver_act':self.F_vbr_hub_mcr_ver_act,'T_vbr_hub_mcr_ver_act':self.T_vbr_hub_mcr_ver_act,'F_vbl_hub_mcl_ver_act':self.F_vbl_hub_mcl_ver_act,'T_vbl_hub_mcl_ver_act':self.T_vbl_hub_mcl_ver_act,'F_vbs_steer_gear_jcs_steer_gear':self.F_vbs_steer_gear_jcs_steer_gear,'T_vbs_steer_gear_jcs_steer_gear':self.T_vbs_steer_gear_jcs_steer_gear}

