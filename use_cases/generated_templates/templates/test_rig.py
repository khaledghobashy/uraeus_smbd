
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
        self.nc = 5
        self.nrows = 5
        self.ncols = 2*7
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_vbr_hub_mcr_ver_act', 'T_vbr_hub_mcr_ver_act', 'F_vbl_hub_mcl_ver_act', 'T_vbl_hub_mcl_ver_act', 'F_vbr_upright_jcr_rev', 'T_vbr_upright_jcr_rev', 'F_vbl_upright_jcl_rev', 'T_vbl_upright_jcl_rev', 'F_vbs_steer_gear_jcs_steer_gear', 'T_vbs_steer_gear_jcs_steer_gear']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.vbs_ground*2, self.vbs_ground*2+1, self.vbr_hub*2, self.vbr_hub*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.vbl_hub*2, self.vbl_hub*2+1, self.vbr_hub*2, self.vbr_hub*2+1, self.vbr_upright*2, self.vbr_upright*2+1, self.vbl_hub*2, self.vbl_hub*2+1, self.vbl_upright*2, self.vbl_upright*2+1, self.vbs_steer_gear*2, self.vbs_steer_gear*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

    

        self.ubar_vbr_hub_mcr_ver_act = (multi_dot([A(config.P_vbr_hub).T,config.pt1_mcr_ver_act]) + -1*multi_dot([A(config.P_vbr_hub).T,config.R_vbr_hub]))
        self.ubar_vbs_ground_mcr_ver_act = (multi_dot([A(config.P_vbs_ground).T,config.pt1_mcr_ver_act]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.ubar_vbl_hub_mcl_ver_act = (multi_dot([A(config.P_vbl_hub).T,config.pt1_mcl_ver_act]) + -1*multi_dot([A(config.P_vbl_hub).T,config.R_vbl_hub]))
        self.ubar_vbs_ground_mcl_ver_act = (multi_dot([A(config.P_vbs_ground).T,config.pt1_mcl_ver_act]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))
        self.Mbar_vbr_upright_jcr_rev = multi_dot([A(config.P_vbr_upright).T,triad(config.ax1_jcr_rev)])
        self.Mbar_vbr_hub_jcr_rev = multi_dot([A(config.P_vbr_hub).T,triad(config.ax1_jcr_rev)])
        self.Mbar_vbl_upright_jcl_rev = multi_dot([A(config.P_vbl_upright).T,triad(config.ax1_jcl_rev)])
        self.Mbar_vbl_hub_jcl_rev = multi_dot([A(config.P_vbl_hub).T,triad(config.ax1_jcl_rev)])
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
        self.L_jcr_rev = Lambda[2:3,0:1]
        self.L_jcl_rev = Lambda[3:4,0:1]
        self.L_jcs_steer_gear = Lambda[4:5,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = np.eye(1,dtype=np.float64)
        x1 = A(self.P_vbr_hub)
        x2 = A(self.P_vbl_hub)
        x3 = A(self.P_vbr_upright).T
        x4 = self.Mbar_vbr_hub_jcr_rev[:,0:1]
        x5 = A(self.P_vbl_upright).T
        x6 = self.Mbar_vbl_hub_jcl_rev[:,0:1]
        x7 = A(self.P_vbs_steer_gear).T
        x8 = A(self.P_vbs_chassis)
        x9 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]

        self.pos_eq_blocks = [(config.AF_mcr_ver_act(t)*-1*x0 + (self.R_vbr_hub + -1*config.pt1_mcr_ver_act + multi_dot([x1,self.ubar_vbr_hub_mcr_ver_act]))[2:3,0:1]),
        (config.AF_mcl_ver_act(t)*-1*x0 + (self.R_vbl_hub + -1*config.pt1_mcl_ver_act + multi_dot([x2,self.ubar_vbl_hub_mcl_ver_act]))[2:3,0:1]),
        (cos(config.AF_jcr_rev(t))*multi_dot([self.Mbar_vbr_upright_jcr_rev[:,1:2].T,x3,x1,x4]) + sin(config.AF_jcr_rev(t))*-1*multi_dot([self.Mbar_vbr_upright_jcr_rev[:,0:1].T,x3,x1,x4])),
        (cos(config.AF_jcl_rev(t))*multi_dot([self.Mbar_vbl_upright_jcl_rev[:,1:2].T,x5,x2,x6]) + sin(config.AF_jcl_rev(t))*-1*multi_dot([self.Mbar_vbl_upright_jcl_rev[:,0:1].T,x5,x2,x6])),
        (cos(config.AF_jcs_steer_gear(t))*multi_dot([self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2].T,x7,x8,x9]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1].T,x7,x8,x9]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [derivative(config.AF_mcr_ver_act,t,0.1,1)*-1*v0,
        derivative(config.AF_mcl_ver_act,t,0.1,1)*-1*v0,
        derivative(config.AF_jcr_rev,t,0.1,1)*v0,
        derivative(config.AF_jcl_rev,t,0.1,1)*v0,
        derivative(config.AF_jcs_steer_gear,t,0.1,1)*v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1,dtype=np.float64)
        a1 = self.Pd_vbr_hub
        a2 = self.Pd_vbl_hub
        a3 = self.Mbar_vbr_hub_jcr_rev[:,0:1]
        a4 = self.P_vbr_hub
        a5 = self.Pd_vbr_upright
        a6 = self.Mbar_vbr_upright_jcr_rev[:,1:2]
        a7 = self.Mbar_vbr_upright_jcr_rev[:,0:1]
        a8 = self.P_vbr_upright
        a9 = A(a8).T
        a10 = self.Mbar_vbl_hub_jcl_rev[:,0:1]
        a11 = self.P_vbl_hub
        a12 = self.Pd_vbl_upright
        a13 = self.Mbar_vbl_upright_jcl_rev[:,1:2]
        a14 = self.Mbar_vbl_upright_jcl_rev[:,0:1]
        a15 = self.P_vbl_upright
        a16 = A(a15).T
        a17 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        a18 = self.P_vbs_chassis
        a19 = self.Pd_vbs_steer_gear
        a20 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        a21 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        a22 = self.P_vbs_steer_gear
        a23 = A(a22).T
        a24 = self.Pd_vbs_chassis

        self.acc_eq_blocks = [(derivative(config.AF_mcr_ver_act,t,0.1,2)*-1*a0 + multi_dot([B(a1,self.ubar_vbr_hub_mcr_ver_act),a1])[2:3,0:1]),
        (derivative(config.AF_mcl_ver_act,t,0.1,2)*-1*a0 + multi_dot([B(a2,self.ubar_vbl_hub_mcl_ver_act),a2])[2:3,0:1]),
        (derivative(config.AF_jcr_rev,t,0.1,2)*a0 + multi_dot([a3.T,A(a4).T,(cos(config.AF_jcr_rev(t))*B(a5,a6) + sin(config.AF_jcr_rev(t))*-1*B(a5,a7)),a5]) + multi_dot([(cos(config.AF_jcr_rev(t))*multi_dot([a6.T,a9]) + sin(config.AF_jcr_rev(t))*-1*multi_dot([a7.T,a9])),B(a1,a3),a1]) + 2*multi_dot([((cos(config.AF_jcr_rev(t))*multi_dot([B(a8,a6),a5])).T + sin(config.AF_jcr_rev(t))*-1*multi_dot([a5.T,B(a8,a7).T])),B(a4,a3),a1])),
        (derivative(config.AF_jcl_rev,t,0.1,2)*a0 + multi_dot([a10.T,A(a11).T,(cos(config.AF_jcl_rev(t))*B(a12,a13) + sin(config.AF_jcl_rev(t))*-1*B(a12,a14)),a12]) + multi_dot([(cos(config.AF_jcl_rev(t))*multi_dot([a13.T,a16]) + sin(config.AF_jcl_rev(t))*-1*multi_dot([a14.T,a16])),B(a2,a10),a2]) + 2*multi_dot([((cos(config.AF_jcl_rev(t))*multi_dot([B(a15,a13),a12])).T + sin(config.AF_jcl_rev(t))*-1*multi_dot([a12.T,B(a15,a14).T])),B(a11,a10),a2])),
        (derivative(config.AF_jcs_steer_gear,t,0.1,2)*a0 + multi_dot([a17.T,A(a18).T,(cos(config.AF_jcs_steer_gear(t))*B(a19,a20) + sin(config.AF_jcs_steer_gear(t))*-1*B(a19,a21)),a19]) + multi_dot([(cos(config.AF_jcs_steer_gear(t))*multi_dot([a20.T,a23]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([a21.T,a23])),B(a24,a17),a24]) + 2*multi_dot([((cos(config.AF_jcs_steer_gear(t))*multi_dot([B(a22,a20),a19])).T + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([a19.T,B(a22,a21).T])),B(a18,a17),a24]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)[2:3,0:3]
        j1 = self.P_vbr_hub
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = np.zeros((1,4),dtype=np.float64)
        j4 = self.P_vbl_hub
        j5 = self.Mbar_vbr_hub_jcr_rev[:,0:1]
        j6 = self.P_vbr_upright
        j7 = self.Mbar_vbr_upright_jcr_rev[:,1:2]
        j8 = self.Mbar_vbr_upright_jcr_rev[:,0:1]
        j9 = A(j6).T
        j10 = self.Mbar_vbl_hub_jcl_rev[:,0:1]
        j11 = self.P_vbl_upright
        j12 = self.Mbar_vbl_upright_jcl_rev[:,1:2]
        j13 = self.Mbar_vbl_upright_jcl_rev[:,0:1]
        j14 = A(j11).T
        j15 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        j16 = self.P_vbs_chassis
        j17 = self.P_vbs_steer_gear
        j18 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        j19 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        j20 = A(j17).T

        self.jac_eq_blocks = [j2,
        j3,
        j0,
        B(j1,self.ubar_vbr_hub_mcr_ver_act)[2:3,0:4],
        j2,
        j3,
        j0,
        B(j4,self.ubar_vbl_hub_mcl_ver_act)[2:3,0:4],
        j2,
        multi_dot([(cos(config.AF_jcr_rev(t))*multi_dot([j7.T,j9]) + sin(config.AF_jcr_rev(t))*-1*multi_dot([j8.T,j9])),B(j1,j5)]),
        j2,
        multi_dot([j5.T,A(j1).T,(cos(config.AF_jcr_rev(t))*B(j6,j7) + sin(config.AF_jcr_rev(t))*-1*B(j6,j8))]),
        j2,
        multi_dot([(cos(config.AF_jcl_rev(t))*multi_dot([j12.T,j14]) + sin(config.AF_jcl_rev(t))*-1*multi_dot([j13.T,j14])),B(j4,j10)]),
        j2,
        multi_dot([j10.T,A(j4).T,(cos(config.AF_jcl_rev(t))*B(j11,j12) + sin(config.AF_jcl_rev(t))*-1*B(j11,j13))]),
        j2,
        multi_dot([j15.T,A(j16).T,(cos(config.AF_jcs_steer_gear(t))*B(j17,j18) + sin(config.AF_jcs_steer_gear(t))*-1*B(j17,j19))]),
        j2,
        multi_dot([(cos(config.AF_jcs_steer_gear(t))*multi_dot([j18.T,j20]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([j19.T,j20])),B(j16,j15)])]

    
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
        Q_vbr_upright_jcr_rev = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcr_rev(t))*B(self.P_vbr_upright,self.Mbar_vbr_upright_jcr_rev[:,0:1]).T + (cos(config.AF_jcr_rev(t))*B(self.P_vbr_upright,self.Mbar_vbr_upright_jcr_rev[:,1:2])).T),A(self.P_vbr_hub),self.Mbar_vbr_hub_jcr_rev[:,0:1]])]]),self.L_jcr_rev])
        self.F_vbr_upright_jcr_rev = Q_vbr_upright_jcr_rev[0:3,0:1]
        Te_vbr_upright_jcr_rev = Q_vbr_upright_jcr_rev[3:7,0:1]
        self.T_vbr_upright_jcr_rev = 0.5*multi_dot([E(self.P_vbr_upright),Te_vbr_upright_jcr_rev])
        Q_vbl_upright_jcl_rev = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcl_rev(t))*B(self.P_vbl_upright,self.Mbar_vbl_upright_jcl_rev[:,0:1]).T + (cos(config.AF_jcl_rev(t))*B(self.P_vbl_upright,self.Mbar_vbl_upright_jcl_rev[:,1:2])).T),A(self.P_vbl_hub),self.Mbar_vbl_hub_jcl_rev[:,0:1]])]]),self.L_jcl_rev])
        self.F_vbl_upright_jcl_rev = Q_vbl_upright_jcl_rev[0:3,0:1]
        Te_vbl_upright_jcl_rev = Q_vbl_upright_jcl_rev[3:7,0:1]
        self.T_vbl_upright_jcl_rev = 0.5*multi_dot([E(self.P_vbl_upright),Te_vbl_upright_jcl_rev])
        Q_vbs_steer_gear_jcs_steer_gear = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcs_steer_gear(t))*B(self.P_vbs_steer_gear,self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]).T + (cos(config.AF_jcs_steer_gear(t))*B(self.P_vbs_steer_gear,self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2])).T),A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]])]]),self.L_jcs_steer_gear])
        self.F_vbs_steer_gear_jcs_steer_gear = Q_vbs_steer_gear_jcs_steer_gear[0:3,0:1]
        Te_vbs_steer_gear_jcs_steer_gear = Q_vbs_steer_gear_jcs_steer_gear[3:7,0:1]
        self.T_vbs_steer_gear_jcs_steer_gear = 0.5*multi_dot([E(self.P_vbs_steer_gear),Te_vbs_steer_gear_jcs_steer_gear])

        self.reactions = {'F_vbr_hub_mcr_ver_act':self.F_vbr_hub_mcr_ver_act,'T_vbr_hub_mcr_ver_act':self.T_vbr_hub_mcr_ver_act,'F_vbl_hub_mcl_ver_act':self.F_vbl_hub_mcl_ver_act,'T_vbl_hub_mcl_ver_act':self.T_vbl_hub_mcl_ver_act,'F_vbr_upright_jcr_rev':self.F_vbr_upright_jcr_rev,'T_vbr_upright_jcr_rev':self.T_vbr_upright_jcr_rev,'F_vbl_upright_jcl_rev':self.F_vbl_upright_jcl_rev,'T_vbl_upright_jcl_rev':self.T_vbl_upright_jcl_rev,'F_vbs_steer_gear_jcs_steer_gear':self.F_vbs_steer_gear_jcs_steer_gear,'T_vbs_steer_gear_jcs_steer_gear':self.T_vbs_steer_gear_jcs_steer_gear}

