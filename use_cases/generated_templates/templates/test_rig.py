
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

        reactions_indicies = ['F_vbr_upright_jcr_rev', 'T_vbr_upright_jcr_rev', 'F_vbl_upright_jcl_rev', 'T_vbl_upright_jcl_rev', 'F_vbs_steer_gear_jcs_steer_gear', 'T_vbs_steer_gear_jcs_steer_gear']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
    
        self.vbl_hub = indicies_map[interface_map[p+'vbl_hub']]
        self.vbs_steer_gear = indicies_map[interface_map[p+'vbs_steer_gear']]
        self.vbl_upright = indicies_map[interface_map[p+'vbl_upright']]
        self.vbr_upright = indicies_map[interface_map[p+'vbr_upright']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbr_hub = indicies_map[interface_map[p+'vbr_hub']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.vbr_hub*2, self.vbr_hub*2+1, self.vbr_upright*2, self.vbr_upright*2+1, self.vbl_hub*2, self.vbl_hub*2+1, self.vbl_upright*2, self.vbl_upright*2+1, self.vbs_steer_gear*2, self.vbs_steer_gear*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

    

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
        self.L_jcr_rev = Lambda[0:1,0:1]
        self.L_jcl_rev = Lambda[1:2,0:1]
        self.L_jcs_steer_gear = Lambda[2:3,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = A(self.P_vbr_upright).T
        x1 = A(self.P_vbr_hub)
        x2 = self.Mbar_vbr_hub_jcr_rev[:,0:1]
        x3 = A(self.P_vbl_upright).T
        x4 = A(self.P_vbl_hub)
        x5 = self.Mbar_vbl_hub_jcl_rev[:,0:1]
        x6 = A(self.P_vbs_steer_gear).T
        x7 = A(self.P_vbs_chassis)
        x8 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]

        self.pos_eq_blocks = [(cos(config.AF_jcr_rev(t))*multi_dot([self.Mbar_vbr_upright_jcr_rev[:,1:2].T,x0,x1,x2]) + sin(config.AF_jcr_rev(t))*-1*multi_dot([self.Mbar_vbr_upright_jcr_rev[:,0:1].T,x0,x1,x2])),
        (cos(config.AF_jcl_rev(t))*multi_dot([self.Mbar_vbl_upright_jcl_rev[:,1:2].T,x3,x4,x5]) + sin(config.AF_jcl_rev(t))*-1*multi_dot([self.Mbar_vbl_upright_jcl_rev[:,0:1].T,x3,x4,x5])),
        (cos(config.AF_jcs_steer_gear(t))*multi_dot([self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2].T,x6,x7,x8]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1].T,x6,x7,x8]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [derivative(config.AF_jcr_rev,t,0.1,1)*-1*v0,
        derivative(config.AF_jcl_rev,t,0.1,1)*-1*v0,
        derivative(config.AF_jcs_steer_gear,t,0.1,1)*-1*v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1,dtype=np.float64)
        a1 = self.Mbar_vbr_hub_jcr_rev[:,0:1]
        a2 = self.P_vbr_hub
        a3 = self.Pd_vbr_upright
        a4 = self.Mbar_vbr_upright_jcr_rev[:,1:2]
        a5 = self.Mbar_vbr_upright_jcr_rev[:,0:1]
        a6 = self.P_vbr_upright
        a7 = A(a6).T
        a8 = self.Pd_vbr_hub
        a9 = self.Mbar_vbl_hub_jcl_rev[:,0:1]
        a10 = self.P_vbl_hub
        a11 = self.Pd_vbl_upright
        a12 = self.Mbar_vbl_upright_jcl_rev[:,1:2]
        a13 = self.Mbar_vbl_upright_jcl_rev[:,0:1]
        a14 = self.P_vbl_upright
        a15 = A(a14).T
        a16 = self.Pd_vbl_hub
        a17 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        a18 = self.P_vbs_chassis
        a19 = self.Pd_vbs_steer_gear
        a20 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        a21 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        a22 = self.P_vbs_steer_gear
        a23 = A(a22).T
        a24 = self.Pd_vbs_chassis

        self.acc_eq_blocks = [(derivative(config.AF_jcr_rev,t,0.1,2)*-1*a0 + multi_dot([a1.T,A(a2).T,(cos(config.AF_jcr_rev(t))*B(a3,a4) + sin(config.AF_jcr_rev(t))*-1*B(a3,a5)),a3]) + multi_dot([(cos(config.AF_jcr_rev(t))*multi_dot([a4.T,a7]) + sin(config.AF_jcr_rev(t))*-1*multi_dot([a5.T,a7])),B(a8,a1),a8]) + 2*multi_dot([((cos(config.AF_jcr_rev(t))*multi_dot([B(a6,a4),a3])).T + sin(config.AF_jcr_rev(t))*-1*multi_dot([a3.T,B(a6,a5).T])),B(a2,a1),a8])),
        (derivative(config.AF_jcl_rev,t,0.1,2)*-1*a0 + multi_dot([a9.T,A(a10).T,(cos(config.AF_jcl_rev(t))*B(a11,a12) + sin(config.AF_jcl_rev(t))*-1*B(a11,a13)),a11]) + multi_dot([(cos(config.AF_jcl_rev(t))*multi_dot([a12.T,a15]) + sin(config.AF_jcl_rev(t))*-1*multi_dot([a13.T,a15])),B(a16,a9),a16]) + 2*multi_dot([((cos(config.AF_jcl_rev(t))*multi_dot([B(a14,a12),a11])).T + sin(config.AF_jcl_rev(t))*-1*multi_dot([a11.T,B(a14,a13).T])),B(a10,a9),a16])),
        (derivative(config.AF_jcs_steer_gear,t,0.1,2)*-1*a0 + multi_dot([a17.T,A(a18).T,(cos(config.AF_jcs_steer_gear(t))*B(a19,a20) + sin(config.AF_jcs_steer_gear(t))*-1*B(a19,a21)),a19]) + multi_dot([(cos(config.AF_jcs_steer_gear(t))*multi_dot([a20.T,a23]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([a21.T,a23])),B(a24,a17),a24]) + 2*multi_dot([((cos(config.AF_jcs_steer_gear(t))*multi_dot([B(a22,a20),a19])).T + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([a19.T,B(a22,a21).T])),B(a18,a17),a24]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_vbr_hub_jcr_rev[:,0:1]
        j2 = self.P_vbr_hub
        j3 = self.P_vbr_upright
        j4 = self.Mbar_vbr_upright_jcr_rev[:,1:2]
        j5 = self.Mbar_vbr_upright_jcr_rev[:,0:1]
        j6 = A(j3).T
        j7 = self.Mbar_vbl_hub_jcl_rev[:,0:1]
        j8 = self.P_vbl_hub
        j9 = self.P_vbl_upright
        j10 = self.Mbar_vbl_upright_jcl_rev[:,1:2]
        j11 = self.Mbar_vbl_upright_jcl_rev[:,0:1]
        j12 = A(j9).T
        j13 = self.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        j14 = self.P_vbs_chassis
        j15 = self.P_vbs_steer_gear
        j16 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        j17 = self.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        j18 = A(j15).T

        self.jac_eq_blocks = [j0,
        multi_dot([(cos(config.AF_jcr_rev(t))*multi_dot([j4.T,j6]) + sin(config.AF_jcr_rev(t))*-1*multi_dot([j5.T,j6])),B(j2,j1)]),
        j0,
        multi_dot([j1.T,A(j2).T,(cos(config.AF_jcr_rev(t))*B(j3,j4) + sin(config.AF_jcr_rev(t))*-1*B(j3,j5))]),
        j0,
        multi_dot([(cos(config.AF_jcl_rev(t))*multi_dot([j10.T,j12]) + sin(config.AF_jcl_rev(t))*-1*multi_dot([j11.T,j12])),B(j8,j7)]),
        j0,
        multi_dot([j7.T,A(j8).T,(cos(config.AF_jcl_rev(t))*B(j9,j10) + sin(config.AF_jcl_rev(t))*-1*B(j9,j11))]),
        j0,
        multi_dot([j13.T,A(j14).T,(cos(config.AF_jcs_steer_gear(t))*B(j15,j16) + sin(config.AF_jcs_steer_gear(t))*-1*B(j15,j17))]),
        j0,
        multi_dot([(cos(config.AF_jcs_steer_gear(t))*multi_dot([j16.T,j18]) + sin(config.AF_jcs_steer_gear(t))*-1*multi_dot([j17.T,j18])),B(j14,j13)])]

    
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

        self.reactions = {'F_vbr_upright_jcr_rev':self.F_vbr_upright_jcr_rev,'T_vbr_upright_jcr_rev':self.T_vbr_upright_jcr_rev,'F_vbl_upright_jcl_rev':self.F_vbl_upright_jcl_rev,'T_vbl_upright_jcl_rev':self.T_vbl_upright_jcl_rev,'F_vbs_steer_gear_jcs_steer_gear':self.F_vbs_steer_gear_jcs_steer_gear,'T_vbs_steer_gear_jcs_steer_gear':self.T_vbs_steer_gear_jcs_steer_gear}

