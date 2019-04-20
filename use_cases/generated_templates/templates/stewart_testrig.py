
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
        self.ncols = 2*4
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_vbs_rocker_1_jcs_rev_1', 'T_vbs_rocker_1_jcs_rev_1', 'F_vbs_rocker_2_jcs_rev_2', 'T_vbs_rocker_2_jcs_rev_2', 'F_vbs_rocker_3_jcs_rev_3', 'T_vbs_rocker_3_jcs_rev_3']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
    
        self.vbs_rocker_2 = indicies_map[interface_map[p+'vbs_rocker_2']]
        self.vbs_rocker_3 = indicies_map[interface_map[p+'vbs_rocker_3']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbs_rocker_1 = indicies_map[interface_map[p+'vbs_rocker_1']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.vbs_ground*2, self.vbs_ground*2+1, self.vbs_rocker_1*2, self.vbs_rocker_1*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.vbs_rocker_2*2, self.vbs_rocker_2*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.vbs_rocker_3*2, self.vbs_rocker_3*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

    

        self.Mbar_vbs_rocker_1_jcs_rev_1 = multi_dot([A(config.P_vbs_rocker_1).T,triad(config.ax1_jcs_rev_1)])
        self.Mbar_vbs_ground_jcs_rev_1 = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_1)])
        self.Mbar_vbs_rocker_2_jcs_rev_2 = multi_dot([A(config.P_vbs_rocker_2).T,triad(config.ax1_jcs_rev_2)])
        self.Mbar_vbs_ground_jcs_rev_2 = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_2)])
        self.Mbar_vbs_rocker_3_jcs_rev_3 = multi_dot([A(config.P_vbs_rocker_3).T,triad(config.ax1_jcs_rev_3)])
        self.Mbar_vbs_ground_jcs_rev_3 = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_3)])

    
    def set_gen_coordinates(self,q):
        pass

    
    def set_gen_velocities(self,qd):
        pass

    
    def set_gen_accelerations(self,qdd):
        pass

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_rev_1 = Lambda[0:1,0:1]
        self.L_jcs_rev_2 = Lambda[1:2,0:1]
        self.L_jcs_rev_3 = Lambda[2:3,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = A(self.P_vbs_rocker_1).T
        x1 = A(self.P_vbs_ground)
        x2 = self.Mbar_vbs_ground_jcs_rev_1[:,0:1]
        x3 = A(self.P_vbs_rocker_2).T
        x4 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        x5 = A(self.P_vbs_rocker_3).T
        x6 = self.Mbar_vbs_ground_jcs_rev_3[:,0:1]

        self.pos_eq_blocks = [(cos(config.AF_jcs_rev_1(t))*multi_dot([self.Mbar_vbs_rocker_1_jcs_rev_1[:,1:2].T,x0,x1,x2]) + sin(config.AF_jcs_rev_1(t))*-1*multi_dot([self.Mbar_vbs_rocker_1_jcs_rev_1[:,0:1].T,x0,x1,x2])),
        (cos(config.AF_jcs_rev_2(t))*multi_dot([self.Mbar_vbs_rocker_2_jcs_rev_2[:,1:2].T,x3,x1,x4]) + sin(config.AF_jcs_rev_2(t))*-1*multi_dot([self.Mbar_vbs_rocker_2_jcs_rev_2[:,0:1].T,x3,x1,x4])),
        (cos(config.AF_jcs_rev_3(t))*multi_dot([self.Mbar_vbs_rocker_3_jcs_rev_3[:,1:2].T,x5,x1,x6]) + sin(config.AF_jcs_rev_3(t))*-1*multi_dot([self.Mbar_vbs_rocker_3_jcs_rev_3[:,0:1].T,x5,x1,x6]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [derivative(config.AF_jcs_rev_1,t,0.1,1)*-1*v0,
        derivative(config.AF_jcs_rev_2,t,0.1,1)*-1*v0,
        derivative(config.AF_jcs_rev_3,t,0.1,1)*-1*v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1,dtype=np.float64)
        a1 = self.Mbar_vbs_ground_jcs_rev_1[:,0:1]
        a2 = self.P_vbs_ground
        a3 = A(a2).T
        a4 = self.Pd_vbs_rocker_1
        a5 = self.Mbar_vbs_rocker_1_jcs_rev_1[:,1:2]
        a6 = self.Mbar_vbs_rocker_1_jcs_rev_1[:,0:1]
        a7 = self.P_vbs_rocker_1
        a8 = A(a7).T
        a9 = self.Pd_vbs_ground
        a10 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        a11 = self.Pd_vbs_rocker_2
        a12 = self.Mbar_vbs_rocker_2_jcs_rev_2[:,1:2]
        a13 = self.Mbar_vbs_rocker_2_jcs_rev_2[:,0:1]
        a14 = self.P_vbs_rocker_2
        a15 = A(a14).T
        a16 = self.Mbar_vbs_ground_jcs_rev_3[:,0:1]
        a17 = self.Pd_vbs_rocker_3
        a18 = self.Mbar_vbs_rocker_3_jcs_rev_3[:,1:2]
        a19 = self.Mbar_vbs_rocker_3_jcs_rev_3[:,0:1]
        a20 = self.P_vbs_rocker_3
        a21 = A(a20).T

        self.acc_eq_blocks = [(derivative(config.AF_jcs_rev_1,t,0.1,2)*-1*a0 + multi_dot([a1.T,a3,(cos(config.AF_jcs_rev_1(t))*B(a4,a5) + sin(config.AF_jcs_rev_1(t))*-1*B(a4,a6)),a4]) + multi_dot([(cos(config.AF_jcs_rev_1(t))*multi_dot([a5.T,a8]) + sin(config.AF_jcs_rev_1(t))*-1*multi_dot([a6.T,a8])),B(a9,a1),a9]) + 2*multi_dot([((cos(config.AF_jcs_rev_1(t))*multi_dot([B(a7,a5),a4])).T + sin(config.AF_jcs_rev_1(t))*-1*multi_dot([a4.T,B(a7,a6).T])),B(a2,a1),a9])),
        (derivative(config.AF_jcs_rev_2,t,0.1,2)*-1*a0 + multi_dot([a10.T,a3,(cos(config.AF_jcs_rev_2(t))*B(a11,a12) + sin(config.AF_jcs_rev_2(t))*-1*B(a11,a13)),a11]) + multi_dot([(cos(config.AF_jcs_rev_2(t))*multi_dot([a12.T,a15]) + sin(config.AF_jcs_rev_2(t))*-1*multi_dot([a13.T,a15])),B(a9,a10),a9]) + 2*multi_dot([((cos(config.AF_jcs_rev_2(t))*multi_dot([B(a14,a12),a11])).T + sin(config.AF_jcs_rev_2(t))*-1*multi_dot([a11.T,B(a14,a13).T])),B(a2,a10),a9])),
        (derivative(config.AF_jcs_rev_3,t,0.1,2)*-1*a0 + multi_dot([a16.T,a3,(cos(config.AF_jcs_rev_3(t))*B(a17,a18) + sin(config.AF_jcs_rev_3(t))*-1*B(a17,a19)),a17]) + multi_dot([(cos(config.AF_jcs_rev_3(t))*multi_dot([a18.T,a21]) + sin(config.AF_jcs_rev_3(t))*-1*multi_dot([a19.T,a21])),B(a9,a16),a9]) + 2*multi_dot([((cos(config.AF_jcs_rev_3(t))*multi_dot([B(a20,a18),a17])).T + sin(config.AF_jcs_rev_3(t))*-1*multi_dot([a17.T,B(a20,a19).T])),B(a2,a16),a9]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_vbs_ground_jcs_rev_1[:,0:1]
        j2 = self.P_vbs_ground
        j3 = A(j2).T
        j4 = self.P_vbs_rocker_1
        j5 = self.Mbar_vbs_rocker_1_jcs_rev_1[:,1:2]
        j6 = self.Mbar_vbs_rocker_1_jcs_rev_1[:,0:1]
        j7 = A(j4).T
        j8 = self.Mbar_vbs_ground_jcs_rev_2[:,0:1]
        j9 = self.P_vbs_rocker_2
        j10 = self.Mbar_vbs_rocker_2_jcs_rev_2[:,1:2]
        j11 = self.Mbar_vbs_rocker_2_jcs_rev_2[:,0:1]
        j12 = A(j9).T
        j13 = self.Mbar_vbs_ground_jcs_rev_3[:,0:1]
        j14 = self.P_vbs_rocker_3
        j15 = self.Mbar_vbs_rocker_3_jcs_rev_3[:,1:2]
        j16 = self.Mbar_vbs_rocker_3_jcs_rev_3[:,0:1]
        j17 = A(j14).T

        self.jac_eq_blocks = [j0,
        multi_dot([(cos(config.AF_jcs_rev_1(t))*multi_dot([j5.T,j7]) + sin(config.AF_jcs_rev_1(t))*-1*multi_dot([j6.T,j7])),B(j2,j1)]),
        j0,
        multi_dot([j1.T,j3,(cos(config.AF_jcs_rev_1(t))*B(j4,j5) + sin(config.AF_jcs_rev_1(t))*-1*B(j4,j6))]),
        j0,
        multi_dot([(cos(config.AF_jcs_rev_2(t))*multi_dot([j10.T,j12]) + sin(config.AF_jcs_rev_2(t))*-1*multi_dot([j11.T,j12])),B(j2,j8)]),
        j0,
        multi_dot([j8.T,j3,(cos(config.AF_jcs_rev_2(t))*B(j9,j10) + sin(config.AF_jcs_rev_2(t))*-1*B(j9,j11))]),
        j0,
        multi_dot([(cos(config.AF_jcs_rev_3(t))*multi_dot([j15.T,j17]) + sin(config.AF_jcs_rev_3(t))*-1*multi_dot([j16.T,j17])),B(j2,j13)]),
        j0,
        multi_dot([j13.T,j3,(cos(config.AF_jcs_rev_3(t))*B(j14,j15) + sin(config.AF_jcs_rev_3(t))*-1*B(j14,j16))])]

    
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

        Q_vbs_rocker_1_jcs_rev_1 = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcs_rev_1(t))*B(self.P_vbs_rocker_1,self.Mbar_vbs_rocker_1_jcs_rev_1[:,0:1]).T + (cos(config.AF_jcs_rev_1(t))*B(self.P_vbs_rocker_1,self.Mbar_vbs_rocker_1_jcs_rev_1[:,1:2])).T),A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_1[:,0:1]])]]),self.L_jcs_rev_1])
        self.F_vbs_rocker_1_jcs_rev_1 = Q_vbs_rocker_1_jcs_rev_1[0:3,0:1]
        Te_vbs_rocker_1_jcs_rev_1 = Q_vbs_rocker_1_jcs_rev_1[3:7,0:1]
        self.T_vbs_rocker_1_jcs_rev_1 = 0.5*multi_dot([E(self.P_vbs_rocker_1),Te_vbs_rocker_1_jcs_rev_1])
        Q_vbs_rocker_2_jcs_rev_2 = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcs_rev_2(t))*B(self.P_vbs_rocker_2,self.Mbar_vbs_rocker_2_jcs_rev_2[:,0:1]).T + (cos(config.AF_jcs_rev_2(t))*B(self.P_vbs_rocker_2,self.Mbar_vbs_rocker_2_jcs_rev_2[:,1:2])).T),A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_2[:,0:1]])]]),self.L_jcs_rev_2])
        self.F_vbs_rocker_2_jcs_rev_2 = Q_vbs_rocker_2_jcs_rev_2[0:3,0:1]
        Te_vbs_rocker_2_jcs_rev_2 = Q_vbs_rocker_2_jcs_rev_2[3:7,0:1]
        self.T_vbs_rocker_2_jcs_rev_2 = 0.5*multi_dot([E(self.P_vbs_rocker_2),Te_vbs_rocker_2_jcs_rev_2])
        Q_vbs_rocker_3_jcs_rev_3 = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T],[multi_dot([(-1*sin(config.AF_jcs_rev_3(t))*B(self.P_vbs_rocker_3,self.Mbar_vbs_rocker_3_jcs_rev_3[:,0:1]).T + (cos(config.AF_jcs_rev_3(t))*B(self.P_vbs_rocker_3,self.Mbar_vbs_rocker_3_jcs_rev_3[:,1:2])).T),A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_rev_3[:,0:1]])]]),self.L_jcs_rev_3])
        self.F_vbs_rocker_3_jcs_rev_3 = Q_vbs_rocker_3_jcs_rev_3[0:3,0:1]
        Te_vbs_rocker_3_jcs_rev_3 = Q_vbs_rocker_3_jcs_rev_3[3:7,0:1]
        self.T_vbs_rocker_3_jcs_rev_3 = 0.5*multi_dot([E(self.P_vbs_rocker_3),Te_vbs_rocker_3_jcs_rev_3])

        self.reactions = {'F_vbs_rocker_1_jcs_rev_1':self.F_vbs_rocker_1_jcs_rev_1,'T_vbs_rocker_1_jcs_rev_1':self.T_vbs_rocker_1_jcs_rev_1,'F_vbs_rocker_2_jcs_rev_2':self.F_vbs_rocker_2_jcs_rev_2,'T_vbs_rocker_2_jcs_rev_2':self.T_vbs_rocker_2_jcs_rev_2,'F_vbs_rocker_3_jcs_rev_3':self.F_vbs_rocker_3_jcs_rev_3,'T_vbs_rocker_3_jcs_rev_3':self.T_vbs_rocker_3_jcs_rev_3}

