
import numpy as np
from numpy import cos, sin
from numpy.linalg import multi_dot
from scipy.misc import derivative

try:
    from smbd.numenv.python.numerics.matrix_funcs import A, B, G, E, triad, skew_matrix as skew
except ModuleNotFoundError:
    print('Failed importing compiled matrices!')
    print('Falling back to python defined matrix functions')
    from smbd.numenv.python.numerics.misc import A, B, G, E, triad, skew_matrix as skew



class topology(object):

    def __init__(self,prefix=''):
        self.t = 0.0
        self.prefix = (prefix if prefix=='' else prefix+'.')
        self.config = None

        self.indicies_map = {'ground': 0, 'rbs_body': 1}

        self.n  = 14
        self.nc = 14
        self.nrows = 9
        self.ncols = 2*2
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a', 'F_ground_jcs_a', 'T_ground_jcs_a']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    

    def initialize(self):
        self.t = 0
        self.assemble(self.indicies_map, {}, 0)
        self.set_initial_states()
        self.eval_constants()

    def assemble(self, indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 8])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2, self.rbs_body*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_body*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)
        self.q0 = self.config.q

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p+'ground']
        self.rbs_body = indicies_map[p+'rbs_body']
    

    
    def eval_constants(self):
        config = self.config

        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_body_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_body]], dtype=np.float64)

        self.Mbar_ground_jcs_a = multi_dot([A(config.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_body_jcs_a = multi_dot([A(config.P_rbs_body).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(config.P_ground).T,config.pt1_jcs_a]) + -1*multi_dot([A(config.P_ground).T,config.R_ground]))
        self.ubar_rbs_body_jcs_a = (multi_dot([A(config.P_rbs_body).T,config.pt1_jcs_a]) + -1*multi_dot([A(config.P_rbs_body).T,config.R_rbs_body]))
        self.Mbar_ground_jcs_a = multi_dot([A(config.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_body_jcs_a = multi_dot([A(config.P_rbs_body).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(config.P_ground).T,config.pt1_jcs_a]) + -1*multi_dot([A(config.P_ground).T,config.R_ground]))
        self.ubar_rbs_body_jcs_a = (multi_dot([A(config.P_rbs_body).T,config.pt1_jcs_a]) + -1*multi_dot([A(config.P_rbs_body).T,config.R_rbs_body]))

    
    def set_gen_coordinates(self,q):
        self.R_ground = q[0:3,0:1]
        self.P_ground = q[3:7,0:1]
        self.R_rbs_body = q[7:10,0:1]
        self.P_rbs_body = q[10:14,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        self.Rd_rbs_body = qd[7:10,0:1]
        self.Pd_rbs_body = qd[10:14,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_ground = qdd[0:3,0:1]
        self.Pdd_ground = qdd[3:7,0:1]
        self.Rdd_rbs_body = qdd[7:10,0:1]
        self.Pdd_rbs_body = qdd[10:14,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_a = Lambda[0:5,0:1]
        self.L_jcs_a = Lambda[5:6,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_ground_jcs_a[:,0:1].T
        x1 = self.P_ground
        x2 = A(x1)
        x3 = x2.T
        x4 = self.P_rbs_body
        x5 = A(x4)
        x6 = self.Mbar_rbs_body_jcs_a[:,2:3]
        x7 = self.Mbar_ground_jcs_a[:,1:2].T
        x8 = self.R_ground
        x9 = (x8 + -1*self.R_rbs_body + multi_dot([x2,self.ubar_ground_jcs_a]) + -1*multi_dot([x5,self.ubar_rbs_body_jcs_a]))
        x10 = np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = [multi_dot([x0,x3,x5,x6]),
        multi_dot([x7,x3,x5,x6]),
        multi_dot([x0,x3,x9]),
        multi_dot([x7,x3,x9]),
        multi_dot([x0,x3,x5,self.Mbar_rbs_body_jcs_a[:,1:2]]),
        (-1*config.UF_jcs_a(t)*x10 + multi_dot([self.Mbar_ground_jcs_a[:,2:3].T,x3,x9])),
        x8,
        (x1 + -1*self.Pg_ground),
        (-1*x10 + multi_dot([x4.T,x4]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,
        v0,
        v0,
        v0,
        v0,
        (v0 + -1*derivative(config.UF_jcs_a, t, 0.1, 1)*np.eye(1, dtype=np.float64)),
        np.zeros((3,1),dtype=np.float64),
        np.zeros((4,1),dtype=np.float64),
        v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_rbs_body_jcs_a[:,2:3]
        a1 = a0.T
        a2 = self.P_rbs_body
        a3 = A(a2).T
        a4 = self.Pd_ground
        a5 = self.Mbar_ground_jcs_a[:,0:1]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_ground
        a9 = A(a8).T
        a10 = self.Pd_rbs_body
        a11 = B(a10,a0)
        a12 = a4.T
        a13 = B(a8,a5).T
        a14 = B(a2,a0)
        a15 = self.Mbar_ground_jcs_a[:,1:2]
        a16 = B(a4,a15)
        a17 = a15.T
        a18 = B(a8,a15).T
        a19 = self.ubar_ground_jcs_a
        a20 = self.ubar_rbs_body_jcs_a
        a21 = (multi_dot([B(a4,a19),a4]) + -1*multi_dot([B(a10,a20),a10]))
        a22 = (self.Rd_ground + -1*self.Rd_rbs_body + multi_dot([B(a8,a19),a4]) + -1*multi_dot([B(a2,a20),a10]))
        a23 = (self.R_ground.T + -1*self.R_rbs_body.T + multi_dot([a19.T,a9]) + -1*multi_dot([a20.T,a3]))
        a24 = self.Mbar_rbs_body_jcs_a[:,1:2]
        a25 = self.Mbar_ground_jcs_a[:,2:3]

        self.acc_eq_blocks = [(multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + 2*multi_dot([a12,a13,a14,a10])),
        (multi_dot([a1,a3,a16,a4]) + multi_dot([a17,a9,a11,a10]) + 2*multi_dot([a12,a18,a14,a10])),
        (multi_dot([a7,a9,a21]) + 2*multi_dot([a12,a13,a22]) + multi_dot([a23,a6,a4])),
        (multi_dot([a17,a9,a21]) + 2*multi_dot([a12,a18,a22]) + multi_dot([a23,a16,a4])),
        (multi_dot([a24.T,a3,a6,a4]) + multi_dot([a7,a9,B(a10,a24),a10]) + 2*multi_dot([a12,a13,B(a2,a24),a10])),
        (-1*derivative(config.UF_jcs_a, t, 0.1, 2)*np.eye(1, dtype=np.float64) + multi_dot([a25.T,a9,a21]) + 2*multi_dot([a12,B(a8,a25).T,a22]) + multi_dot([a23,B(a4,a25),a4])),
        np.zeros((3,1),dtype=np.float64),
        np.zeros((4,1),dtype=np.float64),
        2*multi_dot([a10.T,a10])]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_rbs_body_jcs_a[:,2:3]
        j2 = j1.T
        j3 = self.P_rbs_body
        j4 = A(j3).T
        j5 = self.P_ground
        j6 = self.Mbar_ground_jcs_a[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_ground_jcs_a
        j14 = B(j5,j13)
        j15 = self.ubar_rbs_body_jcs_a
        j16 = (self.R_ground.T + -1*self.R_rbs_body.T + multi_dot([j13.T,j11]) + -1*multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = self.Mbar_rbs_body_jcs_a[:,1:2]
        j20 = B(j3,j1)
        j21 = B(j3,j15)
        j22 = self.Mbar_ground_jcs_a[:,2:3]
        j23 = j22.T
        j24 = multi_dot([j23,j11])

        self.jac_eq_blocks = [j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j10,j11,j20]),
        j0,
        multi_dot([j2,j4,j9]),
        j0,
        multi_dot([j17,j11,j20]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),
        -1*j12,
        -1*multi_dot([j10,j11,j21]),
        j18,
        (multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),
        -1*j18,
        -1*multi_dot([j17,j11,j21]),
        j0,
        multi_dot([j19.T,j4,j7]),
        j0,
        multi_dot([j10,j11,B(j3,j19)]),
        j24,
        (multi_dot([j23,j11,j14]) + multi_dot([j16,B(j5,j22)])),
        -1*j24,
        -1*multi_dot([j23,j11,j21]),
        np.eye(3, dtype=np.float64),
        np.zeros((3,4),dtype=np.float64),
        np.zeros((4,3),dtype=np.float64),
        np.eye(4, dtype=np.float64),
        2*j3.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3, dtype=np.float64)
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_body)

        self.mass_eq_blocks = [self.m_ground*m0,
        4*multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_body*m0,
        4*multi_dot([m2.T,config.Jbar_rbs_body,m2])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_body)

        self.frc_eq_blocks = [np.zeros((3,1),dtype=np.float64),
        np.zeros((4,1),dtype=np.float64),
        self.F_rbs_body_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_body,f0,self.P_rbs_body])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_a = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_ground),self.Mbar_ground_jcs_a[:,0:1]]),multi_dot([A(self.P_ground),self.Mbar_ground_jcs_a[:,1:2]]),np.zeros((1,3),dtype=np.float64).T],[multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_a[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_a[:,2:3]]),(multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,(-1*self.R_rbs_body + multi_dot([A(self.P_ground),self.ubar_ground_jcs_a]) + -1*multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_jcs_a]) + self.R_ground)]) + multi_dot([B(self.P_ground,self.ubar_ground_jcs_a).T,A(self.P_ground),self.Mbar_ground_jcs_a[:,0:1]])),(multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,(-1*self.R_rbs_body + multi_dot([A(self.P_ground),self.ubar_ground_jcs_a]) + -1*multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_jcs_a]) + self.R_ground)]) + multi_dot([B(self.P_ground,self.ubar_ground_jcs_a).T,A(self.P_ground),self.Mbar_ground_jcs_a[:,1:2]])),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_body),self.Mbar_rbs_body_jcs_a[:,1:2]])]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3,0:1]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7,0:1]
        self.T_ground_jcs_a = (-1*multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_a])),self.F_ground_jcs_a]) + 0.5*multi_dot([E(self.P_ground),Te_ground_jcs_a]))
        Q_ground_jcs_a = -1*multi_dot([np.bmat([[multi_dot([A(self.P_ground),self.Mbar_ground_jcs_a[:,2:3]])],[(multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,2:3]).T,(-1*self.R_rbs_body + multi_dot([A(self.P_ground),self.ubar_ground_jcs_a]) + -1*multi_dot([A(self.P_rbs_body),self.ubar_rbs_body_jcs_a]) + self.R_ground)]) + multi_dot([B(self.P_ground,self.ubar_ground_jcs_a).T,A(self.P_ground),self.Mbar_ground_jcs_a[:,2:3]]))]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3,0:1]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7,0:1]
        self.T_ground_jcs_a = 0.5*multi_dot([E(self.P_ground),Te_ground_jcs_a])

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a,
                        'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a}

