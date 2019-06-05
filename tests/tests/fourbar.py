
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

        self.indicies_map = {'ground': 0, 'rbs_crank': 1, 'rbs_conct': 2, 'rbs_rockr': 3}

        self.n  = 28
        self.nc = 27
        self.nrows = 14
        self.ncols = 2*4
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_ground_jcs_a', 'T_ground_jcs_a', 'F_rbs_crank_jcs_b', 'T_rbs_crank_jcs_b', 'F_rbs_conct_jcs_c', 'T_rbs_conct_jcs_c', 'F_rbs_rockr_jcs_d', 'T_rbs_rockr_jcs_d']
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
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 11, 12, 13])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.ground*2, self.ground*2+1, self.rbs_crank*2, self.rbs_crank*2+1, self.ground*2, self.ground*2+1, self.rbs_crank*2, self.rbs_crank*2+1, self.ground*2, self.ground*2+1, self.rbs_crank*2, self.rbs_crank*2+1, self.rbs_crank*2, self.rbs_crank*2+1, self.rbs_conct*2, self.rbs_conct*2+1, self.rbs_conct*2, self.rbs_conct*2+1, self.rbs_rockr*2, self.rbs_rockr*2+1, self.rbs_conct*2, self.rbs_conct*2+1, self.rbs_rockr*2, self.rbs_rockr*2+1, self.ground*2, self.ground*2+1, self.rbs_rockr*2, self.rbs_rockr*2+1, self.ground*2, self.ground*2+1, self.rbs_rockr*2, self.rbs_rockr*2+1, self.ground*2, self.ground*2+1, self.rbs_rockr*2, self.rbs_rockr*2+1, self.ground*2, self.ground*2+1, self.ground*2, self.ground*2+1, self.rbs_crank*2+1, self.rbs_conct*2+1, self.rbs_rockr*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)
        self.q0 = self.config.q

    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.ground = indicies_map[p+'ground']
        self.rbs_crank = indicies_map[p+'rbs_crank']
        self.rbs_conct = indicies_map[p+'rbs_conct']
        self.rbs_rockr = indicies_map[p+'rbs_rockr']
    

    
    def eval_constants(self):
        config = self.config

        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.m_ground = 1.0
        self.Jbar_ground = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.F_rbs_crank_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_crank]], dtype=np.float64)
        self.F_rbs_conct_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_conct]], dtype=np.float64)
        self.F_rbs_rockr_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_rockr]], dtype=np.float64)

        self.Mbar_ground_jcs_a = multi_dot([A(config.P_ground).T,triad(config.ax1_jcs_a)])
        self.Mbar_rbs_crank_jcs_a = multi_dot([A(config.P_rbs_crank).T,triad(config.ax1_jcs_a)])
        self.ubar_ground_jcs_a = (multi_dot([A(config.P_ground).T,config.pt1_jcs_a]) + -1*multi_dot([A(config.P_ground).T,config.R_ground]))
        self.ubar_rbs_crank_jcs_a = (multi_dot([A(config.P_rbs_crank).T,config.pt1_jcs_a]) + -1*multi_dot([A(config.P_rbs_crank).T,config.R_rbs_crank]))
        self.Mbar_rbs_crank_jcs_b = multi_dot([A(config.P_rbs_crank).T,triad(config.ax1_jcs_b)])
        self.Mbar_rbs_conct_jcs_b = multi_dot([A(config.P_rbs_conct).T,triad(config.ax1_jcs_b)])
        self.ubar_rbs_crank_jcs_b = (multi_dot([A(config.P_rbs_crank).T,config.pt1_jcs_b]) + -1*multi_dot([A(config.P_rbs_crank).T,config.R_rbs_crank]))
        self.ubar_rbs_conct_jcs_b = (multi_dot([A(config.P_rbs_conct).T,config.pt1_jcs_b]) + -1*multi_dot([A(config.P_rbs_conct).T,config.R_rbs_conct]))
        self.Mbar_rbs_conct_jcs_c = multi_dot([A(config.P_rbs_conct).T,triad(config.ax1_jcs_c)])
        self.Mbar_rbs_rockr_jcs_c = multi_dot([A(config.P_rbs_rockr).T,triad(config.ax2_jcs_c,triad(config.ax1_jcs_c)[0:3,1:2])])
        self.ubar_rbs_conct_jcs_c = (multi_dot([A(config.P_rbs_conct).T,config.pt1_jcs_c]) + -1*multi_dot([A(config.P_rbs_conct).T,config.R_rbs_conct]))
        self.ubar_rbs_rockr_jcs_c = (multi_dot([A(config.P_rbs_rockr).T,config.pt1_jcs_c]) + -1*multi_dot([A(config.P_rbs_rockr).T,config.R_rbs_rockr]))
        self.Mbar_rbs_rockr_jcs_d = multi_dot([A(config.P_rbs_rockr).T,triad(config.ax1_jcs_d)])
        self.Mbar_ground_jcs_d = multi_dot([A(config.P_ground).T,triad(config.ax1_jcs_d)])
        self.ubar_rbs_rockr_jcs_d = (multi_dot([A(config.P_rbs_rockr).T,config.pt1_jcs_d]) + -1*multi_dot([A(config.P_rbs_rockr).T,config.R_rbs_rockr]))
        self.ubar_ground_jcs_d = (multi_dot([A(config.P_ground).T,config.pt1_jcs_d]) + -1*multi_dot([A(config.P_ground).T,config.R_ground]))

    
    def set_gen_coordinates(self,q):
        self.R_ground = q[0:3,0:1]
        self.P_ground = q[3:7,0:1]
        self.R_rbs_crank = q[7:10,0:1]
        self.P_rbs_crank = q[10:14,0:1]
        self.R_rbs_conct = q[14:17,0:1]
        self.P_rbs_conct = q[17:21,0:1]
        self.R_rbs_rockr = q[21:24,0:1]
        self.P_rbs_rockr = q[24:28,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        self.Rd_rbs_crank = qd[7:10,0:1]
        self.Pd_rbs_crank = qd[10:14,0:1]
        self.Rd_rbs_conct = qd[14:17,0:1]
        self.Pd_rbs_conct = qd[17:21,0:1]
        self.Rd_rbs_rockr = qd[21:24,0:1]
        self.Pd_rbs_rockr = qd[24:28,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_ground = qdd[0:3,0:1]
        self.Pdd_ground = qdd[3:7,0:1]
        self.Rdd_rbs_crank = qdd[7:10,0:1]
        self.Pdd_rbs_crank = qdd[10:14,0:1]
        self.Rdd_rbs_conct = qdd[14:17,0:1]
        self.Pdd_rbs_conct = qdd[17:21,0:1]
        self.Rdd_rbs_rockr = qdd[21:24,0:1]
        self.Pdd_rbs_rockr = qdd[24:28,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_a = Lambda[0:5,0:1]
        self.L_jcs_b = Lambda[5:8,0:1]
        self.L_jcs_c = Lambda[8:12,0:1]
        self.L_jcs_d = Lambda[12:17,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_ground
        x1 = self.R_rbs_crank
        x2 = self.P_ground
        x3 = A(x2)
        x4 = self.P_rbs_crank
        x5 = A(x4)
        x6 = x3.T
        x7 = self.Mbar_rbs_crank_jcs_a[:,2:3]
        x8 = self.R_rbs_conct
        x9 = self.P_rbs_conct
        x10 = A(x9)
        x11 = self.R_rbs_rockr
        x12 = self.P_rbs_rockr
        x13 = A(x12)
        x14 = x13.T
        x15 = self.Mbar_ground_jcs_d[:,2:3]
        x16 = -1*np.eye(1, dtype=np.float64)

        self.pos_eq_blocks = [(x0 + -1*x1 + multi_dot([x3,self.ubar_ground_jcs_a]) + -1*multi_dot([x5,self.ubar_rbs_crank_jcs_a])),
        multi_dot([self.Mbar_ground_jcs_a[:,0:1].T,x6,x5,x7]),
        multi_dot([self.Mbar_ground_jcs_a[:,1:2].T,x6,x5,x7]),
        (x1 + -1*x8 + multi_dot([x5,self.ubar_rbs_crank_jcs_b]) + -1*multi_dot([x10,self.ubar_rbs_conct_jcs_b])),
        (x8 + -1*x11 + multi_dot([x10,self.ubar_rbs_conct_jcs_c]) + -1*multi_dot([x13,self.ubar_rbs_rockr_jcs_c])),
        multi_dot([self.Mbar_rbs_conct_jcs_c[:,0:1].T,x10.T,x13,self.Mbar_rbs_rockr_jcs_c[:,0:1]]),
        (x11 + -1*x0 + multi_dot([x13,self.ubar_rbs_rockr_jcs_d]) + -1*multi_dot([x3,self.ubar_ground_jcs_d])),
        multi_dot([self.Mbar_rbs_rockr_jcs_d[:,0:1].T,x14,x3,x15]),
        multi_dot([self.Mbar_rbs_rockr_jcs_d[:,1:2].T,x14,x3,x15]),
        x0,
        (x2 + -1*self.Pg_ground),
        (x16 + multi_dot([x4.T,x4])),
        (x16 + multi_dot([x9.T,x9])),
        (x16 + multi_dot([x12.T,x12]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,
        v1,
        v1,
        v0,
        v0,
        v1,
        v0,
        v1,
        v1,
        v0,
        np.zeros((4,1),dtype=np.float64),
        v1,
        v1,
        v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_ground
        a1 = self.Pd_rbs_crank
        a2 = self.Mbar_ground_jcs_a[:,0:1]
        a3 = self.P_ground
        a4 = A(a3).T
        a5 = self.Mbar_rbs_crank_jcs_a[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_rbs_crank
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = self.Mbar_ground_jcs_a[:,1:2]
        a13 = self.Pd_rbs_conct
        a14 = self.Pd_rbs_rockr
        a15 = self.Mbar_rbs_conct_jcs_c[:,0:1]
        a16 = self.P_rbs_conct
        a17 = self.Mbar_rbs_rockr_jcs_c[:,0:1]
        a18 = self.P_rbs_rockr
        a19 = A(a18).T
        a20 = a13.T
        a21 = self.Mbar_ground_jcs_d[:,2:3]
        a22 = a21.T
        a23 = self.Mbar_rbs_rockr_jcs_d[:,0:1]
        a24 = B(a0,a21)
        a25 = a14.T
        a26 = B(a3,a21)
        a27 = self.Mbar_rbs_rockr_jcs_d[:,1:2]

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_ground_jcs_a),a0]) + -1*multi_dot([B(a1,self.ubar_rbs_crank_jcs_a),a1])),
        (multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + 2*multi_dot([a10,B(a3,a2).T,a11,a1])),
        (multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + 2*multi_dot([a10,B(a3,a12).T,a11,a1])),
        (multi_dot([B(a1,self.ubar_rbs_crank_jcs_b),a1]) + -1*multi_dot([B(a13,self.ubar_rbs_conct_jcs_b),a13])),
        (multi_dot([B(a13,self.ubar_rbs_conct_jcs_c),a13]) + -1*multi_dot([B(a14,self.ubar_rbs_rockr_jcs_c),a14])),
        (multi_dot([a15.T,A(a16).T,B(a14,a17),a14]) + multi_dot([a17.T,a19,B(a13,a15),a13]) + 2*multi_dot([a20,B(a16,a15).T,B(a18,a17),a14])),
        (multi_dot([B(a14,self.ubar_rbs_rockr_jcs_d),a14]) + -1*multi_dot([B(a0,self.ubar_ground_jcs_d),a0])),
        (multi_dot([a22,a4,B(a14,a23),a14]) + multi_dot([a23.T,a19,a24,a0]) + 2*multi_dot([a25,B(a18,a23).T,a26,a0])),
        (multi_dot([a22,a4,B(a14,a27),a14]) + multi_dot([a27.T,a19,a24,a0]) + 2*multi_dot([a25,B(a18,a27).T,a26,a0])),
        np.zeros((3,1),dtype=np.float64),
        np.zeros((4,1),dtype=np.float64),
        2*multi_dot([a1.T,a1]),
        2*multi_dot([a20,a13]),
        2*multi_dot([a25,a14])]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3, dtype=np.float64)
        j1 = self.P_ground
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_rbs_crank_jcs_a[:,2:3]
        j4 = j3.T
        j5 = self.P_rbs_crank
        j6 = A(j5).T
        j7 = self.Mbar_ground_jcs_a[:,0:1]
        j8 = self.Mbar_ground_jcs_a[:,1:2]
        j9 = -1*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.P_rbs_conct
        j13 = self.Mbar_rbs_rockr_jcs_c[:,0:1]
        j14 = self.P_rbs_rockr
        j15 = A(j14).T
        j16 = self.Mbar_rbs_conct_jcs_c[:,0:1]
        j17 = self.Mbar_ground_jcs_d[:,2:3]
        j18 = j17.T
        j19 = self.Mbar_rbs_rockr_jcs_d[:,0:1]
        j20 = self.Mbar_rbs_rockr_jcs_d[:,1:2]
        j21 = B(j1,j17)

        self.jac_eq_blocks = [j0,
        B(j1,self.ubar_ground_jcs_a),
        j9,
        -1*B(j5,self.ubar_rbs_crank_jcs_a),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j0,
        B(j5,self.ubar_rbs_crank_jcs_b),
        j9,
        -1*B(j12,self.ubar_rbs_conct_jcs_b),
        j0,
        B(j12,self.ubar_rbs_conct_jcs_c),
        j9,
        -1*B(j14,self.ubar_rbs_rockr_jcs_c),
        j2,
        multi_dot([j13.T,j15,B(j12,j16)]),
        j2,
        multi_dot([j16.T,A(j12).T,B(j14,j13)]),
        j9,
        -1*B(j1,self.ubar_ground_jcs_d),
        j0,
        B(j14,self.ubar_rbs_rockr_jcs_d),
        j2,
        multi_dot([j19.T,j15,j21]),
        j2,
        multi_dot([j18,j10,B(j14,j19)]),
        j2,
        multi_dot([j20.T,j15,j21]),
        j2,
        multi_dot([j18,j10,B(j14,j20)]),
        j0,
        np.zeros((3,4),dtype=np.float64),
        np.zeros((4,3),dtype=np.float64),
        np.eye(4, dtype=np.float64),
        2*j5.T,
        2*j12.T,
        2*j14.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3, dtype=np.float64)
        m1 = G(self.P_ground)
        m2 = G(self.P_rbs_crank)
        m3 = G(self.P_rbs_conct)
        m4 = G(self.P_rbs_rockr)

        self.mass_eq_blocks = [self.m_ground*m0,
        4*multi_dot([m1.T,self.Jbar_ground,m1]),
        config.m_rbs_crank*m0,
        4*multi_dot([m2.T,config.Jbar_rbs_crank,m2]),
        config.m_rbs_conct*m0,
        4*multi_dot([m3.T,config.Jbar_rbs_conct,m3]),
        config.m_rbs_rockr*m0,
        4*multi_dot([m4.T,config.Jbar_rbs_rockr,m4])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_crank)
        f1 = G(self.Pd_rbs_conct)
        f2 = G(self.Pd_rbs_rockr)

        self.frc_eq_blocks = [np.zeros((3,1),dtype=np.float64),
        np.zeros((4,1),dtype=np.float64),
        self.F_rbs_crank_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_crank,f0,self.P_rbs_crank]),
        self.F_rbs_conct_gravity,
        8*multi_dot([f1.T,config.Jbar_rbs_conct,f1,self.P_rbs_conct]),
        self.F_rbs_rockr_gravity,
        8*multi_dot([f2.T,config.Jbar_rbs_rockr,f2,self.P_rbs_rockr])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_ground_jcs_a = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_ground,self.ubar_ground_jcs_a).T,multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,0:1]).T,A(self.P_rbs_crank),self.Mbar_rbs_crank_jcs_a[:,2:3]]),multi_dot([B(self.P_ground,self.Mbar_ground_jcs_a[:,1:2]).T,A(self.P_rbs_crank),self.Mbar_rbs_crank_jcs_a[:,2:3]])]]),self.L_jcs_a])
        self.F_ground_jcs_a = Q_ground_jcs_a[0:3,0:1]
        Te_ground_jcs_a = Q_ground_jcs_a[3:7,0:1]
        self.T_ground_jcs_a = (-1*multi_dot([skew(multi_dot([A(self.P_ground),self.ubar_ground_jcs_a])),self.F_ground_jcs_a]) + 0.5*multi_dot([E(self.P_ground),Te_ground_jcs_a]))
        Q_rbs_crank_jcs_b = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64)],[B(self.P_rbs_crank,self.ubar_rbs_crank_jcs_b).T]]),self.L_jcs_b])
        self.F_rbs_crank_jcs_b = Q_rbs_crank_jcs_b[0:3,0:1]
        Te_rbs_crank_jcs_b = Q_rbs_crank_jcs_b[3:7,0:1]
        self.T_rbs_crank_jcs_b = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_crank),self.ubar_rbs_crank_jcs_b])),self.F_rbs_crank_jcs_b]) + 0.5*multi_dot([E(self.P_rbs_crank),Te_rbs_crank_jcs_b]))
        Q_rbs_conct_jcs_c = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_conct,self.ubar_rbs_conct_jcs_c).T,multi_dot([B(self.P_rbs_conct,self.Mbar_rbs_conct_jcs_c[:,0:1]).T,A(self.P_rbs_rockr),self.Mbar_rbs_rockr_jcs_c[:,0:1]])]]),self.L_jcs_c])
        self.F_rbs_conct_jcs_c = Q_rbs_conct_jcs_c[0:3,0:1]
        Te_rbs_conct_jcs_c = Q_rbs_conct_jcs_c[3:7,0:1]
        self.T_rbs_conct_jcs_c = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_conct),self.ubar_rbs_conct_jcs_c])),self.F_rbs_conct_jcs_c]) + 0.5*multi_dot([E(self.P_rbs_conct),Te_rbs_conct_jcs_c]))
        Q_rbs_rockr_jcs_d = -1*multi_dot([np.bmat([[np.eye(3, dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_rockr,self.ubar_rbs_rockr_jcs_d).T,multi_dot([B(self.P_rbs_rockr,self.Mbar_rbs_rockr_jcs_d[:,0:1]).T,A(self.P_ground),self.Mbar_ground_jcs_d[:,2:3]]),multi_dot([B(self.P_rbs_rockr,self.Mbar_rbs_rockr_jcs_d[:,1:2]).T,A(self.P_ground),self.Mbar_ground_jcs_d[:,2:3]])]]),self.L_jcs_d])
        self.F_rbs_rockr_jcs_d = Q_rbs_rockr_jcs_d[0:3,0:1]
        Te_rbs_rockr_jcs_d = Q_rbs_rockr_jcs_d[3:7,0:1]
        self.T_rbs_rockr_jcs_d = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_rockr),self.ubar_rbs_rockr_jcs_d])),self.F_rbs_rockr_jcs_d]) + 0.5*multi_dot([E(self.P_rbs_rockr),Te_rbs_rockr_jcs_d]))

        self.reactions = {'F_ground_jcs_a' : self.F_ground_jcs_a,
                        'T_ground_jcs_a' : self.T_ground_jcs_a,
                        'F_rbs_crank_jcs_b' : self.F_rbs_crank_jcs_b,
                        'T_rbs_crank_jcs_b' : self.T_rbs_crank_jcs_b,
                        'F_rbs_conct_jcs_c' : self.F_rbs_conct_jcs_c,
                        'T_rbs_conct_jcs_c' : self.T_rbs_conct_jcs_c,
                        'F_rbs_rockr_jcs_d' : self.F_rbs_rockr_jcs_d,
                        'T_rbs_rockr_jcs_d' : self.T_rbs_rockr_jcs_d}

