
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

        self.n  = 14
        self.nc = 12
        self.nrows = 10
        self.ncols = 2*3
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbs_block_jcs_cyl_joint', 'T_rbs_block_jcs_cyl_joint', 'F_rbs_dummy_jcs_fixed', 'T_rbs_dummy_jcs_fixed']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_block = indicies_map[p+'rbs_block']
        self.rbs_dummy = indicies_map[p+'rbs_dummy']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 9])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.rbs_block*2, self.rbs_block*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.rbs_block*2, self.rbs_block*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.rbs_block*2, self.rbs_block*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.rbs_block*2, self.rbs_block*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_dummy*2, self.rbs_dummy*2+1, self.rbs_block*2+1, self.rbs_dummy*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_block_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_block]],dtype=np.float64)
        self.F_rbs_dummy_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_dummy]],dtype=np.float64)

        self.Mbar_rbs_block_jcs_cyl_joint = multi_dot([A(config.P_rbs_block).T,triad(config.ax1_jcs_cyl_joint)])
        self.Mbar_rbs_dummy_jcs_cyl_joint = multi_dot([A(config.P_rbs_dummy).T,triad(config.ax1_jcs_cyl_joint)])
        self.ubar_rbs_block_jcs_cyl_joint = (multi_dot([A(config.P_rbs_block).T,config.pt1_jcs_cyl_joint]) + -1*multi_dot([A(config.P_rbs_block).T,config.R_rbs_block]))
        self.ubar_rbs_dummy_jcs_cyl_joint = (multi_dot([A(config.P_rbs_dummy).T,config.pt1_jcs_cyl_joint]) + -1*multi_dot([A(config.P_rbs_dummy).T,config.R_rbs_dummy]))
        self.ubar_rbs_block_fas_spring = (multi_dot([A(config.P_rbs_block).T,config.pt1_fas_spring]) + -1*multi_dot([A(config.P_rbs_block).T,config.R_rbs_block]))
        self.ubar_rbs_dummy_fas_spring = (multi_dot([A(config.P_rbs_dummy).T,config.pt2_fas_spring]) + -1*multi_dot([A(config.P_rbs_dummy).T,config.R_rbs_dummy]))
        self.Mbar_rbs_dummy_jcs_fixed = multi_dot([A(config.P_rbs_dummy).T,triad(config.ax1_jcs_fixed)])
        self.Mbar_vbs_ground_jcs_fixed = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_fixed)])
        self.ubar_rbs_dummy_jcs_fixed = (multi_dot([A(config.P_rbs_dummy).T,config.pt1_jcs_fixed]) + -1*multi_dot([A(config.P_rbs_dummy).T,config.R_rbs_dummy]))
        self.ubar_vbs_ground_jcs_fixed = (multi_dot([A(config.P_vbs_ground).T,config.pt1_jcs_fixed]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))

    
    def set_gen_coordinates(self,q):
        self.R_rbs_block = q[0:3,0:1]
        self.P_rbs_block = q[3:7,0:1]
        self.R_rbs_dummy = q[7:10,0:1]
        self.P_rbs_dummy = q[10:14,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_block = qd[0:3,0:1]
        self.Pd_rbs_block = qd[3:7,0:1]
        self.Rd_rbs_dummy = qd[7:10,0:1]
        self.Pd_rbs_dummy = qd[10:14,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_block = qdd[0:3,0:1]
        self.Pdd_rbs_block = qdd[3:7,0:1]
        self.Rdd_rbs_dummy = qdd[7:10,0:1]
        self.Pdd_rbs_dummy = qdd[10:14,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_cyl_joint = Lambda[0:4,0:1]
        self.L_jcs_fixed = Lambda[4:10,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.Mbar_rbs_block_jcs_cyl_joint[:,0:1].T
        x1 = self.P_rbs_block
        x2 = A(x1)
        x3 = x2.T
        x4 = self.P_rbs_dummy
        x5 = A(x4)
        x6 = self.Mbar_rbs_dummy_jcs_cyl_joint[:,2:3]
        x7 = self.Mbar_rbs_block_jcs_cyl_joint[:,1:2].T
        x8 = self.R_rbs_dummy
        x9 = (self.R_rbs_block + -1*x8 + multi_dot([x2,self.ubar_rbs_block_jcs_cyl_joint]) + -1*multi_dot([x5,self.ubar_rbs_dummy_jcs_cyl_joint]))
        x10 = A(self.P_vbs_ground)
        x11 = self.Mbar_rbs_dummy_jcs_fixed[:,0:1].T
        x12 = x5.T
        x13 = self.Mbar_vbs_ground_jcs_fixed[:,2:3]
        x14 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [multi_dot([x0,x3,x5,x6]),
        multi_dot([x7,x3,x5,x6]),
        multi_dot([x0,x3,x9]),
        multi_dot([x7,x3,x9]),
        (x8 + -1*self.R_vbs_ground + multi_dot([x5,self.ubar_rbs_dummy_jcs_fixed]) + -1*multi_dot([x10,self.ubar_vbs_ground_jcs_fixed])),
        multi_dot([x11,x12,x10,x13]),
        multi_dot([self.Mbar_rbs_dummy_jcs_fixed[:,1:2].T,x12,x10,x13]),
        multi_dot([x11,x12,x10,self.Mbar_vbs_ground_jcs_fixed[:,1:2]]),
        (x14 + multi_dot([x1.T,x1])),
        (x14 + multi_dot([x4.T,x4]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((1,1),dtype=np.float64)

        self.vel_eq_blocks = [v0,
        v0,
        v0,
        v0,
        np.zeros((3,1),dtype=np.float64),
        v0,
        v0,
        v0,
        v0,
        v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_rbs_block_jcs_cyl_joint[:,0:1]
        a1 = a0.T
        a2 = self.P_rbs_block
        a3 = A(a2).T
        a4 = self.Pd_rbs_dummy
        a5 = self.Mbar_rbs_dummy_jcs_cyl_joint[:,2:3]
        a6 = B(a4,a5)
        a7 = a5.T
        a8 = self.P_rbs_dummy
        a9 = A(a8).T
        a10 = self.Pd_rbs_block
        a11 = B(a10,a0)
        a12 = a10.T
        a13 = B(a2,a0).T
        a14 = B(a8,a5)
        a15 = self.Mbar_rbs_block_jcs_cyl_joint[:,1:2]
        a16 = a15.T
        a17 = B(a10,a15)
        a18 = B(a2,a15).T
        a19 = self.ubar_rbs_block_jcs_cyl_joint
        a20 = self.ubar_rbs_dummy_jcs_cyl_joint
        a21 = (multi_dot([B(a10,a19),a10]) + -1*multi_dot([B(a4,a20),a4]))
        a22 = (self.Rd_rbs_block + -1*self.Rd_rbs_dummy + multi_dot([B(a2,a19),a10]) + -1*multi_dot([B(a8,a20),a4]))
        a23 = (self.R_rbs_block.T + -1*self.R_rbs_dummy.T + multi_dot([a19.T,a3]) + -1*multi_dot([a20.T,a9]))
        a24 = self.Pd_vbs_ground
        a25 = self.Mbar_vbs_ground_jcs_fixed[:,2:3]
        a26 = a25.T
        a27 = self.P_vbs_ground
        a28 = A(a27).T
        a29 = self.Mbar_rbs_dummy_jcs_fixed[:,0:1]
        a30 = B(a4,a29)
        a31 = a29.T
        a32 = B(a24,a25)
        a33 = a4.T
        a34 = B(a8,a29).T
        a35 = B(a27,a25)
        a36 = self.Mbar_rbs_dummy_jcs_fixed[:,1:2]
        a37 = self.Mbar_vbs_ground_jcs_fixed[:,1:2]

        self.acc_eq_blocks = [(multi_dot([a1,a3,a6,a4]) + multi_dot([a7,a9,a11,a10]) + 2*multi_dot([a12,a13,a14,a4])),
        (multi_dot([a16,a3,a6,a4]) + multi_dot([a7,a9,a17,a10]) + 2*multi_dot([a12,a18,a14,a4])),
        (multi_dot([a1,a3,a21]) + 2*multi_dot([a12,a13,a22]) + multi_dot([a23,a11,a10])),
        (multi_dot([a16,a3,a21]) + 2*multi_dot([a12,a18,a22]) + multi_dot([a23,a17,a10])),
        (multi_dot([B(a4,self.ubar_rbs_dummy_jcs_fixed),a4]) + -1*multi_dot([B(a24,self.ubar_vbs_ground_jcs_fixed),a24])),
        (multi_dot([a26,a28,a30,a4]) + multi_dot([a31,a9,a32,a24]) + 2*multi_dot([a33,a34,a35,a24])),
        (multi_dot([a26,a28,B(a4,a36),a4]) + multi_dot([a36.T,a9,a32,a24]) + 2*multi_dot([a33,B(a8,a36).T,a35,a24])),
        (multi_dot([a37.T,a28,a30,a4]) + multi_dot([a31,a9,B(a24,a37),a24]) + 2*multi_dot([a33,a34,B(a27,a37),a24])),
        2*multi_dot([a12,a10]),
        2*multi_dot([a33,a4])]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_rbs_dummy_jcs_cyl_joint[:,2:3]
        j2 = j1.T
        j3 = self.P_rbs_dummy
        j4 = A(j3).T
        j5 = self.P_rbs_block
        j6 = self.Mbar_rbs_block_jcs_cyl_joint[:,0:1]
        j7 = B(j5,j6)
        j8 = self.Mbar_rbs_block_jcs_cyl_joint[:,1:2]
        j9 = B(j5,j8)
        j10 = j6.T
        j11 = A(j5).T
        j12 = multi_dot([j10,j11])
        j13 = self.ubar_rbs_block_jcs_cyl_joint
        j14 = B(j5,j13)
        j15 = self.ubar_rbs_dummy_jcs_cyl_joint
        j16 = (self.R_rbs_block.T + -1*self.R_rbs_dummy.T + multi_dot([j13.T,j11]) + -1*multi_dot([j15.T,j4]))
        j17 = j8.T
        j18 = multi_dot([j17,j11])
        j19 = B(j3,j1)
        j20 = B(j3,j15)
        j21 = np.eye(3,dtype=np.float64)
        j22 = self.Mbar_vbs_ground_jcs_fixed[:,2:3]
        j23 = j22.T
        j24 = self.P_vbs_ground
        j25 = A(j24).T
        j26 = self.Mbar_rbs_dummy_jcs_fixed[:,0:1]
        j27 = B(j3,j26)
        j28 = self.Mbar_rbs_dummy_jcs_fixed[:,1:2]
        j29 = self.Mbar_vbs_ground_jcs_fixed[:,1:2]
        j30 = j26.T
        j31 = B(j24,j22)

        self.jac_eq_blocks = [j0,
        multi_dot([j2,j4,j7]),
        j0,
        multi_dot([j10,j11,j19]),
        j0,
        multi_dot([j2,j4,j9]),
        j0,
        multi_dot([j17,j11,j19]),
        j12,
        (multi_dot([j10,j11,j14]) + multi_dot([j16,j7])),
        -1*j12,
        -1*multi_dot([j10,j11,j20]),
        j18,
        (multi_dot([j17,j11,j14]) + multi_dot([j16,j9])),
        -1*j18,
        -1*multi_dot([j17,j11,j20]),
        -1*j21,
        -1*B(j24,self.ubar_vbs_ground_jcs_fixed),
        j21,
        B(j3,self.ubar_rbs_dummy_jcs_fixed),
        j0,
        multi_dot([j30,j4,j31]),
        j0,
        multi_dot([j23,j25,j27]),
        j0,
        multi_dot([j28.T,j4,j31]),
        j0,
        multi_dot([j23,j25,B(j3,j28)]),
        j0,
        multi_dot([j30,j4,B(j24,j29)]),
        j0,
        multi_dot([j29.T,j25,j27]),
        2*j5.T,
        2*j3.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3,dtype=np.float64)
        m1 = G(self.P_rbs_block)
        m2 = G(self.P_rbs_dummy)

        self.mass_eq_blocks = [config.m_rbs_block*m0,
        4*multi_dot([m1.T,config.Jbar_rbs_block,m1]),
        config.m_rbs_dummy*m0,
        4*multi_dot([m2.T,config.Jbar_rbs_dummy,m2])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = self.R_rbs_block
        f1 = self.R_rbs_dummy
        f2 = self.ubar_rbs_block_fas_spring
        f3 = self.P_rbs_block
        f4 = A(f3)
        f5 = self.ubar_rbs_dummy_fas_spring
        f6 = self.P_rbs_dummy
        f7 = A(f6)
        f8 = (f0.T + -1*f1.T + multi_dot([f2.T,f4.T]) + -1*multi_dot([f5.T,f7.T]))
        f9 = multi_dot([f4,f2])
        f10 = multi_dot([f7,f5])
        f11 = (f0 + -1*f1 + f9 + -1*f10)
        f12 = (multi_dot([f8,f11]))**(1.0/2.0)
        f13 = f12**(-1)
        f14 = self.Pd_rbs_block
        f15 = self.Pd_rbs_dummy
        f16 = config.Fd_fas_spring(multi_dot([f13.T,f8,(self.Rd_rbs_block + -1*self.Rd_rbs_dummy + multi_dot([B(f3,f2),f14]) + -1*multi_dot([B(f6,f5),f15]))])) + config.Fs_fas_spring((config.fas_spring_FL + -1*f12))
        f17 = f16*multi_dot([f11,f13])
        f18 = G(f14)
        f19 = G(f15)

        self.frc_eq_blocks = [(self.F_rbs_block_gravity + f17),
        (8*multi_dot([f18.T,config.Jbar_rbs_block,f18,f3]) + 2*multi_dot([G(f3).T,(config.T_rbs_block_fas_spring + f16*multi_dot([skew(f9).T,f11,f13]))])),
        (self.F_rbs_dummy_gravity + np.zeros((3,1),dtype=np.float64) + -1*f17),
        (np.zeros((4,1),dtype=np.float64) + 8*multi_dot([f19.T,config.Jbar_rbs_dummy,f19,f6]) + 2*multi_dot([G(f6).T,(config.T_rbs_dummy_fas_spring + -1*f16*multi_dot([skew(f10).T,f11,f13]))]))]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_block_jcs_cyl_joint = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbs_block),self.Mbar_rbs_block_jcs_cyl_joint[:,0:1]]),multi_dot([A(self.P_rbs_block),self.Mbar_rbs_block_jcs_cyl_joint[:,1:2]])],[multi_dot([B(self.P_rbs_block,self.Mbar_rbs_block_jcs_cyl_joint[:,0:1]).T,A(self.P_rbs_dummy),self.Mbar_rbs_dummy_jcs_cyl_joint[:,2:3]]),multi_dot([B(self.P_rbs_block,self.Mbar_rbs_block_jcs_cyl_joint[:,1:2]).T,A(self.P_rbs_dummy),self.Mbar_rbs_dummy_jcs_cyl_joint[:,2:3]]),(multi_dot([B(self.P_rbs_block,self.Mbar_rbs_block_jcs_cyl_joint[:,0:1]).T,(-1*self.R_rbs_dummy + multi_dot([A(self.P_rbs_block),self.ubar_rbs_block_jcs_cyl_joint]) + -1*multi_dot([A(self.P_rbs_dummy),self.ubar_rbs_dummy_jcs_cyl_joint]) + self.R_rbs_block)]) + multi_dot([B(self.P_rbs_block,self.ubar_rbs_block_jcs_cyl_joint).T,A(self.P_rbs_block),self.Mbar_rbs_block_jcs_cyl_joint[:,0:1]])),(multi_dot([B(self.P_rbs_block,self.Mbar_rbs_block_jcs_cyl_joint[:,1:2]).T,(-1*self.R_rbs_dummy + multi_dot([A(self.P_rbs_block),self.ubar_rbs_block_jcs_cyl_joint]) + -1*multi_dot([A(self.P_rbs_dummy),self.ubar_rbs_dummy_jcs_cyl_joint]) + self.R_rbs_block)]) + multi_dot([B(self.P_rbs_block,self.ubar_rbs_block_jcs_cyl_joint).T,A(self.P_rbs_block),self.Mbar_rbs_block_jcs_cyl_joint[:,1:2]]))]]),self.L_jcs_cyl_joint])
        self.F_rbs_block_jcs_cyl_joint = Q_rbs_block_jcs_cyl_joint[0:3,0:1]
        Te_rbs_block_jcs_cyl_joint = Q_rbs_block_jcs_cyl_joint[3:7,0:1]
        self.T_rbs_block_jcs_cyl_joint = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_block),self.ubar_rbs_block_jcs_cyl_joint])),self.F_rbs_block_jcs_cyl_joint]) + 0.5*multi_dot([E(self.P_rbs_block),Te_rbs_block_jcs_cyl_joint]))
        Q_rbs_dummy_jcs_fixed = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbs_dummy,self.ubar_rbs_dummy_jcs_fixed).T,multi_dot([B(self.P_rbs_dummy,self.Mbar_rbs_dummy_jcs_fixed[:,0:1]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_fixed[:,2:3]]),multi_dot([B(self.P_rbs_dummy,self.Mbar_rbs_dummy_jcs_fixed[:,1:2]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_fixed[:,2:3]]),multi_dot([B(self.P_rbs_dummy,self.Mbar_rbs_dummy_jcs_fixed[:,0:1]).T,A(self.P_vbs_ground),self.Mbar_vbs_ground_jcs_fixed[:,1:2]])]]),self.L_jcs_fixed])
        self.F_rbs_dummy_jcs_fixed = Q_rbs_dummy_jcs_fixed[0:3,0:1]
        Te_rbs_dummy_jcs_fixed = Q_rbs_dummy_jcs_fixed[3:7,0:1]
        self.T_rbs_dummy_jcs_fixed = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_dummy),self.ubar_rbs_dummy_jcs_fixed])),self.F_rbs_dummy_jcs_fixed]) + 0.5*multi_dot([E(self.P_rbs_dummy),Te_rbs_dummy_jcs_fixed]))

        self.reactions = {'F_rbs_block_jcs_cyl_joint':self.F_rbs_block_jcs_cyl_joint,'T_rbs_block_jcs_cyl_joint':self.T_rbs_block_jcs_cyl_joint,'F_rbs_dummy_jcs_fixed':self.F_rbs_dummy_jcs_fixed,'T_rbs_dummy_jcs_fixed':self.T_rbs_dummy_jcs_fixed}

