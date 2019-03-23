
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

        self.n  = 21
        self.nc = 20
        self.nrows = 14
        self.ncols = 2*5
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbr_rocker_jcr_rocker_ch', 'T_rbr_rocker_jcr_rocker_ch', 'F_rbr_rocker_jcs_rc_sph', 'T_rbr_rocker_jcs_rc_sph', 'F_rbl_rocker_jcl_rocker_ch', 'T_rbl_rocker_jcl_rocker_ch', 'F_rbl_rocker_jcs_rc_cyl', 'T_rbl_rocker_jcs_rc_cyl']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_coupler = indicies_map[p+'rbs_coupler']
        self.rbr_rocker = indicies_map[p+'rbr_rocker']
        self.rbl_rocker = indicies_map[p+'rbl_rocker']
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 12, 13])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbr_rocker*2, self.rbr_rocker*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.vbs_chassis*2, self.vbs_chassis*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbs_coupler*2, self.rbs_coupler*2+1, self.rbl_rocker*2, self.rbl_rocker*2+1, self.rbs_coupler*2+1, self.rbr_rocker*2+1, self.rbl_rocker*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_coupler_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_coupler]],dtype=np.float64)
        self.F_rbr_rocker_gravity = np.array([[0], [0], [-9810.0*config.m_rbr_rocker]],dtype=np.float64)
        self.F_rbl_rocker_gravity = np.array([[0], [0], [-9810.0*config.m_rbl_rocker]],dtype=np.float64)

        self.Mbar_rbr_rocker_jcr_rocker_ch = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcr_rocker_ch)])
        self.Mbar_vbs_chassis_jcr_rocker_ch = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcr_rocker_ch)])
        self.ubar_rbr_rocker_jcr_rocker_ch = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcr_rocker_ch]) + -1*multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_vbs_chassis_jcr_rocker_ch = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcr_rocker_ch]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbr_rocker_jcs_rc_sph = multi_dot([A(config.P_rbr_rocker).T,triad(config.ax1_jcs_rc_sph)])
        self.Mbar_rbs_coupler_jcs_rc_sph = multi_dot([A(config.P_rbs_coupler).T,triad(config.ax1_jcs_rc_sph)])
        self.ubar_rbr_rocker_jcs_rc_sph = (multi_dot([A(config.P_rbr_rocker).T,config.pt1_jcs_rc_sph]) + -1*multi_dot([A(config.P_rbr_rocker).T,config.R_rbr_rocker]))
        self.ubar_rbs_coupler_jcs_rc_sph = (multi_dot([A(config.P_rbs_coupler).T,config.pt1_jcs_rc_sph]) + -1*multi_dot([A(config.P_rbs_coupler).T,config.R_rbs_coupler]))
        self.Mbar_rbl_rocker_jcl_rocker_ch = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcl_rocker_ch)])
        self.Mbar_vbs_chassis_jcl_rocker_ch = multi_dot([A(config.P_vbs_chassis).T,triad(config.ax1_jcl_rocker_ch)])
        self.ubar_rbl_rocker_jcl_rocker_ch = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcl_rocker_ch]) + -1*multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_vbs_chassis_jcl_rocker_ch = (multi_dot([A(config.P_vbs_chassis).T,config.pt1_jcl_rocker_ch]) + -1*multi_dot([A(config.P_vbs_chassis).T,config.R_vbs_chassis]))
        self.Mbar_rbl_rocker_jcs_rc_cyl = multi_dot([A(config.P_rbl_rocker).T,triad(config.ax1_jcs_rc_cyl)])
        self.Mbar_rbs_coupler_jcs_rc_cyl = multi_dot([A(config.P_rbs_coupler).T,triad(config.ax1_jcs_rc_cyl)])
        self.ubar_rbl_rocker_jcs_rc_cyl = (multi_dot([A(config.P_rbl_rocker).T,config.pt1_jcs_rc_cyl]) + -1*multi_dot([A(config.P_rbl_rocker).T,config.R_rbl_rocker]))
        self.ubar_rbs_coupler_jcs_rc_cyl = (multi_dot([A(config.P_rbs_coupler).T,config.pt1_jcs_rc_cyl]) + -1*multi_dot([A(config.P_rbs_coupler).T,config.R_rbs_coupler]))

    
    def set_gen_coordinates(self,q):
        self.R_rbs_coupler = q[0:3,0:1]
        self.P_rbs_coupler = q[3:7,0:1]
        self.R_rbr_rocker = q[7:10,0:1]
        self.P_rbr_rocker = q[10:14,0:1]
        self.R_rbl_rocker = q[14:17,0:1]
        self.P_rbl_rocker = q[17:21,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_coupler = qd[0:3,0:1]
        self.Pd_rbs_coupler = qd[3:7,0:1]
        self.Rd_rbr_rocker = qd[7:10,0:1]
        self.Pd_rbr_rocker = qd[10:14,0:1]
        self.Rd_rbl_rocker = qd[14:17,0:1]
        self.Pd_rbl_rocker = qd[17:21,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_coupler = qdd[0:3,0:1]
        self.Pdd_rbs_coupler = qdd[3:7,0:1]
        self.Rdd_rbr_rocker = qdd[7:10,0:1]
        self.Pdd_rbr_rocker = qdd[10:14,0:1]
        self.Rdd_rbl_rocker = qdd[14:17,0:1]
        self.Pdd_rbl_rocker = qdd[17:21,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcr_rocker_ch = Lambda[0:5,0:1]
        self.L_jcs_rc_sph = Lambda[5:8,0:1]
        self.L_jcl_rocker_ch = Lambda[8:13,0:1]
        self.L_jcs_rc_cyl = Lambda[13:17,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.R_rbr_rocker
        x1 = -1*self.R_vbs_chassis
        x2 = self.P_rbr_rocker
        x3 = A(x2)
        x4 = A(self.P_vbs_chassis)
        x5 = x3.T
        x6 = self.Mbar_vbs_chassis_jcr_rocker_ch[:,2:3]
        x7 = -1*self.R_rbs_coupler
        x8 = self.P_rbs_coupler
        x9 = A(x8)
        x10 = self.R_rbl_rocker
        x11 = self.P_rbl_rocker
        x12 = A(x11)
        x13 = x12.T
        x14 = self.Mbar_vbs_chassis_jcl_rocker_ch[:,2:3]
        x15 = self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1].T
        x16 = self.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]
        x17 = self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2].T
        x18 = (x10 + x7 + multi_dot([x12,self.ubar_rbl_rocker_jcs_rc_cyl]) + -1*multi_dot([x9,self.ubar_rbs_coupler_jcs_rc_cyl]))
        x19 = -1*np.eye(1,dtype=np.float64)

        self.pos_eq_blocks = [(x0 + x1 + multi_dot([x3,self.ubar_rbr_rocker_jcr_rocker_ch]) + -1*multi_dot([x4,self.ubar_vbs_chassis_jcr_rocker_ch])),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_ch[:,0:1].T,x5,x4,x6]),
        multi_dot([self.Mbar_rbr_rocker_jcr_rocker_ch[:,1:2].T,x5,x4,x6]),
        (x0 + x7 + multi_dot([x3,self.ubar_rbr_rocker_jcs_rc_sph]) + -1*multi_dot([x9,self.ubar_rbs_coupler_jcs_rc_sph])),
        (x10 + x1 + multi_dot([x12,self.ubar_rbl_rocker_jcl_rocker_ch]) + -1*multi_dot([x4,self.ubar_vbs_chassis_jcl_rocker_ch])),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_ch[:,0:1].T,x13,x4,x14]),
        multi_dot([self.Mbar_rbl_rocker_jcl_rocker_ch[:,1:2].T,x13,x4,x14]),
        multi_dot([x15,x13,x9,x16]),
        multi_dot([x17,x13,x9,x16]),
        multi_dot([x15,x13,x18]),
        multi_dot([x17,x13,x18]),
        (x19 + multi_dot([x8.T,x8])),
        (x19 + multi_dot([x2.T,x2])),
        (x19 + multi_dot([x11.T,x11]))]

    
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
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1,
        v1]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbr_rocker
        a1 = self.Pd_vbs_chassis
        a2 = self.Mbar_vbs_chassis_jcr_rocker_ch[:,2:3]
        a3 = a2.T
        a4 = self.P_vbs_chassis
        a5 = A(a4).T
        a6 = self.Mbar_rbr_rocker_jcr_rocker_ch[:,0:1]
        a7 = self.P_rbr_rocker
        a8 = A(a7).T
        a9 = B(a1,a2)
        a10 = a0.T
        a11 = B(a4,a2)
        a12 = self.Mbar_rbr_rocker_jcr_rocker_ch[:,1:2]
        a13 = self.Pd_rbs_coupler
        a14 = self.Pd_rbl_rocker
        a15 = self.Mbar_rbl_rocker_jcl_rocker_ch[:,0:1]
        a16 = self.P_rbl_rocker
        a17 = A(a16).T
        a18 = self.Mbar_vbs_chassis_jcl_rocker_ch[:,2:3]
        a19 = B(a1,a18)
        a20 = a18.T
        a21 = a14.T
        a22 = B(a4,a18)
        a23 = self.Mbar_rbl_rocker_jcl_rocker_ch[:,1:2]
        a24 = self.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]
        a25 = a24.T
        a26 = self.P_rbs_coupler
        a27 = A(a26).T
        a28 = self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]
        a29 = B(a14,a28)
        a30 = a28.T
        a31 = B(a13,a24)
        a32 = B(a16,a28).T
        a33 = B(a26,a24)
        a34 = self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]
        a35 = B(a14,a34)
        a36 = a34.T
        a37 = B(a16,a34).T
        a38 = self.ubar_rbl_rocker_jcs_rc_cyl
        a39 = self.ubar_rbs_coupler_jcs_rc_cyl
        a40 = (multi_dot([B(a14,a38),a14]) + -1*multi_dot([B(a13,a39),a13]))
        a41 = (self.Rd_rbl_rocker + -1*self.Rd_rbs_coupler + multi_dot([B(a16,a38),a14]) + -1*multi_dot([B(a26,a39),a13]))
        a42 = (self.R_rbl_rocker.T + -1*self.R_rbs_coupler.T + multi_dot([a38.T,a17]) + -1*multi_dot([a39.T,a27]))

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_rbr_rocker_jcr_rocker_ch),a0]) + -1*multi_dot([B(a1,self.ubar_vbs_chassis_jcr_rocker_ch),a1])),
        (multi_dot([a3,a5,B(a0,a6),a0]) + multi_dot([a6.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a6).T,a11,a1])),
        (multi_dot([a3,a5,B(a0,a12),a0]) + multi_dot([a12.T,a8,a9,a1]) + 2*multi_dot([a10,B(a7,a12).T,a11,a1])),
        (multi_dot([B(a0,self.ubar_rbr_rocker_jcs_rc_sph),a0]) + -1*multi_dot([B(a13,self.ubar_rbs_coupler_jcs_rc_sph),a13])),
        (multi_dot([B(a14,self.ubar_rbl_rocker_jcl_rocker_ch),a14]) + -1*multi_dot([B(a1,self.ubar_vbs_chassis_jcl_rocker_ch),a1])),
        (multi_dot([a15.T,a17,a19,a1]) + multi_dot([a20,a5,B(a14,a15),a14]) + 2*multi_dot([a21,B(a16,a15).T,a22,a1])),
        (multi_dot([a23.T,a17,a19,a1]) + multi_dot([a20,a5,B(a14,a23),a14]) + 2*multi_dot([a21,B(a16,a23).T,a22,a1])),
        (multi_dot([a25,a27,a29,a14]) + multi_dot([a30,a17,a31,a13]) + 2*multi_dot([a21,a32,a33,a13])),
        (multi_dot([a25,a27,a35,a14]) + multi_dot([a36,a17,a31,a13]) + 2*multi_dot([a21,a37,a33,a13])),
        (multi_dot([a30,a17,a40]) + 2*multi_dot([a21,a32,a41]) + multi_dot([a42,a29,a14])),
        (multi_dot([a36,a17,a40]) + 2*multi_dot([a21,a37,a41]) + multi_dot([a42,a35,a14])),
        2*multi_dot([a13.T,a13]),
        2*multi_dot([a10,a0]),
        2*multi_dot([a21,a14])]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbr_rocker
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = self.Mbar_vbs_chassis_jcr_rocker_ch[:,2:3]
        j4 = j3.T
        j5 = self.P_vbs_chassis
        j6 = A(j5).T
        j7 = self.Mbar_rbr_rocker_jcr_rocker_ch[:,0:1]
        j8 = self.Mbar_rbr_rocker_jcr_rocker_ch[:,1:2]
        j9 = -1*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = self.P_rbs_coupler
        j13 = self.P_rbl_rocker
        j14 = self.Mbar_vbs_chassis_jcl_rocker_ch[:,2:3]
        j15 = j14.T
        j16 = self.Mbar_rbl_rocker_jcl_rocker_ch[:,0:1]
        j17 = self.Mbar_rbl_rocker_jcl_rocker_ch[:,1:2]
        j18 = A(j13).T
        j19 = B(j5,j14)
        j20 = self.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]
        j21 = j20.T
        j22 = A(j12).T
        j23 = self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]
        j24 = B(j13,j23)
        j25 = self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]
        j26 = B(j13,j25)
        j27 = j23.T
        j28 = multi_dot([j27,j18])
        j29 = self.ubar_rbl_rocker_jcs_rc_cyl
        j30 = B(j13,j29)
        j31 = self.ubar_rbs_coupler_jcs_rc_cyl
        j32 = (self.R_rbl_rocker.T + -1*self.R_rbs_coupler.T + multi_dot([j29.T,j18]) + -1*multi_dot([j31.T,j22]))
        j33 = j25.T
        j34 = multi_dot([j33,j18])
        j35 = B(j12,j20)
        j36 = B(j12,j31)

        self.jac_eq_blocks = [j0,
        B(j1,self.ubar_rbr_rocker_jcr_rocker_ch),
        j9,
        -1*B(j5,self.ubar_vbs_chassis_jcr_rocker_ch),
        j2,
        multi_dot([j4,j6,B(j1,j7)]),
        j2,
        multi_dot([j7.T,j10,j11]),
        j2,
        multi_dot([j4,j6,B(j1,j8)]),
        j2,
        multi_dot([j8.T,j10,j11]),
        j9,
        -1*B(j12,self.ubar_rbs_coupler_jcs_rc_sph),
        j0,
        B(j1,self.ubar_rbr_rocker_jcs_rc_sph),
        j0,
        B(j13,self.ubar_rbl_rocker_jcl_rocker_ch),
        j9,
        -1*B(j5,self.ubar_vbs_chassis_jcl_rocker_ch),
        j2,
        multi_dot([j15,j6,B(j13,j16)]),
        j2,
        multi_dot([j16.T,j18,j19]),
        j2,
        multi_dot([j15,j6,B(j13,j17)]),
        j2,
        multi_dot([j17.T,j18,j19]),
        j2,
        multi_dot([j27,j18,j35]),
        j2,
        multi_dot([j21,j22,j24]),
        j2,
        multi_dot([j33,j18,j35]),
        j2,
        multi_dot([j21,j22,j26]),
        -1*j28,
        -1*multi_dot([j27,j18,j36]),
        j28,
        (multi_dot([j27,j18,j30]) + multi_dot([j32,j24])),
        -1*j34,
        -1*multi_dot([j33,j18,j36]),
        j34,
        (multi_dot([j33,j18,j30]) + multi_dot([j32,j26])),
        2*j12.T,
        2*j1.T,
        2*j13.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = np.eye(3,dtype=np.float64)
        m1 = G(self.P_rbs_coupler)
        m2 = G(self.P_rbr_rocker)
        m3 = G(self.P_rbl_rocker)

        self.mass_eq_blocks = [config.m_rbs_coupler*m0,
        4*multi_dot([m1.T,config.Jbar_rbs_coupler,m1]),
        config.m_rbr_rocker*m0,
        4*multi_dot([m2.T,config.Jbar_rbr_rocker,m2]),
        config.m_rbl_rocker*m0,
        4*multi_dot([m3.T,config.Jbar_rbl_rocker,m3])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_coupler)
        f1 = G(self.Pd_rbr_rocker)
        f2 = G(self.Pd_rbl_rocker)

        self.frc_eq_blocks = [self.F_rbs_coupler_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_coupler,f0,self.P_rbs_coupler]),
        self.F_rbr_rocker_gravity,
        8*multi_dot([f1.T,config.Jbar_rbr_rocker,f1,self.P_rbr_rocker]),
        self.F_rbl_rocker_gravity,
        8*multi_dot([f2.T,config.Jbar_rbl_rocker,f2,self.P_rbl_rocker])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbr_rocker_jcr_rocker_ch = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcr_rocker_ch).T,multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_ch[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_ch[:,2:3]]),multi_dot([B(self.P_rbr_rocker,self.Mbar_rbr_rocker_jcr_rocker_ch[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcr_rocker_ch[:,2:3]])]]),self.L_jcr_rocker_ch])
        self.F_rbr_rocker_jcr_rocker_ch = Q_rbr_rocker_jcr_rocker_ch[0:3,0:1]
        Te_rbr_rocker_jcr_rocker_ch = Q_rbr_rocker_jcr_rocker_ch[3:7,0:1]
        self.T_rbr_rocker_jcr_rocker_ch = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcr_rocker_ch])),self.F_rbr_rocker_jcr_rocker_ch]) + 0.5*multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcr_rocker_ch]))
        Q_rbr_rocker_jcs_rc_sph = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbr_rocker,self.ubar_rbr_rocker_jcs_rc_sph).T]]),self.L_jcs_rc_sph])
        self.F_rbr_rocker_jcs_rc_sph = Q_rbr_rocker_jcs_rc_sph[0:3,0:1]
        Te_rbr_rocker_jcs_rc_sph = Q_rbr_rocker_jcs_rc_sph[3:7,0:1]
        self.T_rbr_rocker_jcs_rc_sph = (-1*multi_dot([skew(multi_dot([A(self.P_rbr_rocker),self.ubar_rbr_rocker_jcs_rc_sph])),self.F_rbr_rocker_jcs_rc_sph]) + 0.5*multi_dot([E(self.P_rbr_rocker),Te_rbr_rocker_jcs_rc_sph]))
        Q_rbl_rocker_jcl_rocker_ch = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64),np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T],[B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcl_rocker_ch).T,multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_ch[:,0:1]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_ch[:,2:3]]),multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcl_rocker_ch[:,1:2]).T,A(self.P_vbs_chassis),self.Mbar_vbs_chassis_jcl_rocker_ch[:,2:3]])]]),self.L_jcl_rocker_ch])
        self.F_rbl_rocker_jcl_rocker_ch = Q_rbl_rocker_jcl_rocker_ch[0:3,0:1]
        Te_rbl_rocker_jcl_rocker_ch = Q_rbl_rocker_jcl_rocker_ch[3:7,0:1]
        self.T_rbl_rocker_jcl_rocker_ch = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcl_rocker_ch])),self.F_rbl_rocker_jcl_rocker_ch]) + 0.5*multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcl_rocker_ch]))
        Q_rbl_rocker_jcs_rc_cyl = -1*multi_dot([np.bmat([[np.zeros((1,3),dtype=np.float64).T,np.zeros((1,3),dtype=np.float64).T,multi_dot([A(self.P_rbl_rocker),self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]]),multi_dot([A(self.P_rbl_rocker),self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]])],[multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]).T,A(self.P_rbs_coupler),self.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]]),multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]).T,A(self.P_rbs_coupler),self.Mbar_rbs_coupler_jcs_rc_cyl[:,2:3]]),(multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]).T,(-1*self.R_rbs_coupler + multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcs_rc_cyl]) + -1*multi_dot([A(self.P_rbs_coupler),self.ubar_rbs_coupler_jcs_rc_cyl]) + self.R_rbl_rocker)]) + multi_dot([B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcs_rc_cyl).T,A(self.P_rbl_rocker),self.Mbar_rbl_rocker_jcs_rc_cyl[:,0:1]])),(multi_dot([B(self.P_rbl_rocker,self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]).T,(-1*self.R_rbs_coupler + multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcs_rc_cyl]) + -1*multi_dot([A(self.P_rbs_coupler),self.ubar_rbs_coupler_jcs_rc_cyl]) + self.R_rbl_rocker)]) + multi_dot([B(self.P_rbl_rocker,self.ubar_rbl_rocker_jcs_rc_cyl).T,A(self.P_rbl_rocker),self.Mbar_rbl_rocker_jcs_rc_cyl[:,1:2]]))]]),self.L_jcs_rc_cyl])
        self.F_rbl_rocker_jcs_rc_cyl = Q_rbl_rocker_jcs_rc_cyl[0:3,0:1]
        Te_rbl_rocker_jcs_rc_cyl = Q_rbl_rocker_jcs_rc_cyl[3:7,0:1]
        self.T_rbl_rocker_jcs_rc_cyl = (-1*multi_dot([skew(multi_dot([A(self.P_rbl_rocker),self.ubar_rbl_rocker_jcs_rc_cyl])),self.F_rbl_rocker_jcs_rc_cyl]) + 0.5*multi_dot([E(self.P_rbl_rocker),Te_rbl_rocker_jcs_rc_cyl]))

        self.reactions = {'F_rbr_rocker_jcr_rocker_ch':self.F_rbr_rocker_jcr_rocker_ch,'T_rbr_rocker_jcr_rocker_ch':self.T_rbr_rocker_jcr_rocker_ch,'F_rbr_rocker_jcs_rc_sph':self.F_rbr_rocker_jcs_rc_sph,'T_rbr_rocker_jcs_rc_sph':self.T_rbr_rocker_jcs_rc_sph,'F_rbl_rocker_jcl_rocker_ch':self.F_rbl_rocker_jcl_rocker_ch,'T_rbl_rocker_jcl_rocker_ch':self.T_rbl_rocker_jcl_rocker_ch,'F_rbl_rocker_jcs_rc_cyl':self.F_rbl_rocker_jcs_rc_cyl,'T_rbl_rocker_jcs_rc_cyl':self.T_rbl_rocker_jcs_rc_cyl}

