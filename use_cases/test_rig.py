
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
        self.ax1_jcr_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.pt1_jcr_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcr_rev = lambda t : 0

        self._set_arguments()

    def _set_arguments(self):
        self.ax1_jcl_rev = Mirror(self.ax1_jcr_rev)
        self.pt1_jcl_rev = Mirror(self.pt1_jcr_rev)
        self.F_jcl_rev = self.F_jcr_rev

    def eval_constants(self):

        c0 = A(self.P_vbr_hub).T
        c1 = self.pt1_jcr_rev
        c2 = A(self.P_vbr_upright).T
        c3 = Triad(self.ax1_jcr_rev,)
        c4 = A(self.P_vbl_hub).T
        c5 = self.pt1_jcl_rev
        c6 = A(self.P_vbl_upright).T
        c7 = Triad(self.ax1_jcl_rev,)

        self.ubar_vbr_hub_jcr_rev = (multi_dot([c0,c1]) + -1.0*multi_dot([c0,self.R_vbr_hub]))
        self.ubar_vbr_upright_jcr_rev = (multi_dot([c2,c1]) + -1.0*multi_dot([c2,self.R_vbr_upright]))
        self.Mbar_vbr_hub_jcr_rev = multi_dot([c0,c3])
        self.Mbar_vbr_upright_jcr_rev = multi_dot([c2,c3])
        self.ubar_vbl_hub_jcl_rev = (multi_dot([c4,c5]) + -1.0*multi_dot([c4,self.R_vbl_hub]))
        self.ubar_vbl_upright_jcl_rev = (multi_dot([c6,c5]) + -1.0*multi_dot([c6,self.R_vbl_upright]))
        self.Mbar_vbl_hub_jcl_rev = multi_dot([c4,c7])
        self.Mbar_vbl_upright_jcl_rev = multi_dot([c6,c7])



class topology(object):

    def __init__(self,config,prefix=''):
        self.t = 0.0
        self.config = config
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)

        self.n = 0
        self.nrows = 8
        self.ncols = 2*5
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
    
        self.vbl_hub = indicies_map[interface_map[p+'vbl_hub']]
        self.vbr_hub = indicies_map[interface_map[p+'vbr_hub']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbr_upright = indicies_map[interface_map[p+'vbr_upright']]
        self.vbl_upright = indicies_map[interface_map[p+'vbl_upright']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbr_hub*2,self.vbr_hub*2+1,self.vbr_upright*2,self.vbr_upright*2+1,self.vbr_hub*2,self.vbr_hub*2+1,self.vbr_upright*2,self.vbr_upright*2+1,self.vbr_hub*2,self.vbr_hub*2+1,self.vbr_upright*2,self.vbr_upright*2+1,self.vbr_hub*2,self.vbr_hub*2+1,self.vbr_upright*2,self.vbr_upright*2+1,self.vbl_hub*2,self.vbl_hub*2+1,self.vbl_upright*2,self.vbl_upright*2+1,self.vbl_hub*2,self.vbl_hub*2+1,self.vbl_upright*2,self.vbl_upright*2+1,self.vbl_hub*2,self.vbl_hub*2+1,self.vbl_upright*2,self.vbl_upright*2+1,self.vbl_hub*2,self.vbl_hub*2+1,self.vbl_upright*2,self.vbl_upright*2+1,])

    
    def set_gen_coordinates(self,q):
        self.

    
    def set_gen_velocities(self,qd):
        self.

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = A(self.P_vbr_hub)
        x1 = A(self.P_vbr_upright)
        x2 = x0.T
        x3 = config.Mbar_vbr_upright_jcr_rev[:,2:3]
        x4 = config.F_jcr_rev(t,)
        x5 = config.Mbar_vbr_upright_jcr_rev[:,0:1]
        x6 = A(self.P_vbl_hub)
        x7 = A(self.P_vbl_upright)
        x8 = x6.T
        x9 = config.Mbar_vbl_upright_jcl_rev[:,2:3]
        x10 = config.F_jcl_rev(t,)
        x11 = config.Mbar_vbl_upright_jcl_rev[:,0:1]

        self.pos_eq_blocks = [(self.R_vbr_hub + -1.0*self.R_vbr_upright + multi_dot([x0,config.ubar_vbr_hub_jcr_rev]) + -1.0*multi_dot([x1,config.ubar_vbr_upright_jcr_rev])),multi_dot([config.Mbar_vbr_hub_jcr_rev[:,0:1].T,x2,x1,x3]),multi_dot([config.Mbar_vbr_hub_jcr_rev[:,1:2].T,x2,x1,x3]),(cos(x4)*multi_dot([config.Mbar_vbr_hub_jcr_rev[:,1:2].T,x2,x1,x5]) + sin(x4)*-1.0*multi_dot([config.Mbar_vbr_hub_jcr_rev[:,0:1].T,x2,x1,x5])),(self.R_vbl_hub + -1.0*self.R_vbl_upright + multi_dot([x6,config.ubar_vbl_hub_jcl_rev]) + -1.0*multi_dot([x7,config.ubar_vbl_upright_jcl_rev])),multi_dot([config.Mbar_vbl_hub_jcl_rev[:,0:1].T,x8,x7,x9]),multi_dot([config.Mbar_vbl_hub_jcl_rev[:,1:2].T,x8,x7,x9]),(cos(x10)*multi_dot([config.Mbar_vbl_hub_jcl_rev[:,1:2].T,x8,x7,x11]) + sin(x10)*-1.0*multi_dot([config.Mbar_vbl_hub_jcl_rev[:,0:1].T,x8,x7,x11]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.zeros((3,1),dtype=np.float64)
        v1 = np.zeros((1,1),dtype=np.float64)
        v2 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [v0,v1,v1,(v1 + derivative(config.F_jcr_rev,t,0.1,1)*-1.0*v2),v0,v1,v1,(v1 + derivative(config.F_jcl_rev,t,0.1,1)*-1.0*v2)]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_vbr_hub
        a1 = self.Pd_vbr_upright
        a2 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        a3 = self.P_vbr_hub
        a4 = A(a3).T
        a5 = config.Mbar_vbr_upright_jcr_rev[:,2:3]
        a6 = B(a1,a5)
        a7 = a5.T
        a8 = self.P_vbr_upright
        a9 = A(a8).T
        a10 = a0.T
        a11 = B(a8,a5)
        a12 = config.Mbar_vbr_hub_jcr_rev[:,1:2]
        a13 = config.F_jcr_rev(t,)
        a14 = np.eye(1,dtype=np.float64)
        a15 = config.Mbar_vbr_upright_jcr_rev[:,0:1]
        a16 = cos(a13)
        a17 = config.Mbar_vbr_hub_jcr_rev[:,1:2]
        a18 = sin(a13)
        a19 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        a20 = self.Pd_vbl_hub
        a21 = self.Pd_vbl_upright
        a22 = config.Mbar_vbl_upright_jcl_rev[:,2:3]
        a23 = a22.T
        a24 = self.P_vbl_upright
        a25 = A(a24).T
        a26 = config.Mbar_vbl_hub_jcl_rev[:,0:1]
        a27 = self.P_vbl_hub
        a28 = A(a27).T
        a29 = B(a21,a22)
        a30 = a20.T
        a31 = B(a24,a22)
        a32 = config.Mbar_vbl_hub_jcl_rev[:,1:2]
        a33 = config.F_jcl_rev(t,)
        a34 = config.Mbar_vbl_upright_jcl_rev[:,0:1]
        a35 = cos(a33)
        a36 = config.Mbar_vbl_hub_jcl_rev[:,1:2]
        a37 = sin(a33)
        a38 = config.Mbar_vbl_hub_jcl_rev[:,0:1]

        self.acc_eq_blocks = [(multi_dot([B(a0,config.ubar_vbr_hub_jcr_rev),a0]) + -1.0*multi_dot([B(a1,config.ubar_vbr_upright_jcr_rev),a1])),(multi_dot([a2.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a2),a0]) + 2.0*multi_dot([a10,B(a3,a2).T,a11,a1])),(multi_dot([a12.T,a4,a6,a1]) + multi_dot([a7,a9,B(a0,a12),a0]) + 2.0*multi_dot([a10,B(a3,a12).T,a11,a1])),(derivative(a13,t,0.1,2)*-1.0*a14 + multi_dot([a15.T,a9,(a16*B(a0,a17) + a18*-1.0*B(a0,a19)),a0]) + multi_dot([(a16*multi_dot([a17.T,a4]) + a18*-1.0*multi_dot([a19.T,a4])),B(a1,a15),a1]) + 2.0*multi_dot([((a16*multi_dot([B(a3,a17),a0])).T + a18*-1.0*multi_dot([a10,B(a3,a19).T])),B(a8,a15),a1])),(multi_dot([B(a20,config.ubar_vbl_hub_jcl_rev),a20]) + -1.0*multi_dot([B(a21,config.ubar_vbl_upright_jcl_rev),a21])),(multi_dot([a23,a25,B(a20,a26),a20]) + multi_dot([a26.T,a28,a29,a21]) + 2.0*multi_dot([a30,B(a27,a26).T,a31,a21])),(multi_dot([a23,a25,B(a20,a32),a20]) + multi_dot([a32.T,a28,a29,a21]) + 2.0*multi_dot([a30,B(a27,a32).T,a31,a21])),(derivative(a33,t,0.1,2)*-1.0*a14 + multi_dot([a34.T,a25,(a35*B(a20,a36) + a37*-1.0*B(a20,a38)),a20]) + multi_dot([(a35*multi_dot([a36.T,a28]) + a37*-1.0*multi_dot([a38.T,a28])),B(a21,a34),a21]) + 2.0*multi_dot([((a35*multi_dot([B(a27,a36),a20])).T + a37*-1.0*multi_dot([a30,B(a27,a38).T])),B(a24,a34),a21]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_vbr_hub
        j2 = np.zeros((1,3),dtype=np.float64)
        j3 = config.Mbar_vbr_upright_jcr_rev[:,2:3]
        j4 = j3.T
        j5 = self.P_vbr_upright
        j6 = A(j5).T
        j7 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        j8 = config.Mbar_vbr_hub_jcr_rev[:,1:2]
        j9 = -1.0*j0
        j10 = A(j1).T
        j11 = B(j5,j3)
        j12 = config.Mbar_vbr_upright_jcr_rev[:,0:1]
        j13 = config.F_jcr_rev(t,)
        j14 = cos(j13)
        j15 = config.Mbar_vbr_hub_jcr_rev[:,1:2]
        j16 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        j17 = self.P_vbl_hub
        j18 = config.Mbar_vbl_upright_jcl_rev[:,2:3]
        j19 = j18.T
        j20 = self.P_vbl_upright
        j21 = A(j20).T
        j22 = config.Mbar_vbl_hub_jcl_rev[:,0:1]
        j23 = config.Mbar_vbl_hub_jcl_rev[:,1:2]
        j24 = A(j17).T
        j25 = B(j20,j18)
        j26 = config.Mbar_vbl_upright_jcl_rev[:,0:1]
        j27 = config.F_jcl_rev(t,)
        j28 = cos(j27)
        j29 = config.Mbar_vbl_hub_jcl_rev[:,1:2]
        j30 = config.Mbar_vbl_hub_jcl_rev[:,0:1]

        self.jac_eq_blocks = [j0,B(j1,config.ubar_vbr_hub_jcr_rev),j9,-1.0*B(j5,config.ubar_vbr_upright_jcr_rev),j2,multi_dot([j4,j6,B(j1,j7)]),j2,multi_dot([j7.T,j10,j11]),j2,multi_dot([j4,j6,B(j1,j8)]),j2,multi_dot([j8.T,j10,j11]),j2,multi_dot([j12.T,j6,(j14*B(j1,j15) + sin(j13)*-1.0*B(j1,j16))]),j2,multi_dot([(j14*multi_dot([j15.T,j10]) + sin(j13)*-1.0*multi_dot([j16.T,j10])),B(j5,j12)]),j0,B(j17,config.ubar_vbl_hub_jcl_rev),j9,-1.0*B(j20,config.ubar_vbl_upright_jcl_rev),j2,multi_dot([j19,j21,B(j17,j22)]),j2,multi_dot([j22.T,j24,j25]),j2,multi_dot([j19,j21,B(j17,j23)]),j2,multi_dot([j23.T,j24,j25]),j2,multi_dot([j26.T,j21,(j28*B(j17,j29) + sin(j27)*-1.0*B(j17,j30))]),j2,multi_dot([(j28*multi_dot([j29.T,j24]) + sin(j27)*-1.0*multi_dot([j30.T,j24])),B(j20,j26)])]
  
