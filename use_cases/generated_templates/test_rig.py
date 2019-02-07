
import numpy as np
import scipy as sc
import pandas as pd
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad
from source.solvers.py_numerical_functions import mirrored, centered, oriented



class configuration(object):

    def __init__(self):
        self.ax1_jcr_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.J_mcr_ver_act = np.array([[0, 0, 0]],dtype=np.float64)
        self.F_mcr_ver_act = lambda t : 0
        self.F_jcr_rev = lambda t : 0
        self.F_jcs_steer_gear = lambda t : 0
        self.ax1_jcs_steer_gear = np.array([[0], [0], [0]],dtype=np.float64)

    def _set_arguments(self):
        self.F_jcl_rev = self.F_jcr_rev
        self.J_mcl_ver_act = mirrored(self.J_mcr_ver_act)
        self.F_mcl_ver_act = self.F_mcr_ver_act
        self.ax1_jcl_rev = mirrored(self.ax1_jcr_rev)

    def load_from_csv(self,csv_file):
        dataframe = pd.read_csv(csv_file,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def eval_constants(self):

        c0 = triad(self.ax1_jcr_rev)
        c1 = triad(self.ax1_jcl_rev)
        c2 = triad(self.ax1_jcs_steer_gear)

        self.Mbar_vbr_upright_jcr_rev = multi_dot([A(self.P_vbr_upright).T,c0])
        self.Mbar_vbr_hub_jcr_rev = multi_dot([A(self.P_vbr_hub).T,c0])
        self.Mbar_vbl_upright_jcl_rev = multi_dot([A(self.P_vbl_upright).T,c1])
        self.Mbar_vbl_hub_jcl_rev = multi_dot([A(self.P_vbl_hub).T,c1])
        self.Mbar_vbs_steer_gear_jcs_steer_gear = multi_dot([A(self.P_vbs_steer_gear).T,c2])
        self.Mbar_vbs_chassis_jcs_steer_gear = multi_dot([A(self.P_vbs_chassis).T,c2])

    @property
    def q(self):
        q = []
        return q

    @property
    def qd(self):
        qd = []
        return qd



class topology(object):

    def __init__(self,config,prefix=''):
        self.t = 0.0
        self.config = config
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 0
        self.nrows = 5
        self.ncols = 2*7
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
    
        self.vbs_steer_gear = indicies_map[interface_map[p+'vbs_steer_gear']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]
        self.vbr_hub = indicies_map[interface_map[p+'vbr_hub']]
        self.vbr_upright = indicies_map[interface_map[p+'vbr_upright']]
        self.vbl_hub = indicies_map[interface_map[p+'vbl_hub']]
        self.vbl_upright = indicies_map[interface_map[p+'vbl_upright']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.vbr_hub*2,self.vbr_hub*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.vbl_hub*2,self.vbl_hub*2+1,self.vbr_hub*2,self.vbr_hub*2+1,self.vbr_upright*2,self.vbr_upright*2+1,self.vbl_hub*2,self.vbl_hub*2+1,self.vbl_upright*2,self.vbl_upright*2+1,self.vbs_steer_gear*2,self.vbs_steer_gear*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def set_gen_coordinates(self,q):
        pass

    
    def set_gen_velocities(self,qd):
        pass

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = np.eye(1,dtype=np.float64)
        x1 = A(self.P_vbr_upright).T
        x2 = A(self.P_vbr_hub)
        x3 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        x4 = A(self.P_vbl_upright).T
        x5 = A(self.P_vbl_hub)
        x6 = config.Mbar_vbl_hub_jcl_rev[:,0:1]
        x7 = A(self.P_vbs_steer_gear).T
        x8 = A(self.P_vbs_chassis)
        x9 = config.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]

        self.pos_eq_blocks = [-1*config.F_mcr_ver_act(t,) + self.R_vbr_hub[2]*x0,-1*config.F_mcl_ver_act(t,) + self.R_vbl_hub[2]*x0,(cos(config.F_jcr_rev(t,))*multi_dot([config.Mbar_vbr_upright_jcr_rev[:,1:2].T,x1,x2,x3]) + sin(config.F_jcr_rev(t,))*-1*multi_dot([config.Mbar_vbr_upright_jcr_rev[:,0:1].T,x1,x2,x3])),(cos(config.F_jcl_rev(t,))*multi_dot([config.Mbar_vbl_upright_jcl_rev[:,1:2].T,x4,x5,x6]) + sin(config.F_jcl_rev(t,))*-1*multi_dot([config.Mbar_vbl_upright_jcl_rev[:,0:1].T,x4,x5,x6])),(cos(config.F_jcs_steer_gear(t,))*multi_dot([config.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2].T,x7,x8,x9]) + sin(config.F_jcs_steer_gear(t,))*-1*multi_dot([config.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1].T,x7,x8,x9]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [derivative(config.F_mcr_ver_act,t,0.1,1)*-1*v0,derivative(config.F_mcl_ver_act,t,0.1,1)*-1*v0,derivative(config.F_jcr_rev,t,0.1,1)*-1*v0,derivative(config.F_jcl_rev,t,0.1,1)*-1*v0,derivative(config.F_jcs_steer_gear,t,0.1,1)*-1*v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1,dtype=np.float64)
        a1 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        a2 = self.P_vbr_hub
        a3 = self.Pd_vbr_upright
        a4 = config.Mbar_vbr_upright_jcr_rev[:,1:2]
        a5 = config.Mbar_vbr_upright_jcr_rev[:,0:1]
        a6 = self.P_vbr_upright
        a7 = A(a6).T
        a8 = self.Pd_vbr_hub
        a9 = config.Mbar_vbl_hub_jcl_rev[:,0:1]
        a10 = self.P_vbl_hub
        a11 = self.Pd_vbl_upright
        a12 = config.Mbar_vbl_upright_jcl_rev[:,1:2]
        a13 = config.Mbar_vbl_upright_jcl_rev[:,0:1]
        a14 = self.P_vbl_upright
        a15 = A(a14).T
        a16 = self.Pd_vbl_hub
        a17 = config.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        a18 = self.P_vbs_chassis
        a19 = self.Pd_vbs_steer_gear
        a20 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        a21 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        a22 = self.P_vbs_steer_gear
        a23 = A(a22).T
        a24 = self.Pd_vbs_chassis

        self.acc_eq_blocks = [derivative(config.F_mcr_ver_act,t,0.1,2)*-1*a0,derivative(config.F_mcl_ver_act,t,0.1,2)*-1*a0,(derivative(config.F_jcr_rev,t,0.1,2)*-1*a0 + multi_dot([a1.T,A(a2).T,(cos(config.F_jcr_rev(t,))*B(a3,a4) + sin(config.F_jcr_rev(t,))*-1*B(a3,a5)),a3]) + multi_dot([(cos(config.F_jcr_rev(t,))*multi_dot([a4.T,a7]) + sin(config.F_jcr_rev(t,))*-1*multi_dot([a5.T,a7])),B(a8,a1),a8]) + 2*multi_dot([((cos(config.F_jcr_rev(t,))*multi_dot([B(a6,a4),a3])).T + sin(config.F_jcr_rev(t,))*-1*multi_dot([a3.T,B(a6,a5).T])),B(a2,a1),a8])),(derivative(config.F_jcl_rev,t,0.1,2)*-1*a0 + multi_dot([a9.T,A(a10).T,(cos(config.F_jcl_rev(t,))*B(a11,a12) + sin(config.F_jcl_rev(t,))*-1*B(a11,a13)),a11]) + multi_dot([(cos(config.F_jcl_rev(t,))*multi_dot([a12.T,a15]) + sin(config.F_jcl_rev(t,))*-1*multi_dot([a13.T,a15])),B(a16,a9),a16]) + 2*multi_dot([((cos(config.F_jcl_rev(t,))*multi_dot([B(a14,a12),a11])).T + sin(config.F_jcl_rev(t,))*-1*multi_dot([a11.T,B(a14,a13).T])),B(a10,a9),a16])),(derivative(config.F_jcs_steer_gear,t,0.1,2)*-1*a0 + multi_dot([a17.T,A(a18).T,(cos(config.F_jcs_steer_gear(t,))*B(a19,a20) + sin(config.F_jcs_steer_gear(t,))*-1*B(a19,a21)),a19]) + multi_dot([(cos(config.F_jcs_steer_gear(t,))*multi_dot([a20.T,a23]) + sin(config.F_jcs_steer_gear(t,))*-1*multi_dot([a21.T,a23])),B(a24,a17),a24]) + 2*multi_dot([((cos(config.F_jcs_steer_gear(t,))*multi_dot([B(a22,a20),a19])).T + sin(config.F_jcs_steer_gear(t,))*-1*multi_dot([a19.T,B(a22,a21).T])),B(a18,a17),a24]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,4),dtype=np.float64)
        j1 = np.zeros((1,3),dtype=np.float64)
        j2 = config.Mbar_vbr_hub_jcr_rev[:,0:1]
        j3 = self.P_vbr_hub
        j4 = self.P_vbr_upright
        j5 = config.Mbar_vbr_upright_jcr_rev[:,1:2]
        j6 = config.Mbar_vbr_upright_jcr_rev[:,0:1]
        j7 = A(j4).T
        j8 = config.Mbar_vbl_hub_jcl_rev[:,0:1]
        j9 = self.P_vbl_hub
        j10 = self.P_vbl_upright
        j11 = config.Mbar_vbl_upright_jcl_rev[:,1:2]
        j12 = config.Mbar_vbl_upright_jcl_rev[:,0:1]
        j13 = A(j10).T
        j14 = config.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        j15 = self.P_vbs_chassis
        j16 = self.P_vbs_steer_gear
        j17 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        j18 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        j19 = A(j16).T

        self.jac_eq_blocks = [j1,j0,config.J_mcr_ver_act,j0,j1,j0,config.J_mcl_ver_act,j0,j1,multi_dot([(cos(config.F_jcr_rev(t,))*multi_dot([j5.T,j7]) + sin(config.F_jcr_rev(t,))*-1*multi_dot([j6.T,j7])),B(j3,j2)]),j1,multi_dot([j2.T,A(j3).T,(cos(config.F_jcr_rev(t,))*B(j4,j5) + sin(config.F_jcr_rev(t,))*-1*B(j4,j6))]),j1,multi_dot([(cos(config.F_jcl_rev(t,))*multi_dot([j11.T,j13]) + sin(config.F_jcl_rev(t,))*-1*multi_dot([j12.T,j13])),B(j9,j8)]),j1,multi_dot([j8.T,A(j9).T,(cos(config.F_jcl_rev(t,))*B(j10,j11) + sin(config.F_jcl_rev(t,))*-1*B(j10,j12))]),j1,multi_dot([j14.T,A(j15).T,(cos(config.F_jcs_steer_gear(t,))*B(j16,j17) + sin(config.F_jcs_steer_gear(t,))*-1*B(j16,j18))]),j1,multi_dot([(cos(config.F_jcs_steer_gear(t,))*multi_dot([j17.T,j19]) + sin(config.F_jcs_steer_gear(t,))*-1*multi_dot([j18.T,j19])),B(j15,j14)])]
  
