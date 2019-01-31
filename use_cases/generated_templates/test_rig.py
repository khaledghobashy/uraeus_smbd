
import numpy as np
import scipy as sc
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad as Triad
from scipy.misc import derivative
from numpy import cos, sin
import pandas as pd

def Mirror(v):
    if v.shape in ((1,3),(4,1)):
        return v
    else:
        m = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=np.float64)
        return m.dot(v)


class configuration(object):

    def __init__(self):
        self.ax1_jcs_steer_gear = np.array([[0], [0], [0]],dtype=np.float64)
        self.F_jcs_steer_gear = lambda t : 0

        self._set_arguments()

    def _set_arguments(self):
        

    def load_from_csv(self,csv_file):
        dataframe = pd.read_csv(csv_file,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def eval_constants(self):

        c0 = Triad(self.ax1_jcs_steer_gear,)

        self.Mbar_vbs_steer_gear_jcs_steer_gear = multi_dot([A(self.P_vbs_steer_gear).T,c0])
        self.Mbar_vbs_chassis_jcs_steer_gear = multi_dot([A(self.P_vbs_chassis).T,c0])

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
        self.nrows = 1
        self.ncols = 2*3
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
    
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbs_steer_gear = indicies_map[interface_map[p+'vbs_steer_gear']]
        self.vbs_chassis = indicies_map[interface_map[p+'vbs_chassis']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_steer_gear*2,self.vbs_steer_gear*2+1,self.vbs_chassis*2,self.vbs_chassis*2+1,])

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

        x0 = config.F_jcs_steer_gear(t,)
        x1 = A(self.P_vbs_steer_gear).T
        x2 = A(self.P_vbs_chassis)
        x3 = config.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]

        self.pos_eq_blocks = [(cos(x0)*multi_dot([config.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2].T,x1,x2,x3]) + sin(x0)*-1.0*multi_dot([config.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1].T,x1,x2,x3]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

    

        self.vel_eq_blocks = [(np.zeros((1,1),dtype=np.float64) + -1*derivative(config.F_jcs_steer_gear,t,0.1,1)*np.eye(1,dtype=np.float64))]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = config.F_jcs_steer_gear(t,)
        a1 = config.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        a2 = self.P_vbs_chassis
        a3 = cos(a0)
        a4 = self.Pd_vbs_steer_gear
        a5 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        a6 = sin(a0)
        a7 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        a8 = self.P_vbs_steer_gear
        a9 = A(a8).T
        a10 = self.Pd_vbs_chassis

        self.acc_eq_blocks = [(derivative(a0,t,0.1,2)*-1.0*np.eye(1,dtype=np.float64) + multi_dot([a1.T,A(a2).T,(a3*B(a4,a5) + a6*-1.0*B(a4,a7)),a4]) + multi_dot([(a3*multi_dot([a5.T,a9]) + a6*-1.0*multi_dot([a7.T,a9])),B(a10,a1),a10]) + 2.0*multi_dot([((a3*multi_dot([B(a8,a5),a4])).T + a6*-1.0*multi_dot([a4.T,B(a8,a7).T])),B(a2,a1),a10]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = config.Mbar_vbs_chassis_jcs_steer_gear[:,0:1]
        j2 = self.P_vbs_chassis
        j3 = config.F_jcs_steer_gear(t,)
        j4 = cos(j3)
        j5 = self.P_vbs_steer_gear
        j6 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,1:2]
        j7 = config.Mbar_vbs_steer_gear_jcs_steer_gear[:,0:1]
        j8 = A(j5).T

        self.jac_eq_blocks = [j0,multi_dot([j1.T,A(j2).T,(j4*B(j5,j6) + sin(j3)*-1.0*B(j5,j7))]),j0,multi_dot([(j4*multi_dot([j6.T,j8]) + sin(j3)*-1.0*multi_dot([j7.T,j8])),B(j2,j1)])]
  
