
import os
import numpy as np
import pandas as pd
from scipy.misc import derivative
from numpy import cos, sin
from numpy.linalg import multi_dot
from source.cython_definitions.matrix_funcs import A, B, triad                



path = os.path.dirname(__file__)

class configuration(object):

    def __init__(self):
        self.ax1_jcs_rev = np.array([[0], [0], [0]],dtype=np.float64)
        self.AF_jcs_rev = lambda t : 0                       

    
    @property
    def q(self):
        q = []
        return q

    @property
    def qd(self):
        qd = []
        return qd

    def load_from_csv(self,csv_file):
        file_path = os.path.join(path,csv_file)
        dataframe = pd.read_csv(file_path,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def _set_arguments(self):
    
        pass




class topology(object):

    def __init__(self,prefix='',cfg=None):
        self.t = 0.0
        self.config = (configuration() if cfg is None else cfg)
        self.prefix = (prefix if prefix=='' else prefix+'.')

        self.n = 0
        self.nrows = 1
        self.ncols = 2*2
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
    
        self.vbs_rocker = indicies_map[interface_map[p+'vbs_rocker']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.vbs_rocker*2,self.vbs_rocker*2+1])

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        c0 = triad(config.ax1_jcs_rev)

        self.Mbar_vbs_rocker_jcs_rev = multi_dot([A(config.P_vbs_rocker).T,c0])
        self.Mbar_vbs_ground_jcs_rev = multi_dot([A(config.P_vbs_ground).T,c0])

    
    def set_gen_coordinates(self,q):
        pass

    
    def set_gen_velocities(self,qd):
        pass

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = A(self.P_vbs_rocker).T
        x1 = A(self.P_vbs_ground)
        x2 = self.Mbar_vbs_ground_jcs_rev[:,0:1]

        self.pos_eq_blocks = [(cos(config.AF_jcs_rev(t,))*multi_dot([self.Mbar_vbs_rocker_jcs_rev[:,1:2].T,x0,x1,x2]) + sin(config.AF_jcs_rev(t,))*-1*multi_dot([self.Mbar_vbs_rocker_jcs_rev[:,0:1].T,x0,x1,x2]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

    

        self.vel_eq_blocks = [-1*derivative(config.AF_jcs_rev,t,0.1,1)*np.eye(1,dtype=np.float64)]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Mbar_vbs_ground_jcs_rev[:,0:1]
        a1 = self.P_vbs_ground
        a2 = self.Pd_vbs_rocker
        a3 = self.Mbar_vbs_rocker_jcs_rev[:,1:2]
        a4 = self.Mbar_vbs_rocker_jcs_rev[:,0:1]
        a5 = self.P_vbs_rocker
        a6 = A(a5).T
        a7 = self.Pd_vbs_ground

        self.acc_eq_blocks = [(-1*derivative(config.AF_jcs_rev,t,0.1,2)*np.eye(1,dtype=np.float64) + multi_dot([a0.T,A(a1).T,(cos(config.AF_jcs_rev(t,))*B(a2,a3) + sin(config.AF_jcs_rev(t,))*-1*B(a2,a4)),a2]) + multi_dot([(cos(config.AF_jcs_rev(t,))*multi_dot([a3.T,a6]) + sin(config.AF_jcs_rev(t,))*-1*multi_dot([a4.T,a6])),B(a7,a0),a7]) + 2*multi_dot([((cos(config.AF_jcs_rev(t,))*multi_dot([B(a5,a3),a2])).T + sin(config.AF_jcs_rev(t,))*-1*multi_dot([a2.T,B(a5,a4).T])),B(a1,a0),a7]))]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,3),dtype=np.float64)
        j1 = self.Mbar_vbs_ground_jcs_rev[:,0:1]
        j2 = self.P_vbs_ground
        j3 = self.P_vbs_rocker
        j4 = self.Mbar_vbs_rocker_jcs_rev[:,1:2]
        j5 = self.Mbar_vbs_rocker_jcs_rev[:,0:1]
        j6 = A(j3).T

        self.jac_eq_blocks = [j0,multi_dot([(cos(config.AF_jcs_rev(t,))*multi_dot([j4.T,j6]) + sin(config.AF_jcs_rev(t,))*-1*multi_dot([j5.T,j6])),B(j2,j1)]),j0,multi_dot([j1.T,A(j2).T,(cos(config.AF_jcs_rev(t,))*B(j3,j4) + sin(config.AF_jcs_rev(t,))*-1*B(j3,j5))])]
  
    
