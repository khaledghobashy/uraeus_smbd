
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
        self.AF_mcr_ver_act = lambda t : 0
        self.J_mcr_ver_act = np.array([[0, 0, 0]],dtype=np.float64)

    def _set_arguments(self):
        self.AF_mcl_ver_act = self.AF_mcr_ver_act
        self.J_mcl_ver_act = mirrored(self.J_mcr_ver_act)

    def load_from_csv(self,csv_file):
        dataframe = pd.read_csv(csv_file,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def eval_constants(self):

        pass

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
        self.nrows = 2
        self.ncols = 2*3
        self.rows = np.arange(self.nrows)

        self.jac_rows = np.array([0,0,0,0,1,1,1,1])                        

    
    def _set_mapping(self,indicies_map,interface_map):
        p = self.prefix
    
        self.vbr_hub = indicies_map[interface_map[p+'vbr_hub']]
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]
        self.vbl_hub = indicies_map[interface_map[p+'vbl_hub']]

    def assemble_template(self,indicies_map,interface_map,rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map,interface_map)
        self.rows += self.rows_offset
        self.jac_rows += self.rows_offset
        self.jac_cols = np.array([self.vbs_ground*2,self.vbs_ground*2+1,self.vbr_hub*2,self.vbr_hub*2+1,self.vbs_ground*2,self.vbs_ground*2+1,self.vbl_hub*2,self.vbl_hub*2+1])

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

        self.pos_eq_blocks = [-1*config.AF_mcr_ver_act(t,) + self.R_vbr_hub[2]*x0,-1*config.AF_mcl_ver_act(t,) + self.R_vbl_hub[2]*x0]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

        v0 = np.eye(1,dtype=np.float64)

        self.vel_eq_blocks = [derivative(config.AF_mcr_ver_act,t,0.1,1)*-1*v0,derivative(config.AF_mcl_ver_act,t,0.1,1)*-1*v0]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = np.eye(1,dtype=np.float64)

        self.acc_eq_blocks = [derivative(config.AF_mcr_ver_act,t,0.1,2)*-1*a0,derivative(config.AF_mcl_ver_act,t,0.1,2)*-1*a0]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.zeros((1,4),dtype=np.float64)
        j1 = np.zeros((1,3),dtype=np.float64)

        self.jac_eq_blocks = [j1,j0,config.J_mcr_ver_act,j0,j1,j0,config.J_mcl_ver_act,j0]
  
