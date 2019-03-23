
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

        self.n  = 7
        self.nc = 1
        self.nrows = 1
        self.ncols = 2*2
        self.rows = np.arange(self.nrows)

        reactions_indicies = []
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_chassis = indicies_map[p+'rbs_chassis']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.rbs_chassis*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_chassis_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_chassis]],dtype=np.float64)

    

    
    def set_gen_coordinates(self,q):
        self.R_rbs_chassis = q[0:3,0:1]
        self.P_rbs_chassis = q[3:7,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_chassis = qd[0:3,0:1]
        self.Pd_rbs_chassis = qd[3:7,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_chassis = qdd[0:3,0:1]
        self.Pdd_rbs_chassis = qdd[3:7,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        pass

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.P_rbs_chassis

        self.pos_eq_blocks = [(-1*np.eye(1,dtype=np.float64) + multi_dot([x0.T,x0]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

    

        self.vel_eq_blocks = [np.zeros((1,1),dtype=np.float64)]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbs_chassis

        self.acc_eq_blocks = [2*multi_dot([a0.T,a0])]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

    

        self.jac_eq_blocks = [2*self.P_rbs_chassis.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = G(self.P_rbs_chassis)

        self.mass_eq_blocks = [config.m_rbs_chassis*np.eye(3,dtype=np.float64),
        4*multi_dot([m0.T,config.Jbar_rbs_chassis,m0])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_chassis)

        self.frc_eq_blocks = [self.F_rbs_chassis_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_chassis,f0,self.P_rbs_chassis])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

    

        self.reactions = {}

