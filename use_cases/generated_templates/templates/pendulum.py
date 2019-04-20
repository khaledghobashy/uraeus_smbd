
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
        self.nc = 4
        self.nrows = 2
        self.ncols = 2*2
        self.rows = np.arange(self.nrows)

        reactions_indicies = ['F_rbs_crank_jcs_rev_crank', 'T_rbs_crank_jcs_rev_crank']
        self.reactions_indicies = ['%s%s'%(self.prefix,i) for i in reactions_indicies]

    
    def _set_mapping(self,indicies_map, interface_map):
        p = self.prefix
        self.rbs_crank = indicies_map[p+'rbs_crank']
        self.vbs_ground = indicies_map[interface_map[p+'vbs_ground']]

    def assemble_template(self,indicies_map, interface_map, rows_offset):
        self.rows_offset = rows_offset
        self._set_mapping(indicies_map, interface_map)
        self.rows += self.rows_offset
        self.jac_rows = np.array([0, 0, 0, 0, 1])
        self.jac_rows += self.rows_offset
        self.jac_cols = [self.vbs_ground*2, self.vbs_ground*2+1, self.rbs_crank*2, self.rbs_crank*2+1, self.rbs_crank*2+1]

    def set_initial_states(self):
        self.set_gen_coordinates(self.config.q)
        self.set_gen_velocities(self.config.qd)

    
    def eval_constants(self):
        config = self.config

        self.F_rbs_crank_gravity = np.array([[0], [0], [-9810.0*config.m_rbs_crank]],dtype=np.float64)

        self.Mbar_rbs_crank_jcs_rev_crank = multi_dot([A(config.P_rbs_crank).T,triad(config.ax1_jcs_rev_crank)])
        self.Mbar_vbs_ground_jcs_rev_crank = multi_dot([A(config.P_vbs_ground).T,triad(config.ax1_jcs_rev_crank)])
        self.ubar_rbs_crank_jcs_rev_crank = (multi_dot([A(config.P_rbs_crank).T,config.pt1_jcs_rev_crank]) + -1*multi_dot([A(config.P_rbs_crank).T,config.R_rbs_crank]))
        self.ubar_vbs_ground_jcs_rev_crank = (multi_dot([A(config.P_vbs_ground).T,config.pt1_jcs_rev_crank]) + -1*multi_dot([A(config.P_vbs_ground).T,config.R_vbs_ground]))

    
    def set_gen_coordinates(self,q):
        self.R_rbs_crank = q[0:3,0:1]
        self.P_rbs_crank = q[3:7,0:1]

    
    def set_gen_velocities(self,qd):
        self.Rd_rbs_crank = qd[0:3,0:1]
        self.Pd_rbs_crank = qd[3:7,0:1]

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_rbs_crank = qdd[0:3,0:1]
        self.Pdd_rbs_crank = qdd[3:7,0:1]

    
    def set_lagrange_multipliers(self,Lambda):
        self.L_jcs_rev_crank = Lambda[0:3,0:1]

    
    def eval_pos_eq(self):
        config = self.config
        t = self.t

        x0 = self.P_rbs_crank

        self.pos_eq_blocks = [(self.R_rbs_crank + -1*self.R_vbs_ground + multi_dot([A(x0),self.ubar_rbs_crank_jcs_rev_crank]) + -1*multi_dot([A(self.P_vbs_ground),self.ubar_vbs_ground_jcs_rev_crank])),
        (-1*np.eye(1,dtype=np.float64) + multi_dot([x0.T,x0]))]

    
    def eval_vel_eq(self):
        config = self.config
        t = self.t

    

        self.vel_eq_blocks = [np.zeros((3,1),dtype=np.float64),
        np.zeros((1,1),dtype=np.float64)]

    
    def eval_acc_eq(self):
        config = self.config
        t = self.t

        a0 = self.Pd_rbs_crank
        a1 = self.Pd_vbs_ground

        self.acc_eq_blocks = [(multi_dot([B(a0,self.ubar_rbs_crank_jcs_rev_crank),a0]) + -1*multi_dot([B(a1,self.ubar_vbs_ground_jcs_rev_crank),a1])),
        2*multi_dot([a0.T,a0])]

    
    def eval_jac_eq(self):
        config = self.config
        t = self.t

        j0 = np.eye(3,dtype=np.float64)
        j1 = self.P_rbs_crank

        self.jac_eq_blocks = [-1*j0,
        -1*B(self.P_vbs_ground,self.ubar_vbs_ground_jcs_rev_crank),
        j0,
        B(j1,self.ubar_rbs_crank_jcs_rev_crank),
        2*j1.T]

    
    def eval_mass_eq(self):
        config = self.config
        t = self.t

        m0 = G(self.P_rbs_crank)

        self.mass_eq_blocks = [config.m_rbs_crank*np.eye(3,dtype=np.float64),
        4*multi_dot([m0.T,config.Jbar_rbs_crank,m0])]

    
    def eval_frc_eq(self):
        config = self.config
        t = self.t

        f0 = G(self.Pd_rbs_crank)

        self.frc_eq_blocks = [self.F_rbs_crank_gravity,
        8*multi_dot([f0.T,config.Jbar_rbs_crank,f0,self.P_rbs_crank])]

    
    def eval_reactions_eq(self):
        config  = self.config
        t = self.t

        Q_rbs_crank_jcs_rev_crank = -1*multi_dot([np.bmat([[np.eye(3,dtype=np.float64)],[B(self.P_rbs_crank,self.ubar_rbs_crank_jcs_rev_crank).T]]),self.L_jcs_rev_crank])
        self.F_rbs_crank_jcs_rev_crank = Q_rbs_crank_jcs_rev_crank[0:3,0:1]
        Te_rbs_crank_jcs_rev_crank = Q_rbs_crank_jcs_rev_crank[3:7,0:1]
        self.T_rbs_crank_jcs_rev_crank = (-1*multi_dot([skew(multi_dot([A(self.P_rbs_crank),self.ubar_rbs_crank_jcs_rev_crank])),self.F_rbs_crank_jcs_rev_crank]) + 0.5*multi_dot([E(self.P_rbs_crank),Te_rbs_crank_jcs_rev_crank]))

        self.reactions = {'F_rbs_crank_jcs_rev_crank':self.F_rbs_crank_jcs_rev_crank,'T_rbs_crank_jcs_rev_crank':self.T_rbs_crank_jcs_rev_crank}

