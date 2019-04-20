
import numpy as np

from use_cases.generated_templates.templates import spatial_fourbar


FB = spatial_fourbar.topology('FB')



class numerical_assembly(object):

    def __init__(self):
        self._t = 0
        self.subsystems = [FB]

        self.interface_map = {'FB.vbs_ground': 'ground'}
        self.indicies_map  = {'ground': 0, 'FB.rbs_crank': 1, 'FB.rbs_rocker': 2, 'FB.rbs_coupler': 3}

        self.R_ground  = np.array([[0],[0],[0]],dtype=np.float64)
        self.P_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
        self.Pg_ground = np.array([[1],[0],[0],[0]],dtype=np.float64)

        self.M_ground = np.eye(3,dtype=np.float64)
        self.J_ground = np.eye(4,dtype=np.float64)

        self.gr_rows = np.array([0,1])
        self.gr_jac_rows = np.array([0,0,1,1])
        self.gr_jac_cols = np.array([0,1,0,1])

        self.nrows = 15
        self.ncols = 8

        self.initialize_assembly()


    @property
    def t(self):
        return self._t
    @t.setter
    def t(self,t):
        self._t = t
        for sub in self.subsystems:
            sub.t = t

    def set_initial_states(self):
        for sub in self.subsystems:
            sub.set_initial_states()
        coordinates = [sub.config.q for sub in self.subsystems if len(sub.config.q)!=0]
        self.q0 = np.concatenate([self.R_ground,self.P_ground,*coordinates])

    def initialize_assembly(self):
        self.t = 0
        self.assemble_system()
        self.set_initial_states()
        self.eval_constants()

    def assemble_system(self):
        offset = 2
        for sub in self.subsystems:
            sub.assemble_template(self.indicies_map,self.interface_map,offset)
            offset += sub.nrows

        self.rows = np.concatenate([s.rows for s in self.subsystems])
        self.jac_rows = np.concatenate([s.jac_rows for s in self.subsystems])
        self.jac_cols = np.concatenate([s.jac_cols for s in self.subsystems])

        self.rows = np.concatenate([self.gr_rows,self.rows])
        self.jac_rows = np.concatenate([self.gr_jac_rows,self.jac_rows])
        self.jac_cols = np.concatenate([self.gr_jac_cols,self.jac_cols])

        self.reactions_indicies = sum([sub.reactions_indicies for sub in self.subsystems],[])

    
    def eval_constants(self):
        FB.config.R_vbs_ground = self.R_ground
        FB.config.P_vbs_ground = self.P_ground

        for sub in self.subsystems:
            sub.eval_constants()

    
    def set_gen_coordinates(self,q):
        self.R_ground = q[0:3,0:1]
        self.P_ground = q[3:7,0:1]
        offset = 7
        for sub in self.subsystems:
            qs = q[offset:sub.n+offset]
            sub.set_gen_coordinates(qs)
            offset += sub.n

        FB.R_vbs_ground = self.R_ground
        FB.P_vbs_ground = self.P_ground

    
    def set_gen_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        offset = 7
        for sub in self.subsystems:
            qs = qd[offset:sub.n+offset]
            sub.set_gen_velocities(qs)
            offset += sub.n

        FB.Rd_vbs_ground = self.Rd_ground
        FB.Pd_vbs_ground = self.Pd_ground

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_ground = qdd[0:3,0:1]
        self.Pdd_ground = qdd[3:7,0:1]
        offset = 7
        for sub in self.subsystems:
            qs = qdd[offset:sub.n+offset]
            sub.set_gen_accelerations(qs)
            offset += sub.n

        FB.Rdd_vbs_ground = self.Rdd_ground
        FB.Pdd_vbs_ground = self.Pdd_ground

    
    def set_lagrange_multipliers(self,Lambda):
        offset = 7
        for sub in self.subsystems:
            l = Lambda[offset:sub.nc+offset]
            sub.set_lagrange_multipliers(l)
            offset += sub.nc

    
    def eval_pos_eq(self):

        pos_ground_eq_blocks = [self.R_ground,(-1*self.Pg_ground + self.P_ground)]

        for sub in self.subsystems:
            sub.eval_pos_eq()
        self.pos_eq_blocks = pos_ground_eq_blocks + sum([s.pos_eq_blocks for s in self.subsystems],[])

    
    def eval_vel_eq(self):

        vel_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_vel_eq()
        self.vel_eq_blocks = vel_ground_eq_blocks + sum([s.vel_eq_blocks for s in self.subsystems],[])

    
    def eval_acc_eq(self):

        acc_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_acc_eq()
        self.acc_eq_blocks = acc_ground_eq_blocks + sum([s.acc_eq_blocks for s in self.subsystems],[])

    
    def eval_jac_eq(self):

        jac_ground_eq_blocks = [np.eye(3,dtype=np.float64),np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_jac_eq()
        self.jac_eq_blocks = jac_ground_eq_blocks + sum([s.jac_eq_blocks for s in self.subsystems],[])

    
    def eval_mass_eq(self):

        mass_ground_eq_blocks = [self.M_ground,self.J_ground]

        for sub in self.subsystems:
            sub.eval_mass_eq()
        self.mass_eq_blocks = mass_ground_eq_blocks + sum([s.mass_eq_blocks for s in self.subsystems],[])

    
    def eval_frc_eq(self):

        frc_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_frc_eq()
        self.frc_eq_blocks = frc_ground_eq_blocks + sum([s.frc_eq_blocks for s in self.subsystems],[])

    
    def eval_reactions_eq(self):
        self.reactions = {}
        for sub in self.subsystems:
            sub.eval_reactions_eq()
            for k,v in sub.reactions.items():
                self.reactions['%s%s'%(sub.prefix,k)] = v

