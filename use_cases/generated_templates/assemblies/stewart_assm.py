
import numpy as np



from use_cases.generated_templates.templates import stewart_gough_3dof
from use_cases.generated_templates.templates import stewart_testrig

SG = stewart_gough_3dof.topology('SG')

TR = stewart_testrig.topology('TR')



class numerical_assembly(object):

    def __init__(self):
        self._t = 0
        self.subsystems = [SG,TR]

        self.interface_map = {'SG.vbs_ground': 'ground', 'TR.vbs_ground': 'ground', 'TR.vbs_rocker_3': 'SG.rbs_rocker_3', 'TR.vbs_rocker_2': 'SG.rbs_rocker_2', 'TR.vbs_rocker_1': 'SG.rbs_rocker_1'}
        self.indicies_map  = {'ground': 0, 'SG.rbs_table': 1, 'SG.rbs_link_1': 2, 'SG.rbs_link_2': 3, 'SG.rbs_link_3': 4, 'SG.rbs_rocker_1': 5, 'SG.rbs_rocker_2': 6, 'SG.rbs_rocker_3': 7}

        self.R_ground  = np.array([[0],[0],[0]],dtype=np.float64)
        self.P_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
        self.Pg_ground = np.array([[1],[0],[0],[0]],dtype=np.float64)

        self.gr_rows = np.array([0,1])
        self.gr_jac_rows = np.array([0,0,1,1])
        self.gr_jac_cols = np.array([0,1,0,1])

        self.nrows = 33
        self.ncols = 16

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
        offset = 0
        for sub in self.subsystems:
            sub.assemble_template(self.indicies_map,self.interface_map,offset)
            offset += sub.nrows

        self.gr_rows += offset
        self.gr_jac_rows += offset

        self.rows = np.concatenate([s.rows for s in self.subsystems])
        self.jac_rows = np.concatenate([s.jac_rows for s in self.subsystems])
        self.jac_cols = np.concatenate([s.jac_cols for s in self.subsystems])

        self.rows = np.concatenate([self.rows,self.gr_rows])
        self.jac_rows = np.concatenate([self.jac_rows,self.gr_jac_rows])
        self.jac_cols = np.concatenate([self.jac_cols,self.gr_jac_cols])

    
    def eval_constants(self):
        SG.config.R_vbs_ground = self.R_ground
        SG.config.P_vbs_ground = self.P_ground
        TR.config.R_vbs_ground = self.R_ground
        TR.config.P_vbs_ground = self.P_ground
        TR.config.R_vbs_rocker_3 = SG.config.R_rbs_rocker_3
        TR.config.P_vbs_rocker_3 = SG.config.P_rbs_rocker_3
        TR.config.R_vbs_rocker_2 = SG.config.R_rbs_rocker_2
        TR.config.P_vbs_rocker_2 = SG.config.P_rbs_rocker_2
        TR.config.R_vbs_rocker_1 = SG.config.R_rbs_rocker_1
        TR.config.P_vbs_rocker_1 = SG.config.P_rbs_rocker_1

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

        SG.R_vbs_ground = self.R_ground
        SG.P_vbs_ground = self.P_ground
        TR.R_vbs_ground = self.R_ground
        TR.P_vbs_ground = self.P_ground
        TR.R_vbs_rocker_3 = SG.R_rbs_rocker_3
        TR.P_vbs_rocker_3 = SG.P_rbs_rocker_3
        TR.R_vbs_rocker_2 = SG.R_rbs_rocker_2
        TR.P_vbs_rocker_2 = SG.P_rbs_rocker_2
        TR.R_vbs_rocker_1 = SG.R_rbs_rocker_1
        TR.P_vbs_rocker_1 = SG.P_rbs_rocker_1

    
    def set_gen_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        offset = 7
        for sub in self.subsystems:
            qs = qd[offset:sub.n+offset]
            sub.set_gen_velocities(qs)
            offset += sub.n

        SG.Rd_vbs_ground = self.Rd_ground
        SG.Pd_vbs_ground = self.Pd_ground
        TR.Rd_vbs_ground = self.Rd_ground
        TR.Pd_vbs_ground = self.Pd_ground
        TR.Rd_vbs_rocker_3 = SG.Rd_rbs_rocker_3
        TR.Pd_vbs_rocker_3 = SG.Pd_rbs_rocker_3
        TR.Rd_vbs_rocker_2 = SG.Rd_rbs_rocker_2
        TR.Pd_vbs_rocker_2 = SG.Pd_rbs_rocker_2
        TR.Rd_vbs_rocker_1 = SG.Rd_rbs_rocker_1
        TR.Pd_vbs_rocker_1 = SG.Pd_rbs_rocker_1

    
    def eval_pos_eq(self):

        pos_ground_eq_blocks = [self.R_ground,(-1*self.Pg_ground + self.P_ground)]

        for sub in self.subsystems:
            sub.eval_pos_eq()
        self.pos_eq_blocks = sum([s.pos_eq_blocks for s in self.subsystems],[])
        self.pos_eq_blocks += pos_ground_eq_blocks

    
    def eval_vel_eq(self):

        vel_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_vel_eq()
        self.vel_eq_blocks = sum([s.vel_eq_blocks for s in self.subsystems],[])
        self.vel_eq_blocks += vel_ground_eq_blocks

    
    def eval_acc_eq(self):

        acc_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_acc_eq()
        self.acc_eq_blocks = sum([s.acc_eq_blocks for s in self.subsystems],[])
        self.acc_eq_blocks += acc_ground_eq_blocks

    
    def eval_jac_eq(self):

        jac_ground_eq_blocks = [np.eye(3,dtype=np.float64),np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4,dtype=np.float64)]

        for sub in self.subsystems:
            sub.eval_jac_eq()
        self.jac_eq_blocks = sum([s.jac_eq_blocks for s in self.subsystems],[])
        self.jac_eq_blocks += jac_ground_eq_blocks
  
