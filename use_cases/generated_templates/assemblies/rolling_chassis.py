
import numpy as np

from use_cases.generated_templates.templates import dwb
from use_cases.generated_templates.templates import front_axle_testrig
from use_cases.generated_templates.templates import rolling_chassis_trg
from use_cases.generated_templates.templates import steer
from use_cases.generated_templates.templates import chassis


SU1 = dwb.topology('SU1')

TR1 = front_axle_testrig.topology('TR1')

SU2 = dwb.topology('SU2')

TR2 = rolling_chassis_trg.topology('TR2')

ST = steer.topology('ST')

CH = chassis.topology('CH')



class numerical_assembly(object):

    def __init__(self):
        self._t = 0
        self.subsystems = [SU1,TR1,SU2,TR2,ST,CH]

        self.interface_map = {'SU1.vbr_steer': 'ST.rbr_rocker', 'SU1.vbs_ground': 'ground', 'SU1.vbl_steer': 'ST.rbl_rocker', 'SU1.vbs_chassis': 'CH.rbs_chassis', 'TR1.vbs_chassis': 'CH.rbs_chassis', 'TR1.vbl_hub': 'SU1.rbl_hub', 'TR1.vbs_ground': 'ground', 'TR1.vbl_upright': 'SU1.rbl_upright', 'TR1.vbr_hub': 'SU1.rbr_hub', 'TR1.vbr_upright': 'SU1.rbr_upright', 'TR1.vbs_steer_gear': 'ST.rbr_rocker', 'SU2.vbs_ground': 'ground', 'SU2.vbl_steer': 'CH.rbs_chassis', 'SU2.vbr_steer': 'CH.rbs_chassis', 'SU2.vbs_chassis': 'CH.rbs_chassis', 'TR2.vbl_hub': 'SU2.rbl_hub', 'TR2.vbr_upright': 'SU2.rbr_upright', 'TR2.vbr_hub': 'SU2.rbr_hub', 'TR2.vbl_upright': 'SU2.rbl_upright', 'TR2.vbs_ground': 'ground', 'ST.vbs_ground': 'ground', 'ST.vbs_chassis': 'CH.rbs_chassis', 'CH.vbs_ground': 'ground'}
        self.indicies_map  = {'ground': 0, 'SU1.rbr_uca': 1, 'SU1.rbl_uca': 2, 'SU1.rbr_lca': 3, 'SU1.rbl_lca': 4, 'SU1.rbr_upright': 5, 'SU1.rbl_upright': 6, 'SU1.rbr_upper_strut': 7, 'SU1.rbl_upper_strut': 8, 'SU1.rbr_lower_strut': 9, 'SU1.rbl_lower_strut': 10, 'SU1.rbr_tie_rod': 11, 'SU1.rbl_tie_rod': 12, 'SU1.rbr_hub': 13, 'SU1.rbl_hub': 14, 'SU2.rbr_uca': 15, 'SU2.rbl_uca': 16, 'SU2.rbr_lca': 17, 'SU2.rbl_lca': 18, 'SU2.rbr_upright': 19, 'SU2.rbl_upright': 20, 'SU2.rbr_upper_strut': 21, 'SU2.rbl_upper_strut': 22, 'SU2.rbr_lower_strut': 23, 'SU2.rbl_lower_strut': 24, 'SU2.rbr_tie_rod': 25, 'SU2.rbl_tie_rod': 26, 'SU2.rbr_hub': 27, 'SU2.rbl_hub': 28, 'ST.rbs_coupler': 29, 'ST.rbr_rocker': 30, 'ST.rbl_rocker': 31, 'CH.rbs_chassis': 32}

        self.R_ground  = np.array([[0],[0],[0]],dtype=np.float64)
        self.P_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
        self.Pg_ground = np.array([[1],[0],[0],[0]],dtype=np.float64)

        self.M_ground = np.eye(3,dtype=np.float64)
        self.J_ground = np.eye(4,dtype=np.float64)

        self.gr_rows = np.array([0,1])
        self.gr_jac_rows = np.array([0,0,1,1])
        self.gr_jac_cols = np.array([0,1,0,1])

        self.nrows = 142
        self.ncols = 66

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
        SU1.config.R_vbr_steer = ST.config.R_rbr_rocker
        SU1.config.P_vbr_steer = ST.config.P_rbr_rocker
        SU1.config.R_vbs_ground = self.R_ground
        SU1.config.P_vbs_ground = self.P_ground
        SU1.config.R_vbl_steer = ST.config.R_rbl_rocker
        SU1.config.P_vbl_steer = ST.config.P_rbl_rocker
        SU1.config.R_vbs_chassis = CH.config.R_rbs_chassis
        SU1.config.P_vbs_chassis = CH.config.P_rbs_chassis
        TR1.config.R_vbs_chassis = CH.config.R_rbs_chassis
        TR1.config.P_vbs_chassis = CH.config.P_rbs_chassis
        TR1.config.R_vbl_hub = SU1.config.R_rbl_hub
        TR1.config.P_vbl_hub = SU1.config.P_rbl_hub
        TR1.config.R_vbs_ground = self.R_ground
        TR1.config.P_vbs_ground = self.P_ground
        TR1.config.R_vbl_upright = SU1.config.R_rbl_upright
        TR1.config.P_vbl_upright = SU1.config.P_rbl_upright
        TR1.config.R_vbr_hub = SU1.config.R_rbr_hub
        TR1.config.P_vbr_hub = SU1.config.P_rbr_hub
        TR1.config.R_vbr_upright = SU1.config.R_rbr_upright
        TR1.config.P_vbr_upright = SU1.config.P_rbr_upright
        TR1.config.R_vbs_steer_gear = ST.config.R_rbr_rocker
        TR1.config.P_vbs_steer_gear = ST.config.P_rbr_rocker
        SU2.config.R_vbs_ground = self.R_ground
        SU2.config.P_vbs_ground = self.P_ground
        SU2.config.R_vbl_steer = CH.config.R_rbs_chassis
        SU2.config.P_vbl_steer = CH.config.P_rbs_chassis
        SU2.config.R_vbr_steer = CH.config.R_rbs_chassis
        SU2.config.P_vbr_steer = CH.config.P_rbs_chassis
        SU2.config.R_vbs_chassis = CH.config.R_rbs_chassis
        SU2.config.P_vbs_chassis = CH.config.P_rbs_chassis
        TR2.config.R_vbl_hub = SU2.config.R_rbl_hub
        TR2.config.P_vbl_hub = SU2.config.P_rbl_hub
        TR2.config.R_vbr_upright = SU2.config.R_rbr_upright
        TR2.config.P_vbr_upright = SU2.config.P_rbr_upright
        TR2.config.R_vbr_hub = SU2.config.R_rbr_hub
        TR2.config.P_vbr_hub = SU2.config.P_rbr_hub
        TR2.config.R_vbl_upright = SU2.config.R_rbl_upright
        TR2.config.P_vbl_upright = SU2.config.P_rbl_upright
        TR2.config.R_vbs_ground = self.R_ground
        TR2.config.P_vbs_ground = self.P_ground
        ST.config.R_vbs_ground = self.R_ground
        ST.config.P_vbs_ground = self.P_ground
        ST.config.R_vbs_chassis = CH.config.R_rbs_chassis
        ST.config.P_vbs_chassis = CH.config.P_rbs_chassis
        CH.config.R_vbs_ground = self.R_ground
        CH.config.P_vbs_ground = self.P_ground

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

        SU1.R_vbr_steer = ST.R_rbr_rocker
        SU1.P_vbr_steer = ST.P_rbr_rocker
        SU1.R_vbs_ground = self.R_ground
        SU1.P_vbs_ground = self.P_ground
        SU1.R_vbl_steer = ST.R_rbl_rocker
        SU1.P_vbl_steer = ST.P_rbl_rocker
        SU1.R_vbs_chassis = CH.R_rbs_chassis
        SU1.P_vbs_chassis = CH.P_rbs_chassis
        TR1.R_vbs_chassis = CH.R_rbs_chassis
        TR1.P_vbs_chassis = CH.P_rbs_chassis
        TR1.R_vbl_hub = SU1.R_rbl_hub
        TR1.P_vbl_hub = SU1.P_rbl_hub
        TR1.R_vbs_ground = self.R_ground
        TR1.P_vbs_ground = self.P_ground
        TR1.R_vbl_upright = SU1.R_rbl_upright
        TR1.P_vbl_upright = SU1.P_rbl_upright
        TR1.R_vbr_hub = SU1.R_rbr_hub
        TR1.P_vbr_hub = SU1.P_rbr_hub
        TR1.R_vbr_upright = SU1.R_rbr_upright
        TR1.P_vbr_upright = SU1.P_rbr_upright
        TR1.R_vbs_steer_gear = ST.R_rbr_rocker
        TR1.P_vbs_steer_gear = ST.P_rbr_rocker
        SU2.R_vbs_ground = self.R_ground
        SU2.P_vbs_ground = self.P_ground
        SU2.R_vbl_steer = CH.R_rbs_chassis
        SU2.P_vbl_steer = CH.P_rbs_chassis
        SU2.R_vbr_steer = CH.R_rbs_chassis
        SU2.P_vbr_steer = CH.P_rbs_chassis
        SU2.R_vbs_chassis = CH.R_rbs_chassis
        SU2.P_vbs_chassis = CH.P_rbs_chassis
        TR2.R_vbl_hub = SU2.R_rbl_hub
        TR2.P_vbl_hub = SU2.P_rbl_hub
        TR2.R_vbr_upright = SU2.R_rbr_upright
        TR2.P_vbr_upright = SU2.P_rbr_upright
        TR2.R_vbr_hub = SU2.R_rbr_hub
        TR2.P_vbr_hub = SU2.P_rbr_hub
        TR2.R_vbl_upright = SU2.R_rbl_upright
        TR2.P_vbl_upright = SU2.P_rbl_upright
        TR2.R_vbs_ground = self.R_ground
        TR2.P_vbs_ground = self.P_ground
        ST.R_vbs_ground = self.R_ground
        ST.P_vbs_ground = self.P_ground
        ST.R_vbs_chassis = CH.R_rbs_chassis
        ST.P_vbs_chassis = CH.P_rbs_chassis
        CH.R_vbs_ground = self.R_ground
        CH.P_vbs_ground = self.P_ground

    
    def set_gen_velocities(self,qd):
        self.Rd_ground = qd[0:3,0:1]
        self.Pd_ground = qd[3:7,0:1]
        offset = 7
        for sub in self.subsystems:
            qs = qd[offset:sub.n+offset]
            sub.set_gen_velocities(qs)
            offset += sub.n

        SU1.Rd_vbr_steer = ST.Rd_rbr_rocker
        SU1.Pd_vbr_steer = ST.Pd_rbr_rocker
        SU1.Rd_vbs_ground = self.Rd_ground
        SU1.Pd_vbs_ground = self.Pd_ground
        SU1.Rd_vbl_steer = ST.Rd_rbl_rocker
        SU1.Pd_vbl_steer = ST.Pd_rbl_rocker
        SU1.Rd_vbs_chassis = CH.Rd_rbs_chassis
        SU1.Pd_vbs_chassis = CH.Pd_rbs_chassis
        TR1.Rd_vbs_chassis = CH.Rd_rbs_chassis
        TR1.Pd_vbs_chassis = CH.Pd_rbs_chassis
        TR1.Rd_vbl_hub = SU1.Rd_rbl_hub
        TR1.Pd_vbl_hub = SU1.Pd_rbl_hub
        TR1.Rd_vbs_ground = self.Rd_ground
        TR1.Pd_vbs_ground = self.Pd_ground
        TR1.Rd_vbl_upright = SU1.Rd_rbl_upright
        TR1.Pd_vbl_upright = SU1.Pd_rbl_upright
        TR1.Rd_vbr_hub = SU1.Rd_rbr_hub
        TR1.Pd_vbr_hub = SU1.Pd_rbr_hub
        TR1.Rd_vbr_upright = SU1.Rd_rbr_upright
        TR1.Pd_vbr_upright = SU1.Pd_rbr_upright
        TR1.Rd_vbs_steer_gear = ST.Rd_rbr_rocker
        TR1.Pd_vbs_steer_gear = ST.Pd_rbr_rocker
        SU2.Rd_vbs_ground = self.Rd_ground
        SU2.Pd_vbs_ground = self.Pd_ground
        SU2.Rd_vbl_steer = CH.Rd_rbs_chassis
        SU2.Pd_vbl_steer = CH.Pd_rbs_chassis
        SU2.Rd_vbr_steer = CH.Rd_rbs_chassis
        SU2.Pd_vbr_steer = CH.Pd_rbs_chassis
        SU2.Rd_vbs_chassis = CH.Rd_rbs_chassis
        SU2.Pd_vbs_chassis = CH.Pd_rbs_chassis
        TR2.Rd_vbl_hub = SU2.Rd_rbl_hub
        TR2.Pd_vbl_hub = SU2.Pd_rbl_hub
        TR2.Rd_vbr_upright = SU2.Rd_rbr_upright
        TR2.Pd_vbr_upright = SU2.Pd_rbr_upright
        TR2.Rd_vbr_hub = SU2.Rd_rbr_hub
        TR2.Pd_vbr_hub = SU2.Pd_rbr_hub
        TR2.Rd_vbl_upright = SU2.Rd_rbl_upright
        TR2.Pd_vbl_upright = SU2.Pd_rbl_upright
        TR2.Rd_vbs_ground = self.Rd_ground
        TR2.Pd_vbs_ground = self.Pd_ground
        ST.Rd_vbs_ground = self.Rd_ground
        ST.Pd_vbs_ground = self.Pd_ground
        ST.Rd_vbs_chassis = CH.Rd_rbs_chassis
        ST.Pd_vbs_chassis = CH.Pd_rbs_chassis
        CH.Rd_vbs_ground = self.Rd_ground
        CH.Pd_vbs_ground = self.Pd_ground

    
    def set_gen_accelerations(self,qdd):
        self.Rdd_ground = qdd[0:3,0:1]
        self.Pdd_ground = qdd[3:7,0:1]
        offset = 7
        for sub in self.subsystems:
            qs = qdd[offset:sub.n+offset]
            sub.set_gen_accelerations(qs)
            offset += sub.n

        SU1.Rdd_vbr_steer = ST.Rdd_rbr_rocker
        SU1.Pdd_vbr_steer = ST.Pdd_rbr_rocker
        SU1.Rdd_vbs_ground = self.Rdd_ground
        SU1.Pdd_vbs_ground = self.Pdd_ground
        SU1.Rdd_vbl_steer = ST.Rdd_rbl_rocker
        SU1.Pdd_vbl_steer = ST.Pdd_rbl_rocker
        SU1.Rdd_vbs_chassis = CH.Rdd_rbs_chassis
        SU1.Pdd_vbs_chassis = CH.Pdd_rbs_chassis
        TR1.Rdd_vbs_chassis = CH.Rdd_rbs_chassis
        TR1.Pdd_vbs_chassis = CH.Pdd_rbs_chassis
        TR1.Rdd_vbl_hub = SU1.Rdd_rbl_hub
        TR1.Pdd_vbl_hub = SU1.Pdd_rbl_hub
        TR1.Rdd_vbs_ground = self.Rdd_ground
        TR1.Pdd_vbs_ground = self.Pdd_ground
        TR1.Rdd_vbl_upright = SU1.Rdd_rbl_upright
        TR1.Pdd_vbl_upright = SU1.Pdd_rbl_upright
        TR1.Rdd_vbr_hub = SU1.Rdd_rbr_hub
        TR1.Pdd_vbr_hub = SU1.Pdd_rbr_hub
        TR1.Rdd_vbr_upright = SU1.Rdd_rbr_upright
        TR1.Pdd_vbr_upright = SU1.Pdd_rbr_upright
        TR1.Rdd_vbs_steer_gear = ST.Rdd_rbr_rocker
        TR1.Pdd_vbs_steer_gear = ST.Pdd_rbr_rocker
        SU2.Rdd_vbs_ground = self.Rdd_ground
        SU2.Pdd_vbs_ground = self.Pdd_ground
        SU2.Rdd_vbl_steer = CH.Rdd_rbs_chassis
        SU2.Pdd_vbl_steer = CH.Pdd_rbs_chassis
        SU2.Rdd_vbr_steer = CH.Rdd_rbs_chassis
        SU2.Pdd_vbr_steer = CH.Pdd_rbs_chassis
        SU2.Rdd_vbs_chassis = CH.Rdd_rbs_chassis
        SU2.Pdd_vbs_chassis = CH.Pdd_rbs_chassis
        TR2.Rdd_vbl_hub = SU2.Rdd_rbl_hub
        TR2.Pdd_vbl_hub = SU2.Pdd_rbl_hub
        TR2.Rdd_vbr_upright = SU2.Rdd_rbr_upright
        TR2.Pdd_vbr_upright = SU2.Pdd_rbr_upright
        TR2.Rdd_vbr_hub = SU2.Rdd_rbr_hub
        TR2.Pdd_vbr_hub = SU2.Pdd_rbr_hub
        TR2.Rdd_vbl_upright = SU2.Rdd_rbl_upright
        TR2.Pdd_vbl_upright = SU2.Pdd_rbl_upright
        TR2.Rdd_vbs_ground = self.Rdd_ground
        TR2.Pdd_vbs_ground = self.Pdd_ground
        ST.Rdd_vbs_ground = self.Rdd_ground
        ST.Pdd_vbs_ground = self.Pdd_ground
        ST.Rdd_vbs_chassis = CH.Rdd_rbs_chassis
        ST.Pdd_vbs_chassis = CH.Pdd_rbs_chassis
        CH.Rdd_vbs_ground = self.Rdd_ground
        CH.Pdd_vbs_ground = self.Pdd_ground

    
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

