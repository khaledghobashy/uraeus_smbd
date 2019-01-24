
                import numpy as np

                import dwb
import steering


            SU1_config = dwb.configuration()
            SU1 = dwb.topology(SU1_config,'SU1')


            SU2_config = dwb.configuration()
            SU2 = dwb.topology(SU2_config,'SU2')


            ST_config = steering.configuration()
            ST = steering.topology(ST_config,'ST')




class numerical_assembly(object):

    def __init__(self):
        self.set_time(0.0)
        self.Pg_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
        self.subsystems = [SU1,SU2,ST]

        self.interface_map = {'SU1.vbs_chassis': 'ground', 'SU1.vbr_steer': 'ST.rbr_rocker', 'SU1.vbl_steer': 'ST.rbl_rocker', 'SU1.vbs_ground': 'ground', 'SU2.vbl_steer': 'ground', 'SU2.vbs_ground': 'ground', 'SU2.vbr_steer': 'ground', 'SU2.vbs_chassis': 'ground', 'ST.vbs_ground': 'ground', 'ST.vbs_chassis': 'ground'}
        self.indicies_map  = {'ground': 0, 'SU1.rbr_lower_strut': 1, 'SU1.rbl_lower_strut': 2, 'SU1.rbr_uca': 3, 'SU1.rbl_lca': 4, 'SU1.rbr_lca': 5, 'SU1.rbl_hub': 6, 'SU1.rbr_tie_rod': 7, 'SU1.rbl_upright': 8, 'SU1.rbl_upper_strut': 9, 'SU1.rbr_hub': 10, 'SU1.rbl_uca': 11, 'SU1.rbr_upper_strut': 12, 'SU1.rbl_tie_rod': 13, 'SU1.rbr_upright': 14, 'SU2.rbr_lower_strut': 15, 'SU2.rbl_lower_strut': 16, 'SU2.rbr_uca': 17, 'SU2.rbl_lca': 18, 'SU2.rbr_lca': 19, 'SU2.rbl_hub': 20, 'SU2.rbr_tie_rod': 21, 'SU2.rbl_upright': 22, 'SU2.rbl_upper_strut': 23, 'SU2.rbr_hub': 24, 'SU2.rbl_uca': 25, 'SU2.rbr_upper_strut': 26, 'SU2.rbl_tie_rod': 27, 'SU2.rbr_upright': 28, 'ST.rbr_rocker': 29, 'ST.rbl_rocker': 30, 'ST.rbs_coupler': 31}

    def set_time(self,t):
        for sub in self.subsystems:
            sub.t = t

    def assemble_system(self):
        offset = 0
        for sub in self.subsystems:
            sub.assemble_template(self.indicies_map,self.interface_map,offset)
            offset += sub.n

    def set_gen_coordinates(self,q):
        offset = 0
        for sub in self.subsystems:
            qs = q[offset:sub.n]
            sub.set_gen_coordinates(qs)
            offset += sub.n

    def set_gen_velocities(self,qd):
        offset = 0
        for sub in self.subsystems:
            qs = qd[offset:sub.n]
            sub.set_gen_velocities(qs)
            offset += sub.n

    
    def eval_pos_eq(self):
        for sub in self.subsystems:
            sub.eval_pos_eq()

    
    def eval_vel_eq(self):
        for sub in self.subsystems:
            sub.eval_vel_eq()

    
    def eval_acc_eq(self):
        for sub in self.subsystems:
            sub.eval_acc_eq()

    
    def eval_jac_eq(self):
        for sub in self.subsystems:
            sub.eval_jac_eq()
  
