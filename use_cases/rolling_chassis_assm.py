
import numpy as np

import dwb
import steering

SU1 = dwb.numerical_assembly()
SU2 = dwb.numerical_assembly()
ST = steering.numerical_assembly()



class numerical_assembly(object):

    def __init__(self):
        self.set_time(0.0)
        self.Pg_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
        self.subsystems = [SU1,SU2,ST]

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
  
