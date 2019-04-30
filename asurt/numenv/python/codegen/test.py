#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:40:45 2019

@author: khaledghobashy
"""

from ..templates import spatial_fourbar

class subsystems(object):
    MOD = spatial_fourbar.topology('MOD')

interface_map = {'MOD.vbs_ground': 'ground'}
indicies_map  = {'ground': 0, 'MOD.rbs_crank': 1, 'MOD.rbs_rocker': 2, 'MOD.rbs_coupler': 3}

from asurt.numenv.python.codegen import numassm

class numerical_assembly(numassm.abstract_assembly):
    
    def __init__(self):
        super().__init__(subsystems, interface_map, indicies_map)
        
    def eval_constants(self):
        MOD = subsystems.MOD
        MOD.config.R_vbs_ground = self.R_ground
        MOD.config.P_vbs_ground = self.P_ground
        super().eval_constants()
    
    def set_gen_coordinates(self, q):
        super().set_gen_coordinates(q)
        MOD = subsystems.MOD
        MOD.R_vbs_ground = self.R_ground
        MOD.P_vbs_ground = self.P_ground

    def set_gen_velocities(self,qd):
        super().set_gen_velocities(qd)
        MOD = subsystems.MOD
        MOD.Rd_vbs_ground = self.Rd_ground
        MOD.Pd_vbs_ground = self.Pd_ground

    def set_gen_accelerations(self,qdd):
        super().set_gen_accelerations(qdd)
        MOD = subsystems.MOD
        MOD.Rdd_vbs_ground = self.Rdd_ground
        MOD.Pdd_vbs_ground = self.Pdd_ground

