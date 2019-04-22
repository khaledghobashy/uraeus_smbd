# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:18:17 2019

@author: khaled.ghobashy
"""
import os
import pickle

from source.symbolic_classes.joints import (revolute,spherical, 
                                                    tripod, universal)
from source.mbs_creators.topology_classes import template_based_topology
from source.code_generators.python_code_generators import template_code_generator

'''
array([[  4.62811840e+08,   1.06627320e+05,   4.66877330e+07],
       [  1.06627320e+05,   1.01388000e+09,   2.31185260e+04],
       [  4.66877330e+07,   2.31185260e+04,   1.11623380e+09]])
'''
topology_name = 'stewart_gough_3dof'

def load():
    global template
    path = os.path.dirname(__file__)
    with open('%s\\%s.stpl'%(path,topology_name),'rb') as f:
        template = pickle.load(f)

def create():   
    global template
    
    template = template_based_topology(topology_name)
    
    template.add_body('table')
    template.add_body('link_1')
    template.add_body('link_2')
    template.add_body('link_3')
    template.add_body('rocker_1')
    template.add_body('rocker_2')
    template.add_body('rocker_3')
    
    template.add_joint(revolute,'rev_1','vbs_ground','rbs_rocker_1')
    template.add_joint(revolute,'rev_2','vbs_ground','rbs_rocker_2')
    template.add_joint(revolute,'rev_3','vbs_ground','rbs_rocker_3')
    
    
    template.add_joint(spherical,'bottom_sph_1','rbs_rocker_1','rbs_link_1')
    template.add_joint(spherical,'bottom_sph_2','rbs_rocker_2','rbs_link_2')
    template.add_joint(spherical,'bottom_sph_3','rbs_rocker_3','rbs_link_3')
    
    template.add_joint(universal,'upper_uni_1','rbs_link_1','rbs_table')
    template.add_joint(universal,'upper_uni_2','rbs_link_2','rbs_table')
    template.add_joint(universal,'upper_uni_3','rbs_link_3','rbs_table')
    
    template.add_joint(tripod,'tripod','rbs_table','vbs_ground')


    template.assemble_model()
    template.save()
    
    numerical_code = template_code_generator(template)
    numerical_code.write_code_file()
    

if __name__ == '__main__':
    create()
else:
    load()
    
