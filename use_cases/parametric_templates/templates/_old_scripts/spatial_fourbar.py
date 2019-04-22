# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:23:45 2019

@author: khaled.ghobashy
"""

from source.interfaces.scripting_interfaces import topology

def create():
    global template    
    template = topology(__file__)
    
    # Adding System Bodies
    template.add_body('crank')
    template.add_body('rocker')
    template.add_body('coupler')
    
    # Adding System Joints
    template.add_joint.revolute('rev_crank','rbs_crank','vbs_ground')
    template.add_joint.revolute('rev_rocker','rbs_rocker','vbs_ground')
    template.add_joint.spherical('sph_coupler_crank','rbs_coupler','rbs_crank')
    template.add_joint.universal('uni_coupler_rocker','rbs_coupler','rbs_rocker')
    template.add_actuator.rotational_actuator('act','jcs_rev_crank')

    # Assembling System
    template.assemble_model()
    template.save()
    
    # Writing Code Files
    template.write_python_code()    

if __name__ == '__main__':
    create()
else:
    template = topology.reload(__file__)
