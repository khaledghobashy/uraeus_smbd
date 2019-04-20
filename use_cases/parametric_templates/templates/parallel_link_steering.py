# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 08:21:51 2019

@author: khaled.ghobashy
"""
from source.interfaces.scripting_interfaces import topology

def create():
    global template
    
    template = topology(__file__)
    
    template.add_body('coupler')
    template.add_body('rocker',mirrored=True)
    template.add_body('chassis',virtual=True)
    
    template.add_joint.revolute('rocker_ch','rbr_rocker','vbs_chassis',mirrored=True)
    template.add_joint.spherical('rc_sph','rbr_rocker','rbs_coupler')
    template.add_joint.cylinderical('rc_cyl','rbl_rocker','rbs_coupler')
    
    template.assemble_model()
    template.save()
    template.write_python_code()
    
if __name__ == '__main__':
    create()
else:
    template = topology.reload(__file__)
