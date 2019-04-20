# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:08:34 2019

@author: khaled.ghobashy
"""

from source.interfaces.scripting_interfaces import topology

def create():
    template = topology(__file__)
    
    template.add_body('hub', mirrored=True, virtual=True)
    template.add_body('upright', mirrored=True, virtual=True)
    template.add_joint.revolute('rev', 'vbr_upright', 'vbr_hub', mirrored=True, virtual=True)
    template.add_actuator.rotational_actuator('act', 'jcr_rev', mirrored=True)
    template.add_actuator.absolute_locator('ver_act', 'vbr_upright', 'z', mirrored=True)
    
    template.add_body('steer_gear',virtual=True)
    template.add_body('chassis',virtual=True)
    template.add_joint.revolute('steer_gear','vbs_steer_gear','vbs_chassis',virtual=True)
    template.add_actuator.rotational_actuator('steer_act','jcs_steer_gear')
    
    template.assemble_model()
    template.save()
    template.write_python_code()
    
if __name__ == '__main__':
    create()
else:
    template = topology.reload(__file__)

