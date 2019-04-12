# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:54:31 2019

@author: khale
"""
from source.interfaces.scripting_interfaces import configuration
import use_cases.parametric_templates.templates.parallel_link_steering as model

def main():
        
    global config
    config = configuration(__file__, model.template)
    
    # Adding Points    
    config.add_point.UserInput('rocker_chassis', mirror=True)
    config.add_point.UserInput('rocker_coupler', mirror=True)
    
    config.add_relation.Equal_to('pt1_jcr_rocker_ch', ('hpr_rocker_chassis',) ,mirror=True)
    config.add_relation.Equal_to('pt1_jcs_rc_sph', ('hpr_rocker_coupler',))
    config.add_relation.Equal_to('pt1_jcs_rc_cyl', ('hpl_rocker_coupler',))
    
    config.add_relation.Centered('R_rbr_rocker', ('hpr_rocker_chassis','hpr_rocker_coupler'),mirror=True)
    config.add_relation.Centered('R_rbs_coupler', ('hpr_rocker_coupler','hpl_rocker_coupler'))
    
    config.add_relation.Oriented('ax1_jcs_rc_cyl', ('hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis'))
    config.add_relation.Oriented('ax1_jcr_rocker_ch', ('hpr_rocker_coupler','hpl_rocker_coupler','hpr_rocker_chassis'),mirror=True)
    
    # GEOMETRIES
    config.add_scalar.UserInput('links_ro')
    config.add_scalar.UserInput('thickness')
    
    config.add_geometry.Cylinder_Geometry('rocker', ('hpr_rocker_chassis','hpr_rocker_coupler','s_links_ro'), mirror=True)
    config.assign_geometry_to_body('rbr_rocker', 'gmr_rocker', mirror=True)
    
    config.add_geometry.Cylinder_Geometry('coupler', ('hpr_rocker_coupler','hpl_rocker_coupler','s_links_ro'))
    config.assign_geometry_to_body('rbs_coupler','gms_coupler')
    
    # Writing Code Files
    config.write_python_code()
    config.write_blender_script()


if __name__ == '__main__':
    main()

