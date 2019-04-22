# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:37:32 2019

@author: khaled.ghobashy
"""

from source.interfaces.scripting_interfaces import configuration
import use_cases.parametric_templates.templates.spatial_fourbar as model

def main():
    
    global config
    config = configuration(__file__, model.template)
    
    # Adding Points
    config.add_point.UserInput('rev_crank')
    config.add_point.UserInput('rev_rocker')
    config.add_point.UserInput('coupler_crank')
    config.add_point.UserInput('coupler_rocker')
    
    config.add_relation.Equal_to('pt1_jcs_rev_crank',('hps_rev_crank',))

    config.add_relation.Equal_to('pt1_jcs_rev_rocker',('hps_rev_rocker',))
    
    config.add_relation.Equal_to('pt1_jcs_sph_coupler_crank',('hps_coupler_crank',))
    
    config.add_relation.Equal_to('pt1_jcs_uni_coupler_rocker',('hps_coupler_rocker',))
    config.add_relation.Oriented('ax1_jcs_uni_coupler_rocker',('hps_coupler_rocker','hps_coupler_crank'))
    config.add_relation.Oriented('ax2_jcs_uni_coupler_rocker',('hps_coupler_crank','hps_coupler_rocker'))
    
    # GEOMETRIES
    config.add_scalar.UserInput('links_ro')
    
    config.add_geometry.Cylinder_Geometry('rocker',('hps_rev_rocker','hps_coupler_rocker','s_links_ro'))
    config.assign_geometry_to_body('rbs_rocker','gms_rocker')
    
    config.add_geometry.Cylinder_Geometry('crank',('hps_rev_crank','hps_coupler_crank','s_links_ro'))
    config.assign_geometry_to_body('rbs_crank','gms_crank')

    config.add_geometry.Cylinder_Geometry('coupler',('hps_coupler_crank','hps_coupler_rocker','s_links_ro'))
    config.assign_geometry_to_body('rbs_coupler','gms_coupler')
    
    # Writing Code Files
    config.write_python_code()
    config.write_blender_script()


if __name__ == '__main__':
    main()

