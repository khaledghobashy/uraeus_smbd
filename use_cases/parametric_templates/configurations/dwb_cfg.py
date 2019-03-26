# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:56:01 2019

@author: khaled.ghobashy
"""
import os

from source.mbs_creators.topology_helpers import parametric_configuration
from source.code_generators.python_code_generators import configuration_code_generator
from source.post_processors.blender.scripter import scripter

import use_cases.parametric_templates.templates.double_wishbone_direct_acting as model

configuration_name = os.path.basename(__file__).split('.')[0]

def main():
    
    global config
    
    config = parametric_configuration(configuration_name, model.template)
    config.assemble_base_layer()
    
    # Adding Points
    config.add_point.UserInput('ucaf', mirror=True)
    config.add_point.UserInput('ucar', mirror=True)
    config.add_point.UserInput('ucao', mirror=True)
    config.add_point.UserInput('lcaf', mirror=True)
    config.add_point.UserInput('lcar', mirror=True)
    config.add_point.UserInput('lcao', mirror=True)
    config.add_point.UserInput('tro', mirror=True)
    config.add_point.UserInput('tri', mirror=True)
    config.add_point.UserInput('strut_chassis', mirror=True)
    config.add_point.UserInput('strut_lca', mirror=True)
    config.add_point.UserInput('strut_mid', mirror=True)
    config.add_point.UserInput('wc', mirror=True)
    
    config.add_point.Centered('hpr_strut_mid', ('hpr_strut_chassis', 'hpr_strut_lca'), mirror=True)

    # JOINTS CONFIGURATIONS
    
    # Spherical Joints:
    config.add_relation.Equal_to('pt1_jcr_uca_upright', ('hpr_ucao',), mirror=True)
    config.add_relation.Equal_to('pt1_jcr_lca_upright', ('hpr_lcao',), mirror=True)
    config.add_relation.Equal_to('pt1_jcr_tie_upright', ('hpr_tro',), mirror=True)
    
    # Upper Control Arm Revolute Joint:
    config.add_relation.Centered('pt1_jcr_uca_chassis', ('hpr_ucaf','hpr_ucar'), mirror=True)
    config.add_relation.Oriented('ax1_jcr_uca_chassis', ('hpr_ucaf','hpr_ucar'), mirror=True)
    
    # Lower Control Arm Revolute Joint:
    config.add_relation.Centered('pt1_jcr_lca_chassis', ('hpr_lcaf','hpr_lcar'), mirror=True)
    config.add_relation.Oriented('ax1_jcr_lca_chassis', ('hpr_lcaf','hpr_lcar'), mirror=True)
    
    # Wheel Hub Revolute Joint:
    config.add_relation.Equal_to('pt1_jcr_hub_bearing', ('hpr_wc',), mirror=True)
    
    # Strut-Chassis Universal Joint:
    config.add_relation.Equal_to('pt1_jcr_strut_chassis', ('hpr_strut_chassis',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_strut_chassis', ('hpr_strut_chassis','hpr_strut_lca'), mirror=True)
    config.add_relation.Oriented('ax2_jcr_strut_chassis', ('hpr_strut_lca','hpr_strut_chassis'), mirror=True)
    
    # Strut-LCA Universal Joint:
    config.add_relation.Equal_to('pt1_jcr_strut_lca', ('hpr_strut_lca',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_strut_lca', ('hpr_strut_chassis','hpr_strut_lca'), mirror=True)
    config.add_relation.Oriented('ax2_jcr_strut_lca', ('hpr_strut_lca','hpr_strut_chassis'), mirror=True)
    
    # Tie-Steer Universal Joint:
    config.add_relation.Equal_to('pt1_jcr_tie_steering', ('hpr_tri',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_tie_steering', ('hpr_tri','hpr_tro'), mirror=True)
    config.add_relation.Oriented('ax2_jcr_tie_steering', ('hpr_tro','hpr_tri'), mirror=True)
    
    # Strut Cylinderical Joint:
    config.add_relation.Equal_to('pt1_jcr_strut', ('hpr_strut_mid',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_strut', ('hpr_strut_lca','hpr_strut_chassis'), mirror=True)
    
    
    # GEOMETRIES
    config.add_scalar.UserInput('links_ro')
    config.add_scalar.UserInput('strut_outer')
    config.add_scalar.UserInput('strut_inner')
    config.add_scalar.UserInput('thickness')
    config.add_scalar.UserInput('hub_radius')
    
    config.add_geometry.Triangular_Prism('uca', ('hpr_ucaf','hpr_ucar','hpr_ucao','s_thickness'), mirror=True)
    config.assign_geometry_to_body('rbr_uca', 'gmr_uca', mirror=True)
    
    config.add_geometry.Triangular_Prism('lca', ('hpr_lcaf','hpr_lcar','hpr_lcao','s_thickness'), mirror=True)
    config.assign_geometry_to_body('rbr_lca', 'gmr_lca', mirror=True)
    
    config.add_geometry.Triangular_Prism('upright', ('hpr_ucao','hpr_wc','hpr_lcao','s_thickness'), mirror=True)
    config.assign_geometry_to_body('rbr_upright', 'gmr_upright', mirror=True)

    config.add_geometry.Cylinder_Geometry('upper_strut', ('hpr_strut_chassis','hpr_strut_mid','s_strut_outer'), mirror=True)
    config.assign_geometry_to_body('rbr_upper_strut', 'gmr_upper_strut', mirror=True)

    config.add_geometry.Cylinder_Geometry('lower_strut', ('hpr_strut_mid','hpr_strut_lca','s_strut_inner'), mirror=True)
    config.assign_geometry_to_body('rbr_lower_strut', 'gmr_lower_strut', mirror=True)
    
    config.add_geometry.Cylinder_Geometry('tie_rod', ('hpr_tri','hpr_tro','s_links_ro'), mirror=True)
    config.assign_geometry_to_body('rbr_tie_rod', 'gmr_tie_rod', mirror=True)
    
    config.add_geometry.Cylinder_Geometry('hub', ('hpr_wc','R_rbr_upright','s_hub_radius'), mirror=True)
    config.assign_geometry_to_body('rbr_hub', 'gmr_hub', mirror=True)
        
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

    geo_code = scripter(config)
    geo_code.write_code_file()


if __name__ == '__main__':
    main()

