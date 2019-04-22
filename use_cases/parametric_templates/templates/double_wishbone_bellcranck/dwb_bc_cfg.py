# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:56:01 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.double_wishbone_bellcranck as model
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  composite_geometry,
                                                  triangular_prism, 
                                                  cylinder_geometry)

def main():
    global config
    
    name = 'dwb_bc_points'
    config = parametric_configuration(model.template)
    config.assemble_base_layer()
    config.name = name
    model.template.cfg_file = name
    
    # Adding Points
    config.add_point('ucaf',mirror=True)
    config.add_point('ucar',mirror=True)
    config.add_point('ucao',mirror=True)
    config.add_point('lcaf',mirror=True)
    config.add_point('lcar',mirror=True)
    config.add_point('lcao',mirror=True)
    config.add_point('tro',mirror=True)
    config.add_point('tri',mirror=True)
    config.add_point('rocker_chassis',mirror=True)
    config.add_point('pushrod_rocker',mirror=True)
    config.add_point('pushrod_uca',mirror=True)
    config.add_point('strut_chassis',mirror=True)
    config.add_point('strut_rocker',mirror=True)
    config.add_point('strut_mid',mirror=True)
    config.add_point('wc',mirror=True)
    
    config.add_relation('hpr_strut_mid',CR.Centered,'hpr_strut_chassis','hpr_strut_rocker',mirror=True)
    
    # JOINTS CONFIGURATIONS
    
    # Spherical Joints:
    config.add_relation('pt1_jcr_uca_upright',CR.Equal_to,'hpr_ucao',mirror=True)
    config.add_relation('pt1_jcr_lca_upright',CR.Equal_to,'hpr_lcao',mirror=True)
    config.add_relation('pt1_jcr_tie_upright',CR.Equal_to,'hpr_tro',mirror=True)
    config.add_relation('pt1_jcr_prod_rocker',CR.Equal_to,'hpr_pushrod_rocker',mirror=True)
    
    # Rocker-Chassis Revolute Joint:
    config.add_relation('ax1_jcr_rocker_chassis',CR.Oriented,'hpr_rocker_chassis','hpr_pushrod_rocker','hpr_strut_rocker',mirror=True)
    config.add_relation('pt1_jcr_rocker_chassis',CR.Equal_to,'hpr_rocker_chassis',mirror=True)
    
    # PushRod-UCA Universal Joint:
    config.add_relation('pt1_jcr_prod_uca',CR.Equal_to,'hpr_pushrod_uca',mirror=True)
    config.add_relation('ax1_jcr_prod_uca',CR.Oriented,'hpr_pushrod_uca','hpr_pushrod_rocker',mirror=True)
    config.add_relation('ax2_jcr_prod_uca',CR.Oriented,'hpr_pushrod_rocker','hpr_pushrod_uca',mirror=True)

    # Upper Control Arm Revolute Joint:
    config.add_relation('pt1_jcr_uca_chassis',CR.Centered,'hpr_ucaf','hpr_ucar',mirror=True)
    config.add_relation('ax1_jcr_uca_chassis',CR.Oriented,'hpr_ucaf','hpr_ucar',mirror=True)
    
    # Lower Control Arm Revolute Joint:
    config.add_relation('pt1_jcr_lca_chassis',CR.Centered,'hpr_lcaf','hpr_lcar',mirror=True)
    config.add_relation('ax1_jcr_lca_chassis',CR.Oriented,'hpr_lcaf','hpr_lcar',mirror=True)
    
    # Wheel Hub Revolute Joint:
    config.add_relation('pt1_jcr_hub_bearing',CR.Equal_to,'hpr_wc',mirror=True)
    
    # Strut-Chassis Universal Joint:
    config.add_relation('pt1_jcr_strut_chassis',CR.Equal_to,'hpr_strut_chassis',mirror=True)
    config.add_relation('ax1_jcr_strut_chassis',CR.Oriented,'hpr_strut_chassis','hpr_strut_mid',mirror=True)
    config.add_relation('ax2_jcr_strut_chassis',CR.Oriented,'hpr_strut_mid','hpr_strut_chassis',mirror=True)
    
    # Strut-Rocker Universal Joint:
    config.add_relation('pt1_jcr_strut_rocker',CR.Equal_to,'hpr_strut_rocker',mirror=True)
    config.add_relation('ax1_jcr_strut_rocker',CR.Oriented,'hpr_strut_chassis','hpr_strut_rocker',mirror=True)
    config.add_relation('ax2_jcr_strut_rocker',CR.Oriented,'hpr_strut_rocker','hpr_strut_chassis',mirror=True)
    
    # Tie-Steer Universal Joint:
    config.add_relation('pt1_jcr_tie_steering',CR.Equal_to,'hpr_tri',mirror=True)
    config.add_relation('ax1_jcr_tie_steering',CR.Oriented,'hpr_tri','hpr_tro',mirror=True)
    config.add_relation('ax2_jcr_tie_steering',CR.Oriented,'hpr_tro','hpr_tri',mirror=True)
    
    # Strut Cylinderical Joint:
    config.add_relation('pt1_jcr_strut',CR.Equal_to,'hpr_strut_mid',mirror=True)
    config.add_relation('ax1_jcr_strut',CR.Oriented,'hpr_strut_rocker','hpr_strut_chassis',mirror=True)
    
    # GEOMETRIES
    config.add_scalar('links_ro')
    config.add_scalar('strut_outer')
    config.add_scalar('strut_inner')
    config.add_scalar('thickness')
    config.add_scalar('tire_radius')
    
    config.add_geometry('uca',mirror=True)
    config.add_relation('gmr_uca',triangular_prism,'hpr_ucaf','hpr_ucar','hpr_ucao','s_thickness',mirror=True)
    config.assign_geometry_to_body('rbr_uca','gmr_uca',mirror=True)
    
    config.add_geometry('lca',mirror=True)
    config.add_relation('gmr_lca',triangular_prism,'hpr_lcaf','hpr_lcar','hpr_lcao','s_thickness',mirror=True)
    config.assign_geometry_to_body('rbr_lca','gmr_lca',mirror=True)
    
    config.add_geometry('rocker',mirror=True)
    config.add_relation('gmr_rocker',triangular_prism,'hpr_strut_rocker','hpr_pushrod_rocker','hpr_rocker_chassis','s_thickness',mirror=True)
    config.assign_geometry_to_body('rbr_rocker','gmr_rocker',mirror=True)

    config.add_geometry('upright',mirror=True)
    config.add_relation('gmr_upright',triangular_prism,'hpr_ucao','hpr_wc','hpr_lcao','s_thickness',mirror=True)
    config.assign_geometry_to_body('rbr_upright','gmr_upright',mirror=True)

    config.add_geometry('upper_strut',mirror=True)
    config.add_relation('gmr_upper_strut',cylinder_geometry,'hpr_strut_chassis','hpr_strut_mid','s_strut_outer',mirror=True)
    config.assign_geometry_to_body('rbr_upper_strut','gmr_upper_strut',mirror=True)

    config.add_geometry('lower_strut',mirror=True)
    config.add_relation('gmr_lower_strut',cylinder_geometry,'hpr_strut_mid','hpr_strut_rocker','s_strut_inner',mirror=True)
    config.assign_geometry_to_body('rbr_lower_strut','gmr_lower_strut',mirror=True)
    
    config.add_geometry('tie_rod',mirror=True)
    config.add_relation('gmr_tie_rod',cylinder_geometry,'hpr_tri','hpr_tro','s_links_ro',mirror=True)
    config.assign_geometry_to_body('rbr_tie_rod','gmr_tie_rod',mirror=True)
    
    config.add_geometry('pushrod',mirror=True)
    config.add_relation('gmr_pushrod',cylinder_geometry,'hpr_pushrod_uca','hpr_pushrod_rocker','s_links_ro',mirror=True)
    config.assign_geometry_to_body('rbr_pushrod','gmr_pushrod',mirror=True)
    
    config.add_geometry('tire',mirror=True)
    config.add_relation('gmr_tire',cylinder_geometry,'hpr_wc','R_rbr_upright','s_tire_radius',mirror=True)
    config.assign_geometry_to_body('rbr_hub','gmr_tire',mirror=True)
        
    config.topology.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()

    from source.post_processors.blender.scripter import scripter
    geo_code = scripter(config)
    geo_code.write_code_file()


if __name__ == '__main__':
    main()

