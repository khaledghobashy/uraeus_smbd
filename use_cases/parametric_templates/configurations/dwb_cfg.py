# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:56:01 2019

@author: khaled.ghobashy
"""

from source.code_generators.python_code_generators import configuration_code_generator
from source.mbs_creators.topology_helpers import parametric_configuration

import use_cases.parametric_templates.templates.double_wishbone_direct_acting as model


def main():
    
    global config
    
    name = 'dwb_da_cfg'
    config = parametric_configuration(name,model.template)
    config.assemble_base_layer()
    
    # Adding Points
    config.add_point('ucaf',mirror=True)
    config.add_point('ucar',mirror=True)
    config.add_point('ucao',mirror=True)
    config.add_point('lcaf',mirror=True)
    config.add_point('lcar',mirror=True)
    config.add_point('lcao',mirror=True)
    config.add_point('tro',mirror=True)
    config.add_point('tri',mirror=True)
    config.add_point('strut_chassis',mirror=True)
    config.add_point('strut_lca',mirror=True)
    config.add_point('strut_mid',mirror=True)
    config.add_point('wc',mirror=True)
    
    config.add_relation.Centered('hpr_strut_mid', args=('hpr_strut_chassis','hpr_strut_lca'), mirror=True)

    # JOINTS CONFIGURATIONS
    
    # Spherical Joints:
    config.add_relation.Equal_to('pt1_jcr_uca_upright', args=('hpr_ucao',), mirror=True)
    config.add_relation.Equal_to('pt1_jcr_lca_upright', args=('hpr_lcao',), mirror=True)
    config.add_relation.Equal_to('pt1_jcr_tie_upright', args=('hpr_tro',), mirror=True)
    
    # Upper Control Arm Revolute Joint:
    config.add_relation.Centered('pt1_jcr_uca_chassis', args=('hpr_ucaf','hpr_ucar'), mirror=True)
    config.add_relation.Oriented('ax1_jcr_uca_chassis', args=('hpr_ucaf','hpr_ucar'), mirror=True)
    
    # Lower Control Arm Revolute Joint:
    config.add_relation.Centered('pt1_jcr_lca_chassis', args=('hpr_lcaf','hpr_lcar'), mirror=True)
    config.add_relation.Oriented('ax1_jcr_lca_chassis', args=('hpr_lcaf','hpr_lcar'), mirror=True)
    
    # Wheel Hub Revolute Joint:
    config.add_relation.Equal_to('pt1_jcr_hub_bearing', args=('hpr_wc',), mirror=True)
    
    # Strut-Chassis Universal Joint:
    config.add_relation.Equal_to('pt1_jcr_strut_chassis', args=('hpr_strut_chassis',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_strut_chassis', args=('hpr_strut_chassis','hpr_strut_lca'), mirror=True)
    config.add_relation.Oriented('ax2_jcr_strut_chassis', args=('hpr_strut_lca','hpr_strut_chassis'), mirror=True)
    
    # Strut-LCA Universal Joint:
    config.add_relation.Equal_to('pt1_jcr_strut_lca', args=('hpr_strut_lca',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_strut_lca', args=('hpr_strut_chassis','hpr_strut_lca'), mirror=True)
    config.add_relation.Oriented('ax2_jcr_strut_lca', args=('hpr_strut_lca','hpr_strut_chassis'), mirror=True)
    
    # Tie-Steer Universal Joint:
    config.add_relation.Equal_to('pt1_jcr_tie_steering', args=('hpr_tri',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_tie_steering', args=('hpr_tri','hpr_tro'), mirror=True)
    config.add_relation.Oriented('ax2_jcr_tie_steering', args=('hpr_tro','hpr_tri'), mirror=True)
    
    # Strut Cylinderical Joint:
    config.add_relation.Equal_to('pt1_jcr_strut', args=('hpr_strut_mid',), mirror=True)
    config.add_relation.Oriented('ax1_jcr_strut', args=('hpr_strut_lca','hpr_strut_chassis'), mirror=True)
    
    
    # GEOMETRIES
#    config.add_scalar('links_ro')
#    config.add_scalar('strut_outer')
#    config.add_scalar('strut_inner')
#    config.add_scalar('thickness')
#    config.add_scalar('tire_radius')
#    
#    config.add_geometry('uca',mirror=True)
#    config.add_relation('gmr_uca',triangular_prism,'hpr_ucaf','hpr_ucar','hpr_ucao','s_thickness',mirror=True)
#    config.assign_geometry_to_body('rbr_uca','gmr_uca',mirror=True)
#    
#    config.add_geometry('lca',mirror=True)
#    config.add_relation('gmr_lca',triangular_prism,'hpr_lcaf','hpr_lcar','hpr_lcao','s_thickness',mirror=True)
#    config.assign_geometry_to_body('rbr_lca','gmr_lca',mirror=True)
#    
#    config.add_geometry('upright',mirror=True)
#    config.add_relation('gmr_upright',triangular_prism,'hpr_ucao','hpr_wc','hpr_lcao','s_thickness',mirror=True)
#    config.assign_geometry_to_body('rbr_upright','gmr_upright',mirror=True)
#
#    config.add_geometry('upper_strut',mirror=True)
#    config.add_relation('gmr_upper_strut',cylinder_geometry,'hpr_strut_chassis','hpr_strut_mid','s_strut_outer',mirror=True)
#    config.assign_geometry_to_body('rbr_upper_strut','gmr_upper_strut',mirror=True)
#
#    config.add_geometry('lower_strut',mirror=True)
#    config.add_relation('gmr_lower_strut',cylinder_geometry,'hpr_strut_mid','hpr_strut_lca','s_strut_inner',mirror=True)
#    config.assign_geometry_to_body('rbr_lower_strut','gmr_lower_strut',mirror=True)
#    
#    config.add_geometry('tie_rod',mirror=True)
#    config.add_relation('gmr_tie_rod',cylinder_geometry,'hpr_tri','hpr_tro','s_links_ro',mirror=True)
#    config.assign_geometry_to_body('rbr_tie_rod','gmr_tie_rod',mirror=True)
#    
#    config.add_geometry('tire',mirror=True)
#    config.add_relation('gmr_tire',cylinder_geometry,'hpr_wc','R_rbr_upright','s_tire_radius',mirror=True)
#    config.assign_geometry_to_body('rbr_hub','gmr_tire',mirror=True)
#        
#    
#    config_code = configuration_code_generator(config)
#    config_code.write_code_file()
#
#    from source.post_processors.blender.scripter import scripter
#    geo_code = scripter(config)
#    geo_code.write_code_file()
#


if __name__ == '__main__':
    main()

