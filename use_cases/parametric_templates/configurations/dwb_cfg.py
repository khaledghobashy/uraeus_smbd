# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:56:01 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.double_wishbone_direct_acting as model
from source.mbs_creators.topology_helpers import (parametric_configuration,
                                                  composite_geometry,
                                                  triangular_prism, 
                                                  cylinder_geometry)


def main():
    
    name = 'dwb_simple_points'
    config = parametric_configuration(model.template)
    config.assemble_base_layer()
    config.name = name
    model.template.cfg_file = name
    
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
    
    config.add_relation('hpr_strut_mid',CR.Centered,'hpr_strut_chassis','hpr_strut_lca',mirror=True)
    
    config.add_relation('R_rbr_uca',CR.Centered,'hpr_ucao','hpr_ucaf','hpr_ucar',mirror=True)
    config.add_relation('R_rbr_lca',CR.Centered,'hpr_lcao','hpr_lcaf','hpr_lcar',mirror=True)
    config.add_relation('R_rbr_upright',CR.Centered,'hpr_ucao','hpr_lcao','hpr_wc',mirror=True)
    config.add_relation('R_rbr_upper_strut',CR.Centered,'hpr_strut_chassis','hpr_strut_mid',mirror=True)
    config.add_relation('R_rbr_lower_strut',CR.Centered,'hpr_strut_lca','hpr_strut_mid',mirror=True)
    config.add_relation('R_rbr_tie_rod',CR.Centered,'hpr_tro','hpr_tri',mirror=True)
    config.add_relation('R_rbr_hub',CR.Centered,'hpr_wc','R_rbr_upright',mirror=True)
    
    config.add_relation('pt1_jcr_uca_upright',CR.Equal_to,'hpr_ucao',mirror=True)
    config.add_relation('pt1_jcr_lca_upright',CR.Equal_to,'hpr_lcao',mirror=True)
    config.add_relation('pt1_jcr_tie_upright',CR.Equal_to,'hpr_tro',mirror=True)
    
    config.add_relation('pt1_jcr_uca_chassis',CR.Centered,'hpr_ucaf','hpr_ucar',mirror=True)
    config.add_relation('ax1_jcr_uca_chassis',CR.Oriented,'hpr_ucaf','hpr_ucar',mirror=True)
    
    config.add_relation('pt1_jcr_lca_chassis',CR.Centered,'hpr_lcaf','hpr_lcar',mirror=True)
    config.add_relation('ax1_jcr_lca_chassis',CR.Oriented,'hpr_lcaf','hpr_lcar',mirror=True)
    
    config.add_relation('pt1_jcr_hub_bearing',CR.Equal_to,'hpr_wc',mirror=True)
    
    config.add_relation('pt1_jcr_strut_chassis',CR.Equal_to,'hpr_strut_chassis',mirror=True)
    config.add_relation('ax1_jcr_strut_chassis',CR.Oriented,'hpr_strut_chassis','hpr_strut_lca',mirror=True)
    config.add_relation('ax2_jcr_strut_chassis',CR.Oriented,'hpr_strut_lca','hpr_strut_chassis',mirror=True)
    
    config.add_relation('pt1_jcr_strut_lca',CR.Equal_to,'hpr_strut_lca',mirror=True)
    config.add_relation('ax1_jcr_strut_lca',CR.Oriented,'hpr_strut_chassis','hpr_strut_lca',mirror=True)
    config.add_relation('ax2_jcr_strut_lca',CR.Oriented,'hpr_strut_lca','hpr_strut_chassis',mirror=True)
    
    config.add_relation('pt1_jcr_tie_steering',CR.Equal_to,'hpr_tri',mirror=True)
    config.add_relation('ax1_jcr_tie_steering',CR.Oriented,'hpr_tri','hpr_tro',mirror=True)
    config.add_relation('ax2_jcr_tie_steering',CR.Oriented,'hpr_tro','hpr_tri',mirror=True)
    
    config.add_relation('pt1_jcr_strut',CR.Equal_to,'hpr_strut_mid',mirror=True)
    config.add_relation('ax1_jcr_strut',CR.Oriented,'hpr_strut_lca','hpr_strut_chassis',mirror=True)
    
#    config.topology.save()
#    
#    config_code = configuration_code_generator(config)
#    config_code.write_code_file()
    
    # Adding Geometries
    config.add_scalar('links_ro')
    config.add_scalar('strut_outer')
    config.add_scalar('strut_inner')
    config.add_scalar('thickness')
    config.add_scalar('tire_radius')
    
    config.add_geometry('uca',mirror=True)
    config.add_relation('gmr_uca',triangular_prism,'hpr_ucaf','hpr_ucar','hpr_ucao','s_thickness')
    config.assign_geometry_to_body('rbr_uca','gmr_uca')
    
    config.add_geometry('lca',mirror=True)
    config.add_relation('gmr_lca',triangular_prism,'hpr_lcaf','hpr_lcar','hpr_lcao','s_thickness')
    config.assign_geometry_to_body('rbr_lca','gmr_lca')
    
    config.add_geometry('upright',mirror=True)
    config.add_relation('gmr_upright',triangular_prism,'hpr_ucao','hpr_wc','hpr_lcao','s_thickness')
    config.assign_geometry_to_body('rbr_upright','gmr_upright')

    config.add_geometry('upper_strut',mirror=True)
    config.add_relation('gmr_upper_strut',cylinder_geometry,'hpr_strut_chassis','hpr_strut_mid','s_strut_outer')
    config.assign_geometry_to_body('rbr_upper_strut','gmr_upper_strut')

    config.add_geometry('lower_strut',mirror=True)
    config.add_relation('gmr_lower_strut',cylinder_geometry,'hpr_strut_mid','hpr_strut_lca','s_strut_inner')
    config.assign_geometry_to_body('rbr_lower_strut','gmr_lower_strut')
    
    config.add_geometry('tie_rod',mirror=True)
    config.add_relation('gmr_tie_rod',cylinder_geometry,'hpr_tri','hpr_tro','s_links_ro')
    config.assign_geometry_to_body('rbr_tie_rod','gmr_tie_rod')
    
    config.add_geometry('tire',mirror=True)
    config.add_relation('gmr_tire',cylinder_geometry,'hpr_wc','R_rbr_upright','s_tire_radius')
    config.assign_geometry_to_body('rbr_hub','gmr_tire')
        
    
    from source.post_processors.blender.scripter import scripter
    geo_code = scripter(config)
    geo_code.write_code_file()



if __name__ == '__main__':
    main()

