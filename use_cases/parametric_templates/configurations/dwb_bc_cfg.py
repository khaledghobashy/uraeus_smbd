# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:56:01 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import Config_Relations as CR
from source.code_generators.python_code_generators import configuration_code_generator

import use_cases.parametric_templates.templates.double_wishbone_bellcranck as model

def main():
    
    name = 'dwb_bc_points'
    config = model.template.param_config
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
    config.add_point('rocker_chassis',mirror=True)
    config.add_point('pushrod_rocker',mirror=True)
    config.add_point('pushrod_uca',mirror=True)
    config.add_point('strut_chassis',mirror=True)
    config.add_point('strut_rocker',mirror=True)
    config.add_point('strut_mid',mirror=True)
    config.add_point('wc',mirror=True)
    
    config.add_relation(CR.Centered,'hpr_strut_mid',['hpr_strut_chassis','hpr_strut_rocker'],True)
    
    config.add_relation(CR.Centered,'R_rbr_uca',['hpr_ucao','hpr_ucaf','hpr_ucar'],True)
    config.add_relation(CR.Centered,'R_rbr_lca',['hpr_lcao','hpr_lcaf','hpr_lcar'],True)
    config.add_relation(CR.Centered,'R_rbr_upright',['hpr_ucao','hpr_lcao','hpr_wc'],True)
    config.add_relation(CR.Centered,'R_rbr_upper_strut',['hpr_strut_chassis','hpr_strut_mid'],True)
    config.add_relation(CR.Centered,'R_rbr_lower_strut',['hpr_strut_rocker','hpr_strut_mid'],True)
    config.add_relation(CR.Centered,'R_rbr_tie_rod',['hpr_tro','hpr_tri'],True)
    config.add_relation(CR.Centered,'R_rbr_hub',['hpr_wc','R_rbr_upright'],True)
    config.add_relation(CR.Centered,'R_rbr_pushrod',['hpr_pushrod_uca','hpr_pushrod_rocker'],True)
    config.add_relation(CR.Centered,'R_rbr_rocker',['hpr_pushrod_rocker','hpr_strut_rocker','hpr_rocker_chassis'],True)

    
    config.add_relation(CR.Equal_to,'pt1_jcr_uca_upright',['hpr_ucao'],True)
    config.add_relation(CR.Equal_to,'pt1_jcr_lca_upright',['hpr_lcao'],True)
    config.add_relation(CR.Equal_to,'pt1_jcr_tie_upright',['hpr_tro'],True)
    config.add_relation(CR.Equal_to,'pt1_jcr_prod_rocker',['hpr_pushrod_rocker'],True)
    
    config.add_relation(CR.Oriented,'ax1_jcr_rocker_chassis',['hpr_rocker_chassis','hpr_pushrod_rocker','hpr_strut_rocker'],True)
    config.add_relation(CR.Equal_to,'pt1_jcr_rocker_chassis',['hpr_rocker_chassis'],True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_prod_uca',['hpr_pushrod_uca'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_prod_uca',['hpr_pushrod_uca','hpr_pushrod_rocker'],True)
    config.add_relation(CR.Oriented,'ax2_jcr_prod_uca',['hpr_pushrod_rocker','hpr_pushrod_uca'],True)

    
    config.add_relation(CR.Centered,'pt1_jcr_uca_chassis',['hpr_ucaf','hpr_ucar'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_uca_chassis',['hpr_ucaf','hpr_ucar'],True)
    
    config.add_relation(CR.Centered,'pt1_jcr_lca_chassis',['hpr_lcaf','hpr_lcar'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_lca_chassis',['hpr_lcaf','hpr_lcar'],True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_hub_bearing',['hpr_wc'],True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_strut_chassis',['hpr_strut_chassis'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_strut_chassis',['hpr_strut_chassis','hpr_strut_rocker'],True)
    config.add_relation(CR.Oriented,'ax2_jcr_strut_chassis',['hpr_strut_rocker','hpr_strut_chassis'],True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_strut_rocker',['hpr_strut_rocker'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_strut_rocker',['hpr_strut_chassis','hpr_strut_rocker'],True)
    config.add_relation(CR.Oriented,'ax2_jcr_strut_rocker',['hpr_strut_rocker','hpr_strut_chassis'],True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_tie_steering',['hpr_tri'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_tie_steering',['hpr_tri','hpr_tro'],True)
    config.add_relation(CR.Oriented,'ax2_jcr_tie_steering',['hpr_tro','hpr_tri'],True)
    
    config.add_relation(CR.Equal_to,'pt1_jcr_strut',['hpr_strut_mid'],True)
    config.add_relation(CR.Oriented,'ax1_jcr_strut',['hpr_strut_rocker','hpr_strut_chassis'],True)
    
    config.topology.save()
    
    config_code = configuration_code_generator(config)
    config_code.write_code_file()


if __name__ == '__main__':
    main()

