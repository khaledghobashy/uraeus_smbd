# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 20:45:05 2019

@author: khale
"""
import numpy as np

from source import pkg_path
import use_cases.generated_templates.assemblies.front_axle_assembly as f

from use_cases.generated_templates.configurations import dwb_cfg
from use_cases.generated_templates.configurations import steer_cfg
from use_cases.generated_templates.configurations import front_axle_testrig_bcfg

f.SU.config = dwb_cfg.configuration()
f.ST.config = steer_cfg.configuration()
f.TR.config = front_axle_testrig_bcfg.configuration()

f.SU.config.load_from_csv(pkg_path + r'\use_cases\generated_templates\configurations\csv_files\dwb_st500_axletech.csv')
f.ST.config.load_from_csv(pkg_path + r'\use_cases\generated_templates\configurations\csv_files\steer_st500_axletech.csv')
f.TR.config.load_from_csv(pkg_path + r'\use_cases\generated_templates\configurations\csv_files\front_axle_testrig_bcfg_1.csv')

f.TR.config.AF_jcs_steer_gear = lambda t : np.deg2rad(15)*np.sin(t)
f.TR.config.AF_mcr_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_mcl_ver_act = lambda t : 170*np.sin(t)
f.TR.config.AF_jcr_rev = lambda t :  np.deg2rad(360)*t
f.TR.config.AF_jcl_rev = lambda t : -np.deg2rad(360)*t

assm = f.numerical_assembly()
