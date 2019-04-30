#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:29:25 2019

@author: khaledghobashy
"""
# Standard library imports
import os

# Local applicataion imports
import asurt.utilities.blender.codegen as bpygen
from asurt.numenv.python.interfaces.simenv import codegen as pygen


class topology_generators(object):
    
    def write_python_code(mbs, proj_dir):
        pygen.write_template_code(mbs, proj_dir)


class configuration_generators(object):
    
    def write_python_code(cfg, proj_dir):
        pygen.write_configuration_code(cfg, proj_dir)
        

class assembly_generators(object):
    
    def write_python_code(mbs, proj_dir):
        pygen.write_assembly_code(mbs, proj_dir)
        
class visuals_generator(object):
    
    def blender(cfg, proj_dir):
        relative_path = 'visenv.blender.gen_scripts'.split('.')
        dir_path = os.path.join(proj_dir, *relative_path)
        blender_code = bpygen.script_generator(cfg)
        blender_code.write_code_file(dir_path)

    
