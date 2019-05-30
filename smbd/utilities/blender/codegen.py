# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:17:15 2019

@author: khale
"""
import os
import re
import textwrap
import itertools

from .printer import printer

class script_generator(object):
        
    def __init__(self, config, _printer=printer()):
        
        self.config  = config
        self.name = self.config.name
        self.printer = _printer

        data = self.config.get_geometries_graph_data()
        self.input_args   = data['input_nodes']
        self.output_args  = data['output_nodes']
        self.input_equalities  = data['input_equal']
        self.output_equalities = data['output_equal']
        self.geometries_map = data['geometries_map']
        
        equalities = itertools.chain(self.input_equalities,self.output_equalities)
        self.args_str = [self.printer._print(expr.lhs) for expr in equalities]
                
    def write_imports(self):
        text = '''
                import numpy as np
                
                from smbd.utilities.blender.numcls import bpy_scene
                from smbd.utilities.numerics.spatial_alg import centered, mirrored
                from smbd.utilities.blender.objects import (cylinder_geometry,
                                                             composite_geometry,
                                                             triangular_prism,
                                                             sphere_geometry)
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
        
    def write_class_init(self):
        text = '''
                class blender_scene(bpy_scene):

                    def __init__(self, prefix='', scale=1/20):
                        self.prefix = prefix
                        self.scale = scale
                        
                        {inputs}
                        
                        self._inputs = {inputs_attr}
                        self.geometries = {outputs}
                '''
        p = self.printer
        indent = 8*' '
        inputs  = self.input_equalities
        
        pattern = '|'.join([p._print(arg) for arg in self.input_args])
        self_inserter = self._insert_string('self.')
        
        inputs = '\n'.join(['%s*scale'%p._print(exp) for exp in inputs])
        inputs = re.sub(pattern,self_inserter,inputs)
        inputs = textwrap.indent(inputs,indent).lstrip()
                
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        text = text.format(inputs = inputs,
                           inputs_attr = self.input_args,
                           outputs = self.geometries_map)
        return text
    
    def write_helpers(self):
        text = '''
                def create_scene(self):
                    {outputs}
                    
                    self.setup_VIEW_3D()
                '''
        p = self.printer
        indent = 4*' '
        
        outputs = self.output_equalities
        
        pattern_items = self.args_str
        pattern = '|'.join([p._print(arg) for arg in pattern_items])
        self_inserter = self._insert_string('self.')
                
        outputs = '\n'.join([p._print(exp) for exp in outputs])
        outputs = re.sub(pattern,self_inserter,outputs)
        outputs = textwrap.indent(outputs,indent).lstrip()
                
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(outputs = outputs)
        text = textwrap.indent(text,indent)
        return text
    
    
    def write_system_class(self):
        text = '''
                {class_init}
                    {class_helpers}
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        class_init = self.write_class_init()
        class_helpers = self.write_helpers()
                
        text = text.format(class_init = class_init,
                           class_helpers = class_helpers)
        return text
        
    def write_code_file(self, dir_path=''):
        file_path = os.path.join(dir_path, self.name)
        imports = self.write_imports()
        config_class = self.write_system_class()
        text = ''.join([imports,config_class])
        with open('%s_bpy.py'%file_path, 'w') as file:
            file.write(text)
        

    @staticmethod
    def _insert_string(string):
        def inserter(x): return string + x.group(0).strip("'")
        return inserter
###############################################################################
###############################################################################

