# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:17:15 2019

@author: khale
"""
import os
import re
import textwrap
import itertools
from source.code_generators.code_printers import numerical_printer

class scripter(object):
        
    def __init__(self,config,printer=numerical_printer()):
        
        self.config  = config
        self.printer = printer
        self.name = self.config.name
        
        data = config.get_geometries_graph_data()
        self.input_args   = data['input_nodes']
        self.output_args  = data['output_nodes']
        self.input_equalities  = data['input_equal']
        self.output_equalities = data['output_equal']
        self.geometries_map = data['geometries_map']
        
        equalities = itertools.chain(self.input_equalities,self.output_equalities)
        self.args_str = [printer._print(expr.lhs) for expr in equalities]
                
    def write_imports(self):
        text = '''
                import csv
                import numpy as np
                import bpy
                from source.solvers.py_numerical_functions import centered, mirrored
                from source.post_processors.blender.objects import (cylinder_geometry,
                                                                    composite_geometry,
                                                                    triangular_prism)
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
        
    def write_class_init(self):
        text = '''
                try:
                    bpy.context.scene.objects.active = bpy.data.objects['Cube']
                    bpy.ops.object.delete()
                except KeyError:
                    pass

                class blender_scene(object):

                    def __init__(self,prefix=''):
                        self.prefix = prefix
                        scale = 1/20
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
        
        inputs = '*scale\n'.join([p._print(exp) for exp in inputs])
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
                def get_data(self,csv_file):
                    with open(csv_file, newline='') as csvfile:
                        content = csv.reader(csvfile)
                        next(content)
                        for row in content:
                            attr = row[0]
                            if attr in self._inputs:
                                value = np.array(row[1:],dtype=np.float64)
                                value = np.resize(value,(3,1))*self.scale
                                setattr(self,attr,value)
                    
                def load_anim_data(self,csv_file):
                    with open(csv_file, newline='') as csvfile:
                        content = csv.reader(csvfile)
                        keys = {{k:i for i,k in enumerate(next(content)[1:])}}
                        arr = np.array(list(content))[:,1:]
                        arr = np.array(arr,dtype=np.float64)
                    
                    scale = self.scale
                    for i,row in enumerate(arr):
                        for g,b in self.geometries.items():
                            k = keys['%s%s.x'%(self.prefix,b)]
                            obj = getattr(self,g).obj
                            obj.location = [float(n)*scale for n in row[k:k+3]]
                            obj.rotation_quaternion = [float(n) for n in row[k+3:k+7]]
                            obj.keyframe_insert('location', frame=i)
                            obj.keyframe_insert('rotation_quaternion', frame=i)
                    
                    bpy.context.scene.render.frame_map_old = i+1
                    bpy.context.scene.render.frame_map_new = 24*2
                    bpy.context.scene.frame_end = bpy.context.scene.render.frame_map_new
                        
                def create_scene(self):
                    {outputs}
                    
                    self.setup_VIEW_3D()
                    
                @staticmethod
                def setup_VIEW_3D():
                    for area in bpy.context.screen.areas:
                        if area.type == 'VIEW_3D':
                            for region in area.regions:
                                if region.type == 'WINDOW':
                                    override = {{'area': area, 'region': region, 'edit_object': bpy.context.edit_object}}
                                    bpy.ops.view3d.view_all(override)
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
    
    def write_scene_creator(self):
        text = '''
                def create_scene(prefix=''):
                    conf_data = bpy.data.scenes["Scene"].cfg_path
                    anim_data = bpy.data.scenes["Scene"].sim_path
                    blend = blender_scene(prefix)
                    blend.get_data(conf_data)
                    blend.create_scene()
                    blend.load_anim_data(anim_data)
               '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
                        
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
        
    def write_code_file(self):
        os.chdir('..\..')
        path = os.getcwd() + r'\generated_templates\blender\gen_scripts'
        os.chdir(path)
        
        imports = self.write_imports()
        config_class = self.write_system_class()
        scene_creator = self.write_scene_creator()
        text = '\n'.join([imports,config_class,scene_creator])
        with open('%s.py'%self.name,'w') as file:
            file.write(text)
        

    @staticmethod
    def _insert_string(string):
        def inserter(x): return string + x.group(0).strip("'")
        return inserter
###############################################################################
###############################################################################

