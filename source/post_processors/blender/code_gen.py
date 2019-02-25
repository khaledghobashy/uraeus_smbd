# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:17:15 2019

@author: khale
"""
import os
import re
import textwrap
from source.code_generators.code_printers import numerical_printer

class scripter(object):
        
    def __init__(self,config,printer=numerical_printer()):
        
        self.config  = config
        self.printer = printer
        
        self.name = self.config.name
        self.graph = self.config.graph.subgraph(config.geometry_nodes).copy()
                
        self.config_vars = [printer._print(i) for i in self.config.arguments_symbols]
        self.input_args  = self.config.input_equalities
        self.output_args = self.config.output_equalities
        
        self.gen_coordinates_sym = [printer._print(exp.lhs) 
        for exp in self.config.topology.mapped_gen_coordinates]
        self.gen_velocities_sym  = [printer._print(exp.lhs) 
        for exp in self.config.topology.mapped_gen_velocities]
    
    def extract_geometries_graph(self):
        pass
    
    def get_inputs(self):
        g = self.graph
        nodes = [i for i,d in g.in_degree() if d == 0]
        equalities = [g.nodes[i]['func'] for i,d in g.in_degree(nodes) if d==0]
        return equalities
    
    def mid_equalities(self):
        mid_layer = []
        for n in self.output_nodes:
            self._get_node_dependencies(n,mid_layer)
        return [self.graph.nodes[n]['func'] for n in mid_layer]
        
    def write_imports(self):
        text = '''
                import os
                import numpy as np
                from source.post_processors.blender.objects import (cylinder_geometry,
                                                                    composite_geometry,
                                                                    triangular_prism)
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
        
    def write_class_init(self):
        text = '''
                path = os.path.dirname(__file__)
                
                class blender_scene(object):

                    def __init__(self):
                        {inputs}                       
                '''
        p = self.printer
        indent = 8*' '
        inputs  = self.input_args
        
        pattern = '|'.join(self.config_vars)
        self_inserter = self._insert_string('self.')
        
        inputs = '\n'.join([p._print(exp) for exp in inputs])
        inputs = re.sub(pattern,self_inserter,inputs)
        inputs = textwrap.indent(inputs,indent).lstrip()
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        text = text.format(inputs = inputs)
        return text
    
    def write_helpers(self):
        text = '''
                        
                def load_from_csv(self,csv_file):
                    file_path = os.path.join(path,csv_file)
                    dataframe = pd.read_csv(file_path,index_col=0)
                    for ind in dataframe.index:
                        shape = getattr(self,ind).shape
                        v = np.array(dataframe.loc[ind],dtype=np.float64)
                        v = np.resize(v,shape)
                        setattr(self,ind,v)
                    self._set_arguments()
                
                def _set_arguments(self):
                    {outputs}
                    {pass_text}
                '''
        p = self.printer
        indent = 4*' '
        
        outputs = self.output_args
        
        pattern = '|'.join(self.config_vars)
        self_inserter = self._insert_string('self.')
                
        outputs = '\n'.join([p._print(exp) for exp in outputs])
        outputs = re.sub(pattern,self_inserter,outputs)
        outputs = textwrap.indent(outputs,indent).lstrip()
        
        pass_text = ('pass' if len(outputs)==0 else '')
        
        coordinates = ','.join(self.gen_coordinates_sym)
        coordinates = re.sub(pattern,self_inserter,coordinates)
        coordinates = ('np.concatenate([%s])'%coordinates if len(coordinates)!=0 else '[]')
        coordinates = textwrap.indent(coordinates,indent).lstrip()
        
        velocities = ','.join(self.gen_velocities_sym)
        velocities = re.sub(pattern,self_inserter,velocities)
        velocities = ('np.concatenate([%s])'%velocities if len(velocities)!=0 else '[]')
        velocities = textwrap.indent(velocities,indent).lstrip()
        
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(outputs = outputs,
                           pass_text = pass_text,
                           coordinates = coordinates,
                           velocities = velocities)
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
        
    def write_code_file(self):
        os.chdir('..\..')
        path = os.getcwd() + '\\generated_templates\\blender_scripts'
        os.chdir(path)
        
        imports = self.write_imports()
        config_class = self.write_system_class()
        text = '\n'.join([imports,config_class])
        with open('%s.py'%self.name,'w') as file:
            file.write(text)
        
        inputs_dataframe = self._create_inputs_dataframe()
        inputs_dataframe.to_csv('%s.csv'%self.name)

    
###############################################################################
###############################################################################

