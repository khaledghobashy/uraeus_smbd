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
        
        data = config.get_geometries_graph_data()
        self.input_args  = data['input_nodes']
        self.input_equalities  = data['input_equal']
        self.output_equalities = data['output_equal']
        self.input_equalities  = data['input_equal']
                
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
        inputs  = self.input_equalities
        
        pattern = '|'.join([p._print(arg) for arg in self.input_args])
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
        
        outputs = self.output_equalities
        
        pattern = '|'.join([p._print(arg) for arg in self.input_args])
        self_inserter = self._insert_string('self.')
                
        outputs = '\n'.join([p._print(exp) for exp in outputs])
        outputs = re.sub(pattern,self_inserter,outputs)
        outputs = textwrap.indent(outputs,indent).lstrip()
        
        pass_text = ('pass' if len(outputs)==0 else '')
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(outputs = outputs,
                           pass_text = pass_text)
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
        
#        inputs_dataframe = self._create_inputs_dataframe()
#        inputs_dataframe.to_csv('%s.csv'%self.name)

    @staticmethod
    def _insert_string(string):
        def inserter(x): return string + x.group(0).strip("'")
        return inserter
###############################################################################
###############################################################################

