#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:36:49 2019

@author: khaledghobashy
"""

import os
import shutil
import textwrap

from . import generators


class project_generator(object):
    
    def __init__(self, sym_model, sym_config):
        
        self.sym_model = sym_model
        self.sym_config = sym_config
        
        self.name = sym_model.name
    
    
    def generate_project(self, parent_dir='', dir_name='', overwrite=False):
        dir_name = self.name if dir_name =='' else dir_name
        self.parent_dir = os.path.join(parent_dir,'numenv', 'python', dir_name)
        if overwrite:
            if os.path.exists(self.parent_dir):
                shutil.rmtree(self.parent_dir)
        
        self._create_dirs()
        self._write_mainfile()
        self._generate_files()
    
    
    def _create_dirs(self):
        for d in ['src', 'results']:
            os.makedirs(os.path.join(self.parent_dir, d))
            
    
    def _write_mainfile(self):
        text = '''
                import numpy as np
                import pandas as pd
                
                try:
                    from smbd.numenv.python.interfaces.scripting import multibody_system, simulation
                except ModuleNotFoundError:
                    import sys
                    sys.path.append('{pkg_path}')
                    from smbd.numenv.python.interfaces.scripting import multibody_system, simulation

                import {model_name}, {model_name}_cfg
                
                
                num_model = multibody_system({model_name})
                num_model.topology.config = {model_name}_cfg.configuration()
                
                inputs_df = pd.read_csv('path/to/config.csv', index_col=0)
                # input the configuration data here ...
                inputs_df.loc['P_ground'] = [1, 0, 0, 0]
                
                
                # Saving the configuration as a .csv file.
                inputs_df.to_csv('path/to/new.csv')
                
                num_model.topology.config.load_from_dataframe(inputs_df)
                
                # Setting actuation data
                #num_model.topology.config.UF_mcs_act_1 = lambda t :  np.deg2rad(360)*t
           
                sim = simulation('sim1', num_model, 'kds')
                sim.set_time_array(1, 100)
                sim.solve()
            
        '''
        pkg_path = os.path.dirname(__file__)
        pkg_path = os.path.abspath(os.path.join(pkg_path, '../../../../'))
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(model_name = self.name,
                           pkg_path = pkg_path)
        
        
        file_path = os.path.join(self.parent_dir, 'src', 'main')
        with open('%s.py'%file_path, 'w') as file:
            file.write(text)
        print('File full path : %s.py'%file_path)
        
        
    def _generate_files(self):
        src_path = os.path.join(self.parent_dir, 'src')
        
        mbs_code = generators.template_codegen(self.sym_model)
        mbs_code.write_code_file(src_path)

        cfg_code = generators.configuration_codegen(self.sym_config)
        cfg_code.write_code_file(src_path)        

