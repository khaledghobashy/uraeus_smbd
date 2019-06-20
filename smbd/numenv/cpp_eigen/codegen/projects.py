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
    
    def __init__(self, sym_model, sym_config=None):
        
        self.sym_model = sym_model
        self.sym_config = sym_config
        
        self.name = sym_model.name
    
    
    def generate_project(self, parent_dir='', dir_name='', overwrite=False):
        dir_name = self.name if dir_name =='' else dir_name
        self.parent_dir = os.path.join(parent_dir, dir_name)
        if overwrite:
            if os.path.exists(self.parent_dir):
                shutil.rmtree(self.parent_dir)
        self._create_dirs()
        self._write_makefile()
        self._write_mainfile()
        self._generate_files()
    
    
    def _create_dirs(self):
        for d in ['build', 'src', 'results']:
            os.makedirs(os.path.join(self.parent_dir, d))
    
    
    def _write_makefile(self):
        text = '''
                MODEL := {model_name}

                BUILD := build/
                SRC := src/
                
                SMBD_SRC := {cpp_src}/src/
                SMBD_BUILD := {cpp_src}/build/
                SMBD_FILES := $(SMBD_BUILD)*.o
                
                DEPS := $(BUILD)$(MODEL).o $(SRC)main.cpp $(SRC){model_name}_cfg.hpp $(SMBD_FILES)
                
                INC := -I {cpp_src}/src
                CC := g++
                
                $(MODEL): $(DEPS) $(SMBD_SRC)smbd/solvers.hpp
                	$(CC) $(INC) $(DEPS) -o $(MODEL)
                
                $(BUILD)$(MODEL).o: $(SRC)$(MODEL).cpp $(SRC)$(MODEL).hpp 
                	$(CC) $(INC) -c -o $@ $<
                
                clear:
                	rm $(BUILD)*.o $(MODEL)
        
        '''
        cpp_src = os.path.dirname(__file__)
        cpp_src = os.path.join(cpp_src, '../numerics')
        
        #text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(model_name = self.name,
                           cpp_src = cpp_src)
        
        file_path = os.path.join(self.parent_dir, 'Makefile')
        with open('%s'%file_path, 'w') as file:
            file.write(text)
        print('File full path : %s'%file_path)
        
    
    def _write_mainfile(self):
        text = '''
                #include <iostream>
                
                #include "smbd/solvers.hpp"
                
                #include "{model_name}.hpp"
                #include "{model_name}_cfg.hpp"
                
                
                int main()
                {{
                    Topology model("");
                    auto Config = ConfigurationInputs<Configuration>(model.config);
                    
                    // assign the configuration inputs needed ...
                    
                    Config.assemble();
                    model.config.set_inital_configuration();
                    
                    Solver<Topology> Soln(model);
                    Soln.set_time_array(1, 100);
                    Soln.Solve();
                    Soln.ExportResultsCSV(0);
                
                }};
        '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(model_name = self.name)
        
        file_path = os.path.join(self.parent_dir, 'src', 'main')
        with open('%s.cpp'%file_path, 'w') as file:
            file.write(text)
        print('File full path : %s.cpp'%file_path)
        
        
    def _generate_files(self):
        src_path = os.path.join(self.parent_dir, 'src')
        
        mbs_code = generators.template_codegen(self.sym_model)
        mbs_code.write_header_file(src_path)
        mbs_code.write_source_file(src_path)

        if self.sym_config:
            cfg_code = generators.configuration_codegen(self.sym_config)
            cfg_code.write_source_file(src_path)
        else:
            print('Configuration file was not generated as No configuration Found!')
        

