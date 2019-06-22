#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:36:49 2019

@author: khaledghobashy
"""

# Standard library imports
import os
import shutil
import textwrap

# Local applicataion imports
from smbd import pkg_path

# Local directories imports
from . import generators


class standalone_project(object):
    
    def __init__(self, parent_dir=''):
        
        self.parent_dir = parent_dir
        self.code_dir = os.path.join(self.parent_dir, 'numenv', 'cpp_eigen')
        
    def _create_subdirs(self):
        for d in ['build', 'src', 'results']:
            subdir = os.path.join(self.code_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        
    
    def create_dirs(self, overwrite=False):
        if os.path.exists(self.code_dir):
            if overwrite:
                shutil.rmtree(self.code_dir)
                self._create_subdirs()
            else:
                self._create_subdirs()
        
    def write_topology_code(self, topology):
        src_path = os.path.join(self.code_dir, 'src')
        codegen = generators.template_codegen(topology)
        codegen.write_header_file(src_path)
        codegen.write_source_file(src_path)
            
    
    def write_configuration_code(self, config):
        src_path = os.path.join(self.code_dir, 'src')
        codegen = generators.configuration_codegen(config)
        codegen.write_source_file(src_path)
        
    
    def write_mainfile(self, filename='main'):
        text = '''
                #include <iostream>
                
                #include "smbd/solvers.hpp"
                
                #include "src/topology.hpp"
                #include "src/configuration.hpp"
                
                
                int main()
                {{
                    Topology model("");
                    auto Config = ConfigurationInputs<Configuration>(model.config);
                    
                    // assign the configuration inputs needed ...
                    
                    Config.assemble();
                    
                    Solver<Topology> Soln(model);
                    Soln.set_time_array(1, 100);
                    Soln.Solve();
                    Soln.ExportResultsCSV("results/", 0);
                
                }};
        '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        file_path = os.path.join(self.code_dir, filename)
        full_name = '%s.cpp'%file_path
        with open(full_name, 'w') as f:
            f.write(text)
        print('File full path : %s'%full_name)

        
    
    def write_makefile(self):
        text = '''
                # Change MODEL, CONFG and MAIN to match the source files you want to build
                MODEL := topology
                CONFG := configuration
                MAIN := main.cpp

                BUILD := build/
                SRC := src/
                
                SMBD_SRC := {cpp_src}/src/
                SMBD_BUILD := {cpp_src}/build/
                SMBD_FILES := $(SMBD_BUILD)*.o
                
                DEPS := $(BUILD)$(MODEL).o $(MAIN) $(SRC)$(CONFG).hpp $(SMBD_FILES)
                
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
        cpp_src = os.path.abspath(os.path.join(pkg_path, 'numenv', 'cpp_eigen', 'numerics'))
        
        text = textwrap.dedent(text)
        text = text.format(cpp_src = cpp_src)
        
        file_path = os.path.join(self.code_dir, 'Makefile')
        full_name = '%s'%file_path
        with open(full_name, 'w') as f:
            f.write(text)
        print('File full path : %s'%full_name)

    