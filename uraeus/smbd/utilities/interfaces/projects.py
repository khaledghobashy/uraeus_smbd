# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 08:16:56 2019

@author: khaled.ghobashy
"""

# Standard library imports
import os

###############################################################################

class standalone_project(object):
    
    def __init__(self, parent_dir=''):
        
        self.parent_dir = parent_dir
        self.subdirs = ['numenv', 'model_data']
        self.code_dir = os.path.join(self.parent_dir, 'numenv')
        
    def create(self):
        self._create_common_dirs()
    
    def _create_common_dirs(self):
        for d in self.subdirs:
            subdir = os.path.join(self.parent_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        if self.parent_dir !='':
            os.chdir(self.parent_dir)
            cwdir = os.path.abspath(self.parent_dir)
            print('Current working directory is %s'%cwdir)

###############################################################################

class templatebased_project(object):
    
    def __init__(self, parent_dir=''):
        
        self.parent_dir = parent_dir
        self.code_dir = os.path.join(self.parent_dir, 'numenv')
        self.symbolic_dir = os.path.join(self.parent_dir, 'symenv')
        
    def create(self):
        self._create_common_dirs()
        self._create_symbolic_dirs()
    
    def _create_common_dirs(self):
        for d in ['config_inputs', 'results', 'numenv', 'symenv']:
            subdir = os.path.join(self.parent_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        if self.parent_dir !='':
            os.chdir(self.parent_dir)
            print('Current working directory is %s'%os.path.abspath(self.parent_dir))
    
    def _create_symbolic_dirs(self):
        for d in ['templates/scripts', 'templates/objects', 'assemblies/scripts']:
            subdir = os.path.join(self.symbolic_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

###############################################################################

class smbd_database(object):
    
    def __init__(self, name, parent_dir=''):
        
        self.name = name
        self.path = os.path.join(os.path.abspath(parent_dir), name)
        self.parent_dir = parent_dir
        
        self._dirs = ['symenv.templates.config_inputs',
                      'symenv.templates.objects',
                      'symenv.assemblies',
                      'numenv.visenv',
                      'simulations']
        
        
    def create(self):
        for d in self._dirs:
            subdir = os.path.join(self.path, *d.split('.'))
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        
        self._write_init_file(self.path)

    
    def _write_init_file(self, parent_dir):
        file_path = os.path.join(parent_dir, '__init__.py')
        file_name = file_path
        with open(file_name, 'w') as file:
            file.write('#')
                       
    