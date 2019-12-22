# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 08:16:56 2019

@author: khaled.ghobashy
"""

import os


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
                       
    