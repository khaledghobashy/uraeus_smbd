#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:28:16 2019

@author: khaledghobashy
"""
# Standard library imports
import os
import nbformat as nbf

# Local applicataion imports
from ..codegen.generators import (assembly_codegen, 
                                  template_codegen,
                                  configuration_codegen)
###############################################################################
"""
/numenv
    /python
        /models
            /templates
                template_1.py
                template_2.py
                ...
            /configurations
                config_1.py
                config_2.py
                ...
            /assemblies
                assembly_1.py
                assembly_2.py
                ...
        /simulations
            /sim_1
                /results
                /configuration_files
                sim_1.ipynb
            /...
"""
###############################################################################

def _nbtext_parser(text, formats):
    text = text.format(**formats)
    nb_cells = []
    celltype = None
    text_block = ''
    for line in text.splitlines():
        if line == '$':
            continue
        if line.startswith('$markdown'):
            if celltype is not None:
                nb_cells.append(celltype('\n'.join(text_block)))
            celltype = nbf.v4.new_markdown_cell
            text_block = []
            continue
        elif line.startswith('$code'):
            if celltype is not None:
                nb_cells.append(celltype('\n'.join(text_block)))
            celltype = nbf.v4.new_code_cell
            text_block = []
            continue
        else:
            text_block.append(line)
    
    nb_cells.append(celltype('\n'.join(text_block)))
    return nb_cells

###############################################################################

class codegen(object):
    
    def write_standalone_code(mbs, proj_dir):
        relative_path = 'numenv.python.models.standalones'.split('.')
        dir_path = os.path.join(proj_dir, *relative_path)
        numerical_code = template_codegen(mbs)
        numerical_code.write_code_file(dir_path)
    
    def write_template_code(mbs, proj_dir):
        relative_path = 'numenv.python.models.templates'.split('.')
        dir_path = os.path.join(proj_dir, *relative_path)
        numerical_code = template_codegen(mbs)
        numerical_code.write_code_file(dir_path)
    
    def write_configuration_code(cfg, proj_dir):
        relative_path = 'numenv.python.models.configurations'.split('.')
        dir_path = os.path.join(proj_dir, *relative_path)
        numerical_code = configuration_codegen(cfg)
        numerical_code.write_code_file(dir_path)
        
    def write_assembly_code(mbs, proj_dir):
        relative_path = 'numenv.python.models.assemblies'.split('.')
        dir_path = os.path.join(proj_dir, *relative_path)
        numerical_code = assembly_codegen(mbs)
        numerical_code.write_code_file(dir_path)


###############################################################################

_simulation_nbtext = """\
$markdown
# SIMULATION STUDY
## **{system_title}**
-----------------
$markdown
### STUDY DISCRIPTION
---------------------
$markdown
_Double click to write a discription here ..._
$markdown
-----------------------------------------------
$code
import sys
pkg_path = ''
sys.path.append(pkg_path)
$
$code
project_dir = '{project_dir}'
$code
import numpy as np
from smbd.numenv.python.interfaces.scripting import multibody_system, simulation
$
$markdown
## ASSEMBLY IMPORT & CONFIGURATION ASSIGNMENT
---------------------------------------------
$
$code
#import smbd_projects.numenv.python.models.assemblies as models
#import smbd_projects.numenv.python.models.standalones as models
#import smbd_projects.numenv.python.models.configurations as configs
$code
#model = multibody_system(models.'model_name')
$code
#model.Subsystems.'SUB1'.set_configuration_file(configs.'configuration1')
#model.Subsystems.'SUB2'.set_configuration_file(configs.'configuration2')

$code

$markdown
## SETTING MODEL ACTUATION FUNCTIONS
------------------------------------
$
$code
#model.Subsystems.'SUB1'.config.'ACTUATOR' = lambda t : 25.4*np.sin(t)
$
$markdown
## MODEL CONFIGURATIONS
-----------------------
$markdown
### CONFIGURATION VARIANT #1
$markdown
#### SETTING CONFIGURATION DATA
$
$code
#model.Subsystems.'SUB1'.set_configuration_data('configuration_1.csv')
$
$markdown
#### CREATING A SIMULATION INSTANCE
$
$code
#sim1 = simulation('v1', model, 'kds')
#sim1.set_time_array(2*np.pi, 100)
#sim1.solve()
$
$markdown
#### RESULTS' PLOTS
$code
#sim1.plot([('SUB.body.x', 'pos'), ('SUB.body.x', 'vel')])
#sim1.plot([('SUB.body.y', 'pos'), ('SUB.body.y', 'vel')])
$
"""

class simulation_project(object):
    
    def __init__(self, name):
        self.name = name
        relative_path  = 'simulations'.split('.')
        self.directory = os.path.join(*relative_path, self.name)
    
    def create_project(self):
        self._create_directories()
        self._write_notebook_text()
        print('Project %r created at %r'%(self.name, self.directory))
    
    def _create_directories(self):
        directory = self.directory
        try:
            os.mkdir(directory)
            os.mkdir(os.path.join(directory, 'configuration_files'))
            os.mkdir(os.path.join(directory, 'results'))
        except FileExistsError:
            raise FileExistsError('Project Already Exists!')
        init_file = os.path.join(directory, '__init__.py')
        with open(init_file, 'w') as file:
            file.write('#')
    
    def _write_notebook_text(self):
        project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        formats = {'system_name': self.name,
                   'system_title': self.name.upper(),
                   'project_dir': project_dir}
        cells = _nbtext_parser(_simulation_nbtext, formats)
        
        nb = nbf.v4.new_notebook()
        nb['cells'] = cells
        
        notebook_path = os.path.join(self.directory, 'SIM_%s'%self.name)
        nbf.write(nb, '%s.ipynb'%notebook_path)

