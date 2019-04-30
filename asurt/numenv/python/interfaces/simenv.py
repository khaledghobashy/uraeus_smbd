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
# {system_title}
----------------
$markdown
## SYSTEM DISCRIPTION
---------------------
$markdown
_Double click to write a discription here ..._
$markdown
-----------------------------------------------
$code
import sys
pkg_path = None
sys.path.append(pkg_path)
$
$code
project_dir = '{project_dir}'
$code
import asurt.interfaces.scripting as sui
$
$markdown
## IMPORTING TOPLOGIES
----------------------
$
$code
# template_1 = sui.load_stpl_file(project_dir, 'template_name')
$
$markdown
## CREATING ASSEMBLY
--------------------
$
$code
# model = sui.assembly({system_name})
$
$markdown
### CREATING SUBSYSTEMS
$
$code
#model.add_subsystem('SUBSYSTEM_INITIALS', template)
$
$markdown
### ASSIGNING VIRTUAL BODIES
$
$code
#model.assign_virtual_body('SU1.vbr_body', 'SU2.rbr_body')
$
$markdown
### ASSEMBLING AND SAVING STSYEM
$
$code
model.assemble_model()
$code
model.draw_constraints_topology()
$code
model.write_python_code(project_dir)
"""

class simulation_project(object):
    
    def __init__(self, name):
        self.name = name
        relative_path  = 'symbolic_models.assemblies'.split('.')
        self.directory = os.path.join(*relative_path, self.name)
    
    def create_project(self):
        self._create_directories()
        self._write_notebook_text()
    
    def _create_directories(self):
        directory = self.directory
        try:
            os.mkdir(directory)
        except FileExistsError:
            raise FileExistsError('Project Already Exists!')
        init_file = os.path.join(directory, '__init__.py')
        with open(init_file, 'w') as file:
            file.write('#')
    
    def _write_notebook_text(self):
        
        formats = {'system_name': self.name,
                   'system_title': self.name.upper(),
                   'project_dir': os.getcwd()}
        cells = _nbtext_parser(_simulation_nbtext, formats)
        
        nb = nbf.v4.new_notebook()
        nb['cells'] = cells
        
        notebook_path = os.path.join(self.directory, self.name)
        nbf.write(nb, '%s.ipynb'%notebook_path)

