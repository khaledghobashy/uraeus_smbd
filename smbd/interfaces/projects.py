#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:32:58 2019

@author: khaledghobashy
"""

# Standard library imports
import os
import nbformat as nbf

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

_template_nbtext = """\
$markdown
# SYMBOLIC STUDY
## **{system_title}**
----------------
$markdown
### STUDY DISCRIPTION
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
import smbd.interfaces.scripting as sui
$
$
$markdown
## SYSTEM TOPOLOGY
-----------------
$
$code
model = sui.standalone_topology('{system_name}')
$
$markdown
### ADDING SYSTEM BODIES
$code
#model.add_body("body_name", mirror=False, virtual=False)
$
$markdown
### ADDING SYSTEM JOINTS
$code
#model.joint.spherical('joint_name', 'body_1', 'body_2', mirror=False, virtual=False)
$
$markdown
### ADDING SYSTEM ACTUATORS
$code
#model.actuator.rotational_actuator('actuator_name', 'joint_name', mirror=False)
$
$markdown
### ADDING SYSTEM FORCES
$code
#model.add_force.internal_force('force_name', 'body_1', 'body_2', mirror=False)
$
$markdown
### ASSEMBLING AND SAVING SYSTEM
$code
model.assemble_model()
model.save()
$code
model.write_python_code(project_dir)
$
$
$
$markdown
## SYSTEM CONFIGURATION
-----------------------
$code
config_name = '{system_name}_cfg'
config = sui.configuration(config_name, model)
$
$markdown
### CONFIGURATION INPUTS
$code
#config.add_point.UserInput('point_name', mirror=False)
$markdown
### CONFIGURATION RELATIONS
$code
#config.add_relation.Equal_to('node_1', ('node_2',), mirror=False)
$markdown
### CONFIGURATION GEOMETRIES
$code
#config.add_geometry.Cylinder_Geometry('name', ('p1','p2','radius'), mirror=False)
#config.assign_geometry_to_body('body_name', 'geom_name', mirror=False)
$
$markdown
### ASSEMBLING AND SAVING CONFIGURATION
$code
config.assemble_model()
config.extract_inputs_to_csv()
$code
config.write_python_code(project_dir)
$code
config.write_blender_script(project_dir)
$code

"""

class topology_project(object):
    
    def __init__(self, template_name, typ='s'):
        self.name = template_name
        assert typ in 'st', '%r not a valid input. %r or %r is expected'%(typ,'r','s')
        relative_path  = ['templates'] if typ == 't' else ['standalones']
        self.directory = os.path.join(*relative_path, template_name)
    
    def create_project(self):
        self._create_directories()
        self._write_notebook_text()
        print('Project %r created at %r'%(self.name, self.directory))
    
    def _create_directories(self):
        directory = self.directory
        try:
            os.mkdir(directory)
            os.mkdir(os.path.join(directory, 'csv_files'))
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
        cells = _nbtext_parser(_template_nbtext, formats)
        
        nb = nbf.v4.new_notebook()
        nb['cells'] = cells
        
        notebook_path = os.path.join(self.directory, self.name)
        nbf.write(nb, '%s.ipynb'%notebook_path)
        

###############################################################################
###############################################################################

_assm_nbtext = """\
$markdown
# SYMBOLIC STUDY
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
pkg_path = None
sys.path.append(pkg_path)
$
$code
project_dir = '{project_dir}'
$code
import smbd.interfaces.scripting as sui
$
$markdown
## IMPORTING TOPLOGIES
----------------------
$
$code
# template_1 = sui.load_template(project_dir, 'template_name')
$
$markdown
## CREATING ASSEMBLY
--------------------
$
$code
# model = sui.assembly('{system_name}')
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

class assembly_project(object):
    
    def __init__(self, name):
        self.name = name
        relative_path  = 'assemblies'.split('.')
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
        project_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        formats = {'system_name': self.name,
                   'system_title': self.name.upper(),
                   'project_dir': project_dir}
        cells = _nbtext_parser(_assm_nbtext, formats)
        
        nb = nbf.v4.new_notebook()
        nb['cells'] = cells
        
        notebook_path = os.path.join(self.directory, self.name)
        nbf.write(nb, '%s.ipynb'%notebook_path)

###############################################################################
###############################################################################

