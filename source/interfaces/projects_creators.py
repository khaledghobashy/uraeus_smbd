#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:32:58 2019

@author: khaledghobashy
"""

import nbformat as nbf
import os

class template_based_project(object):
    
    _database_directory = None
    
    def __init__(self, template_name):
        
        self.name = template_name
        relative_path  = 'symbolic_models.templates'.split('.')
        self.directory = os.path.join(*relative_path, template_name)
    
    
    @property
    @classmethod
    def database_directory(cls):
        return cls._database_directory
    @database_directory.setter
    def database_directory(cls, directory):
        if not os.path.isdir(directory):
            raise ValueError('Not a valid directory!')
        cls._database_directory = directory
    
    
    def create_project(self):
        if self.database_directory is None:
            raise ValueError('Database directory was not properly set!')
        self._create_directories()
        self._write_notebook_text()
    
    
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
        
        separator = 8*'-'
        
        system_title = ('# %s'%self.name).upper()
        discription_title = '\n'.join(['## SYSTEM DISCRIPTION', separator])
        discription_body  = '_Double click to write a discription here ..._'
        
        system_path = '\n'.join(['import sys',
                                 'pkg_path = ""',
                                 'sys.path.append(pkg_path)'])
        
        topology_section = '\n'.join(['## SYSTEM TOPOLOGY', separator])
        
        topology_imports = 'from source.interfaces.scripting_interfaces import topology'
        template_code = 'template = topology(%r)'%self.name
        
        bodies_mdc  = '### ADDING SYSTEM BODIES'
        bodies_code = '#template.add_body("body_name", mirror=False, virtual=False)'

        joints_mdc  = '### ADDING SYSTEM JOINTS'
        joints_code = '#template.joint.spherical("joint_name", "body_1", "body_2", mirror=False, virtual=False)'

        actuators_mdc  = '### ADDING SYSTEM ACTUATORS'
        actuators_code = "#template.actuator.rotational_actuator('actuator_name', 'joint_name', mirror=False)"

        forces_mdc  = '### ADDING SYSTEM FORCES'
        forces_code = "#template.add_force.internal_force('force_name', 'body_1', 'body_2', mirror=False)"

        save_mdc  = '### ASSEMBLING AND SAVING SYSTEM'
        save_code = '\n'.join(['template.assemble_model()',
                               'template.save()',
                               'template.write_python_code()'])
        
        
        config_section = '\n'.join(['## SYSTEM CONFIGURATION', separator])
        
        config_imports = "from source.interfaces.scripting_interfaces import configuration"
        config_code = '\n'.join(["config_name = ''",
                                 "config = configuration(config_name, template)"])
        
        points_mdc  = '### CONFIGURATION INPUTS'
        points_code = "#config.add_point.UserInput('point_name', mirror=False)"
        
        relations_mdc  = "### CONFIGURATION RELATIONS"
        relations_code = "#config.add_relation.Equal_to('node_1', ('node_2',), mirror=False)"
        
        geometries_mdc  = "### CONFIGURATION GEOMETRIES"
        geometries_code = '\n'.join(["#config.add_geometry.Cylinder_Geometry('name', ('p1','p2','radius'), mirror=False)",
                                     "#config.assign_geometry_to_body('body_name', 'geom_name', mirror=False)"])
        
        config_save_mdc  = '### ASSEMBLING AND SAVING CONFIGURATION'
        config_save_code = '\n'.join(['config.write_python_code()',
                                      'config.write_blender_script()'])
        
        

        
        nb = nbf.v4.new_notebook()
        nb['cells'] = [nbf.v4.new_markdown_cell(system_title),
                       nbf.v4.new_markdown_cell(discription_title),
                       nbf.v4.new_markdown_cell(discription_body),
                       nbf.v4.new_markdown_cell(separator),
                       
                       nbf.v4.new_code_cell(system_path),
                       nbf.v4.new_markdown_cell(topology_section),
                       nbf.v4.new_code_cell(topology_imports),
                       nbf.v4.new_code_cell(template_code),
                       
                       nbf.v4.new_markdown_cell(bodies_mdc),
                       nbf.v4.new_code_cell(bodies_code),
                       
                       nbf.v4.new_markdown_cell(joints_mdc),
                       nbf.v4.new_code_cell(joints_code),
                       
                       nbf.v4.new_markdown_cell(actuators_mdc),
                       nbf.v4.new_code_cell(actuators_code),
                       
                       nbf.v4.new_markdown_cell(forces_mdc),
                       nbf.v4.new_code_cell(forces_code),
                       
                       nbf.v4.new_markdown_cell(save_mdc),
                       nbf.v4.new_code_cell(save_code),
                       nbf.v4.new_code_cell(''),
                       
                       nbf.v4.new_markdown_cell(separator),
                      
                       nbf.v4.new_markdown_cell(config_section),
                       nbf.v4.new_code_cell(config_imports),
                       nbf.v4.new_code_cell(config_code),
                      
                       nbf.v4.new_markdown_cell(points_mdc),
                       nbf.v4.new_code_cell(points_code),
                      
                       nbf.v4.new_markdown_cell(relations_mdc),
                       nbf.v4.new_code_cell(relations_code),
                       
                       nbf.v4.new_markdown_cell(geometries_mdc),
                       nbf.v4.new_code_cell(geometries_code),
                      
                       nbf.v4.new_markdown_cell(config_save_mdc),
                       nbf.v4.new_code_cell(config_save_code),
                      
                       nbf.v4.new_code_cell(''),]
        
        notebook_path = os.path.join(self.directory, self.name)
        nbf.write(nb, '%s.ipynb'%notebook_path)

