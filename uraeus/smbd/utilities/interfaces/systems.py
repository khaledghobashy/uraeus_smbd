# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
# Standard library imports
import os

# 3rd party library imports
import cloudpickle
import sympy as sm

# Local applicataion imports
from . import _decorated_containers as containers
from ..serialization.structural.json.configuration_encoder import generator
from ...symbolic.systems import topology_classes as topology_classes
from ...symbolic.systems import configuration_classes as cfg_cls

###############################################################################

def get_file_name(script_path):
    name = os.path.basename(script_path).split('.')[0]
    return name

def load_pickled_data(file):
    with open(file, 'rb') as f:
        instance = cloudpickle.load(f)
    return instance

###############################################################################
###############################################################################

class template_topology(object):
    
    def __init__(self, name):
        self.name = name
        self.topology = topology_classes.template_based_topology(self.name)
        
        self._joints = containers.joints_container(self.topology)
        self._actuators = containers.actuators_container(self.topology)
        self._forces = containers.forces_container(self.topology)
    
    
    def add_body(self, *args, **kwargs):
        self.topology.add_body(*args, **kwargs)
    
    @property
    def add_joint(self):
        return self._joints
    
    @property
    def add_actuator(self):
        return self._actuators
    
    @property
    def add_force(self):
        return self._forces
    
    def assemble(self):
        self.topology.assemble_model()
            
    def save(self, dir_path=''):
        file = os.path.join(dir_path, '%s.stpl'%self.name)
        with open(file, 'wb') as f:
            cloudpickle.dump(self, f)
    
###############################################################################
###############################################################################

class standalone_topology(template_topology):
    
    def __init__(self, name):
        self.name = name
        self.topology = topology_classes.standalone_topology(self.name)
        
        self._joints = containers.joints_container(self.topology)
        self._actuators = containers.actuators_container(self.topology)
        self._forces = containers.forces_container(self.topology)
        

###############################################################################
###############################################################################

class assembly(object):
    
    def __init__(self, name):
        self.name = get_file_name(name)
        self.topology = topology_classes.assembly(self.name)
        
    def add_assembly(self, assm):
        try :
            assm = assm.topology
        except AttributeError:
            pass
        self.topology.add_assembly(assm)
        
    def add_subsystem(self, subsystem_name, template_instance):
        try :
            template_instance = template_instance.topology
        except AttributeError:
            pass
        subsystem = topology_classes.subsystem(subsystem_name, template_instance)
        self.topology.add_subsystem(subsystem)
        
    def assign_virtual_body(self, virtual_node, actual_node):
        self.topology.assign_virtual_body(virtual_node, actual_node)
    
    def assemble(self):
        self.topology.assemble_model()
        
    def save(self):
        file = '%s.sasm'%self.name
        with open(file, 'wb') as f:
            cloudpickle.dump(self, f)
        
    def draw_constraints_topology(self):
        self.topology.draw_constraints_topology()
    
    def draw_interface_graph(self):
        self.topology.draw_interface_graph()
        
###############################################################################
###############################################################################

class configuration(object):
    def __init__(self, name, model_instance):
        self.name = get_file_name(name)
        self.config = cfg_cls.abstract_configuration(self.name, model_instance.topology)
        self._decorate_methods()

    @property
    def add_point(self):
        """
        Add a spatial point.

        Availabe Methods:
            'UserInput', 'Mirrored', 'Centered', 'Equal_to'
        """
        return self._points_constructors
    
    @property
    def add_vector(self):
        return self._vectors_constructors
    
    @property
    def add_scalar(self):
        return self._scalars_constructors
    
    @property
    def add_geometry(self):
        return self._geometries_constructors
    
    @property
    def add_relation(self):
        return self._relations_methods
    
    def assign_geometry_to_body(self, body, geo, eval_inertia=True, mirror=False):
        self.config.assign_geometry_to_body(body, geo, eval_inertia, mirror)
    

    def extract_inputs_to_csv(self, path):
        file_path = os.path.join(path, self.name)
        inputs_dataframe = self.config.create_inputs_dataframe()
        inputs_dataframe.to_csv('%s.csv'%file_path)
    
    def export_JSON_file(self, path=''):
        config_constructor = generator(self.config)
        config_constructor.write_JSON_file(path)
    
    def save(self):
        file = '%s.scfg'%self.name
        with open(file, 'wb') as f:
            cloudpickle.dump(self, f)
    
    def _decorate_methods(self):
        self._scalars_constructors = containers.scalar_nodes(self.config)
        self._vectors_constructors = containers.vector_nodes(self.config)
        self._points_constructors  = containers.points_nodes(self.config)
        self._geometries_constructors = containers.geometries_nodes(self.config)
        self._relations_methods = containers.relations_methods(self.config)

    
###############################################################################
###############################################################################
