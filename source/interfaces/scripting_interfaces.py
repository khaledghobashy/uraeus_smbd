# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
import os
import pickle
import cloudpickle
import sympy as sm

import source.mbs_creators.topology_classes as topology_classes
import source.symbolic_classes.joints as joints
import source.symbolic_classes.forces as forces
from source.code_generators.python_code_generators import (assembly_code_generator,
                                                           template_code_generator,
                                                           configuration_code_generator)
from source.post_processors.blender.scripter import scripter

from source.symbolic_classes.abstract_matrices import vector
from source.mbs_creators.configuration_classes import (abstract_configuration, 
                                                       Geometries,
                                                       CR, Geometry)


def get_file_name(script_path):
    name = os.path.basename(script_path).split('.')[0]
    return name

import types
e = types.SimpleNamespace(ro=1)

###############################################################################
###############################################################################    
class topology_edges_container(object):
    
    def __init__(self, mbs):
        self._mbs = mbs
        self._decorate_items()
        
    @property
    def _items(self):
        members = {i:getattr(self,i) for i in dir(self) if not i.startswith('_') and not i.startswith("__")}
        return members
    
    def _decorate_items(self):
        for attr,obj in self._items.items():
            setattr(self, attr, self._decorate(obj))

class joints_container(topology_edges_container):
    def __init__(self, mbs):
        self.spherical = joints.spherical
        self.revolute  = joints.revolute
        self.universal = joints.universal
        self.translational = joints.translational
        self.cylinderical  = joints.cylinderical
        self.tripod = joints.tripod
        self.fixed  = joints.fixed
        super().__init__(mbs)
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
                self._mbs.add_joint(edge_component, *args, **kwargs)
        return decorated

    
class actuators_container(topology_edges_container):
    def __init__(self, mbs):
        self.rotational_actuator = joints.rotational_actuator
        self.absolute_locator = joints.absolute_locator
        super().__init__(mbs)
    
    def _decorate(self, edge_component):
        if issubclass(edge_component, joints.absolute_locator):
            def decorated(*args, **kwargs):
                self._mbs.add_absolute_actuator(edge_component, *args, **kwargs)
        elif issubclass(edge_component, joints.rotational_actuator):
            def decorated(*args, **kwargs):
                self._mbs.add_joint_actuator(edge_component, *args, **kwargs)
        return decorated


class forces_container(topology_edges_container):
    def __init__(self, mbs):
        self.internal_force = forces.internal_force
        super().__init__(mbs)
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
                self._mbs.add_force(edge_component, *args, **kwargs)
        return decorated
    
###############################################################################    
class topology(object):
    
    def __init__(self, script_path):
        self._script_path = script_path
        self._name = get_file_name(script_path)
        self._mbs = topology_classes.template_based_topology(self._name)
        
        self._joints = joints_container(self._mbs)
        self._actuators = actuators_container(self._mbs)
        self._forces = forces_container(self._mbs)
        
    def add_body(self, *args, **kwargs):
        self._mbs.add_body(*args, **kwargs)
    
    @property
    def add_joint(self):
        return self._joints
    
    @property
    def add_actuator(self):
        return self._actuators
    
    @property
    def add_force(self):
        return self._forces
    
    def assemble_model(self):
        self._mbs.assemble_model()
        
    def write_python_code(self):
        numerical_code = template_code_generator(self._mbs)
        numerical_code.write_code_file()
        self._python_code_gen = numerical_code
    
    def save(self):
        file = '%s.stpl'%self._name
        with open(file,'wb') as f:
            cloudpickle.dump(self, f)
    
    @staticmethod
    def reload(file_path):
        file = '%s.stpl'%file_path
        with open(file, 'rb') as f:
            template = pickle.load(f)
        return template
    

###############################################################################
###############################################################################

class assembly(object):
    
    def __init__(self, script_path):
        self.script_path = script_path
        self.name = get_file_name(script_path)
        self._mbs = topology_classes.assembly(self.name)
        
    def add_subsystem(self, subsystem_name, template_instance):
        try :
            template_instance = template_instance._mbs
        except AttributeError:
            pass
        subsystem = topology_classes.subsystem(subsystem_name, template_instance)
        self._mbs.add_subsystem(subsystem)
        
    def assign_virtual_body(self, virtual_node, actual_node):
        self._mbs.assign_virtual_body(virtual_node, actual_node)
    
    def assemble_model(self):
        self._mbs.assemble_model()
    
    def write_python_code(self):
        code = assembly_code_generator(self._mbs)
        code.write_code_file()
    
    def draw_constraints_topology(self):
        self._mbs.draw_constraints_topology()
    
    def draw_interface_graph(self):
        self._mbs.draw_interface_graph()
        
###############################################################################
###############################################################################

class configuration(object):
    
    def __init__(self, script_path, model_instance):
        self._script_path = script_path
        self._name = get_file_name(script_path)
        
        self._config = abstract_configuration(self._name, model_instance._mbs)
        self._config.assemble_base_layer()
        self._decorate_methods()
    
    @property
    def add_point(self):
        """
        Add a spatial point.
        Availabe Methodes:
            'UserInput', 'Mirrored', 'Centered', 'Equal_to'
        """
        return self._point_methods
    
    @property
    def add_vector(self):
        return self._vector_methods
    
    @property
    def add_scalar(self):
        return self._scalar_methods
    
    @property
    def add_geometry(self):
        return self._geometry_methods
    
    @property
    def add_relation(self):
        return self._relation_methods
    
    def assign_geometry_to_body(self, body, geo, eval_inertia=True, mirror=False):
        self._config.assign_geometry_to_body(body, geo, eval_inertia, mirror)

            
    def write_python_code(self):
        self._config.assemble_equalities()
        numerical_code = configuration_code_generator(self)
        numerical_code.write_code_file()
        
    def write_blender_script(self):
        blender_code_gen = scripter(self)
        blender_code_gen.write_code_file()
        
    def extract_inputs_to_csv(self):
        file_path = os.path.join('csv_files', self._name)
        inputs_dataframe = self._config.create_inputs_dataframe()
        inputs_dataframe.to_csv('%s.csv'%file_path)
        
    
    def _decorate_methods(self):
        self._decorate_point_methods()
        self._decorate_vector_methods()
        self._decorate_scalar_methods()
        self._decorate_geometry_methods()
        self._decorate_relation_methods()

    def _decorate_point_methods(self):
        sym = 'hp'
        node_type = vector
        methods = ['Mirrored', 'Centered', 'Equal_to', 'UserInput']
        self._point_methods = self._decorate_components(node_type, sym, methods, CR)
        
    def _decorate_vector_methods(self):
        sym = 'vc'
        node_type = vector
        methods = ['Mirrored', 'Oriented', 'Equal_to', 'UserInput']
        self._vector_methods = self._decorate_components(node_type, sym, methods, CR)

    def _decorate_scalar_methods(self):
        sym = ''
        node_type = sm.symbols
        methods = ['Equal_to', 'UserInput']
        self._scalar_methods = self._decorate_components(node_type, sym, methods, CR)
            
    def _decorate_geometry_methods(self):
        sym = 'gm'
        node_type = Geometry
        methods = ['Composite_Geometry', 'Cylinder_Geometry', 'Triangular_Prism']
        self._geometry_methods = self._decorate_components(node_type, sym, methods, Geometries)

    def _decorate_relation_methods(self):
        sym = None
        node_type = None
        methods = ['Mirrored', 'Centered', 'Equal_to', 'Oriented', 'UserInput']
        self._relation_methods = self._decorate_components(node_type, sym, methods, CR)

    def _decorate_components(self, node_type, sym, methods_list, methods_class):   
        container_class = type('container', (object,), {})
        def dummy_init(dself): pass
        container_class.__init__ = dummy_init
        container_instance = container_class()

        for name in methods_list:
            method = getattr(methods_class, name)
            decorated_method = self._decorate_as_attr(node_type, sym, method)
            setattr(container_instance, name, decorated_method)
        
        return container_instance
    
    def _decorate_as_attr(self, symbolic_type, sym, construction_method):
        
        if construction_method is None:
            def decorated(*args, **kwargs):
                name = args[0]
                self._add_node(name, symbolic_type , sym=sym, **kwargs)
            decorated.__doc__ = ''
        
        elif symbolic_type is None:
            def decorated(*args, **kwargs):
                self._add_relation(construction_method, *args, **kwargs)
            decorated.__doc__ = construction_method.__doc__
       
        else:
            def decorated(*args, **kwargs):
                name = args[0]
                node = self._add_node(name, symbolic_type, sym=sym, **kwargs)
                self._add_relation(construction_method, node, *args[1:], **kwargs)
            decorated.__doc__ = construction_method.__doc__
        
        return decorated
    
    def _add_node(self, name, symbolic_type, **kwargs):
        return self._config.add_node(name, symbolic_type, **kwargs)

    def _add_relation(self, relation, node, arg_nodes, **kwargs):
        self._config.add_relation(relation, node, arg_nodes, **kwargs)

###############################################################################
###############################################################################

class numerical_subsystem(object):
    
    def __init__(self, topology_instance):
        self._name = topology_instance.prefix[:-1]
        self._topology = topology_instance
        
    def set_configuration_file(self, config_module):
        self._topology.config = config_module.configuration()
        
    def set_configuration_data(self, file):
        path = os.path.join('configuration_files', file)
        self._topology.config.load_from_csv(path)
    
    @property
    def config(self):
        return self._topology.config

###############################################################################
###############################################################################

class Subsystems(object):
    
    def __init__(self, subsystems_list):
        for sub in subsystems_list:
            sub = numerical_subsystem(sub)
            setattr(self, sub._name, sub)

class multibody_system(object):
    
    def __init__(self, system):
        
        self.system = system.numerical_assembly()        
        self.Subsystems = Subsystems(self.system.subsystems)
        

###############################################################################
###############################################################################

from source.numerical_classes.python_solver import kds_solver, dds_solver
import matplotlib.pyplot as plt
import numpy as np

class simulation(object):
    
    def __init__(self, name, model, typ='kds'):
        
        self.name = name
        self.assembly = model.system
        
        if typ == 'kds':
            self.soln = kds_solver(self.assembly)
        elif typ == 'dds':
            self.soln = dds_solver(self.assembly)
        else:
            raise ValueError('Bad simulation type argument : %r'%typ)
    
    def set_time_array(self, duration, spacing):
        self.soln.set_time_array(duration, spacing)
        
    def solve(self, run_id=None, save=True):
        run_id = '%s_temp'%self.name if run_id is None else run_id
        self.soln.solve(run_id)
        if save:
            filename = run_id
            self.save_results(filename)
    
    def save_results(self, filename):
        path = os.path.join('results', filename)
        self.soln.pos_dataframe.to_csv('%s.csv'%path, index=True)
    
    def plot(self, y_args, x=None):
        
        if x is None:
            x_data = self.soln.time_array 
        elif isinstance(x, tuple):
            x_string, level = x
            data = getattr(self.soln, '%s_dataframe'%level)
            x_data = data[x_string]
        elif isinstance(x, np.ndarray):
            x_data = x
        
        plt.figure(figsize=(8,4))
        for y_string, level in y_args:
            data = getattr(self.soln, '%s_dataframe'%level)
            y_data = data[y_string]
            plt.plot(x_data, y_data)
        
        plt.legend()
        plt.grid()
        plt.show()

        
