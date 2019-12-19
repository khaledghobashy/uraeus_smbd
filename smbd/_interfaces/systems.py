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
import smbd.symbolic.components.joints as joints
import smbd.symbolic.components.forces as forces
import smbd.symbolic.systems.topology_classes as topology_classes
import smbd.symbolic.systems.configuration_classes  as cfg_cls
from smbd.symbolic.components.matrices import vector
from smbd.utilities.serialization.structural.json.configuration_encoder import graph_data_extractor

###############################################################################

def get_file_name(script_path):
    name = os.path.basename(script_path).split('.')[0]
    return name

def load_pickled_data(file):
    with open(file, 'rb') as f:
        instance = cloudpickle.load(f)
    return instance


class standalone_project(object):
    
    def __init__(self, parent_dir=''):
        
        self.parent_dir = parent_dir
        self.code_dir = os.path.join(self.parent_dir, 'numenv')
        
    def create(self):
        self._create_common_dirs()
    
    def _create_common_dirs(self):
        for d in ['config_inputs', 'results', 'numenv']:
            subdir = os.path.join(self.parent_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        if self.parent_dir !='':
            os.chdir(self.parent_dir)
            print('Current working directory is %s'%os.path.abspath(self.parent_dir))

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

class topology_edges_container(object):
    
    def __init__(self, topology):
        self._topology = topology
        self._decorate_items()
        
    @property
    def _items(self):
        members = {i:getattr(self,i) for i in dir(self) if not i.startswith('_') and not i.startswith("__")}
        return members
    
    def _decorate_items(self):
        for attr,obj in self._items.items():
            setattr(self, attr, self._decorate(obj))
    
    def _decorate(self, edge_component):
        raise NotImplementedError


class joints_container(topology_edges_container):
    
    def __init__(self, topology):
        self.spherical = joints.spherical
        self.revolute  = joints.revolute
        self.universal = joints.universal
        self.translational = joints.translational
        self.cylinderical  = joints.cylinderical
        self.tripod = joints.tripod
        self.fixed  = joints.fixed
        self.fixed_orientation = joints.fixed_orientation
        self.inline = joints.inline
        
        super().__init__(topology)
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
            self._topology.add_joint(edge_component, *args, **kwargs)
        return decorated

    
class actuators_container(topology_edges_container):
    
    def __init__(self, topology):
        self.rotational_actuator = joints.rotational_actuator
        self.absolute_locator = joints.absolute_locator
        self.translational_actuator = joints.translational_actuator
        self.absolute_rotator = joints.absolute_rotator
        super().__init__(topology)
    
    def _decorate(self, edge_component):
        if issubclass(edge_component, joints.absolute_locator):
            def decorated(*args, **kwargs):
                self._topology.add_absolute_actuator(edge_component, *args, **kwargs)
        elif issubclass(edge_component, joints.absolute_rotator):
            def decorated(*args, **kwargs):
                self._topology.add_absolute_actuator(edge_component, *args, **kwargs)
        else:
            def decorated(*args, **kwargs):
                self._topology.add_joint_actuator(edge_component, *args, **kwargs)
        return decorated


class forces_container(topology_edges_container):
    
    def __init__(self, topology):
        self.internal_force = forces.internal_force
        self.force = forces.force
        self.torque = forces.torque
        self.generic_force = forces.generic_force
        self.bushing = forces.bushing
        
        super().__init__(topology)
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
            self._topology.add_force(edge_component, *args, **kwargs)
        return decorated
    
###############################################################################

class template_topology(object):
    
    def __init__(self, name):
        self.name = name
        self.topology = topology_classes.template_based_topology(self.name)
        
        self._joints = joints_container(self.topology)
        self._actuators = actuators_container(self.topology)
        self._forces = forces_container(self.topology)
    
    
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
        
        self._joints = joints_container(self.topology)
        self._actuators = actuators_container(self.topology)
        self._forces = forces_container(self.topology)
        

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
        self.config.assign_geometry_to_body(body, geo, eval_inertia, mirror)
    
    def assemble(self):
        self.config.assemble_equalities()


    def extract_inputs_to_csv(self, path):
        file_path = os.path.join(path, self.name)
        inputs_dataframe = self.config.create_inputs_dataframe()
        inputs_dataframe.to_csv('%s.csv'%file_path)
    
    def export_JSON_file(self, path=''):
        config_constructor = graph_data_extractor(self.config)
        config_constructor.write_JSON_file(path)
    
    def save(self):
        file = '%s.scfg'%self.name
        with open(file, 'wb') as f:
            cloudpickle.dump(self, f)
    
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
        self._point_methods = self._decorate_components(node_type, sym, 
                                                        methods, cfg_cls.CR)
        
    def _decorate_vector_methods(self):
        sym = 'vc'
        node_type = vector
        methods = ['Mirrored', 'Oriented', 'Equal_to', 'UserInput']
        self._vector_methods = self._decorate_components(node_type, sym, 
                                                         methods, cfg_cls.CR)

    def _decorate_scalar_methods(self):
        sym = ''
        node_type = sm.symbols
        methods = ['Equal_to', 'UserInput']
        self._scalar_methods = self._decorate_components(node_type, sym, 
                                                         methods, cfg_cls.CR)
            
    def _decorate_geometry_methods(self):
        sym = 'gm'
        node_type = cfg_cls.Geometry
        methods = ['Composite_Geometry', 
                   'Cylinder_Geometry', 
                   'Triangular_Prism',
                   'Sphere_Geometry']
        self._geometry_methods = self._decorate_components(node_type, sym, 
                                                           methods, cfg_cls.Geometries)

    def _decorate_relation_methods(self):
        sym = None
        node_type = None
        methods = ['Mirrored', 'Centered', 'Equal_to', 'Oriented', 'UserInput']
        self._relation_methods = self._decorate_components(node_type, sym, 
                                                           methods, cfg_cls.CR)

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
        return self.config.add_node(name, symbolic_type, **kwargs)

    def _add_relation(self, relation, node, arg_nodes, **kwargs):
        self.config.add_relation(relation, node, arg_nodes, **kwargs)


###############################################################################
###############################################################################

