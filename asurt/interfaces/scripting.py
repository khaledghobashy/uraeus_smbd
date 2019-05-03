# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
# Standard library imports
import os
import pickle

# 3rd party library imports
import cloudpickle
import sympy as sm

# Local applicataion imports
import asurt.symbolic.symbolic_classes.joints as joints
import asurt.symbolic.symbolic_classes.forces as forces
import asurt.symbolic.mbs_creators.topology_classes as topology_classes
import asurt.symbolic.mbs_creators.configuration_classes  as cfg_cls
from asurt.symbolic.symbolic_classes.matrices import vector

# Local directory imports
from . import codegens

###############################################################################

def get_file_name(script_path):
    name = os.path.basename(script_path).split('.')[0]
    return name

def load_pickled_data(file):
    with open(file, 'rb') as f:
        instance = pickle.load(f)
    return instance

def load_template(project_dir, file_name):
    relative_file_path = 'symenv.templates'.split('.')
    file = os.path.join(project_dir, *relative_file_path, file_name, file_name)
    instance = load_pickled_data('%s.stpl'%file)
    return instance
    
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
    
    def __init__(self, mbs, _execluded_kwargs=[]):
        self.spherical = joints.spherical
        self.revolute  = joints.revolute
        self.universal = joints.universal
        self.translational = joints.translational
        self.cylinderical  = joints.cylinderical
        self.tripod = joints.tripod
        self.fixed  = joints.fixed
        
        self._execluded_kwargs = _execluded_kwargs
        super().__init__(mbs)
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
            for kw in kwargs:
                if kw in self._execluded_kwargs:
                    raise ValueError('%r not allowed here!.'%kw)
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

class template_topology(object):
    
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
            
    def save(self):
        file = '%s.stpl'%self._name
        with open(file, 'wb') as f:
            cloudpickle.dump(self, f)
    
    def write_python_code(self, proj_dir):
        codegens.topology_generators.write_python_code(self._mbs, proj_dir)

###############################################################################
###############################################################################

class standalone_topology(template_topology):
    
    def __init__(self, script_path):
        self._script_path = script_path
        self._name = get_file_name(script_path)
        self._mbs = topology_classes.standalone_topology(self._name)
        
        self._joints = joints_container(self._mbs)
        self._actuators = actuators_container(self._mbs)
        self._forces = forces_container(self._mbs)
    
    def write_python_code(self, proj_dir):
        codegens.standalone_generators.write_python_code(self._mbs, proj_dir)
    

###############################################################################
###############################################################################

class assembly(object):
    
    def __init__(self, script_path):
        self._script_path = script_path
        self._name = get_file_name(script_path)
        self._mbs = topology_classes.assembly(self._name)
        
    def add_assembly(self, assm):
        try :
            assm = assm._mbs
        except AttributeError:
            pass
        self._mbs.add_assembly(assm)
        
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
        
    def save(self):
        file = '%s.sasm'%self._name
        with open(file, 'wb') as f:
            cloudpickle.dump(self, f)
    
    def write_python_code(self, proj_dir):
        codegens.assembly_generators.write_python_code(self._mbs, proj_dir)
    
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
        
        self._config = cfg_cls.abstract_configuration(self._name, model_instance._mbs)
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
    
    def assemble_model(self):
        self._config.assemble_equalities()

    def write_python_code(self, proj_dir):
        codegens.configuration_generators.write_python_code(self._config, proj_dir)
        
    def write_blender_script(self, proj_dir):
        codegens.visuals_generator.blender(self._config, proj_dir)

    def extract_inputs_to_csv(self):
        file_path = os.path.join('csv_files', self._name)
        inputs_dataframe = self._config.create_inputs_dataframe()
        inputs_dataframe.to_csv('%s.csv'%file_path)
    
    def save(self):
        file = '%s.scfg'%self._name
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
        methods = ['Composite_Geometry', 'Cylinder_Geometry', 'Triangular_Prism']
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
        return self._config.add_node(name, symbolic_type, **kwargs)

    def _add_relation(self, relation, node, arg_nodes, **kwargs):
        self._config.add_relation(relation, node, arg_nodes, **kwargs)


###############################################################################
###############################################################################

