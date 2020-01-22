# -*- coding: utf-8 -*-
"""
@author: Khaled Ghobashy
"""
# Standard library imports
import os
import sys
import json
import inspect

# 3rd party library imports
import sympy as sm

# Local applicataion imports
from .....symbolic.components.matrices import AbstractMatrix, vector, quatrenion
from .....symbolic.systems.configuration_classes import Simple_geometry, Equal_to

################################################################################

class Encoder(json.JSONEncoder):
    """
    A subclass of the `json.JSONEncoder` that over-rides the `default` method
    that calls a custom `JSONify` function that returns a compatibale type
    that can be serialzed in JSON.
    """
    
    def default(self, obj):
        return JSONify(obj)

################################################################################
################################################################################

def JSONify(instance):
    """
    A function that takes in a symbolic object or a class and returns a 
    compatibale type that can be serialzed in JSON.

    TODO:
        DataTypes map
    """

    # check if the given instance is a class
    if inspect.isclass(instance):
        constructor = instance.__name__
        return constructor
    
    # check if the given instance is a basic scalar data type that can be 
    # understod by the JSON encoder directly.
    if isinstance(instance, (str, float, int, bool)):
        return instance
    
    # check if the given instance is a basic sequence/iterable data type that 
    # can be understod by the JSON encoder directly.
    elif isinstance(instance, dict):
        return {k: JSONify(v) for k,v in instance.items()}
    
    elif isinstance(instance, list):
        alias = [JSONify(value) for value in instance]
        return alias
    
    elif isinstance(instance, (tuple, sm.Tuple)):
        alias = tuple(JSONify(value) for value in instance)
        return alias
    
    # Conversions of basic symbolic scalars / symbols to JSON
    elif isinstance(instance, (sm.Number,)):
        return float(instance)
    
    elif isinstance(instance, (vector, quatrenion, sm.Symbol)):
        text = str(instance)
        return text
    
    # Conversion of sympy matrices.
    elif isinstance(instance, (sm.ImmutableDenseMatrix, sm.MutableDenseMatrix)):
        if 1 in instance.shape:
            alias = [JSONify(value) for value in instance]
        else:
            alias = [JSONify(value) for value in instance.tolist()]
        data_object = {'constructor': 'array', 'args':  alias}
        return data_object
    
    # Conversion of symbolic geometries.
    elif isinstance(instance, tuple(Simple_geometry.__subclasses__())):
        constructor = JSONify(instance.__class__)
        args = [JSONify(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args':  args}
        return data_object
    
    # Conversion of symbolic geometries.
    elif isinstance(instance, tuple(AbstractMatrix.__subclasses__())):
        constructor = JSONify(instance.__class__)
        args = [JSONify(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args':  args}
        return data_object
    
    # Conversion of Lambda functions.
    elif isinstance(instance, (sm.Function, sm.Lambda)):
        constructor = JSONify(instance.__class__)
        args = [JSONify(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args':  args}
        return data_object
    
    # Fall back to basic string message if datatype not included in previous
    # casses.
    else:
        return 'Data type not supported'
        

################################################################################
################################################################################
def flatten_assembly(assm, attr):
    if len(assm.assemblies) == 0:
        return getattr(assm, attr)
    else:
        nested = {}
        for _assm in assm.assemblies.values():
            nested.update(flatten_assembly(_assm, attr))
        nested.update(getattr(assm, attr))
        return nested

def flatten_equalities(assm, attr):
    if len(assm.assemblies) == 0:
        return getattr(assm, attr)
    else:
        nested = []
        for _assm in assm.assemblies.values():
            nested += flatten_equalities(_assm, attr)
        nested += getattr(assm, attr)
        return nested

class generator(object):
    """
    This class serves as a 
    """

    def __init__(self, sym_assembly):

        self.mbs = sym_assembly
        self.name = self.mbs.name
        self.subsystems    = flatten_assembly(self.mbs, 'subsystems')
        self.interface_map = flatten_assembly(self.mbs, 'interface_map')
        self.nodes_indicies = self.mbs.nodes_indicies
        self.mapped_vir_coordinates = flatten_equalities(self.mbs, 'mapped_vir_coordinates')
        self.mapped_vir_velocities  = flatten_equalities(self.mbs, 'mapped_vir_velocities')
        self.mapped_vir_accelerations = flatten_equalities(self.mbs, 'mapped_vir_accelerations')
    
    def write_JSON_file(self, file_path=''):
        name = '%s.json'%self.name
        file_name = os.path.join(file_path, name)
        json_text = self.dump_JSON_text()

        with open(file_name, 'w') as f:
            f.write(json_text)
        
    def dump_JSON_text(self):
        data = self.construct()
        json_text = json.dumps(data, cls=Encoder, indent=4)
        return json_text
    
    def construct(self):
        assembly_info = {}
        assembly_info['assembly_name'] = self.name
        assembly_info['subsystems'] = {k: sub.template.name for k, sub in self.subsystems.items()}
        assembly_info['interface_map'] = self.interface_map
        assembly_info['nodes_indicies'] = self.nodes_indicies
        assembly_info['mapped_vir_coordinates'] = {str(eq.lhs): str(eq.rhs) for eq in self.mapped_vir_coordinates}
        assembly_info['mapped_vir_velocities'] = {str(eq.lhs): str(eq.rhs) for eq in self.mapped_vir_velocities}
        assembly_info['mapped_vir_accelerations'] = {str(eq.lhs): str(eq.rhs) for eq in self.mapped_vir_accelerations}
        return assembly_info

    
    def construct_data_dict(self, nodes):
        storage_dict = {}
        for node in nodes:
            feeding_nodes = self.get_feeding_nodes(node)

            if len(feeding_nodes) == 1 and issubclass(self.graph.nodes[node]['rhs_function'], Equal_to):
                n = feeding_nodes[0]
                storage_dict[node] = self.check_attribute_access((n, node))
                
            else:
                sym_equality = self.graph.nodes[node]['equality']
                storage_dict[node] = JSONify(sym_equality.rhs)
        return storage_dict
    

    def check_attribute_access(self, edge):
        parent_node = edge[0]
        attribute = self.graph.edges[edge]['passed_attr']
        if attribute:
            data_dict = {'constructor': 'getattribute', 
                         'args': [parent_node, attribute]}
            return data_dict
        else:
            return parent_node
    
    def get_feeding_nodes(self, node):
        return list(self.graph.predecessors(node))
    

    
    




