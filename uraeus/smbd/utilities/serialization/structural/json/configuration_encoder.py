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

class generator(object):
    """
    This class serves as a 
    """

    def __init__(self, sym_config):

        self.config = sym_config

        self.configuration_name = self.config.name
        self.topology_name = self.config.topology.name
        
        self.graph  = self.config.graph

        self.input_nodes  = self.config.input_nodes
        self.output_nodes = self.config.output_nodes
        self.intermediat_nodes = self.config.intermediat_nodes

        self.primary_equalities = self.config.primary_equalities
        self.geometries_map = self.config.geometries_map

        self.data = self.construct()

    def write_JSON_file(self, file_path=''):
        name = '%s.json'%self.configuration_name
        file_name = os.path.join(file_path, name)
        json_text = self.dump_JSON_text()

        with open(file_name, 'w') as f:
            f.write(json_text)
        
    def dump_JSON_text(self):
        data = self.construct()
        json_text = json.dumps(data, cls=Encoder, indent=4)
        return json_text
    
    def construct(self):
        config_info = {}
        config_info['topology_name'] = self.topology_name
        config_info['configuration_name'] = self.configuration_name
        config_info['subsystem_name'] = ''
        
        data = {}
        data['information'] = config_info
        data['user_inputs'] = self.construct_data_dict(self.input_nodes)
        data['evaluations'] = self.construct_data_dict(self.intermediat_nodes)
        data['outputs'] = self.construct_data_dict(self.output_nodes)
        data['geometries_map'] = self.geometries_map
        return data

    
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
    

    
    




