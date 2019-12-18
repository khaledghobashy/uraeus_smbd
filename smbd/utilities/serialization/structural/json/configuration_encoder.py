# -*- coding: utf-8 -*-
"""
@author: Khaled Ghobashy
"""

import os
import sys
import json
import inspect

import sympy as sm
from smbd.symbolic.components.matrices import AbstractMatrix, vector, quatrenion
from smbd.symbolic.systems.configuration_classes import Simple_geometry

class Encoder(json.JSONEncoder):
    
    def default(self, obj):
        return JSONify(obj)


def JSONify(instance):

    if inspect.isclass(instance):
        constructor = instance.__name__
        return constructor
    
    if isinstance(instance, (str, float, int, bool)):
        return instance
    
    elif isinstance(instance, dict):
        return {k: JSONify(v) for k,v in instance.items()}
    
    elif isinstance(instance, list):
        alias = [JSONify(value) for value in instance]
        return alias
    
    elif isinstance(instance, (tuple, sm.Tuple)):
        alias = tuple(JSONify(value) for value in instance)
        return alias
    
    elif isinstance(instance, (sm.Number,)):
        return float(instance)
    
    elif isinstance(instance, (sm.ImmutableDenseMatrix, sm.MutableDenseMatrix)):
        if 1 in instance.shape:
            alias = [JSONify(value) for value in instance]
        else:
            alias = [JSONify(value) for value in instance.tolist()]
        data_object = {'constructor': 'array', 'args':  alias}
        return data_object
    
    elif isinstance(instance, (vector, quatrenion, sm.Symbol)):
        text = str(instance)
        return text
    
    elif isinstance(instance, tuple(Simple_geometry.__subclasses__())):
        constructor = JSONify(instance.__class__)
        args = [JSONify(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args':  args}
        return data_object
    
    elif isinstance(instance, (sm.Function, sm.Lambda)):
        constructor = JSONify(instance.__class__)
        args = [JSONify(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args':  args}
        return data_object
    
    elif isinstance(instance, tuple(AbstractMatrix.__subclasses__())):
        constructor = JSONify(instance.__class__)
        args = [JSONify(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args':  args}
        return data_object
    
    else:
        #print(instance)
        return 'Data type not supported'
        



class graph_data_extractor(object):

    def __init__(self, sym_config):

        self.config = sym_config

        self.configuration_name = self.config.name
        self.topology_name = self.config.topology.name
        
        self.graph  = self.config.graph

        self.input_nodes  = self.config.input_nodes
        self.output_nodes = self.config.output_nodes
        self.intermediat_nodes = self.config.intermediat_nodes

        self.primary_equalities = self.config.primary_equalities

    def dump_JSON_text(self):
        data = self.construct()
        json_text = json.dumps(data, cls=Encoder, indent=4)
        return json_text
    
    def write_JSON_file(self, file_path=''):
        name = '%s.json'%self.configuration_name
        file_name = os.path.join(file_path, name)
        json_text = self.dump_JSON_text()

        with open(file_name, 'w') as f:
            f.write(json_text)
    
    def construct(self):
        config_info = {}
        config_info['topology_name'] = self.topology_name
        config_info['configuration_name'] = self.configuration_name
        
        data = {}
        data['information'] = config_info
        data['user_inputs'] = self.construct_data_dict(self.input_nodes)
        #data['raw_inputs']  = self.construt_raw_inputs(self.primary_equalities)
        data['evaluations'] = self.construct_data_dict(self.intermediat_nodes)
        data['outputs']     = self.construct_data_dict(self.output_nodes)
        return data

    def construt_raw_inputs(self, equalities_dict):
        storage_dict = {}
        for node, equality in equalities_dict.items():
            storage_dict[node] = JSONify(equality.rhs)
        return storage_dict
    
    def construct_data_dict(self, nodes):
        storage_dict = {}
        for node in nodes:
            feeding_nodes = self.get_feeding_nodes(node)

            if len(feeding_nodes) == 1:
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
            data_dict = {'constructor': 'getattribute', 'args': [parent_node, attribute]}
            return data_dict
        else:
            return parent_node
    
    def get_feeding_nodes(self, node):
        return list(self.graph.predecessors(node))
    

    
    




