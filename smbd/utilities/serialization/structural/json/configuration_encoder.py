# -*- coding: utf-8 -*-
"""
@author: Khaled Ghobashy
"""

import sys
import json
import inspect

import sympy as sm
from smbd.symbolic.components.matrices import AbstractMatrix, vector, quatrenion
from smbd.symbolic.systems.configuration_classes import Simple_geometry



def to_JSON(instance):

    if inspect.isclass(instance):
        pass
    
    if isinstance(instance, (str, float, int, bool)):
        return instance
    
    elif isinstance(instance, dict):
        return {k: to_JSON(v) for k,v in instance.items()}
    
    elif isinstance(instance, list):
        alias = [to_JSON(value) for value in instance]
        return alias
    
    elif isinstance(instance, tuple):
        alias = tuple(to_JSON(value) for value in instance)
        return alias
    
    elif isinstance(instance, (sm.Number,)):
        return float(instance)
    
    elif isinstance(instance, (sm.ImmutableDenseMatrix, sm.MutableDenseMatrix)):
        if 1 in instance.shape:
            alias = [to_JSON(value) for value in instance]
        else:
            alias = [to_JSON(value) for value in instance.tolist()]
        data_object = {'constructor': 'array', 'args':  alias}
        return data_object
    
    elif isinstance(instance, (vector, quatrenion, sm.Symbol)):
        text = str(instance)
        if '.' in text:
            splitted = text.split('.')
            object_data = {'object': splitted[0], 'attribute': splitted[1]}
            return object_data
        else:
            return text
    
    
    elif isinstance(instance, tuple(Simple_geometry.__subclasses__())):
        return instance
        '''constructor = instance.__class__.__name__
        args = [get_type_alias(arg) for arg in instance.args]
        data_object = {'constructor': constructor, 'args': args}
        return data_object'''
    
    elif isinstance(instance, (sm.Function, sm.Lambda)):
        return instance.__class__.__name__
    
    elif isinstance(instance, tuple(AbstractMatrix.__subclasses__())):
        return instance.__class__.__name__
    
    else:
        #print(instance)
        return instance#'Data type not supported'
        


class Encoder(json.JSONEncoder):
    
    def default(self, obj):
        return to_JSON(obj)
