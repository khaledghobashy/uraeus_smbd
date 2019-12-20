# -*- coding: utf-8 -*-
"""
@author: khaled.ghobashy
"""
# Standard library imports
import json

# 3rd party library imports
import numpy as np

# Local applicataion imports
from ..math_funcs.spatial_alg import centered, oriented, mirrored
from ..math_funcs.geometries import cylinder_geometry

###############################################################################

class constructors(object):

    @classmethod
    def Mirrored(cls, args):
        return mirrored(*args)
    
    @classmethod
    def Oriented(cls, args):
        return oriented(*args)
    
    @classmethod
    def Centered(cls, args):
        return centered(*args)
    
    @classmethod
    def Cylinder_Geometry(cls, args):
        return cylinder_geometry(*args)
    
    @classmethod
    def Lambda(cls, args):
        arguments, return_value = args
        str_args = ', '.join([i for i in arguments])
        text = 'dummy = lambda %s: %s'%(str_args, return_value)
        exec(text)
        return eval('dummy')
    
    @classmethod
    def array(cls, args):
        value = np.array(args, dtype=np.float64)
        value = value[:, None]
        return value
    
    @classmethod
    def getattribute(cls, args):
        return getattr(*args)

###############################################################################
###############################################################################

class JSON_Decoder(object):

    def __init__(self, json_file):
        self.file = json_file
        self._initialize()
    
    def assemble(self):
        self._construct_data(self.evaluations)
        self._construct_data(self.outputs)

    def _initialize(self):
        data_dict = self._get_data_from_file(self.file)
        self.user_inputs = data_dict['user_inputs']
        self.evaluations = data_dict['evaluations']
        self.outputs     = data_dict['outputs']
        self._construct_data(self.user_inputs, is_inputs=True)
    
    def _construct_data(self, data_dict, is_inputs=False):
        for key, data in data_dict.items():
            if isinstance(data, dict):
                constructor_name = data['constructor']
                constructor_args = data['args']

                if is_inputs:
                    args = constructor_args
                else:
                    if constructor_name == 'getattribute':
                        obj  = getattr(self, constructor_args[0])
                        attr = constructor_args[1]
                        args = [obj, attr]
                    else:
                        args = [getattr(self, arg) for arg in constructor_args]
                
                constructor = self.get_constructor(constructor_name)
                value = constructor(args)
                
            elif isinstance(data, (int, float, str, bool)):
                if isinstance(data, str):
                    value = getattr(self, data)
                else:
                    value = data
            
            setattr(self, key, value)


    @staticmethod
    def _get_data_from_file(json_file):
        with open(json_file, 'r') as f:
            json_text = f.read()
        data_dict = json.loads(json_text)
        return data_dict
    
    @staticmethod
    def get_constructor(constructor_name):
        return getattr(constructors, constructor_name)


###############################################################################
###############################################################################
