# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
# Standard library imports
import os
import json
import pickle
import itertools

# 3rd party library imports
import numpy as np

# Local applicataion imports
from .solvers import kds_solver, dds_solver
from smbd.utilities.numerics.spatial_alg import centered, oriented, mirrored
from smbd.utilities.numerics.geometries import cylinder_geometry

###############################################################################

def load_pickled_data(file):
    with open(file, 'rb') as f:
        instance = pickle.load(f)
    return instance

###############################################################################
###############################################################################

class multibody_system(object):
    
    def __init__(self, system):
        self.topology = system.topology()
        try:
            self.Subsystems = system.subsystems
        except AttributeError:
            pass
        
###############################################################################
###############################################################################

class simulation(object):
    
    def __init__(self, name, model, typ='kds'):
        self.name = name
        self.assembly = model.topology
        if typ == 'kds':
            self.soln = kds_solver(self.assembly)
        elif typ == 'dds':
            self.soln = dds_solver(self.assembly)
        else:
            raise ValueError('Bad simulation type argument : %r'%typ)
    
    def set_time_array(self, duration, spacing):
        self.soln.set_time_array(duration, spacing)
        
    def solve(self, run_id=None):
        run_id = '%s_temp'%self.name if run_id is None else run_id
        self.soln.solve(run_id)
    
    def save_results(self, path, filename):
        path = os.path.abspath(path)
        filepath = os.path.join(path, filename)
        self.soln.pos_dataframe.to_csv('%s.csv'%filepath, index=True)
        print('results saved as %s.csv at %s'%(filename, path))
        
    def eval_reactions(self):
        self.soln.eval_reactions()
    

###############################################################################
###############################################################################

class JSON_configuration(object):

    _construction_map = {'array': np.array,
                         'getattribute': getattr,
                         'Oriented': oriented,
                         'Mirrored': mirrored,
                         'Centered': centered,
                         'Cylinder_Geometry': cylinder_geometry,}

    def __init__(self, json_file):
        self.file = json_file
        self.construct_toplevel_dicts()
        self.construct_inputs()

    def construct_toplevel_dicts(self):
        json_text = self.file
        data_dict = json.loads(json_text)

        self.user_inputs = data_dict['user_inputs']
        self.evaluations = data_dict['evaluations']
        self.outputs     = data_dict['outputs']
    
    def construct_inputs(self):
        user_inputs_dict = self.user_inputs
        for key, data in user_inputs_dict.items():
            if isinstance(data, dict):
                constructor = self._construction_map[data['constructor']]
                args = data['args']
                value = constructor(args)[:, None]
                setattr(self, key, value)
            elif isinstance(data, (int, float, str, bool)):
                setattr(self, key, data)
    
    def perform_evaluations(self):
        
        evaluations_dict = self.evaluations
        outputs_dict = self.outputs
        
        for key, data in evaluations_dict.items():
            if isinstance(data, dict):
                constructor = self._construction_map[data['constructor']]
                args = [getattr(self, arg) for arg in data['args']]
                value = constructor(*args)
                setattr(self, key, value)
            elif isinstance(data, (int, float, str, bool)):
                setattr(self, key, data)
        
        for key, data in outputs_dict.items():
            if isinstance(data, dict):
                constructor = self._construction_map[data['constructor']]
                if constructor is getattr:
                    args = [getattr(self, data['args'][0]), data['args'][1]]
                else:
                    args = [getattr(self, arg) for arg in data['args']]
                value = constructor(*args)
                setattr(self, key, value)
            elif isinstance(data, (int, float, str, bool)):
                if isinstance(data, str):
                    value = getattr(self, data)
                else:
                    value = data
                setattr(self, key, value)


###############################################################################
###############################################################################

class configuration(object):
    
    def __init__(self, name):
        self.name = name

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)

        self.Rd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
    
    def construct_from_json(self, json_file, assemble=False):

        json_config = JSON_configuration(json_file)
        self.json_config = json_config

        if not assemble:
            _attributes = json_config.user_inputs.keys()
            for key in _attributes:
                value = getattr(json_config, key)
                setattr(self, key, value)
        else:
            self.assemble()
        
        
    
    def assemble(self):
        self.json_config.perform_evaluations()

        _attributes = itertools.chain(self.json_config.evaluations.keys(),
                                      self.json_config.outputs.keys())
        
        for key in _attributes:
            value = getattr(self.json_config, key)
            setattr(self, key, value)

