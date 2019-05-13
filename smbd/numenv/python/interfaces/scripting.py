# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
# Standard library imports
import os
import pickle

# 3rd party library imports
import numpy as np
import matplotlib.pyplot as plt

# Local applicataion imports
from ..solvers.solvers import kds_solver, dds_solver

###############################################################################

def load_pickled_data(file):
    with open(file, 'rb') as f:
        instance = pickle.load(f)
    return instance

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
        self.system = system.topology()
        try:
            self.Subsystems = Subsystems(self.system.subsystems)
        except AttributeError:
            pass
        
###############################################################################
###############################################################################

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

