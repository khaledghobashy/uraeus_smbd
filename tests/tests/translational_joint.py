#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:52:59 2019

@author: khaledghobashy
"""

from smbd.interfaces.scripting import standalone_topology, configuration
from smbd.numenv.python.codegen import generators

model = standalone_topology('translational')

model.add_body('body')
model.add_joint.translational('a', 'ground', 'rbs_body')

model.add_actuator.translational_actuator('act', 'jcs_a')


model.assemble()
model.save()

code = generators.template_codegen(model._mbs)
code.write_code_file()


config = configuration('%s_cfg'%model._name, model)
print(config._config.input_nodes)
config.assemble()

config_code = generators.configuration_codegen(config._config)
config_code.write_code_file()

import pandas as pd
import numpy as np

import translational, translational_cfg
from smbd.numenv.python.interfaces.scripting import multibody_system, simulation

num_model = multibody_system(translational)
num_model.topology.config = translational_cfg.configuration()

num_model.topology.config.P_ground = np.array([[1],[0],[0],[0]], dtype=np.float64)

num_model.topology.config.R_rbs_body = np.array([[0],[0],[200]], dtype=np.float64)
num_model.topology.config.P_rbs_body = np.array([[1],[0],[0],[0]], dtype=np.float64)

num_model.topology.config.pt1_jcs_a = np.array([[0],[0],[0]], dtype=np.float64)
num_model.topology.config.ax1_jcs_a = np.array([[0],[0],[1]], dtype=np.float64)

num_model.topology.config.pt1_mcs_act = np.array([[0],[0],[0]], dtype=np.float64)
num_model.topology.config.ax1_mcs_act = np.array([[0],[0],[1]], dtype=np.float64)

num_model.topology.config.UF_jcs_a = lambda t : 50*np.sin(t)

sim1 = simulation('sim1', num_model, 'kds')
sim1.set_time_array(2*np.pi, 100)
sim1.solve(save=False)


sim1.soln.pos_dataframe.plot(x='time', y='rbs_body.z')

