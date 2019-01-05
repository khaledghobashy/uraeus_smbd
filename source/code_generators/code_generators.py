# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:25:26 2019

@author: khale
"""

import sympy as sm
from source.code_generators.code_printers import numerical_printer

import textwrap
class python_generator(object):
    global ccode_print
    global enclose
    ccode_print = True
    enclose = True
    
    def __init__(self,assembled_model,printer=numerical_printer()):
        
        self.assembly = assembled_model
        self.graph = assembled_model.graph.copy()
        self.printer = printer
        
        self.numerical_arguments = [printer._print(e.lhs) for e in self.assembly.numerical_arguments]
        self.configuration_constants = [printer._print(e.lhs) for e in self.assembly.configuration_constants]
        
        #self.configuration_coordinates = '\n'.join([printer._print(e) for e in self.assembly.numerical_arguments])
        
        self.generalized_coordinates_equalities = self.assembly.q_maped.copy()
        self.generalized_velocities_equalities = self.assembly.qd_maped.copy()
        
        self.generalized_coordinates_lhs = [printer._print(exp.lhs) for exp in self.generalized_coordinates_equalities]
        self.generalized_velocities_lhs = [printer._print(exp.lhs) for exp in self.generalized_velocities_equalities]

    
    def write_imports(self):
        text = '''
                import numpy as np
                import scipy as sc
                from numpy.linalg import multi_dot
                from matrix_funcs import A, B, sparse_assembler, triad as Triad
                from scipy.misc import derivative
                from numpy import cos, sin
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
    
    def write_config_inputs(self):
        text = '''
                class inputs(object):

                    def __init__(self):
                        {declared_inputs}
                '''
        
        p = self.printer
        indent = 8*' '
        typing_prefix = 'self.'
        
        inputs = self.assembly.numerical_arguments
        
        self.inputs_equalities = [p._print(exp) for exp in inputs]
        self.inputs_variables  = [p._print(exp.lhs) for exp in inputs]
        declared_inputs  = '\n'.join([typing_prefix + i for i in self.inputs_equalities])
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        declared_inputs = textwrap.indent(declared_inputs,indent).lstrip()
        text = text.format(declared_inputs = declared_inputs)
        
        return text
    
    def write_coordinates_setter(self):
        text = '''
                def set_coordinates(self,q):
                    {maped}
                    
               '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        maped_coordinates = '\n'.join(['self.' + p._print(i) for i in self.assembly.q_maped])
        maped_coordinates = textwrap.indent(maped_coordinates,indent).lstrip()
        text = text.format(maped = maped_coordinates)
        return text
    
    def write_velocities_setter(self):
        text = '''
                def set_velocities(self,qd):
                    {maped}
                    
               '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        maped_coordinates = '\n'.join(['self.' + p._print(i) for i in self.assembly.qd_maped])
        maped_coordinates = textwrap.indent(maped_coordinates,indent).lstrip()
        text = text.format(maped = maped_coordinates)
        return text
    
    def write_initial_configuration_setter(self):
        text = '''
                def set_initial_configuration(self):
                    config = self.config
                    
                    q = np.concatenate([{coordinates}])
                    
                    qd = np.concatenate([{velocities}])
                    
                    self.set_coordinates(q)
                    self.set_velocities(qd)
                    self.q_initial = q.copy()
                    
               '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indent = 4*' '
        
        coordinates = ',\n'.join(self.generalized_coordinates_lhs)
        for exp in self.generalized_coordinates_lhs:
            coordinates = coordinates.replace(exp,'config.'+exp[1:-1])
        coordinates = textwrap.indent(coordinates,indent).lstrip()
        
        velocities = ',\n'.join(self.generalized_velocities_lhs)
        for exp in self.generalized_velocities_lhs:
            velocities = velocities.replace(exp,'config.'+exp[1:-1])
        velocities = textwrap.indent(velocities,indent).lstrip()
        
        text = text.format(coordinates = coordinates,
                           velocities  = velocities)
        return text
    
    def write_assembly_class(self):
        text = '''
                class numerical_assembly(object):

                    def __init__(self,config):
                        self.F = config.F
                        self.t = 0
                    
                        self.config = config
                        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
                        
                        {pos_rows}
                        {pos_cols}
                        
                        {vel_rows}
                        {vel_cols}
                        
                        {acc_rows}
                        {acc_cols}
                        
                        {jac_rows}
                        {jac_cols}
                    
                    def eval_constants(self):
                        config = self.config
                        
                        {cse_variables}

                        {cse_expressions}
                    
                    {coord_setter}
                    {velocities_setter}
                    {initial_config_setter}
                    
                    {eval_pos}
                    
                    {eval_vel}
                    
                    {eval_acc}
                    
                    {eval_jac}
                    
                '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 8*' '
        
        configuration_constants = self.assembly.configuration_constants
        
        constants_cse = sm.cse(configuration_constants,symbols=sm.iterables.numbered_symbols('c'))
        cse_variables = '\n'.join([p._print(exp) for exp in constants_cse[0]])
        cse_expressions  = '\n'.join(['self.' + p._print(exp) for exp in constants_cse[1]])

        for exp in self.numerical_arguments:
            cse_variables = cse_variables.replace(exp,'config.'+exp[1:-1])
            
        cse_variables   = textwrap.indent(cse_variables,indent).lstrip()
        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
        
        coord_setter = self.write_coordinates_setter()
        velocities_setter = self.write_velocities_setter()
        initial_config_setter = self.write_initial_configuration_setter()
        
        eval_pos = self.write_pos_level_eq()
        eval_vel = self.write_vel_level_eq()
        eval_acc = self.write_acc_level_eq()
        eval_jac = self.write_jacobian()
        
        text = text.format(cse_variables = cse_variables,
                           cse_expressions = cse_expressions,
                           pos_rows = self.pos_level_rows,
                           pos_cols = self.pos_level_cols,
                           vel_rows = self.vel_level_rows,
                           vel_cols = self.vel_level_cols,
                           acc_rows = self.acc_level_rows,
                           acc_cols = self.acc_level_cols,
                           jac_rows = self.jacobian_rows,
                           jac_cols = self.jacobian_cols,
                           eval_pos = eval_pos,
                           eval_vel = eval_vel,
                           eval_acc = eval_acc,
                           eval_jac = eval_jac,
                           coord_setter = coord_setter,
                           velocities_setter = velocities_setter,
                           initial_config_setter = initial_config_setter)
        
        return text
    
    def write_pos_level_eq(self):
        
        text = '''
                def eval_pos_eq(self):
                    F = self.F
                    t = self.t

                    {cse_variables}

                    {pos_level_data}
                    
                    self.pos_level_rows_explicit = []
                    self.pos_level_cols_explicit = []
                    self.pos_level_data_explicit = []
                    
                    sparse_assembler(self.pos_level_data_blocks, self.pos_level_rows_blocks, self.pos_level_cols_blocks,
                                     self.pos_level_data_explicit, self.pos_level_rows_explicit, self.pos_level_cols_explicit)
                
                    self.pos_rhs = sc.sparse.coo_matrix(
                    (self.pos_level_data_explicit,
                    (self.pos_level_rows_explicit,self.pos_level_cols_explicit)),
                    (28,1))
                
                '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        pos_level_equations = self.assembly.pos_level_equations
        
        cse_symbols = sm.iterables.numbered_symbols('x')
        cse = sm.cse(pos_level_equations,symbols=cse_symbols)
        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])

        
        for exp in self.generalized_coordinates_lhs:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        for exp in self.configuration_constants:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        
        cse_variables = textwrap.indent(cse_variables,indent).lstrip() 
        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
        
        rows,cols,data = p._print(cse_expressions).split('\n')
        
        self.pos_level_rows = 'self.pos_level_rows_blocks = %s'%rows.lstrip()
        self.pos_level_cols = 'self.pos_level_cols_blocks = %s'%cols.lstrip()
        data = 'self.pos_level_data_blocks = %s'%data.lstrip()
        
        text = text.format(cse_variables  = cse_variables,
                           pos_level_data = data)
        
        return text
    
    def write_vel_level_eq(self):
        
        text = '''
                def eval_vel_eq(self):
                    F = self.F
                    t = self.t

                    {cse_variables}

                    {vel_level_data}
                    
                    self.vel_level_rows_explicit = []
                    self.vel_level_cols_explicit = []
                    self.vel_level_data_explicit = []
                    
                    sparse_assembler(self.vel_level_data_blocks, self.vel_level_rows_blocks, self.vel_level_cols_blocks,
                                     self.vel_level_data_explicit, self.vel_level_rows_explicit, self.vel_level_cols_explicit)
                
                    self.vel_rhs = sc.sparse.coo_matrix(
                    (self.vel_level_data_explicit,
                    (self.vel_level_rows_explicit,self.vel_level_cols_explicit)),
                    (28,1))
                
                '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        vel_level_equations = self.assembly.vel_level_equations
        
        cse_symbols = sm.iterables.numbered_symbols('x')
        cse = sm.cse(vel_level_equations,symbols=cse_symbols)
        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])

        
        for exp in self.generalized_coordinates_lhs:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        for exp in self.configuration_constants:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        
        cse_variables = textwrap.indent(cse_variables,indent).lstrip() 
        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
        
        rows,cols,data = p._print(cse_expressions).split('\n')
        
        self.vel_level_rows = 'self.vel_level_rows_blocks = %s'%rows.lstrip()
        self.vel_level_cols = 'self.vel_level_cols_blocks = %s'%cols.lstrip()
        data = 'self.vel_level_data_blocks = %s'%data.lstrip()
        
        text = text.format(cse_variables  = cse_variables,
                           vel_level_data = data)
        
        return text
    
    def write_jacobian(self):
        
        text = '''
                def eval_jacobian(self):

                    {cse_variables}

                    {jacobian_data}
                    
                    self.jacobian_rows_explicit = []
                    self.jacobian_cols_explicit = []
                    self.jacobian_data_explicit = []
                    
                    sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
                                     self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)
                
                    self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))
                
                '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        jacobian = self.assembly.jacobian
        
        cse_symbols = sm.iterables.numbered_symbols('j')
        cse = sm.cse(jacobian,symbols=cse_symbols)
        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])

        
        for exp in self.generalized_coordinates_lhs:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        for exp in self.configuration_constants:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        
        cse_variables = textwrap.indent(cse_variables,indent).lstrip()  
        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
        
        rows,cols,data = p._print(cse_expressions).split('\n')
        
        self.jacobian_rows = ('self.jacobian_rows_blocks = %s'%rows.lstrip())
        self.jacobian_cols = ('self.jacobian_cols_blocks = %s'%cols.lstrip())
        data = ('self.jacobian_data_blocks = %s'%data.lstrip())
        
        text = text.format(cse_variables = cse_variables,
                           jacobian_data = data)
        
        return text
    
    def write_acc_level_eq(self):
        
        text = '''
                def eval_acc_eq(self):
                    F = self.F
                    t = self.t

                    {cse_variables}

                    {acc_level_data}
                    
                    self.acc_level_rows_explicit = []
                    self.acc_level_cols_explicit = []
                    self.acc_level_data_explicit = []
                    
                    sparse_assembler(self.acc_level_data_blocks, self.acc_level_rows_blocks, self.acc_level_cols_blocks,
                                     self.acc_level_data_explicit, self.acc_level_rows_explicit, self.acc_level_cols_explicit)
                
                    self.acc_rhs = sc.sparse.coo_matrix(
                    (self.acc_level_data_explicit,
                    (self.acc_level_rows_explicit,self.acc_level_cols_explicit)),
                    (28,1))
                
                '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        acc_level_equations = self.assembly.acc_level_equations
        
        cse_symbols = sm.iterables.numbered_symbols('a')
        cse = sm.cse(acc_level_equations,symbols=cse_symbols)
        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])

        
        for exp in self.generalized_velocities_lhs+self.generalized_coordinates_lhs:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        for exp in self.configuration_constants:
            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
        
        cse_variables = textwrap.indent(cse_variables,indent).lstrip()  
        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
        
        rows,cols,data = p._print(cse_expressions).split('\n')
        
        self.acc_level_rows = 'self.acc_level_rows_blocks = %s'%rows.lstrip()
        self.acc_level_cols = 'self.acc_level_cols_blocks = %s'%cols.lstrip()
        data = 'self.acc_level_data_blocks = %s'%data.lstrip()
        
        text = text.format(cse_variables  = cse_variables,
                           acc_level_data = data)
        
        return text
    
    
    def dump_code(self):
        imports = self.write_imports()
        inputs  = self.write_config_inputs()
        assembly  = self.write_assembly_class()
        
        text = '\n'.join([imports,inputs,assembly])
        return text
        