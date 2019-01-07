# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:23:39 2019

@author: khale
"""
import sympy as sm
import textwrap
import re
import pandas as pd
import numpy as np
from source.code_generators.code_printers import numerical_printer

class abstract_generator(object):
    
    import source.symbolic_classes.abstract_matrices as temp
    temp.ccode_print = True
    temp.enclose = True

    def __init__(self,multibody_system,printer=numerical_printer()):
        
        self.mbs = multibody_system
        self.printer = printer
        
        self.numerical_arguments = num_args_sym = self.mbs.numerical_arguments.copy()
        self.configuration_constants = cfig_cons_sym = self.mbs.configuration_constants.copy()
        
        self.num_args_sym = [printer._print(e.lhs) for e in num_args_sym]
        self.cfig_cons_sym = [printer._print(e.lhs) for e in cfig_cons_sym]
                
        self.generalized_coordinates_equalities = self.mbs.q_maped.copy()
        self.generalized_velocities_equalities = self.mbs.qd_maped.copy()
        
        self.generalized_coordinates_lhs = [printer._print(exp.lhs) for exp in self.generalized_coordinates_equalities]
        self.generalized_velocities_lhs = [printer._print(exp.lhs) for exp in self.generalized_velocities_equalities]
    
    def create_config_dataframe(self):
        indecies = [i[1:-1] for i in self.num_args_sym]
        dataframe = pd.DataFrame(index=indecies,columns=range(0,4,1),dtype=np.float64)
        self.config_dataframe = dataframe
    
    def _generate_cse(self,equations,symbol):
        p = self.printer
        cse_symbols = sm.iterables.numbered_symbols(symbol)
        cse = sm.cse(equations,symbols=cse_symbols)
        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])
        return cse_variables, cse_expressions    
    
    def _setup_x_equations(self,xstring,sym='x'):
        p = self.printer
        equations = getattr(self.mbs,'%s_equations'%xstring)
        cse_variables, cse_expressions = self._generate_cse(equations,sym)
        rows,cols,data = p._print(cse_expressions).split('\n')
        setattr(self,'%s_eq_csev'%xstring,cse_variables)
        setattr(self,'%s_eq_rows'%xstring,rows)
        setattr(self,'%s_eq_cols'%xstring,cols)
        setattr(self,'%s_eq_data'%xstring,data)
    
    def setup_pos_equations(self):
        self._setup_x_equations('pos','x')
    
    def setup_vel_equations(self):
        self._setup_x_equations('vel','v')
    
    def setup_acc_equations(self):
        self._setup_x_equations('acc','a')
        
    def setup_jac_equations(self):
        self._setup_x_equations('jac','j')
    
    def setup_equations(self):
        self.setup_pos_equations()
        self.setup_vel_equations()
        self.setup_acc_equations()
        self.setup_jac_equations()

    
    def write_imports(self):
        pass
    def write_pos_equations(self):
        pass
    def write_vel_equations(self):
        pass
    def write_acc_equations(self):
        pass
    def write_jac_equations(self):
        pass

###############################################################################
###############################################################################

class python_code_generator(abstract_generator):
    
    def _insert_self(self,x): 
        return 'self.' + x.group(0)[1:-1]
    
    def _insert_config(self,x): 
        return 'config.' + x.group(0)[1:-1]
    
    def _write_x_setter(self,xstring):
        text = '''
                def set_%s(self,%s):
                    {maped}
               '''%(xstring,xstring)
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        coordinates = getattr(self.mbs,'%s_maped'%xstring)
        maped_coordinates = '\n'.join(['self.' + p._print(i) for i in coordinates])
        maped_coordinates = textwrap.indent(maped_coordinates,indent).lstrip()
        text = text.format(maped = maped_coordinates)
        text = textwrap.indent(text,indent)
        return text
    
    def write_coordinates_setter(self):
        return self._write_x_setter('q')
    
    def write_velocities_setter(self):
        return self._write_x_setter('qd')
    
    def _write_x_equations(self,xstring):
        text = '''
                def eval_%s_eq(self):

                    {cse_var_txt}

                    {cse_exp_txt}
                '''%xstring
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indent = 4*' '
        
        cse_var_txt = getattr(self,'%s_eq_csev'%xstring)
        cse_exp_txt = getattr(self,'%s_eq_data'%xstring)
        
        gen_coord_pattern = '|'.join(self.generalized_coordinates_lhs)
        cse_var_txt = re.sub(gen_coord_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(gen_coord_pattern,self._insert_self,cse_exp_txt)
        
        gen_veloc_pattern = '|'.join(self.generalized_velocities_lhs)
        cse_var_txt = re.sub(gen_veloc_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(gen_veloc_pattern,self._insert_self,cse_exp_txt)

        config_var_pattern = '|'.join(self.cfig_cons_sym)
        cse_var_txt = re.sub(config_var_pattern,self._insert_config,cse_var_txt)
        cse_exp_txt = re.sub(config_var_pattern,self._insert_config,cse_exp_txt)
        
        cse_var_txt = textwrap.indent(cse_var_txt,indent).lstrip() 
        cse_exp_txt = textwrap.indent(cse_exp_txt,indent).lstrip()
        
        cse_exp_txt = 'self.%s_eq_blocks = %s'%(xstring,cse_exp_txt.lstrip())
        
        text = text.format(cse_var_txt = cse_var_txt,
                           cse_exp_txt = cse_exp_txt)
        
        text = textwrap.indent(text,indent)
        return text
    
    def write_pos_equations(self):
        return self._write_x_equations('pos')
    
    def write_vel_equations(self):
        return self._write_x_equations('vel')
    
    def write_acc_equations(self):
        return self._write_x_equations('acc')
    
    def write_jac_equations(self):
        return self._write_x_equations('jac')
    
    def write_imports(self):
        text = '''
                import numpy as np
                import scipy as sc
                from numpy.linalg import multi_dot
                from source.cython_definitions.matrix_funcs import A, B, triad as Triad
                from scipy.misc import derivative
                from numpy import cos, sin
                from source.solvers.python_solver import solver
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
    
    def write_class_init(self):
        text = '''
                class numerical_assembly(object):

                    def __init__(self,config):                    
                        self.config = config
                        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
                        
                        self.pos_rows = {pos_rows}
                        self.pos_rows = {pos_cols}
                        
                        self.vel_rows = {vel_rows}
                        self.vel_rows = {vel_cols}
                        
                        self.acc_rows = {acc_rows}
                        self.acc_rows = {acc_cols}
                        
                        self.jac_rows = {jac_rows}
                        self.jac_rows = {jac_cols}
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
                                        
        text = text.format(pos_rows = self.pos_eq_rows,
                           pos_cols = self.pos_eq_cols,
                           vel_rows = self.vel_eq_rows,
                           vel_cols = self.vel_eq_cols,
                           acc_rows = self.acc_eq_rows,
                           acc_cols = self.acc_eq_cols,
                           jac_rows = self.jac_eq_rows,
                           jac_cols = self.jac_eq_cols)
        return text
                
    
    def write_system_class(self):
        text = '''
                {class_init}
                    {coord_setter}
                    {veloc_setter}
                    {eval_pos}
                    {eval_vel}
                    {eval_acc}
                    {eval_jac}  
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        class_init = self.write_class_init()
        
        coord_setter = self.write_coordinates_setter()
        veloc_setter = self.write_velocities_setter()
        
        eval_pos = self.write_pos_equations()
        eval_vel = self.write_vel_equations()
        eval_acc = self.write_acc_equations()
        eval_jac = self.write_jac_equations()
        
        text = text.format(class_init = class_init,
                           eval_pos = eval_pos,
                           eval_vel = eval_vel,
                           eval_acc = eval_acc,
                           eval_jac = eval_jac,
                           coord_setter = coord_setter,
                           veloc_setter = veloc_setter)
        return text
    
    def write_code_file(self):
        self.setup_equations()
        imports = self.write_imports()
        system_class  = self.write_system_class()
        
        text = '\n'.join([imports,system_class])
        with open('%s.py'%self.mbs.name,'w') as file:
            file.write(text)
    
#    def write_config_class(self):
#        text = '''
#                class inputs(object):
#
#                    def __init__(self,config_file):
#                        {inputs}
#                '''
#        
#        p = self.printer
#        indent = 8*' '
#        
#        inputs = self.numerical_arguments
#        consts = self.configuration_constants
#        
#        inputs_equalities = [p._print(exp) for exp in inputs]
#        inputs_variables  = [p._print(exp.lhs) for exp in inputs]
#        declared_inputs  = '\n'.join([typing_prefix + i for i in self.inputs_equalities])
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        declared_inputs = textwrap.indent(declared_inputs,indent).lstrip()
#        text = text.format(declared_inputs = declared_inputs)
#        
#        return text
    
    
    
    
    
    
    
#    
#    
#    
#    
#    def write_imports(self):
#        text = '''
#                import numpy as np
#                import scipy as sc
#                from numpy.linalg import multi_dot
#                from matrix_funcs import A, B, sparse_assembler, triad as Triad
#                from scipy.misc import derivative
#                from numpy import cos, sin
#                '''
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        return text
#    
#    def write_config_inputs(self):
#        text = '''
#                class inputs(object):
#
#                    def __init__(self):
#                        {declared_inputs}
#                '''
#        
#        p = self.printer
#        indent = 8*' '
#        typing_prefix = 'self.'
#        
#        inputs = self.assembly.numerical_arguments
#        
#        self.inputs_equalities = [p._print(exp) for exp in inputs]
#        self.inputs_variables  = [p._print(exp.lhs) for exp in inputs]
#        declared_inputs  = '\n'.join([typing_prefix + i for i in self.inputs_equalities])
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        declared_inputs = textwrap.indent(declared_inputs,indent).lstrip()
#        text = text.format(declared_inputs = declared_inputs)
#        
#        return text
#    
#    def write_coordinates_setter(self):
#        text = '''
#                def set_coordinates(self,q):
#                    {maped}
#                    
#               '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 4*' '
#        
#        maped_coordinates = '\n'.join(['self.' + p._print(i) for i in self.assembly.q_maped])
#        maped_coordinates = textwrap.indent(maped_coordinates,indent).lstrip()
#        text = text.format(maped = maped_coordinates)
#        return text
#    
#    def write_velocities_setter(self):
#        text = '''
#                def set_velocities(self,qd):
#                    {maped}
#                    
#               '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 4*' '
#        
#        maped_coordinates = '\n'.join(['self.' + p._print(i) for i in self.assembly.qd_maped])
#        maped_coordinates = textwrap.indent(maped_coordinates,indent).lstrip()
#        text = text.format(maped = maped_coordinates)
#        return text
#    
#    def write_initial_configuration_setter(self):
#        text = '''
#                def set_initial_configuration(self):
#                    config = self.config
#                    
#                    q = np.concatenate([{coordinates}])
#                    
#                    qd = np.concatenate([{velocities}])
#                    
#                    self.set_coordinates(q)
#                    self.set_velocities(qd)
#                    self.q_initial = q.copy()
#                    
#               '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        indent = 4*' '
#        
#        coordinates = ',\n'.join(self.generalized_coordinates_lhs)
#        for exp in self.generalized_coordinates_lhs:
#            coordinates = coordinates.replace(exp,'config.'+exp[1:-1])
#        coordinates = textwrap.indent(coordinates,indent).lstrip()
#        
#        velocities = ',\n'.join(self.generalized_velocities_lhs)
#        for exp in self.generalized_velocities_lhs:
#            velocities = velocities.replace(exp,'config.'+exp[1:-1])
#        velocities = textwrap.indent(velocities,indent).lstrip()
#        
#        text = text.format(coordinates = coordinates,
#                           velocities  = velocities)
#        return text
#    
#    def write_assembly_class(self):
#        text = '''
#                class numerical_assembly(object):
#
#                    def __init__(self,config):
#                        self.F = config.F
#                        self.t = 0
#                    
#                        self.config = config
#                        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
#                        
#                        {pos_rows}
#                        {pos_cols}
#                        
#                        {vel_rows}
#                        {vel_cols}
#                        
#                        {acc_rows}
#                        {acc_cols}
#                        
#                        {jac_rows}
#                        {jac_cols}
#                    
#                    def eval_constants(self):
#                        config = self.config
#                        
#                        {cse_variables}
#
#                        {cse_expressions}
#                    
#                    {coord_setter}
#                    {velocities_setter}
#                    {initial_config_setter}
#                    
#                    {eval_pos}
#                    
#                    {eval_vel}
#                    
#                    {eval_acc}
#                    
#                    {eval_jac}
#                    
#                '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 8*' '
#        
#        configuration_constants = self.assembly.configuration_constants
#        
#        constants_cse = sm.cse(configuration_constants,symbols=sm.iterables.numbered_symbols('c'))
#        cse_variables = '\n'.join([p._print(exp) for exp in constants_cse[0]])
#        cse_expressions  = '\n'.join(['self.' + p._print(exp) for exp in constants_cse[1]])
#
#        for exp in self.numerical_arguments:
#            cse_variables = cse_variables.replace(exp,'config.'+exp[1:-1])
#            
#        cse_variables   = textwrap.indent(cse_variables,indent).lstrip()
#        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
#        
#        coord_setter = self.write_coordinates_setter()
#        velocities_setter = self.write_velocities_setter()
#        initial_config_setter = self.write_initial_configuration_setter()
#        
#        eval_pos = self.write_pos_level_eq()
#        eval_vel = self.write_vel_level_eq()
#        eval_acc = self.write_acc_level_eq()
#        eval_jac = self.write_jacobian()
#        
#        text = text.format(cse_variables = cse_variables,
#                           cse_expressions = cse_expressions,
#                           pos_rows = self.pos_level_rows,
#                           pos_cols = self.pos_level_cols,
#                           vel_rows = self.vel_level_rows,
#                           vel_cols = self.vel_level_cols,
#                           acc_rows = self.acc_level_rows,
#                           acc_cols = self.acc_level_cols,
#                           jac_rows = self.jacobian_rows,
#                           jac_cols = self.jacobian_cols,
#                           eval_pos = eval_pos,
#                           eval_vel = eval_vel,
#                           eval_acc = eval_acc,
#                           eval_jac = eval_jac,
#                           coord_setter = coord_setter,
#                           velocities_setter = velocities_setter,
#                           initial_config_setter = initial_config_setter)
#        
#        return text
#    
#    def write_pos_level_eq(self):
#        
#        text = '''
#                def eval_pos_eq(self):
#                    F = self.F
#                    t = self.t
#
#                    {cse_variables}
#
#                    {pos_level_data}
#                    
#                    self.pos_level_rows_explicit = []
#                    self.pos_level_cols_explicit = []
#                    self.pos_level_data_explicit = []
#                    
#                    sparse_assembler(self.pos_level_data_blocks, self.pos_level_rows_blocks, self.pos_level_cols_blocks,
#                                     self.pos_level_data_explicit, self.pos_level_rows_explicit, self.pos_level_cols_explicit)
#                
#                    self.pos_rhs = sc.sparse.coo_matrix(
#                    (self.pos_level_data_explicit,
#                    (self.pos_level_rows_explicit,self.pos_level_cols_explicit)),
#                    (28,1))
#                
#                '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 4*' '
#        
#        pos_level_equations = self.assembly.pos_level_equations
#        
#        cse_symbols = sm.iterables.numbered_symbols('x')
#        cse = sm.cse(pos_level_equations,symbols=cse_symbols)
#        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
#        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])
#
#        
#        for exp in self.generalized_coordinates_lhs:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        for exp in self.configuration_constants:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        
#        cse_variables = textwrap.indent(cse_variables,indent).lstrip() 
#        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
#        
#        rows,cols,data = p._print(cse_expressions).split('\n')
#        
#        self.pos_level_rows = 'self.pos_level_rows_blocks = %s'%rows.lstrip()
#        self.pos_level_cols = 'self.pos_level_cols_blocks = %s'%cols.lstrip()
#        data = 'self.pos_level_data_blocks = %s'%data.lstrip()
#        
#        text = text.format(cse_variables  = cse_variables,
#                           pos_level_data = data)
#        
#        return text
#    
#    def write_vel_level_eq(self):
#        
#        text = '''
#                def eval_vel_eq(self):
#                    F = self.F
#                    t = self.t
#
#                    {cse_variables}
#
#                    {vel_level_data}
#                    
#                    self.vel_level_rows_explicit = []
#                    self.vel_level_cols_explicit = []
#                    self.vel_level_data_explicit = []
#                    
#                    sparse_assembler(self.vel_level_data_blocks, self.vel_level_rows_blocks, self.vel_level_cols_blocks,
#                                     self.vel_level_data_explicit, self.vel_level_rows_explicit, self.vel_level_cols_explicit)
#                
#                    self.vel_rhs = sc.sparse.coo_matrix(
#                    (self.vel_level_data_explicit,
#                    (self.vel_level_rows_explicit,self.vel_level_cols_explicit)),
#                    (28,1))
#                
#                '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 4*' '
#        
#        vel_level_equations = self.assembly.vel_level_equations
#        
#        cse_symbols = sm.iterables.numbered_symbols('x')
#        cse = sm.cse(vel_level_equations,symbols=cse_symbols)
#        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
#        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])
#
#        
#        for exp in self.generalized_coordinates_lhs:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        for exp in self.configuration_constants:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        
#        cse_variables = textwrap.indent(cse_variables,indent).lstrip() 
#        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
#        
#        rows,cols,data = p._print(cse_expressions).split('\n')
#        
#        self.vel_level_rows = 'self.vel_level_rows_blocks = %s'%rows.lstrip()
#        self.vel_level_cols = 'self.vel_level_cols_blocks = %s'%cols.lstrip()
#        data = 'self.vel_level_data_blocks = %s'%data.lstrip()
#        
#        text = text.format(cse_variables  = cse_variables,
#                           vel_level_data = data)
#        
#        return text
#    
#    def write_jacobian(self):
#        
#        text = '''
#                def eval_jacobian(self):
#
#                    {cse_variables}
#
#                    {jacobian_data}
#                    
#                    self.jacobian_rows_explicit = []
#                    self.jacobian_cols_explicit = []
#                    self.jacobian_data_explicit = []
#                    
#                    sparse_assembler(self.jacobian_data_blocks, self.jacobian_rows_blocks, self.jacobian_cols_blocks,
#                                     self.jacobian_data_explicit, self.jacobian_rows_explicit, self.jacobian_cols_explicit)
#                
#                    self.jacobian = sc.sparse.coo_matrix((self.jacobian_data_explicit,(self.jacobian_rows_explicit,self.jacobian_cols_explicit)))
#                
#                '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 4*' '
#        
#        jacobian = self.assembly.jacobian
#        
#        cse_symbols = sm.iterables.numbered_symbols('j')
#        cse = sm.cse(jacobian,symbols=cse_symbols)
#        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
#        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])
#
#        
#        for exp in self.generalized_coordinates_lhs:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        for exp in self.configuration_constants:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        
#        cse_variables = textwrap.indent(cse_variables,indent).lstrip()  
#        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
#        
#        rows,cols,data = p._print(cse_expressions).split('\n')
#        
#        self.jacobian_rows = ('self.jacobian_rows_blocks = %s'%rows.lstrip())
#        self.jacobian_cols = ('self.jacobian_cols_blocks = %s'%cols.lstrip())
#        data = ('self.jacobian_data_blocks = %s'%data.lstrip())
#        
#        text = text.format(cse_variables = cse_variables,
#                           jacobian_data = data)
#        
#        return text
#    
#    def write_acc_level_eq(self):
#        
#        text = '''
#                def eval_acc_eq(self):
#                    F = self.F
#                    t = self.t
#
#                    {cse_variables}
#
#                    {acc_level_data}
#                    
#                    self.acc_level_rows_explicit = []
#                    self.acc_level_cols_explicit = []
#                    self.acc_level_data_explicit = []
#                    
#                    sparse_assembler(self.acc_level_data_blocks, self.acc_level_rows_blocks, self.acc_level_cols_blocks,
#                                     self.acc_level_data_explicit, self.acc_level_rows_explicit, self.acc_level_cols_explicit)
#                
#                    self.acc_rhs = sc.sparse.coo_matrix(
#                    (self.acc_level_data_explicit,
#                    (self.acc_level_rows_explicit,self.acc_level_cols_explicit)),
#                    (28,1))
#                
#                '''
#        
#        text = text.expandtabs()
#        text = textwrap.dedent(text)
#        
#        p = self.printer
#        indent = 4*' '
#        
#        acc_level_equations = self.assembly.acc_level_equations
#        
#        cse_symbols = sm.iterables.numbered_symbols('a')
#        cse = sm.cse(acc_level_equations,symbols=cse_symbols)
#        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
#        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])
#
#        
#        for exp in self.generalized_velocities_lhs+self.generalized_coordinates_lhs:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        for exp in self.configuration_constants:
#            cse_variables = cse_variables.replace(exp,'self.'+exp[1:-1])
#            cse_expressions = cse_expressions.replace(exp,'self.'+exp[1:-1])
#        
#        cse_variables = textwrap.indent(cse_variables,indent).lstrip()  
#        cse_expressions = textwrap.indent(cse_expressions,indent).lstrip()
#        
#        rows,cols,data = p._print(cse_expressions).split('\n')
#        
#        self.acc_level_rows = 'self.acc_level_rows_blocks = %s'%rows.lstrip()
#        self.acc_level_cols = 'self.acc_level_cols_blocks = %s'%cols.lstrip()
#        data = 'self.acc_level_data_blocks = %s'%data.lstrip()
#        
#        text = text.format(cse_variables  = cse_variables,
#                           acc_level_data = data)
#        
#        return text
#    
#    
#    def dump_code(self):
#        imports = self.write_imports()
#        inputs  = self.write_config_inputs()
#        assembly  = self.write_assembly_class()
#        
#        text = '\n'.join([imports,inputs,assembly])
#        return text
        