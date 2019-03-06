# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:23:39 2019

@author: khale
"""
import os
import sympy as sm
import textwrap
import re
import pandas as pd
import numpy as np
import itertools
from source.code_generators.code_printers import numerical_printer

class abstract_generator(object):
    
    def __init__(self,multibody_system,printer=numerical_printer()):
        
        self.mbs     = multibody_system
        self.config  = self.mbs.param_config
        self.printer = printer
        
        self.primary_arguments = self.config.primary_arguments
        
        self.arguments_symbols = [printer._print(exp) for exp in self.mbs.arguments_symbols]
        self.constants_symbols = [printer._print(exp) for exp in self.mbs.constants_symbols]
        self.runtime_symbols   = [printer._print(exp) for exp in self.mbs.runtime_symbols]
        
        self.constants_symbolic_expr = self.mbs.constants_symbolic_expr
        self.constants_numeric_expr  = self.mbs.constants_numeric_expr
        
        
        self.gen_coordinates_exp = self.mbs.mapped_gen_coordinates
        self.gen_coordinates_sym = [printer._print(exp.lhs) for exp in self.gen_coordinates_exp]

        self.gen_velocities_exp  = self.mbs.mapped_gen_velocities
        self.gen_velocities_sym  = [printer._print(exp.lhs) for exp in self.gen_velocities_exp]
        
        self.gen_accelerations_exp  = self.mbs.mapped_gen_accelerations
        self.gen_accelerations_sym  = [printer._print(exp.lhs) for exp in self.gen_accelerations_exp]
        
        self.lagrange_multipliers_exp  = self.mbs.mapped_lagrange_multipliers
        self.lagrange_multipliers_sym  = [printer._print(exp.lhs) for exp in self.lagrange_multipliers_exp]

        self.joint_reactions_sym = [printer._print(exp) for exp in self.mbs.reactions_symbols]

        self.virtual_coordinates = [printer._print(exp) for exp in self.mbs.virtual_coordinates]
        
        self.config_vars = [printer._print(i) for i in self.mbs.param_config.arguments_symbols]
        self.input_args  = self.config.input_equalities
        self.output_args = self.config.output_equalities
        
        self.bodies = self.mbs.bodies
        self.jac_cols = self.mbs.jac_cols
        
    
    def setup_equations(self):
        self._setup_pos_equations()
        self._setup_vel_equations()
        self._setup_acc_equations()
        self._setup_jac_equations()
        self._setup_frc_equations()
        self._setup_mass_equations()
    
    def _setup_pos_equations(self):
        self._setup_x_equations('pos','x')
    
    def _setup_vel_equations(self):
        self._setup_x_equations('vel','v')
    
    def _setup_acc_equations(self):
        self._setup_x_equations('acc','a')
        
    def _setup_jac_equations(self):
        self._setup_x_equations('jac','j')
        scols = ','.join(['self.%s'%i for i in self.jac_cols])
        self.jac_eq_cols = 'np.array([%s])'%scols
    
    def _setup_frc_equations(self):
        self._setup_x_equations('frc','f')
        
    def _setup_mass_equations(self):
        self._setup_x_equations('mass','m')


    def _setup_x_equations(self,xstring,sym='x'):
        p = self.printer
        equations = getattr(self.mbs,'%s_equations'%xstring)
        cse_variables, cse_expressions = self._generate_cse(equations,sym)
        rows,cols,data = p._print(cse_expressions).split('\n')
        setattr(self,'%s_eq_csev'%xstring,cse_variables)
        setattr(self,'%s_eq_rows'%xstring,rows)
        setattr(self,'%s_eq_cols'%xstring,cols)
        setattr(self,'%s_eq_data'%xstring,data)
    
    def _generate_cse(self,equations,symbol):
        p = self.printer
        cse_symbols = sm.iterables.numbered_symbols(symbol)
        cse = sm.cse(equations,symbols=cse_symbols,ignore=(sm.S('t'),))
        cse_variables   = '\n'.join([p._print(exp) for exp in cse[0]])
        cse_expressions = '\n'.join([p._print(exp) for exp in cse[1]])
        return cse_variables, cse_expressions

    
    def _create_inputs_dataframe(self):
        indecies = [str(i.lhs) for i in self.input_args if isinstance(i.lhs,sm.MatrixSymbol)]
        indecies.sort()
        shape = (len(indecies),4)
        dataframe = pd.DataFrame(np.zeros(shape),index=indecies,dtype=np.float64)
        return dataframe

    @staticmethod
    def _insert_string(string):
        def inserter(x): return string + x.group(0)
        return inserter

###############################################################################
###############################################################################

class configuration_code_generator(abstract_generator):
        
    def __init__(self,config,printer=numerical_printer()):
        
        self.config  = config
        self.printer = printer
        self.name = self.config.name
                
        self.config_vars = [printer._print(i) for i in self.config.arguments_symbols]
        self.input_args  = self.config.input_equalities
        self.output_args = self.config.output_equalities
        
        self.gen_coordinates_sym = [printer._print(exp.lhs) 
        for exp in self.config.topology.mapped_gen_coordinates]
        self.gen_velocities_sym  = [printer._print(exp.lhs) 
        for exp in self.config.topology.mapped_gen_velocities]
        
    def write_imports(self):
        text = '''
                import os
                import numpy as np
                import pandas as pd
                from source.solvers.py_numerical_functions import (mirrored, centered, oriented, 
                                                                   cylinder_geometry,
                                                                   composite_geometry,
                                                                   triangular_prism)
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
        
    def write_class_init(self):
        text = '''
                path = os.path.dirname(__file__)
                
                class configuration(object):

                    def __init__(self):
                        {inputs}                       
                '''
        p = self.printer
        indent = 8*' '
        inputs  = self.input_args
        
        pattern = '|'.join(self.config_vars)
        self_inserter = self._insert_string('self.')
        
        inputs = '\n'.join([p._print(exp) for exp in inputs])
        inputs = re.sub(pattern,self_inserter,inputs)
        inputs = textwrap.indent(inputs,indent).lstrip()
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        text = text.format(inputs = inputs)
        return text
    
    def write_helpers(self):
        text = '''
                @property
                def q(self):
                    q = {coordinates}
                    return q
                    
                @property
                def qd(self):
                    qd = {velocities}
                    return qd
                                        
                def load_from_csv(self,csv_file):
                    file_path = os.path.join(path,csv_file)
                    dataframe = pd.read_csv(file_path,index_col=0)
                    for ind in dataframe.index:
                        shape = getattr(self,ind).shape
                        v = np.array(dataframe.loc[ind],dtype=np.float64)
                        v = np.resize(v,shape)
                        setattr(self,ind,v)
                    self._set_arguments()
                
                def _set_arguments(self):
                    {outputs}
                    {pass_text}
                '''
        p = self.printer
        indent = 4*' '
        
        outputs = self.output_args
        
        pattern = '|'.join(self.config_vars)
        self_inserter = self._insert_string('self.')
                
        outputs = '\n'.join([p._print(exp) for exp in outputs])
        outputs = re.sub(pattern,self_inserter,outputs)
        outputs = textwrap.indent(outputs,indent).lstrip()
        
        pass_text = ('pass' if len(outputs)==0 else '')
        
        coordinates = ','.join(self.gen_coordinates_sym)
        coordinates = re.sub(pattern,self_inserter,coordinates)
        coordinates = ('np.concatenate([%s])'%coordinates if len(coordinates)!=0 else '[]')
        coordinates = textwrap.indent(coordinates,indent).lstrip()
        
        velocities = ','.join(self.gen_velocities_sym)
        velocities = re.sub(pattern,self_inserter,velocities)
        velocities = ('np.concatenate([%s])'%velocities if len(velocities)!=0 else '[]')
        velocities = textwrap.indent(velocities,indent).lstrip()
        
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(outputs = outputs,
                           pass_text = pass_text,
                           coordinates = coordinates,
                           velocities = velocities)
        text = textwrap.indent(text,indent)
        return text
    
    def write_system_class(self):
        text = '''
                {class_init}
                    {class_helpers}
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        class_init = self.write_class_init()
        class_helpers = self.write_helpers()
                
        text = text.format(class_init = class_init,
                           class_helpers = class_helpers)
        return text
        
    def write_code_file(self):
        os.chdir('..\..')
        path = os.getcwd() + '\\generated_templates\\configurations'
        os.chdir(path)
        
        imports = self.write_imports()
        config_class = self.write_system_class()
        text = '\n'.join([imports,config_class])
        with open('%s.py'%self.name,'w') as file:
            file.write(text)
        
        inputs_dataframe = self._create_inputs_dataframe()
        inputs_dataframe.to_csv('%s.csv'%self.name)

        
###############################################################################
###############################################################################

class template_code_generator(abstract_generator):
    
        
    def write_imports(self):
        text = '''
                import os
                import numpy as np
                import pandas as pd
                from scipy.misc import derivative
                from numpy import cos, sin
                from numpy.linalg import multi_dot
                from source.cython_definitions.matrix_funcs import A, B, G, E, triad, skew_matrix as skew
                from source.solvers.py_numerical_functions import mirrored
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
    
    def write_config_class(self):
        config_code_gen = configuration_code_generator(self.config)
        text = config_code_gen.write_system_class()
        return text

    
    def write_class_init(self):
        text = '''
                class topology(object):

                    def __init__(self,prefix='',cfg=None):
                        self.t = 0.0
                        self.config = (configuration() if cfg is None else cfg)
                        self.prefix = (prefix if prefix=='' else prefix+'.')
                                                
                        self.n = {n}
                        self.nrows = {nve}
                        self.ncols = 2*{nodes}
                        self.rows = np.arange(self.nrows)
                        
                        self.jac_rows = {jac_rows}                        
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        text = text.format(rows = self.pos_eq_rows,
                           jac_rows = self.jac_eq_rows,
                           jac_cols = self.jac_eq_cols,
                           nve = self.mbs.nve,
                           n = self.mbs.n,
                           nodes = len(self.mbs.nodes))
        return text

    def write_template_assembler(self):
        text = '''
                def _set_mapping(self,indicies_map,interface_map):
                    p = self.prefix
                    {maped}
                    {virtuals}
                
                def assemble_template(self,indicies_map,interface_map,rows_offset):
                    self.rows_offset = rows_offset
                    self._set_mapping(indicies_map,interface_map)
                    self.rows += self.rows_offset
                    self.jac_rows += self.rows_offset
                    self.jac_cols = {jac_cols}
                
                def set_initial_states(self):
                    self.set_gen_coordinates(self.config.q)
                    self.set_gen_velocities(self.config.qd)
               '''
        
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        nodes = '\n'.join(['%s = indicies_map[p+%r]'%('self.%s'%i,i) for i in self.bodies])
        nodes = textwrap.indent(nodes,indent).lstrip()
        
        virtuals = '\n'.join(['%s = indicies_map[interface_map[p+%r]]'%('self.%s'%i,i) for i in self.mbs.virtual_bodies])
        virtuals = textwrap.indent(virtuals,indent).lstrip()
        
        text = text.format(maped = nodes,
                           virtuals = virtuals,
                           jac_cols = self.jac_eq_cols)
        text = textwrap.indent(text,indent)
        return text
    
    def write_constants_eval(self):
        text = '''
                def eval_constants(self):
                    config = self.config
                    
                    {num_constants}
                    
                    {sym_constants}
                '''
        indent = 4*' '
        p = self.printer
                
        
        config_pattern_iter = itertools.chain(self.arguments_symbols,
                                              self.virtual_coordinates)
        config_pattern = '|'.join(config_pattern_iter)
        config_inserter = self._insert_string('config.')
        
        self_pattern_iter = itertools.chain(self.constants_symbols)
        self_pattern = '|'.join(self_pattern_iter)
        self_inserter = self._insert_string('self.')
        
        sym_consts = self.constants_symbolic_expr
        cse_var_txt, cse_exp_txt = self._generate_cse(sym_consts,'c')
        cse_var_txt = re.sub(config_pattern,config_inserter,cse_var_txt)
        cse_exp_txt = re.sub(config_pattern,config_inserter,cse_exp_txt)
        cse_var_txt = re.sub(self_pattern,self_inserter,cse_var_txt)
        cse_exp_txt = re.sub(self_pattern,self_inserter,cse_exp_txt)
        sym_constants = '\n'.join([cse_var_txt,'',cse_exp_txt])
        sym_constants = textwrap.indent(sym_constants,indent).lstrip()
        
        num_constants_list = self.constants_numeric_expr
        num_constants_text = '\n'.join((p._print(i) for i in num_constants_list))
        num_constants_text = re.sub(config_pattern,config_inserter,num_constants_text)
        num_constants_text = re.sub(self_pattern,self_inserter,num_constants_text)
        num_constants_text = textwrap.indent(num_constants_text,indent).lstrip()

        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(sym_constants = sym_constants,
                           num_constants = num_constants_text)
        text = textwrap.indent(text,indent)
        return text

    def write_coordinates_setter(self):
        return self._write_x_setter('gen_coordinates','q')
    
    def write_velocities_setter(self):
        return self._write_x_setter('gen_velocities','qd')
    
    def write_accelerations_setter(self):
        return self._write_x_setter('gen_accelerations','qdd')
    
    def write_lagrange_setter(self):
        return self._write_x_setter('lagrange_multipliers','Lambda')
        
    def write_pos_equations(self):
        return self._write_x_equations('pos')
    
    def write_vel_equations(self):
        return self._write_x_equations('vel')
    
    def write_acc_equations(self):
        return self._write_x_equations('acc')
    
    def write_jac_equations(self):
        return self._write_x_equations('jac')
    
    def write_forces_equations(self):
        return self._write_x_equations('frc')
    
    def write_mass_equations(self):
        return self._write_x_equations('mass')
    
    def write_reactions_equations(self):
        text = '''
                def eval_reactions_equations(self):

                    {equations_text}
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indent = 4*' '
        p = self.printer
        
        equations = self.mbs.reactions_equalities
        equations_text = '\n'.join([p._print(expr) for expr in equations])
        
        self_pattern = itertools.chain(self.runtime_symbols,
                                       self.constants_symbols,
                                       self.joint_reactions_sym,
                                       self.lagrange_multipliers_sym)
        self_pattern = '|'.join(self_pattern)
        self_inserter = self._insert_string('self.')
        equations_text = re.sub(self_pattern,self_inserter,equations_text)
                
        equations_text = textwrap.indent(equations_text,indent).lstrip() 
        
#        cse_exp_txt = 'self.%s_eq_blocks = %s'%(xstring,cse_exp_txt.lstrip())
        
        text = text.format(equations_text = equations_text)
        text = textwrap.indent(text,indent)
        return text


    def write_system_class(self):
        text = '''
                {class_init}
                    {assembler}
                    {constants}
                    {coord_setter}
                    {veloc_setter}
                    {accel_setter}
                    {lagrg_setter}
                    {eval_pos}
                    {eval_vel}
                    {eval_acc}
                    {eval_jac}
                    {eval_mass}
                    {eval_frc}
                    {eval_rct}
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        class_init = self.write_class_init()
        assembler  = self.write_template_assembler()
        constants  = self.write_constants_eval()
        coord_setter = self.write_coordinates_setter()
        veloc_setter = self.write_velocities_setter()
        accel_setter = self.write_accelerations_setter()
        lagrg_setter = self.write_lagrange_setter()
        
        eval_pos = self.write_pos_equations()
        eval_vel = self.write_vel_equations()
        eval_acc = self.write_acc_equations()
        eval_jac = self.write_jac_equations()
        eval_frc = self.write_forces_equations()
        eval_mass = self.write_mass_equations()
        eval_rct = self.write_reactions_equations()
        
        text = text.format(class_init = class_init,
                           assembler = assembler,
                           constants = constants,
                           eval_pos = eval_pos,
                           eval_vel = eval_vel,
                           eval_acc = eval_acc,
                           eval_jac = eval_jac,
                           eval_frc = eval_frc,
                           eval_rct = eval_rct,
                           eval_mass = eval_mass,
                           coord_setter = coord_setter,
                           veloc_setter = veloc_setter,
                           accel_setter = accel_setter,
                           lagrg_setter = lagrg_setter)
        return text
    
    
    def write_code_file(self):
        os.chdir('..\..')
        path = os.getcwd() + '\\generated_templates\\templates'
        os.chdir(path)
        
        self.setup_equations()
        imports = self.write_imports()
        base_cfg = self.write_config_class()
        system_class = self.write_system_class()
        text = '\n'.join([imports,base_cfg,system_class])
        with open('%s.py'%self.mbs.name,'w') as file:
            file.write(text)
        
        inputs_dataframe = self._create_inputs_dataframe()
        inputs_dataframe.to_csv('%s_base_cfg.csv'%self.mbs.name)
        
    ###########################################################################
    ###########################################################################

    def _write_x_setter(self,func_name,var='q'):
        text = '''
                def set_%s(self,%s):
                    {equalities}
               '''%(func_name,var)
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        symbolic_equality  = getattr(self,'%s_exp'%func_name)
        
        if len(symbolic_equality) !=0:
            pattern = '|'.join(getattr(self,'%s_sym'%func_name))
            self_inserter = self._insert_string('self.')
            
            numerical_equality = '\n'.join([p._print(i) for i in symbolic_equality])
            numerical_equality = re.sub(pattern,self_inserter,numerical_equality)
            numerical_equality = textwrap.indent(numerical_equality,indent).lstrip()
        else:
            numerical_equality = 'pass'
        
        text = text.format(equalities = numerical_equality)
        text = textwrap.indent(text,indent)
        return text
    
    def _write_x_equations(self,xstring):
        text = '''
                def eval_%s_eq(self):
                    config = self.config
                    t = self.t

                    {cse_var_txt}

                    {cse_exp_txt}
                '''%xstring
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indent = 4*' '
        
        cse_var_txt = getattr(self,'%s_eq_csev'%xstring)
        cse_exp_txt = getattr(self,'%s_eq_data'%xstring)
        
        self_pattern = itertools.chain(self.runtime_symbols,
                                       self.constants_symbols)
        self_pattern = '|'.join(self_pattern)
        self_inserter = self._insert_string('self.')
        cse_var_txt = re.sub(self_pattern,self_inserter,cse_var_txt)
        cse_exp_txt = re.sub(self_pattern,self_inserter,cse_exp_txt)
        
        config_pattern = self.primary_arguments - set(self.runtime_symbols)  #\w*(?<!.)
        config_pattern = '|'.join([r'%s'%i for i in config_pattern])
        config_inserter = self._insert_string('config.')
        cse_var_txt = re.sub(config_pattern,config_inserter,cse_var_txt)
        cse_exp_txt = re.sub(config_pattern,config_inserter,cse_exp_txt)
        
        cse_var_txt = textwrap.indent(cse_var_txt,indent).lstrip() 
        cse_exp_txt = textwrap.indent(cse_exp_txt,indent).lstrip()
        
        cse_exp_txt = 'self.%s_eq_blocks = %s'%(xstring,cse_exp_txt.lstrip())
        
        text = text.format(cse_var_txt = cse_var_txt,
                           cse_exp_txt = cse_exp_txt)
        text = textwrap.indent(text,indent)
        return text
    

###############################################################################
###############################################################################

class assembly_code_generator(template_code_generator):
    
    def __init__(self,multibody_system,printer=numerical_printer()):
        self.mbs  = multibody_system
        self.printer = printer
        self.name = self.mbs.name
        self.templates = []
        self.configs = []
        self.subsystems_templates = {}
        for name,obj in self.mbs.subsystems.items():
            topology_name = obj.topology.name
            cfg_file_name = obj.topology.cfg_file
            self.subsystems_templates[name] = (topology_name,cfg_file_name)
            if topology_name not in self.templates:
                self.templates.append(topology_name)
                self.configs.append(cfg_file_name)
    
    
    def write_imports(self):
        text = '''
                import numpy as np
                
                {configs_imports}
                
                {templates_imports}
                {subsystems}
                '''
        tpl_import_prefix = 'from use_cases.generated_templates.templates '
        templates_imports = '\n'.join(['%simport %s'%(tpl_import_prefix,i)
                                        for i in self.templates])
        cfg_import_prefix = 'from use_cases.generated_templates.configurations '
        configs_imports = '\n'.join(['%simport %s'%(cfg_import_prefix,i)
                                        for i in self.configs if i is not None])
        
        
        subsystems = []
        for subsys, topology in self.subsystems_templates.items():
            template, config = topology
            config = ('' if config is None else ',%s.configuration()'%config)
            assignment = f'''
            {subsys} = {template}.topology('{subsys}'{config})
            '''
            subsystems.append(assignment)
            
        subsystems = ''.join(subsystems)
        subsystems = textwrap.dedent(subsystems)
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(configs_imports = configs_imports,
                           templates_imports = templates_imports,
                           subsystems = subsystems)
        return text
    
    def write_class_init(self):
        text = '''
                class numerical_assembly(object):

                    def __init__(self):
                        self._t = 0
                        self.subsystems = [{subsystems}]
                        
                        self.interface_map = {interface_map}
                        self.indicies_map  = {indicies_map}
                        
                        self.R_ground  = np.array([[0],[0],[0]],dtype=np.float64)
                        self.P_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
                        self.Pg_ground = np.array([[1],[0],[0],[0]],dtype=np.float64)
                        
                        self.gr_rows = np.array([0,1])
                        self.gr_jac_rows = np.array([0,0,1,1])
                        self.gr_jac_cols = np.array([0,1,0,1])
                        
                        self.nrows = {nrows}
                        self.ncols = {ncols}
                        
                        self.initialize_assembly()
                        
                '''
        
        subsystems = ','.join(self.mbs.subsystems.keys())
        interface_map = self.mbs.interface_map
        indicies_map  = self.mbs.nodes_indicies
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(subsystems = subsystems,
                           interface_map = interface_map,
                           indicies_map  = indicies_map,
                           nrows = self.mbs.nve,
                           ncols = 2*len(self.mbs.nodes))
        return text

    def write_class_helpers(self):
        text = '''
                @property
                def t(self):
                    return self._t
                @t.setter
                def t(self,t):
                    self._t = t
                    for sub in self.subsystems:
                        sub.t = t
                
                def set_initial_states(self):
                    for sub in self.subsystems:
                        sub.set_initial_states()
                    coordinates = [sub.config.q for sub in self.subsystems if len(sub.config.q)!=0]
                    self.q0 = np.concatenate([self.R_ground,self.P_ground,*coordinates])
                
                def initialize_assembly(self):
                    self.t = 0
                    self.assemble_system()
                    self.set_initial_states()
                    self.eval_constants()
                '''
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = textwrap.indent(text,indent).lstrip()
        text = text.format(subsystems = self.mbs.subsystems)
        return text
    
    def write_assembler(self):
        text = '''
                def assemble_system(self):
                    offset = 0
                    for sub in self.subsystems:
                        sub.assemble_template(self.indicies_map,self.interface_map,offset)
                        offset += sub.nrows
                    
                    self.gr_rows += offset
                    self.gr_jac_rows += offset
                    
                    self.rows = np.concatenate([s.rows for s in self.subsystems])
                    self.jac_rows = np.concatenate([s.jac_rows for s in self.subsystems])
                    self.jac_cols = np.concatenate([s.jac_cols for s in self.subsystems])
                    
                    self.rows = np.concatenate([self.rows,self.gr_rows])
                    self.jac_rows = np.concatenate([self.jac_rows,self.gr_jac_rows])
                    self.jac_cols = np.concatenate([self.jac_cols,self.gr_jac_cols])
               '''
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = textwrap.indent(text,indent).lstrip()
        return text
        
    def write_constants_evaluator(self):
        text = '''
                def eval_constants(self):
                    {virtuals_map}
                    
                    for sub in self.subsystems:
                        sub.eval_constants()
                '''
        indent = 4*' '
        p = self.printer
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        ground_map = self.mbs.mapped_gen_coordinates[0:2]
        pattern = '|'.join([p._print(i.lhs) for i in ground_map])

        virtuals_map = self.mbs.mapped_vir_coordinates
        pattern1 = '|'.join([p._print(i.lhs) for i in virtuals_map])
        pattern2 = '|'.join([p._print(i.rhs) for i in virtuals_map])
        
        def sub(x):
            l = x.group(0).split('.')
            try:
                s = '%s.config.%s'%(*l,)
            except TypeError:
                s = '%s'%x.group(0)
            return s
        self_inserter = self._insert_string('self.')
        
        virtuals_map = '\n'.join([p._print(expr) for expr in virtuals_map])
        virtuals_map = re.sub(pattern,self_inserter,virtuals_map)
        virtuals_map = re.sub(pattern1,sub,virtuals_map)
        virtuals_map = re.sub(pattern2,sub,virtuals_map)
        virtuals_map = textwrap.indent(virtuals_map,indent).lstrip()
        
        text = text.format(virtuals_map = virtuals_map)
        text = textwrap.indent(text,indent)
        return text
    
    def write_coordinates_setter(self):
        return self._write_x_setter('coordinates','q')
    
    def write_velocities_setter(self):
        return self._write_x_setter('velocities','qd')
    
    def write_accelerations_setter(self):
        return self._write_x_setter('gen_accelerations','qdd')
    
    def write_lagrange_setter(self):
        return self._write_x_setter('lagrange_multipliers','Lambda')
        
    def write_pos_equations(self):
        return self._write_x_equations('pos')
    
    def write_vel_equations(self):
        return self._write_x_equations('vel')
    
    def write_acc_equations(self):
        return self._write_x_equations('acc')
    
    def write_jac_equations(self):
        return self._write_x_equations('jac')
    
    def write_forces_equations(self):
        return self._write_x_equations('frc')
    
    def write_mass_equations(self):
        return self._write_x_equations('mass')
    
    def write_reactions_equations(self):
        func_name = 'reactions_equations'
        text = '''
                def eval_{func_name}_eq(self):
                    
                    for sub in self.subsystems:
                        sub.eval_{func_name}_eq()
                '''
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        text = text.format(func_name = func_name)
        text = textwrap.indent(text,indent)
        return text
    
    
    def write_system_class(self):
        text = '''
                {class_init}
                    {class_helpers}
                    {assembler}
                    {constants}
                    {coord_setter}
                    {veloc_setter}
                    {accel_setter}
                    {lagrg_setter}
                    {eval_pos}
                    {eval_vel}
                    {eval_acc}
                    {eval_jac}
                    {eval_mass}
                    {eval_frc}
                    {eval_rct}
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        class_init = self.write_class_init()
        class_helpers = self.write_class_helpers()
        assembler = self.write_assembler()
        constants = self.write_constants_evaluator()
        coord_setter = self.write_coordinates_setter()
        veloc_setter = self.write_velocities_setter()
        accel_setter = self.write_accelerations_setter()
        lagrg_setter = self.write_lagrange_setter()
        
        eval_pos = self.write_pos_equations()
        eval_vel = self.write_vel_equations()
        eval_acc = self.write_acc_equations()
        eval_jac = self.write_jac_equations()
        eval_frc = self.write_forces_equations()
        eval_mass = self.write_mass_equations()
        eval_rct = self.write_reactions_equations()
        
        text = text.format(class_init = class_init,
                           class_helpers = class_helpers,
                           assembler = assembler,
                           constants = constants,
                           eval_pos = eval_pos,
                           eval_vel = eval_vel,
                           eval_acc = eval_acc,
                           eval_jac = eval_jac,
                           eval_frc = eval_frc,
                           eval_rct = eval_rct,
                           eval_mass = eval_mass,
                           coord_setter = coord_setter,
                           veloc_setter = veloc_setter,
                           accel_setter = accel_setter,
                           lagrg_setter = lagrg_setter)
        return text
    
    def write_code_file(self):
        os.chdir('..\..')
        path = os.getcwd() + '\\generated_templates\\assemblies'
        os.chdir(path)
        imports = self.write_imports()
        system_class = self.write_system_class()
        
        text = ''.join([imports,system_class])
        with open('%s.py'%self.mbs.name,'w') as file:
            file.write(text)

    ###########################################################################
    def _write_x_setter(self,func_name,var='q'):
        text = '''
                def set_gen_{func_name}(self,{var}):
                    {ground_map}
                    offset = 7
                    for sub in self.subsystems:
                        qs = {var}[offset:sub.n+offset]
                        sub.set_gen_{func_name}(qs)
                        offset += sub.n
                        
                    {virtuals_map}
               '''
        indent = 4*' '
        p = self.printer
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        self_inserter = self._insert_string('self.')
        
        ground_map = getattr(self.mbs,'mapped_gen_%s'%func_name)[0:2]
        pattern = '|'.join([p._print(i.lhs) for i in ground_map])
        ground_map = '\n'.join([p._print(i) for i in ground_map])
        ground_map = re.sub(pattern,self_inserter,ground_map)
        ground_map = textwrap.indent(ground_map,indent).lstrip()
        
        virtuals_map = getattr(self.mbs,'mapped_vir_%s'%func_name)
        pattern1 = '|'.join([p._print(i.lhs) for i in virtuals_map])
        pattern2 = '|'.join([p._print(i.rhs) for i in virtuals_map])
        sub = self._insert_string('')
        virtuals_map = '\n'.join([str(p._print(i)) for i in virtuals_map])
        virtuals_map = re.sub(pattern,self_inserter,virtuals_map)
        virtuals_map = re.sub(pattern1,sub,virtuals_map)
        virtuals_map = re.sub(pattern2,sub,virtuals_map)
        virtuals_map = textwrap.indent(virtuals_map,indent).lstrip()
                
        text = text.format(func_name = func_name,
                           var = var,
                           ground_map = ground_map,
                           virtuals_map = virtuals_map)
        text = textwrap.indent(text,indent)
        return text
    
    def _write_x_equations(self,func_name):
        text = '''
                def eval_{func_name}_eq(self):

                    {ground_data}
                    
                    for sub in self.subsystems:
                        sub.eval_{func_name}_eq()
                    self.{func_name}_eq_blocks = sum([s.{func_name}_eq_blocks for s in self.subsystems],[])
                    self.{func_name}_eq_blocks += {func_name}_ground_eq_blocks
                '''
        indent = 4*' '
        p = self.printer
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        matrix = p._print(getattr(self.mbs,'%s_equations'%func_name)).split('\n')
        rows,cols,ground_data = matrix
        
        pattern = '|'.join([p._print(i) for i in self.mbs.nodes['ground']['arguments_symbols']])
        self_inserter = self._insert_string('self.')
        ground_data = re.sub(pattern,self_inserter,ground_data)
                
        ground_data = textwrap.indent(ground_data,indent).lstrip()
        ground_data = '%s_ground_eq_blocks = %s'%(func_name,ground_data.lstrip())
        
        text = text.format(func_name = func_name,
                           ground_data = ground_data)
        text = textwrap.indent(text,indent)
        return text
    