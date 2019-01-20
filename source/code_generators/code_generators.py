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
import networkx as nx
from source.code_generators.code_printers import numerical_printer

class abstract_generator(object):
    
    import source.symbolic_classes.abstract_matrices as temp
    temp.ccode_print = True
    temp.enclose = True

    def __init__(self,multibody_system,printer=numerical_printer()):
        
        self.mbs = multibody_system
        self.printer = printer
        
        self.arguments = num_args_sym = self.mbs.arguments.copy()
        self.constants = cfig_cons_sym = self.mbs.constants.copy()
        
        self.num_args_sym = [printer._print(e.lhs) for e in num_args_sym]
        self.cfig_cons_sym = [printer._print(e.lhs) for e in cfig_cons_sym]
        
        self.generalized_coordinates_equalities = self.mbs.q_maped.copy()
        self.generalized_velocities_equalities = self.mbs.qd_maped.copy()
        
        self.generalized_coordinates_lhs = [printer._print(exp.lhs) for exp in self.generalized_coordinates_equalities]
        self.generalized_velocities_lhs = [printer._print(exp.lhs) for exp in self.generalized_velocities_equalities]
    
        self.virtual_bodies = nx.get_node_attributes(self.mbs.graph.subgraph(self.mbs.virtual_bodies),'obj')
    
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
        return 'self.' + x.group(0).strip("'")
    
    def _insert_config(self,x): 
        return 'config.' + x.group(0).strip("'")
    
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
        maped_coordinates = '\n'.join([p._print(i) for i in coordinates])
        pattern = '|'.join(self.generalized_coordinates_lhs+self.generalized_velocities_lhs)
        maped_coordinates = re.sub(pattern,self._insert_self,maped_coordinates)
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
                
        gen_coord_pattern = '|'.join(self.generalized_coordinates_lhs)
        cse_var_txt = re.sub(gen_coord_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(gen_coord_pattern,self._insert_self,cse_exp_txt)
        
        gen_veloc_pattern = '|'.join(self.generalized_velocities_lhs)
        cse_var_txt = re.sub(gen_veloc_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(gen_veloc_pattern,self._insert_self,cse_exp_txt)
        
        num_args_pattern = '|'.join(self.num_args_sym)
        cse_var_txt = re.sub(num_args_pattern,self._insert_config,cse_var_txt)
        cse_exp_txt = re.sub(num_args_pattern,self._insert_config,cse_exp_txt)
        
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
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
    
    def write_class_init(self):
        text = '''
                class numerical_assembly(object):

                    def __init__(self,config):
                        self.t = 0.0
                        self.config = config
                        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
                        
                        self.pos_rows = {pos_rows}
                        self.pos_cols = {pos_cols}
                        
                        self.vel_rows = {vel_rows}
                        self.vel_cols = {vel_cols}
                        
                        self.acc_rows = {acc_rows}
                        self.acc_cols = {acc_cols}
                        
                        self.jac_rows = {jac_rows}
                        self.jac_cols = {jac_cols}
                        
                        self.nrows = max(self.pos_rows)
                        self.ncols = max(self.jac_cols)
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
                                        
        text = text.format(nve = self.mbs.nve,
                           pos_rows = self.pos_eq_rows,
                           pos_cols = self.pos_eq_cols,
                           vel_rows = self.vel_eq_rows,
                           vel_cols = self.vel_eq_cols,
                           acc_rows = self.acc_eq_rows,
                           acc_cols = self.acc_eq_cols,
                           jac_rows = self.jac_eq_rows,
                           jac_cols = self.jac_eq_cols)
        return text
                
    def write_config_class(self):
        text = '''
                class inputs(object):

                    def __init__(self):
                        {inputs}
                        
                    def eval_constants(self):
                        
                        {cse_var_txt}

                        {cse_exp_txt}
                    
                    @property
                    def q_initial(self):
                        q = np.concatenate([{q_initial}])
                        return q
                    
                    @property
                    def qd_initial(self):
                        qd = np.concatenate([{qd_initial}])
                        return qd
                '''
        
        p = self.printer
        indent = 8*' '
        
        inputs = self.arguments
        consts = self.constants
        
        pattern = '|'.join(self.num_args_sym+self.cfig_cons_sym)
        
        inputs = '\n'.join([p._print(exp) for exp in inputs])
        inputs = re.sub(pattern,self._insert_self,inputs)
        
        cse_var_txt, cse_exp_txt = self._generate_cse(consts,'c')
        cse_var_txt = re.sub(pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(pattern,self._insert_self,cse_exp_txt)
        
        q_text = ','.join(self.generalized_coordinates_lhs)
        q_text = re.sub(pattern,self._insert_self,q_text)
        
        qd_text = ','.join(self.generalized_velocities_lhs)
        qd_text = re.sub(pattern,self._insert_self,qd_text)
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        inputs = textwrap.indent(inputs,indent).lstrip()
        cse_var_txt = textwrap.indent(cse_var_txt,indent).lstrip()
        cse_exp_txt = textwrap.indent(cse_exp_txt,indent).lstrip()
        text = text.format(inputs = inputs,
                           cse_exp_txt = cse_exp_txt,
                           cse_var_txt = cse_var_txt,
                           q_initial  = q_text,
                           qd_initial = qd_text)
        
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
        config_class = self.write_config_class()
        system_class = self.write_system_class()
        
        text = '\n'.join([imports,config_class,system_class])
        with open('%s.py'%self.mbs.name,'w') as file:
            file.write(text)

    
###############################################################################
###############################################################################
            
class assembly_generator(python_code_generator):
    
    def __init__(self,multibody_system,printer=numerical_printer()):
        
        self.mbs = multibody_system
        self.subsystems = multibody_system.subsystems
        self.printer = printer
        
        self.arguments = num_args_sym = self.mbs.arguments.copy()
        self.constants = cfig_cons_sym = self.mbs.constants.copy()
        
        self.num_args_sym = [printer._print(e.lhs) for e in num_args_sym]
        self.cfig_cons_sym = [printer._print(e.lhs) for e in cfig_cons_sym]
        
    
    def _write_x_setter(self,xstring):
        text = '''
                def set_%s(self,%s):
                    {maped}
               '''%(xstring,xstring)
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = 4*' '
        
        text = '\n'.join(['self.'+sub.name+'.set_coordinates(q)' for sub in self.subsystems.values()])
            
        
        text = text.format(maped = text)
        text = textwrap.indent(text,indent)
        return text
    
    def write_coordinates_setter(self):
        return self._write_x_setter('q')
    
    def write_velocities_setter(self):
        return self._write_x_setter('qd')
    
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
                
        gen_coord_pattern = '|'.join(self.generalized_coordinates_lhs)
        cse_var_txt = re.sub(gen_coord_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(gen_coord_pattern,self._insert_self,cse_exp_txt)
        
        gen_veloc_pattern = '|'.join(self.generalized_velocities_lhs)
        cse_var_txt = re.sub(gen_veloc_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(gen_veloc_pattern,self._insert_self,cse_exp_txt)
        
        num_args_pattern = '|'.join(self.num_args_sym)
        cse_var_txt = re.sub(num_args_pattern,self._insert_config,cse_var_txt)
        cse_exp_txt = re.sub(num_args_pattern,self._insert_config,cse_exp_txt)
        
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
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
    
    
    def write_class_init(self):
        indent = 8*' '
        text1 = super().write_class_init()
        text2 = '''
                        {subsystems}
                '''
        text2 = text2.expandtabs()
        text2 = textwrap.dedent(text2)
        
        subsystems = enumerate(self.subsystems.values())
        subsystems = '\n'.join(['self.sub_%s = %s'%(i,sub.name) for i,sub in subsystems])
        subsystems = textwrap.indent(subsystems,indent)

        text = '\n'.join([text1,text2])
        text = text.format(subsystems = subsystems)
        return text
                
    def write_config_class(self):
        text = '''
                class inputs(object):

                    def __init__(self):
                        {inputs}
                        
                    def eval_constants(self):
                        
                        {cse_var_txt}

                        {cse_exp_txt}
                    
                    @property
                    def q_initial(self):
                        q = np.concatenate([{q_initial}])
                        return q
                    
                    @property
                    def qd_initial(self):
                        qd = np.concatenate([{qd_initial}])
                        return qd
                '''
        
        p = self.printer
        indent = 8*' '
        
        inputs = self.arguments
        consts = self.constants
        
        pattern = '|'.join(self.num_args_sym+self.cfig_cons_sym)
        
        inputs = '\n'.join([p._print(exp) for exp in inputs])
        inputs = re.sub(pattern,self._insert_self,inputs)
        
        cse_var_txt, cse_exp_txt = self._generate_cse(consts,'c')
        cse_var_txt = re.sub(pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(pattern,self._insert_self,cse_exp_txt)
        
        q_text = ','.join(self.generalized_coordinates_lhs)
        q_text = re.sub(pattern,self._insert_self,q_text)
        
        qd_text = ','.join(self.generalized_velocities_lhs)
        qd_text = re.sub(pattern,self._insert_self,qd_text)
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        inputs = textwrap.indent(inputs,indent).lstrip()
        cse_var_txt = textwrap.indent(cse_var_txt,indent).lstrip()
        cse_exp_txt = textwrap.indent(cse_exp_txt,indent).lstrip()
        text = text.format(inputs = inputs,
                           cse_exp_txt = cse_exp_txt,
                           cse_var_txt = cse_var_txt,
                           q_initial  = q_text,
                           qd_initial = qd_text)
        
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
#        config_class = self.write_config_class()
        system_class = self.write_system_class()
        
        text = '\n'.join([imports,config_class,system_class])
        with open('%s.py'%self.mbs.name,'w') as file:
            file.write(text)
            
    
    
    
        