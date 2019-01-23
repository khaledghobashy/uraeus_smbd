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
        
        self.mbs     = multibody_system
        self.config  = self.mbs.param_config
        self.printer = printer
        
        self.gen_coordinates_exp = self.mbs.mapped_gen_coordinates
        self.gen_velocities_exp  = self.mbs.mapped_gen_velocities
        self.edges_constants_exp = self.mbs.constants

        self.gen_coordinates_sym = [printer._print(exp.lhs) for exp in self.gen_coordinates_exp]
        self.gen_velocities_sym  = [printer._print(exp.lhs) for exp in self.gen_velocities_exp]
        self.edges_arguments_sym = [printer._print(exp) for exp in self.mbs.edges_arguments]
        self.edges_constants_sym = [printer._print(exp.lhs) for exp in self.edges_constants_exp]

        self.virtual_coordinates = [printer._print(exp) for exp in self.mbs.virtual_coordinates]
                
        self.input_args  = self.config.inputs_layer()
        self.output_args = self.config.outputs_layer()
                
        self.bodies = sum([list(e) for e in self.mbs.bodies],[])
        self.jac_cols = sum([e for e in self.mbs.cols],[])
            
    
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
    
    def _setup_pos_equations(self):
        self._setup_x_equations('pos','x')
    
    def _setup_vel_equations(self):
        self._setup_x_equations('vel','v')
    
    def _setup_acc_equations(self):
        self._setup_x_equations('acc','a')
        
    def _setup_jac_equations(self):
        self._setup_x_equations('jac','j')
        scols = ','.join(['self.%s*2,self.%s*2+1'%(i,i) for i in self.jac_cols])
        scols += ','+','.join(['self.%s*2+1'%i for i in self.bodies])
        self.jac_eq_cols = 'np.array([%s])'%scols

        
    def setup_equations(self):
        self._setup_pos_equations()
        self._setup_vel_equations()
        self._setup_acc_equations()
        self._setup_jac_equations()

###############################################################################
###############################################################################

class template_code_generator(abstract_generator):
    
    def _insert_self(self,x): 
        return 'self.' + x.group(0).strip("'")
    
    def _insert_config(self,x): 
        return 'config.' + x.group(0).strip("'")
    
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
        pattern = '|'.join(getattr(self,'%s_sym'%func_name))
        
        numerical_equality = '\n'.join([p._print(i) for i in symbolic_equality])
        numerical_equality = re.sub(pattern,self._insert_self,numerical_equality)
        numerical_equality = textwrap.indent(numerical_equality,indent).lstrip()
        
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
        
        self_pattern = sum([self.virtual_coordinates,self.gen_coordinates_sym,
                            self.gen_velocities_sym],[])
        self_pattern = '|'.join(self_pattern)
        cse_var_txt = re.sub(self_pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(self_pattern,self._insert_self,cse_exp_txt)
        
        config_pattern = sum([self.edges_arguments_sym,self.edges_constants_sym],[])
        config_pattern = '|'.join(config_pattern)
        cse_var_txt = re.sub(config_pattern,self._insert_config,cse_var_txt)
        cse_exp_txt = re.sub(config_pattern,self._insert_config,cse_exp_txt)
        
        cse_var_txt = textwrap.indent(cse_var_txt,indent).lstrip() 
        cse_exp_txt = textwrap.indent(cse_exp_txt,indent).lstrip()
        
        cse_exp_txt = 'self.%s_eq_blocks = %s'%(xstring,cse_exp_txt.lstrip())
        
        text = text.format(cse_var_txt = cse_var_txt,
                           cse_exp_txt = cse_exp_txt)
        text = textwrap.indent(text,indent)
        return text
    
    ###########################################################################
    ###########################################################################

    def write_imports(self):
        text = '''
                import numpy as np
                import scipy as sc
                from numpy.linalg import multi_dot
                from source.cython_definitions.matrix_funcs import A, B, triad as Triad, Mirror
                from scipy.misc import derivative
                from numpy import cos, sin
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        return text
    
    def write_config_class(self):
        text = '''
                class inputs(object):

                    def __init__(self):
                        {inputs}
                    
                    def _set_arguments(self):
                        {outputs}
                        
                    def eval_constants(self):
                        self._set_arguments()
                        
                        {cse_var_txt}

                        {cse_exp_txt}
                '''
        
        p = self.printer
        indent = 8*' '
        
        inputs  = [sm.Eq(i,sm.MutableDenseMatrix([0,0,0])) for i in self.input_args]
        outputs = self.output_args
        consts = self.edges_constants_exp
        
        pattern = '|'.join( self.edges_arguments_sym + self.edges_constants_sym
                           +self.virtual_coordinates + self.gen_coordinates_sym)
        
        inputs = '\n'.join([p._print(exp) for exp in inputs])
        inputs = re.sub(pattern,self._insert_self,inputs)
        inputs = textwrap.indent(inputs,indent).lstrip()
        
        outputs = '\n'.join([p._print(exp) for exp in outputs])
        outputs = re.sub(pattern,self._insert_self,outputs)
        outputs = textwrap.indent(outputs,indent).lstrip()
        
        cse_var_txt, cse_exp_txt = self._generate_cse(consts,'c')
        cse_var_txt = re.sub(pattern,self._insert_self,cse_var_txt)
        cse_exp_txt = re.sub(pattern,self._insert_self,cse_exp_txt)
        cse_var_txt = textwrap.indent(cse_var_txt,indent).lstrip()
        cse_exp_txt = textwrap.indent(cse_exp_txt,indent).lstrip()
                
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(inputs  = inputs,
                           outputs = outputs,
                           cse_var_txt = cse_var_txt,
                           cse_exp_txt = cse_exp_txt)
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
                    self.jac_cols += self.rows_offset
                    self.jac_cols = {jac_cols}
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
    
    def write_coordinates_setter(self):
        return self._write_x_setter('gen_coordinates','q')
    
    def write_velocities_setter(self):
        return self._write_x_setter('gen_velocities','qd')
        
    def write_pos_equations(self):
        return self._write_x_equations('pos')
    
    def write_vel_equations(self):
        return self._write_x_equations('vel')
    
    def write_acc_equations(self):
        return self._write_x_equations('acc')
    
    def write_jac_equations(self):
        return self._write_x_equations('jac')
        
    def write_class_init(self):
        text = '''
                class numerical_assembly(object):

                    def __init__(self,config,prefix=''):
                        self.t = 0.0
                        self.config = config
                        self.prefix = prefix
                        
                        self.Pg_ground = np.array([[1], [0], [0], [0]],dtype=np.float64)
                        
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
                

    def write_system_class(self):
        text = '''
                {class_init}
                    {jac_mappings}
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
        jac_mappings = self.write_template_assembler()
        coord_setter = self.write_coordinates_setter()
        veloc_setter = self.write_velocities_setter()
        
        eval_pos = self.write_pos_equations()
        eval_vel = self.write_vel_equations()
        eval_acc = self.write_acc_equations()
        eval_jac = self.write_jac_equations()
        
        text = text.format(class_init = class_init,
                           jac_mappings = jac_mappings,
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

class assembly_code_generator(template_code_generator):
    
    def __init__(self,multibody_system,printer=numerical_printer()):
        self.mbs  = multibody_system
        self.name = self.mbs.name
        self.templates = []
        self.subsystems_templates = {}
        for name,obj in self.mbs.subsystems.items():
            topology_name = obj.topology.name
            self.subsystems_templates[name] = topology_name
            if topology_name not in self.templates:
                self.templates.append(topology_name)
    
    def _write_x_setter(self,func_name,var='q'):
        text = f'''
                def set_{func_name}(self,{var}):
                    offset = 0
                    for sub in self.subsystems:
                        qs = {var}[offset:sub.n]
                        sub.set_{func_name}(qs)
                        offset += sub.n
               '''
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = textwrap.indent(text,indent).lstrip()
        return text
    
    def _write_x_equations(self,func_name):
        text = f'''
                def eval_{func_name}_eq(self):
                    for sub in self.subsystems:
                        sub.eval_{func_name}_eq()
                '''
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = textwrap.indent(text,indent)
        return text
    
    def write_imports(self):
        text = '''
                import numpy as np
                
                {templates_imports}
                
                {subsystems}
                '''
        templates_imports = '\n'.join(['import %s'%i for i in self.templates])
        subsystems = '\n'.join(['%s = %s.numerical_assembly()'%d for d in self.subsystems_templates.items()])
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(templates_imports = templates_imports,
                           subsystems = subsystems)
        return text
    
    def write_class_init(self):
        text = '''
                class numerical_assembly(object):

                    def __init__(self):
                        self.set_time(0.0)
                        self.Pg_ground  = np.array([[1],[0],[0],[0]],dtype=np.float64)
                        self.subsystems = [{subsystems}]
                '''
        subsystems = ','.join(self.mbs.subsystems.keys())
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(subsystems = subsystems)
        return text

    def write_class_helpers(self):
        text = '''
                def set_time(self,t):
                    for sub in self.subsystems:
                        sub.t = t
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
                        offset += sub.n
               '''
        indent = 4*' '
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = textwrap.indent(text,indent).lstrip()
        return text
        
    def write_coordinates_setter(self):
        return self._write_x_setter('gen_coordinates','q')
    
    def write_velocities_setter(self):
        return self._write_x_setter('gen_velocities','qd')
        
    def write_pos_equations(self):
        return self._write_x_equations('pos')
    
    def write_vel_equations(self):
        return self._write_x_equations('vel')
    
    def write_acc_equations(self):
        return self._write_x_equations('acc')
    
    def write_jac_equations(self):
        return self._write_x_equations('jac')
    
    def write_system_class(self):
        text = '''
                {class_init}
                    {class_helpers}
                    {assembler}
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
        class_helpers = self.write_class_helpers()
        assembler = self.write_assembler()
        coord_setter = self.write_coordinates_setter()
        veloc_setter = self.write_velocities_setter()
        
        eval_pos = self.write_pos_equations()
        eval_vel = self.write_vel_equations()
        eval_acc = self.write_acc_equations()
        eval_jac = self.write_jac_equations()
        
        text = text.format(class_init = class_init,
                           class_helpers = class_helpers,
                           assembler = assembler,
                           eval_pos = eval_pos,
                           eval_vel = eval_vel,
                           eval_acc = eval_acc,
                           eval_jac = eval_jac,
                           coord_setter = coord_setter,
                           veloc_setter = veloc_setter)
        return text
    
    def write_code_file(self):
        imports = self.write_imports()
        system_class = self.write_system_class()
        
        text = '\n'.join([imports,system_class])
        with open('%s.py'%self.mbs.name,'w') as file:
            file.write(text)

