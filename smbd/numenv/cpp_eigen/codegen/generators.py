#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 21:41:25 2019

@author: khaledghobashy
"""

# Standard library imports
import os
import re
import textwrap
import itertools

# Local application imports
from .printer import printer


class abstract_generator(object):
    """
    A C++ code-generator class that generates numerical code with an OOP
    structure using Eigen Library.
    
    Parameters
    ----------
    mbs : topology
        An instance of a topology class.
    
    Methods
    -------
    write_code_file()
        Write the structured code-files in a .py file format. These code-files
        are saved in the corresponding sub-directory in the //generated_templates
        directory.
        
    Notes
    -----
    This generator structures the numerical code of the system as follows:
        - Creating a 'model_name'.py file that is designed  be used as an 
          imported  module. The file is saved in a sub-directory inside the
          'generated_templates' directory.
        - This module contains one class only, the 'topology' class.
        - This class provides the interface to be used either directly by the 
          python_solver or indirectly by the assembly class that assembles 
          several 'topology' classes together.

    """
    
    def __init__(self,mbs, printer=printer()):
        
        self.mbs     = mbs
        self.printer = printer
        self.name    = self.mbs.name
                
        self.arguments_symbols = [printer._print(exp) for exp in self.mbs.arguments_symbols]
        self.constants_symbols = [printer._print(exp) for exp in self.mbs.constants_symbols]
        self.runtime_symbols   = [printer._print(exp) for exp in self.mbs.runtime_symbols]
        
        self.primary_arguments = self.arguments_symbols
        
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
                
        self.bodies = self.mbs.bodies

    @staticmethod
    def _insert_string(string):
        def inserter(x): return string + x.group(0)
        return inserter

###############################################################################
###############################################################################

class configuration_codegen(abstract_generator):
        
    def __init__(self, config, printer=printer()):
        
        self.config  = config
        self.printer = printer
        self.name = self.config.name
        
        self.primary_args = [printer._print(i) for i in config.topology.arguments_symbols]
        self.config_vars  = [printer._print(i) for i in self.config.arguments_symbols]
        self.input_nodes  = self.config.input_nodes
        
        self.input_args  = self.config.input_equalities
        self.output_args = self.config.intermediat_equalities \
                         + self.config.output_equalities
        
        self.gen_coordinates_sym = [printer._print(exp.lhs) 
        for exp in self.config.topology.mapped_gen_coordinates]
        self.gen_velocities_sym  = [printer._print(exp.lhs) 
        for exp in self.config.topology.mapped_gen_velocities]
        
    
    def write_source_content(self):
        text = '''
                #include "smbd/geometries.hpp"
                #include "smbd/spatial_algebra.hpp"
                                
                {assembler}
                
               '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        assembler  = self.write_template_assembler()
                
        text = text.format(file_name = self.name,
                           assembler = assembler)
        return text
    
    
    def write_template_assembler(self):
        text = '''
                template<class T>
                class ConfigurationInputs
                {{
                public:
                    T* config;
                    {inputs}
                
                public:
                    ConfigurationInputs(T& config)
                    {{
                        this-> config = &config;
                    }};
                    
                    void assemble();
                
                }};
                
                template<class T>
                void ConfigurationInputs<T>::assemble()
                {{
                    auto& config = *this-> config;
                    
                    {primary_inputs}
                    
                    {intmd_eq}
                    
                    {outputs}

                    config.set_inital_configuration();
                }};
                '''
        
        p = self.printer
        indent = ''
        
        outputs = self.output_args
        
        
        self_pattern = '|'.join(self.input_nodes)
        self_inserter = self._insert_string('this-> ')
        
        config_pattern = '|'.join(self.primary_args)
        config_inserter = self._insert_string('config.')
        
        inputs = '\n'.join(['%s ;'%p._print(exp.lhs, declare=True) for exp in 
                              self.input_args])
        inputs = textwrap.indent(inputs, 4*' ').lstrip()
        
        intmd_eq = '\n'.join([p._print(exp, declare=True) for exp in 
                              self.config.intermediat_equalities])
        intmd_eq = re.sub(config_pattern, config_inserter, intmd_eq)
        intmd_eq = re.sub(self_pattern, self_inserter, intmd_eq)
        intmd_eq = textwrap.indent(intmd_eq, 4*' ').lstrip()
        
        direct_inputs = set.intersection(set(self.input_nodes), set(self.primary_args)) 
        primary_inputs = '\n'.join(['config.%s = this-> %s ;'%(exp,exp) for exp in 
                              direct_inputs])
        primary_inputs = textwrap.indent(primary_inputs, 4*' ').lstrip()
        
        
        outputs = '\n'.join([p._print(exp) for exp in 
                             self.config.output_equalities])
        outputs = re.sub(config_pattern, config_inserter, outputs)
        outputs = re.sub(self_pattern, self_inserter, outputs)
        outputs = textwrap.indent(outputs, 4*' ').lstrip()
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(inputs = inputs,
                           primary_inputs = primary_inputs,
                           intmd_eq = intmd_eq,
                           outputs = outputs)
        text = textwrap.indent(text,indent)
        return text

        
    def write_source_file(self, dir_path=''):
        file_path = os.path.join(dir_path, self.name)
        system_class = self.write_source_content()
        with open('%s.hpp'%file_path, 'w') as file:
            file.write(system_class)
        print('File full path : %s.hpp'%file_path)


###############################################################################
###############################################################################

class template_codegen(abstract_generator):
    
            
    def write_header_class(self):
        text = '''
                #include <map>
                #include <vector>
                
                #include "smbd/euler_parameters.hpp"
                #include "smbd/spatial_algebra.hpp"
                #include "smbd/helpers.hpp"
                
                typedef std::map<std::string, std::string> Dict_SS;
                typedef std::map<std::string, int> Dict_SI;
                
                class Configuration
                {{
                
                public:
                    Eigen::VectorXd q;
                    Eigen::VectorXd qd;
                    void set_inital_configuration();                
                
                public:
                    {primary_arguments}
                    
                }};


                // Declaring the Topology Class and its numerical objects.
                // =======================================================
                class Topology
                {{
                
                public:
                    // Topology constants.
                    int n = {n};
                    int nc = {nc};
                    int nrows = {nve};
                    int ncols = 2*{nodes};
                    
                    // Topology variables
                    std::string prefix;
                    double t = 0;
                    
                    Eigen::VectorXd q0;
                    
                    Eigen::VectorXd pos_eq;
                    Eigen::VectorXd vel_eq;
                    Eigen::VectorXd acc_eq;
                    
                    std::vector<Eigen::MatrixXd> jac_eq;
                    std::vector<Eigen::MatrixXd> mas_eq;
                    
                    Eigen::VectorXd rows = Eigen::VectorXd::LinSpaced(nrows, 0, nrows-1);

                    Eigen::VectorXd jac_rows;
                    Eigen::VectorXd jac_cols;
                    
                    Dict_SI indicies_map;
                    
                    
                public:
                
                    // Topology Constructors.
                    Topology();
                    Topology(std::string);
                    
                    // Base Configuration object.
                    Configuration config;
                
                    // Topology initializing functions.
                    void initialize();
                    void assemble(Dict_SI& indicies_map, Dict_SS& interface_map, int rows_offset);
                    void set_initial_states();
                    void eval_constants();
                    
                    // Topology Equations Evaluators.
                    void eval_pos_eq();
                    void eval_vel_eq();
                    void eval_acc_eq();
                    void eval_jac_eq();
                    
                    // Topology States Setters.
                    void set_gen_coordinates(Eigen::VectorXd& q);
                    void set_gen_velocities(Eigen::VectorXd& qd);
                    void set_gen_accelerations(Eigen::VectorXd& qdd);
                
                private:
                    void set_mapping(Dict_SI& indicies_map, Dict_SS& interface_map);
                
                // Topology Bodies Indicies from the network graph.                    
                public:
                    {bodies}
                
                // Topology Generalized Coordinates (R and P vectors).
                public:
                    {coordinates}
                
                // Topology Generalized Velocities (dR/dt and dP/dt vectors).
                public:
                    {velocities}
                
                // Topology Generalized Accelerations (dR2/dt2 and dP2/dt2 vectors).
                public:
                    {accelerations}
                
                // Configuration Constants.
                public:    
                    {constants}
                
                }};
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        
        primary_aruments = '\n'.join(['%s ;'%p._print(i, declare=True) for i in 
                                      set(self.mbs.arguments_symbols)])
        primary_aruments = textwrap.indent(primary_aruments, 4*' ').lstrip()
        
        bodies_indices = '\n'.join(['int %s ;'%i for i in self.bodies])
        bodies_indices = textwrap.indent(bodies_indices, 4*' ').lstrip()

        
        coordinates = '\n'.join(['%s ;'%p._print(i.lhs, declare=True) for i in
                                 self.gen_coordinates_exp])
        coordinates = textwrap.indent(coordinates, 4*' ').lstrip()

        velocities = '\n'.join(['%s ;'%p._print(i.lhs, declare=True) for i in 
                                self.gen_velocities_exp])
        velocities = textwrap.indent(velocities, 4*' ').lstrip()

        accelerations = '\n'.join(['%s ;'%p._print(i.lhs, declare=True) for i in
                                   self.gen_accelerations_exp])
        accelerations = textwrap.indent(accelerations, 4*' ').lstrip()

        constants = '\n'.join(['%s ;'%p._print(i, declare=True) for i in 
                               set(self.mbs.constants_symbols)])
        constants = textwrap.indent(constants, 4*' ').lstrip()
        
        text = text.format(primary_arguments = primary_aruments,
                           bodies = bodies_indices,
                           coordinates = coordinates,
                           velocities = velocities,
                           accelerations = accelerations,
                           constants = constants,
                           n = self.mbs.n,
                           nc = self.mbs.nc,
                           nve = self.mbs.nve,
                           nodes = len(self.bodies))        
        return text
    
    def write_header_file(self, dir_path=''):
        file_path = os.path.join(dir_path, self.name)
        system_class = self.write_header_class()
        with open('%s.hpp'%file_path, 'w') as file:
            file.write(system_class)
        print('File full path : %s.hpp'%file_path)
    
    
    def write_source_file(self, dir_path=''):
        file_path = os.path.join(dir_path, self.name)
        system_class = self.write_source_content()
        with open('%s.cpp'%file_path, 'w') as file:
            file.write(system_class)
        print('File full path : %s.cpp'%file_path)


    def write_source_content(self):
        text = '''
                #include "{file_name}.hpp"
                
                {configuration_setter}
                
                {constructor}
                
                {assemblers}
                
                {setters}
                
                {evaluators}
                
               '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        configuration_setter = self.write_configuration_setter()
        
        constructor = self.write_class_constructor()
        assemblers  = self.write_template_assembler()
        
        setters = ''.join([self.write_coordinates_setter(),
                           self.write_velocities_setter(),
                           self.write_accelerations_setter()])
            
        evaluators = ''.join([self.write_constants_eval(),
                              self.write_pos_equations(),
                              self.write_vel_equations(),
                              self.write_acc_equations(),
                              self.write_jac_equations()])
        
        text = text.format(file_name = self.name,
                           configuration_setter = configuration_setter,
                           constructor = constructor,
                           assemblers = assemblers,
                           setters = setters,
                           evaluators = evaluators)
        return text


    def write_class_constructor(self):
        text = '''
                Topology::Topology(){{}};
                Topology::Topology(std::string prefix = "")
                {{
                    this-> prefix = prefix;
                    
                    this-> q0.resize(this-> n);
                    
                    this-> pos_eq.resize(this-> nc);
                    this-> vel_eq.resize(this-> nc);
                    this-> acc_eq.resize(this-> nc);
                    
                    this-> jac_eq.reserve(this-> jac_rows.size());
                    
                    {indicies_map}
                }};
               '''
        
        indent = ''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indicies_map = '\n'.join(['this-> indicies_map[prefix + "%s"] = %s;'%(n,i) for i,n in enumerate(self.bodies)])
        indicies_map = textwrap.indent(indicies_map, 4*' ').lstrip()
        
        text = text.format(indicies_map = indicies_map)
        text = textwrap.indent(text,indent)
        
        return text
        
        
    def write_template_assembler(self):
        text = '''
                void Topology::initialize()
                {{
                    Dict_SS interface_map;
                    this-> t = 0;
                    this-> assemble(this-> indicies_map, interface_map, 0);
                    this-> set_initial_states();
                    this-> eval_constants();
                    
                }};
                    
                                
                void Topology::assemble(Dict_SI& indicies_map, Dict_SS& interface_map, int rows_offset)
                {{
                    this-> set_mapping(indicies_map, interface_map);
                    this-> rows += (rows_offset * Eigen::VectorXd::Ones(this ->rows.size()) );
                    
                    this-> jac_rows.resize({nnz});
                    this-> jac_rows << {jac_rows};
                    this-> jac_rows += (rows_offset * Eigen::VectorXd::Ones(this ->jac_rows.size()) );
                    
                    this-> jac_cols.resize({nnz});
                    this-> jac_cols << 
                        {jac_cols};
                }};
                    
                
                void Topology::set_initial_states()
                {{
                    this-> set_gen_coordinates(this-> config.q);
                    this-> set_gen_velocities(this-> config.qd);
                    this-> q0 = this-> config.q;
                }};
                
                
                void Topology::set_mapping(Dict_SI& indicies_map, Dict_SS& interface_map)
                {{
                    auto& p = this-> prefix;
                    
                    {indicies_map}
                    
                    {virtuals}
                }};
                
               '''
        
        indent = ''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indicies_map = '\n'.join(['%s = indicies_map[p+"%s"];'%('this-> %s'%i, i) for i in self.bodies])
        indicies_map = textwrap.indent(indicies_map, 4*' ').lstrip()
        
        virtuals = '\n'.join(['%s = indicies_map[interface_map[p+"%s"];'%('this-> %s'%i, i) for i in self.mbs.virtual_bodies])
        virtuals = textwrap.indent(virtuals, indent).lstrip()
        
        ind_body = {v:k for k,v in self.mbs.nodes_indicies.items()}
        rows, cols, data = zip(*self.mbs.jac_equations.row_list())
        string_cols = [('this-> %s*2'%ind_body[i//2] if i%2==0 else 'this-> %s*2+1'%ind_body[i//2]) for i in cols]
        string_cols_text = ', \n'.join(string_cols)
        string_cols_text = textwrap.indent(string_cols_text, 8*' ').lstrip()
        
        jac_rows = str(rows)
        
        text = text.format(nnz = len(data),
                           indicies_map = indicies_map,
                           virtuals = virtuals,
                           jac_rows = jac_rows[1:-1],
                           jac_cols = string_cols_text)
        text = textwrap.indent(text,indent)
        return text
    
    def write_constants_eval(self):
        text = '''
                void Topology::eval_constants()
                {{
                    auto& config = this-> config;
                    
                    {num_constants}
                    
                    {sym_constants}
                }};
                '''
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        printer = self.printer
        indent = ''
                
        config_pattern_iter = itertools.chain(self.arguments_symbols,
                                              self.virtual_coordinates)
        config_pattern = '|'.join(config_pattern_iter)
        config_inserter = self._insert_string('config.')
        
        self_pattern_iter = itertools.chain(self.constants_symbols)
        self_pattern = '|'.join(self_pattern_iter)
        self_inserter = self._insert_string('this-> ')
        
        num_constants_list = self.constants_numeric_expr
        num_constants_text = '\n'.join((printer._print(i) for i in num_constants_list))
        num_constants_text = re.sub(config_pattern, config_inserter, num_constants_text)
        num_constants_text = re.sub(self_pattern, self_inserter, num_constants_text)
        num_constants_text = textwrap.indent(num_constants_text, 4*' ').lstrip()

        
        sym_constants = self.constants_symbolic_expr
        sym_constants_text = '\n'.join((printer._print(i) for i in sym_constants))
        sym_constants_text = re.sub(config_pattern, config_inserter, sym_constants_text)
        sym_constants_text = re.sub(self_pattern, self_inserter, sym_constants_text)
        sym_constants_text = textwrap.indent(sym_constants_text, 4*' ').lstrip()
        

        text = text.expandtabs()
        text = textwrap.dedent(text)
        text = text.format(sym_constants = sym_constants_text,
                           num_constants = num_constants_text)
        text = textwrap.indent(text,indent)
        return text

    def write_configuration_setter(self):
        text = '''
                void Configuration::set_inital_configuration()
                {{
                    this-> q.resize({n});
                    this-> q << 
                        {q_equalities};
                    
                    this-> qd.resize({n});
                    this-> qd << 
                        {qd_equalities};
                }};
               '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        indent = ''
                
        
        self_inserter = self._insert_string('this-> ')
        
        q_pattern = '|'.join(self.gen_coordinates_sym)
        q_equalities = ', \n'.join(self.gen_coordinates_sym)
        q_equalities = re.sub(q_pattern, self_inserter, q_equalities)
        q_equalities = textwrap.indent(q_equalities, 8*' ').lstrip()
        
        qd_pattern = '|'.join(self.gen_velocities_sym)
        qd_equalities = ', \n'.join(self.gen_velocities_sym)
        qd_equalities = re.sub(qd_pattern, self_inserter, qd_equalities)
        qd_equalities = textwrap.indent(qd_equalities, 8*' ').lstrip()
        
        text = text.format(n = self.mbs.n,
                           q_equalities = q_equalities,
                           qd_equalities = qd_equalities)
        text = textwrap.indent(text, indent)
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
        return self._write_x_equations('jac', std_vector=True)
    
    def write_forces_equations(self):
        return self._write_x_equations('frc')
    
    def write_mass_equations(self):
        return self._write_x_equations('mass', std_vector=True)
    
    def write_reactions_equations(self):
        text = '''
                def eval_reactions_eq(self):
                    config  = self.config
                    t = self.t
                    
                    {equations_text}
                    
                    self.reactions = {reactions}
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
        
        config_pattern = set(self.primary_arguments) - set(self.runtime_symbols)
        config_pattern = '|'.join([r'%s'%i for i in config_pattern])
        config_inserter = self._insert_string('config.')
        equations_text = re.sub(config_pattern,config_inserter,equations_text)
                
        equations_text = textwrap.indent(equations_text,indent).lstrip() 
        
        reactions = ',\n'.join(['%r : self.%s'%(i,i) for i in self.joint_reactions_sym])
        reactions = textwrap.indent(reactions, 5*indent).lstrip()
        reactions = '{%s}'%reactions
        
        text = text.format(equations_text = equations_text,
                           reactions = reactions)
        text = textwrap.indent(text,indent)
        return text

    
    ###########################################################################
    ###########################################################################

    def _write_x_setter(self,func_name,var='q'):
        text = '''
                void Topology::set_%s(Eigen::VectorXd &%s)
                {{
                    {equalities}
                }};
               '''%(func_name, var)
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        p = self.printer
        indent = ''
        
        symbolic_equality  = getattr(self,'%s_exp'%func_name)
        
        if len(symbolic_equality) !=0:
            pattern = '|'.join(getattr(self,'%s_sym'%func_name))
            self_inserter = self._insert_string('this-> ')
            
            numerical_equality = '\n'.join([p._print(i) for i in symbolic_equality])
            numerical_equality = re.sub(pattern,self_inserter,numerical_equality)
            numerical_equality = textwrap.indent(numerical_equality, 4*' ').lstrip()
        else:
            numerical_equality = 'pass'
        
        text = text.format(equalities = numerical_equality)
        text = textwrap.indent(text,indent)
        return text

    
    def _write_x_equations(self, eq_initial, std_vector=False):
        text = '''
                 void Topology::eval_{eq_initial}_eq()
                 {{
                     auto& config = this-> config;
                     auto& t = this-> t;

                     {replacements}
                                        
                     this-> {eq_initial}_eq {operator} 
                         {expressions};
                 }};
                '''
        
        text = text.expandtabs()
        text = textwrap.dedent(text)
        
        printer = self.printer
        indent = ''
        
        operator = '=' if std_vector else '<<'
        
        # Geting the optimized equations' vector/matrix from the topology class.
        # The expected format is two lists [replacements] and [expressions].
        replacements_list = getattr(self.mbs, '%s_rep'%eq_initial)
        expressions_list  = getattr(self.mbs, '%s_exp'%eq_initial)
        # Extracting the vector/matrix from the returned expressions list.
        vector_expr = expressions_list[0]
        # Extracting the Non-Zero values of the vector/matrix.
        vector_data = [i[-1] for i in vector_expr.row_list()]
        
        # Extract the numerical format of the replacements and expressions into
        # a list of string expressions.
        num_repl_list = [printer._print(exp, declare=True) for exp in replacements_list]
        num_expr_list = [printer._print(exp) for exp in vector_data]
        
        # Joining the extracted strings to form a valid text block.
        num_repl_text = '\n'.join(num_repl_list)
        num_expr_text = ',\n'.join(num_expr_list)

        # Creating a regex pattern of strings that represents the variables
        # which need to be perfixed by a 'self.' to be referenced correctly.
        self_pattern = itertools.chain(self.runtime_symbols,
                                       self.constants_symbols)
        self_pattern = '|'.join(self_pattern)
        
        # Creating a regex pattern of strings that represents the variables
        # which need to be perfixed by a 'config.' to be referenced correctly.
        config_pattern = set(self.primary_arguments) - set(self.runtime_symbols)
        config_pattern = '|'.join([r'%s'%i for i in config_pattern])
        
        # Performing the regex substitution with 'self.'.
        self_inserter = self._insert_string('this-> ')
        num_repl_text = re.sub(self_pattern, self_inserter, num_repl_text)
        num_expr_text = re.sub(self_pattern, self_inserter, num_expr_text)
        
        # Performing the regex substitution with 'config.'.
        config_inserter = self._insert_string('config.')
        num_repl_text = re.sub(config_pattern, config_inserter, num_repl_text)
        num_expr_text = re.sub(config_pattern, config_inserter, num_expr_text)
        
        # Indenting the text block for propper class and function indentation.
        num_repl_text = textwrap.indent(num_repl_text, 4*' ').lstrip() 
        num_expr_text = textwrap.indent(num_expr_text, 8*' ').lstrip()
        
        num_expr_text = '{%s}'%num_expr_text if std_vector else num_expr_text
        
        text = text.format(eq_initial  = eq_initial,
                           replacements = num_repl_text,
                           expressions  = num_expr_text,
                           operator = operator)
        text = textwrap.indent(text,indent)
        return text
    

###############################################################################
###############################################################################


