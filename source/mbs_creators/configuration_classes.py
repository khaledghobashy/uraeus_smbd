# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:44:13 2019

@author: khaled.ghobashy
"""
import itertools
import sympy as sm
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from source.symbolic_classes.abstract_matrices import (AbstractMatrix, vector, 
                                                       quatrenion, matrix_symbol)

###############################################################################
###############################################################################
    
class Mirrored(AbstractMatrix):
    """
    A symbolic matrix function that represents a mirrored vector about the 
    Y-Axis.
    
    Parameters
    ----------
    v : vector
    
    """
    def __init__(self,v):
        super().__init__(v)
        self.shape = v.shape
    def _latex(self,expr):
        return r'{Mirrored(%s)}'%self.args[0].name

class Centered(AbstractMatrix):
    """
    A symbolic matrix function that represents the center point of a collection
    of points in 3D.
    
    Parameters
    ----------
    args : Collection of vectors
    
    """
    shape = (3,1)
    def __init__(self,*args):
        super().__init__(*args)
    def _latex(self,expr):
        return r'{Centered%s}'%(self.args,)

class Oriented(AbstractMatrix):
    """
    A symbolic matrix function that represents an oriented vector based on
    the given parameters, either oriented along two points or normal to the
    plane given by three points.
    
    Parameters
    ----------
    args : Collection of vectors
    
    """
    shape = (3,1)
    def __init__(self,*args):
        super().__init__(*args)
    def _latex(self,expr):
        return r'{Oriented%s}'%(self.args,)

class Equal_to(AbstractMatrix):
    """
    A symbolic matrix function that functions as a place holder that reference
    the value of a vector to that of another.
    
    Parameters
    ----------
    v : vector
    
    """
    def __new__(cls,arg):
        return arg
    def __init__(self,arg):
        super().__init__(arg)
    def _latex(self,expr):
        return r'{Equal\_to%s}'%(self.args,)


class Config_Relations(object):
    """
    A container class that holds the relational classes as its' attributes
    for convienient access and import.    
    """
    Mirrored = Mirrored
    Centered = Centered
    Oriented = Oriented
    Equal_to = Equal_to
    UserInput = None

CR = Config_Relations
###############################################################################
###############################################################################

class Geometry(sm.Symbol):
    """
    A symbolic geometry class.
    
    Parameters
    ----------
    name : str
        Name of the geometry object
    
    """
    def __new__(cls, name, *args):
        return super().__new__(cls, name)
    
    def __init__(self, name, *args):
        self.name = name
        self._args = args

        self.R = vector('%s.R'%name)
        self.P = quatrenion('%s.P'%name)
        self.m = sm.symbols('%s.m'%name)
        self.J = matrix_symbol('%s.J'%name,3,3)
        
    def __call__(self,*args):
        return Geometry(self.name,*args)


class Simple_geometry(sm.Function):
    """
    A symbolic geometry class representing simple geometries of well-known,
    easy to calculate properties.
    
    Parameters
    ----------
    name : str
        Name of the geometry object
    
    """
    
    def _latex(self,expr):
        name = self.__class__.__name__
        name = '\_'.join(name.split('_'))
        return r'%s%s'%(name, (*self.args,))
    
    def _ccode(self, printer):
        expr_lowerd = self.__class__.__name__.lower()
        args = ','.join([printer._print(i) for i in self.args])
        return '%s(%s)'%(expr_lowerd,args)

class Composite_Geometry(Simple_geometry):
    """
    A symbolic geometry class representing a composite geometry instance that 
    can be composed of other simple geometries of well-known, easy to calculate
    properties.
    
    Parameters
    ----------
    name : str
        Name of the geometry object
    
    args : sequence of simple_geometry
    
    """
    pass

class Cylinder_Geometry(Simple_geometry):
    
    def __init__(self, arg1, arg2, ro=10, ri=0):
        self.arg1 = arg1
        self.arg2 = arg2
        self.ri = ri
        self.ro = ro

class Triangular_Prism(Simple_geometry):
    
    def __init__(self, arg1, arg2, arg3, l=10):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.l = l

class Geometries(object):
    
    Triangular_Prism   = Triangular_Prism
    Cylinder_Geometry  = Cylinder_Geometry
    Composite_Geometry = Composite_Geometry

###############################################################################
###############################################################################

class relational_graph(object):
    
    def __init__(self, name):
        self.name = name
        self.graph = nx.DiGraph(name=self.name)
        
    @property
    def input_nodes(self):
        graph = self.graph
        nodes = [i for i,d in graph.in_degree() if d==0]
        return nodes
    
    @property
    def intermediat_nodes(self):
        nodes = self.output_nodes
        edges = itertools.chain(*[self._get_node_predecessors(n) for n in nodes])
        mid_nodes = {e[0]:i for i,e in enumerate(edges)}
        return mid_nodes

    @property
    def output_nodes(self):
        graph = self.graph
        condition = lambda i,d : d==0 and graph.in_degree(i)!=0
        nodes = [i for i,d in graph.out_degree() if condition(i,d)]
        return nodes
    
    def add_node(self, name, **kwargs):
        self.graph.add_node(name, **kwargs)

    def add_relation(self, node, arg_nodes):
        name_attribute = [self._extract_name_and_attr(n) for n in arg_nodes]
        arguments_names, nested_attributes = zip(*name_attribute)
        passed_attrs = [{'passed_attr':i if i!='' else None } for i in nested_attributes]
        self._update_in_edges(node, arguments_names, passed_attrs)

        
    def draw_node_dependencies(self, node):
        graph = self.graph
        edges = self._get_node_predecessors(node)
        sub_graph = graph.edge_subgraph(edges)
        plt.figure(figsize=(10, 6))
        nx.draw_networkx(sub_graph, with_labels=True)
        plt.show() 
        
    def draw_graph(self):
        plt.figure(figsize=(10, 6))
        nx.draw_circular(self.graph, with_labels=True)
        plt.show()
    
    def _update_in_edges(self, node, nbunch, edges_attrs):
        graph = self.graph
        old_edges = list(graph.in_edges(node))
        graph.remove_edges_from(old_edges)
        new_edges = [(i, node, d) for i, d in zip(nbunch, edges_attrs)]
        graph.add_edges_from(new_edges)
        
    def _extract_name_and_attr(self, argument):
        graph = self.graph
        splitted_attributes = argument.split('.')
        node_name = splitted_attributes[0]
        attribute_string = '.'.join(splitted_attributes[1:])
        if node_name not in graph.nodes:
            raise ValueError('Node %r is not is the graph.'%node_name)
        return node_name, attribute_string
    
    def _get_nodes_attribute(self, nodes, attribute):
        graph = self.graph
        sub_graph = graph.subgraph(nodes)
        attr_list = list(nx.get_node_attributes(sub_graph, attribute).values())
        return attr_list
    
    def _get_node_predecessors(self, node):
        graph = self.graph
        edges = reversed([e[:-1] for e in nx.edge_bfs(graph, node, 'reverse')])
        return edges

###############################################################################
###############################################################################

class abstract_configuration(relational_graph):
                    
    def add_node(self, name, symbolic_type, sym='', mirror=False):
        if mirror:
            node1 = '%sr_%s'%(sym, name)
            node2 = '%sl_%s'%(sym, name)
            node1_attr_dict = self._create_node_dict(node1, symbolic_type, node2, align='r')
            node2_attr_dict = self._create_node_dict(node2, symbolic_type, node1, align='l')
            super().add_node(node1, **node1_attr_dict)
            super().add_node(node2, **node2_attr_dict)
            self.add_relation(Mirrored, node2, (node1,))
        else:
            node1 = '%ss_%s'%(sym, name)
            node1_attr_dict = self._create_node_dict(node1, symbolic_type, node1, align='s')
            super().add_node(node1, **node1_attr_dict)
        return node1
    
    def add_relation(self, relation, node, arg_nodes, mirror=False):
        if mirror:
            node1 = node
            node2 = self.graph.nodes[node1]['mirr']
            args1 = arg_nodes
            args2 = [self.graph.nodes[i]['mirr'] for i in args1]
            super().add_relation(node1, args1)
            super().add_relation(node2, args2)
            self._update_node_rhs(node1, relation)
            self._update_node_rhs(node2, relation)
        else:
            super().add_relation(node, arg_nodes)
            self._update_node_rhs(node, relation)
    
    
    def _update_node_rhs(self, node, rhs_function):
        self.graph.nodes[node]['rhs_function'] = rhs_function
    
    def _create_node_dict(self, name, symbolic_type, mirr='',align='s'):
        node_object = symbolic_type(name)
        function = self._get_initial_equality(node_object)
        attributes_dict = {'lhs_value':node_object, 'rhs_function':function,
                           'mirr':mirr, 'align':align}
        return attributes_dict

    @staticmethod
    def _get_initial_equality(node_object):
        if isinstance(node_object, sm.MatrixSymbol):
            return sm.Eq(node_object, sm.zeros(*node_object.shape))
        elif isinstance(node_object, sm.Symbol):
            return sm.Eq(node_object, 1.0)
        elif issubclass(node_object, sm.Function):
            t = sm.symbols('t')
            return sm.Eq(node_object, sm.Lambda(t, 0.0))
    
###############################################################################
###############################################################################

class configuration(abstract_configuration):                

    def __init__(self, name, model_instance):
        super().__init__(name)
        self.topology = model_instance._mbs
        self.geometries_map = {}

        
    @property
    def arguments_symbols(self):
        nodes = self.graph.nodes(data='lhs_value')
        return [n[-1] for n in nodes]

    @property
    def primary_arguments(self):
        return self.topology.arguments_symbols
    
    @property
    def geometry_nodes(self):
        nodes = self.graph.nodes
        geometries = [n for n in nodes if isinstance(n['lhs_value'], Geometry)]
        return geometries
    
    def assemble_geometries_graph_data(self):
        graph = self.graph
        geo_graph = graph.subgraph(self.geometry_nodes)
        
        input_nodes = self.get_input_nodes(geo_graph)
        input_equal = self.get_nodes_attributes(graph, input_nodes, 'rhs_function')

        mid_nodes = self.get_intermediat_nodes(geo_graph)
        mid_equal = self.get_nodes_attributes(graph, mid_nodes, 'rhs_function')

        output_nodes = self.get_output_nodes(geo_graph)
        output_equal = self.get_nodes_attributes(graph, output_nodes, 'rhs_function')
        
        data = {'input_nodes':input_nodes,
                'input_equal':input_equal,
                'output_nodes':output_nodes,
                'output_equal':output_equal + mid_equal,
                'geometries_map':self.geometries_map}
        return data

    def create_inputs_dataframe(self):
        nodes  = self.graph.nodes
        inputs = self.input_nodes
        condition = lambda i:  isinstance(nodes[i]['lhs_value'], sm.MatrixSymbol)\
                            or isinstance(nodes[i]['lhs_value'], sm.Symbol)
        indecies = list(filter(condition, inputs))
        indecies.sort()
        shape = (len(indecies),4)
        dataframe = pd.DataFrame(np.zeros(shape),index=indecies,dtype=np.float64)
        return dataframe

    def assemble_base_layer(self):
        edges_data = list(zip(*self.topology.edges(data=True)))
        edges_arguments = self._extract_primary_arguments(edges_data[-1])
        self._add_primary_nodes(edges_arguments)
        
        nodes_data = list(zip(*self.topology.nodes(data=True)))
        nodes_arguments = self._extract_primary_arguments(nodes_data[-1])
        self._add_primary_nodes(nodes_arguments)

        self.bodies = {n:self.topology.nodes[n] for n in self.topology.bodies}

            
    def assign_geometry_to_body(self, body, geo, eval_inertia=True, mirror=False):
        b1 = body
        g1 = geo
        b2 = self.bodies[body]['mirr']
        g2 = self.graph.nodes[geo]['mirr']
        self._assign_geometry_to_body(b1, g1, eval_inertia)
        if b1 != b2 : self._assign_geometry_to_body(b2, g2, eval_inertia)


    def _add_primary_nodes(self, arguments_lists):
        single_args, right_args, left_args = arguments_lists
        for arg in single_args:
            node = str(arg)
            self._add_primary_node(arg, mirr=node, align='s')
        for arg1, arg2 in zip(right_args, left_args):
            node1 = str(arg1)
            node2 = str(arg2)
            self._add_primary_node(arg1, mirr=node2, align='r')
            self._add_primary_node(arg2, mirr=node1, align='l')
            relation = self._get_primary_mirrored_relation(arg1)
            self.add_relation(relation, node2, (node1,))
        
    def _add_primary_node(self, node_object, mirr='', align='s'):
        name = str(node_object)
        function = self._get_initial_equality(node_object)
        attributes_dict = {'lhs_value':node_object, 'rhs_function':function,
                           'mirr':mirr, 'align':align}
        self.graph.add_node(name, **attributes_dict)

    def _assign_geometry_to_body(self, body, geo, eval_inertia=True):
        b = self.bodies[body]['obj']
        R, P, m, J = [str(getattr(b,i)) for i in 'R,P,m,Jbar'.split(',')]
        self.geometries_map[geo] = body
        if eval_inertia:
            self.add_relation(CR.Equal_to, R, ('%s.R'%geo,))
            self.add_relation(CR.Equal_to, P, ('%s.P'%geo,))
            self.add_relation(CR.Equal_to, J, ('%s.J'%geo,))
            self.add_relation(CR.Equal_to, m, ('%s.m'%geo,))


    @staticmethod
    def _extract_primary_arguments(data_dict):
        s_args = [n['arguments_symbols'] for n in data_dict if n['align']=='s']
        r_args = [n['arguments_symbols'] for n in data_dict if n['align']=='r']
        l_args = [n['arguments_symbols'] for n in data_dict if n['align']=='l']
        arguments = [itertools.chain(*i) for i in (s_args, r_args, l_args)]
        return arguments

    @staticmethod
    def _get_primary_mirrored_relation(arg1):
        if isinstance(arg1, sm.MatrixSymbol):
            return Mirrored
        elif isinstance(arg1, sm.Symbol):
            return Equal_to
        elif issubclass(arg1, sm.Function):
            return Equal_to

###############################################################################
###############################################################################
