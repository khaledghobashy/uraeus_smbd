# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:44:13 2019

@author: khaled.ghobashy
"""

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
    
equalities_map = {vector: sm.Matrix}
    
class relational_graph(object):
    
    def __init__(self, name):
        self.name = name
        self.graph = nx.DiGraph(name=self.name)
        
    @property
    def input_nodes(self):
        return self.get_input_nodes(self.graph)
    @property
    def input_equalities(self):
        g = self.graph
        nodes = self.input_nodes
        equalities = nx.get_node_attributes(g, nodes, 'rhs_function')
        return equalities
    
    @property
    def output_nodes(self):
        return self.get_output_nodes(self.graph)
    @property
    def output_equalities(self):
        g = self.graph
        nodes = self.output_nodes
        equalities = nx.get_node_attributes(g, nodes, 'rhs_function')
        return equalities
        
    def add_node(self, name, symbolic_type, ):
        node_object    = symbolic_type(name)
        node_attr_dict = self._create_node_dict(node_object)
        self.graph.add_node(name, **node_attr_dict)
        
    def add_relation(self, relation, node, args):
        graph = self.graph
        name_object = [self._extract_name_and_object(graph, n) for n in args]
        arguments_names, arguments_objects = zip(*name_object)
        graph.nodes[node]['rhs_function'] = relation(*arguments_objects)
        self._update_in_edges(graph, node, arguments_names)
    
    
    def _create_node_dict(self, node_object):
        function = self._get_initial_equality(node_object)
        attr_dict = {'lhs_value':node_object, 'rhs_function':function}
        return attr_dict
    
    
    
    @staticmethod
    def _extract_name_and_object(graph, argument):
        splitted_attributes = argument.split('.')
        node_name = splitted_attributes[0]
        if node_name not in graph.nodes:
            raise ValueError('Node %r is not is the graph.'%node_name)
        
        node_object = graph.nodes[node_name]['lhs_value']
        if len(splitted_attributes) == 1:
            symbolic_object = node_object
        else:
            attribute_string = '.'.join(splitted_attributes[1:])
            symbolic_object  = getattr(node_object, attribute_string)
        
        return node_name, symbolic_object
        
    @staticmethod    
    def _update_in_edges(graph, node, nbunch):
        old_edges = list(graph.in_edges(node))
        graph.remove_edges_from(old_edges)
        new_edges = [(i, node) for i in nbunch]
        graph.add_edges_from(new_edges)

    @staticmethod
    def _get_initial_equality(node_object):
        if isinstance(node_object, sm.MatrixSymbol):
            return sm.zeros(*node_object.shape)
        elif isinstance(node_object, sm.Symbol):
            return 1.0
        elif issubclass(node_object, sm.Function):
            t = sm.symbols('t')
            return sm.Lambda(t, 0.0)

    @staticmethod
    def get_input_nodes(graph):
        nodes = [i for i,d in graph.in_degree() if d == 0]
        return nodes
    
    @staticmethod
    def get_output_nodes(graph):
        condition = lambda i,d : d==0 and graph.in_degree(i)!=0
        nodes = [i for i,d in graph.out_degree() if condition(i,d)]
        return nodes
    
    @staticmethod
    def get_node_deps(graph, node):
        edges = reversed([e[:-1] for e in nx.edge_bfs(graph, node, 'reverse')])
        return edges
    
    
###############################################################################
###############################################################################

class abstract_configuration(object):
    
    def __init__(self,name,mbs_instance):
        self.name  = name
        self.topology = mbs_instance
        self.graph = nx.DiGraph(name=self.name)
        self.geometries_map = {}

    @property
    def arguments_symbols(self):
        return set(nx.get_node_attributes(self.graph,'obj').values())

    @property
    def primary_arguments(self):
        graph = self.graph
        cond = lambda n : graph.nodes[n]['primary']
        args = filter(cond,graph.nodes)
        return set(args)
    
    @property
    def input_nodes(self):
        return self.get_input_nodes(self.graph)
    @property
    def input_equalities(self):
        g = self.graph
        nodes = self.input_nodes
        return self.get_input_equalities(g,nodes)
    
    @property
    def output_nodes(self):
        return self.get_output_nodes(self.graph)
    @property
    def output_equalities(self):
        g = self.graph
        nodes = self.output_nodes
        equalities = self.get_output_equalities(g,nodes)
        return self.mid_equalities(g,nodes) + equalities
            
    @property
    def geometry_nodes(self):
        return set(i[0] for i in self.graph.nodes(data='obj') if isinstance(i[-1],Geometry))

    def mid_equalities(self,graph,output_nodes):
        mid_layer = []
        for n in output_nodes:
            self._get_node_dependencies(n,mid_layer)
        return [graph.nodes[n]['func'] for n in mid_layer]
    
    def assemble_base_layer(self):
        self._get_nodes_arguments()
        self._get_edges_arguments()
        nx.set_node_attributes(self.graph,True,'primary')
        self.bodies = {n:self.topology.nodes[n] for n in self.topology.bodies}
        self._get_topology_args()
        
    def get_node_dependencies(self,n):
        edges = [e[:-1] for e in nx.edge_bfs(self.graph,n,'reverse')]
        g = self.graph.edge_subgraph(edges)
        return g
    
    def draw_node_dependencies(self,n):
        g = self.get_node_dependencies(n)
        plt.figure(figsize=(10,6))
        nx.draw_networkx(g,with_labels=True)
        plt.show() 
        
    def draw_graph(self):
        plt.figure(figsize=(10,6))
        nx.draw_circular(self.graph,with_labels=True)
        plt.show()
    
    def get_geometries_graph_data(self):
        geo_graph = self.get_node_dependencies(self.geometry_nodes)
        input_nodes  = self.get_input_nodes(geo_graph)
        input_equal  = self.get_input_equalities(geo_graph,input_nodes)
        output_nodes = self.get_output_nodes(geo_graph)
        output_equal = self.get_output_equalities(geo_graph,output_nodes)
        mid_equal = self.mid_equalities(geo_graph,output_nodes)
        output_equal = mid_equal + output_equal
        data = {'input_nodes':input_nodes,
                'input_equal':input_equal,
                'output_nodes':output_nodes,
                'output_equal':output_equal,
                'geometries_map':self.geometries_map}
        return data

    def create_inputs_dataframe(self):
        nodes  = self.graph.nodes
        inputs = self.input_nodes
        condition = lambda i:  isinstance(nodes[i]['obj'], sm.MatrixSymbol)\
                            or isinstance(nodes[i]['obj'], sm.Symbol)
        indecies = list(filter(condition,inputs))
        indecies.sort()
        shape = (len(indecies),4)
        dataframe = pd.DataFrame(np.zeros(shape),index=indecies,dtype=np.float64)
        return dataframe


    def _get_topology_args(self):
        args = {}
        nodes_args = dict(self.topology.nodes(data='arguments_symbols'))
        nodes_args = {n:arg for n,arg in nodes_args.items()}
        edges = self.topology.edges
        edges_args = {e:edges[e]['arguments_symbols'] for e in edges}
        args.update(nodes_args)
        args.update(edges_args)
        self._base_args = args
        self._base_nodes = sum(args.values(),[])
        
    def _get_nodes_arguments(self):
        Eq = self._set_base_equality
        graph   = self.graph
        t_nodes = self.topology.nodes
        
        def filter_cond(n):
            cond = (n not in self.topology.virtual_bodies 
                    and t_nodes[n]['align'] in 'sr')
            return cond
        filtered_nodes = filter(filter_cond,t_nodes)

        for n in filtered_nodes:
            m      = t_nodes[n]['mirr']
            args_n = t_nodes[n]['arguments_symbols']
            nodes_args_n = [(str(i),{'func':Eq(i),'obj':i}) for i in args_n]
            if m == n:
                graph.add_nodes_from(nodes_args_n)
                mirr = {i[0]:i[0] for i in nodes_args_n}
                nx.set_node_attributes(self.graph, mirr, 'mirr')
            else:
                args_m = t_nodes[m]['arguments_symbols']
                args_c = zip(args_n,args_m)
                nodes_args_m = [(str(m),{'func':Eq(n,m),'obj':m}) for n,m in args_c]
                graph.add_nodes_from(nodes_args_n+nodes_args_m)
                edges = [(str(n),str(m)) for n,m in zip(args_n,args_m)]
                graph.add_edges_from(edges)    
                mirr = {m:n for n,m in edges}
                nx.set_node_attributes(self.graph, mirr, 'mirr')
                mirr = {n:m for n,m in edges}
                nx.set_node_attributes(self.graph, mirr, 'mirr')
    
    def _get_edges_arguments(self):
        Eq = self._set_base_equality
        graph   = self.graph
        t_edges = self.topology.edges
        
        def filter_cond(e):
            cond = (e not in self.topology.virtual_edges
                    and t_edges[e]['align'] in 'sr')
            return cond
        filtered_edges = filter(filter_cond,t_edges)
        
        for e in filtered_edges:
            n = t_edges[e]['name']
            m = t_edges[e]['mirr']
            args_n = t_edges[e]['arguments_symbols']
            nodes_args_n = [(str(i),{'func':Eq(i),'obj':i}) for i in args_n]
            if m == n:
                graph.add_nodes_from(nodes_args_n)
                mirr = {i[0]:i[0] for i in nodes_args_n}
                nx.set_node_attributes(self.graph,mirr,'mirr')
            else:
                e2 = self.topology._edges_map[m]
                args_m = t_edges[e2]['arguments_symbols']
                args_c = zip(args_n,args_m)
                nodes_args_m = [(str(m),{'func':Eq(n,m),'obj':m}) for n,m in args_c]
                graph.add_nodes_from(nodes_args_n+nodes_args_m)
                edges = [(str(n),str(m)) for n,m in zip(args_n,args_m)]
                graph.add_edges_from(edges)    
                mirr = {m:n for n,m in edges}
                nx.set_node_attributes(self.graph,mirr,'mirr')
                mirr = {n:m for n,m in edges}
                nx.set_node_attributes(self.graph,mirr,'mirr')
    
    def _get_node_dependencies(self,n,mid_layer):
        self.get_node_deps(self.graph, n, self.input_nodes, mid_layer)

    def _set_base_equality(self, sym1, sym2=None):
        if sym1 and sym2:
            if isinstance(sym1, sm.MatrixSymbol):
                return sm.Eq(sym2, CR.Mirrored(sym1))
            
            elif isinstance(sym1, sm.Symbol):
                return sm.Eq(sym2, sym1)
            
            elif issubclass(sym1, sm.Function):
                return sm.Eq(sym2, sym1)
        
        else:
            if isinstance(sym1, sm.MatrixSymbol):
                return sm.Eq(sym1, sm.zeros(*sym1.shape))
            
            elif isinstance(sym1, sm.Symbol):
                return sm.Eq(sym1, 1)
            
            elif issubclass(sym1, sm.Function):
                t = sm.symbols('t')
                return sm.Eq(sym1, sm.Lambda(t, 0))

    @staticmethod
    def get_input_nodes(graph):
        nodes = [i for i,d in graph.in_degree() if d == 0]
        return nodes
    @staticmethod
    def get_input_equalities(graph,nodes):
        g = graph
        equalities = [g.nodes[i]['func'] for i,d in g.in_degree(nodes) if d==0]
        return equalities
    
    @staticmethod
    def get_output_nodes(graph):
        condition = lambda i,d : d==0 and graph.in_degree(i)!=0
        nodes = [i for i,d in graph.out_degree() if condition(i,d)]
        return nodes
    @staticmethod
    def get_output_equalities(graph,nodes):
        g = graph
        equalities = [g.nodes[i]['func'] for i,d in g.out_degree(nodes) if d==0]
        return equalities
    
    @staticmethod
    def get_node_deps(graph,n,input_nodes,mid_layer):
        edges = reversed([e[:-1] for e in nx.edge_bfs(graph,n,'reverse')])
        for e in edges:
            node = e[0]
            if node not in input_nodes and node not in mid_layer:
                mid_layer.append(node)

###############################################################################
###############################################################################

class standalone_configuration(abstract_configuration):
    
    def __init__(self, name, mbs_instance):
        super().__init__(name, mbs_instance)
        self._decorate_methods()
    
    def assemble_base_layer(self):
        super().assemble_base_layer()
        base_nodes_class = type('base_nodes_class', (), {})
        def dummy_init(dself): pass
        base_nodes_class.__init__ = dummy_init
        base_nodes_instance = base_nodes_class()
        for n in self.input_nodes:
            setattr(base_nodes_instance, n, n)
        self.primary_variable = base_nodes_instance
    
    @property
    def add_point(self):
        return self._point_methods
    
    @property
    def add_vector(self):
        return self._vector_methods
    
    @property
    def add_scalar(self):
        return self._scalar_methods
    
    @property
    def add_geometry(self):
        return self._geometry_methods
    
    @property
    def add_relation(self):
        return self._relation_methods
    
    def assign_geometry_to_body(self, body, geo, eval_inertia):
        b = self.bodies[body]['obj']
        R, P, m, J = [str(getattr(b,i)) for i in 'R,P,m,Jbar'.split(',')]
        self.geometries_map[geo] = body
        if eval_inertia:
            self._add_relation(CR.Equal_to, R, ('%s.R'%geo,))
            self._add_relation(CR.Equal_to, P, ('%s.P'%geo,))
            self._add_relation(CR.Equal_to, J, ('%s.J'%geo,))
            self._add_relation(CR.Equal_to, m, ('%s.m'%geo,))

    
    def _decorate_methods(self):
        self._decorate_point_methods()
        self._decorate_vector_methods()
        self._decorate_scalar_methods()
        self._decorate_geometry_methods()
        self._decorate_relation_methods()
        
    def _add_node(self, typ, name):
        obj = typ(name)
        attr_dict = self._obj_attr_dict(obj)
        self.graph.add_node(name, **attr_dict)
    
    def _obj_attr_dict(self ,obj):
        Eq = self._set_base_equality
        attr_dict = {'obj':obj, 'mirr':None, 'align':'s', 'func':Eq(obj),
                     'primary':False}
        return attr_dict

    def _add_relation(self, relation, node, args):
        graph = self.graph
        
        node_object = graph.nodes[node]['obj']
        name_object = [self._extract_name_and_object(n) for n in args]
        arguments_names, arguments_objects = zip(*[item for item in name_object])
        
        graph.nodes[node]['func'] = sm.Equality(node_object, relation(*arguments_objects))        
        
        old_edges = list(graph.in_edges(node))
        graph.remove_edges_from(old_edges)
        new_edges = [(i, node) for i in arguments_names]
        graph.add_edges_from(new_edges)
        
        
    def _extract_name_and_object(self, argument):
        splitted_attributes = argument.split('.')
        node_name = splitted_attributes[0]
        if len(splitted_attributes) == 1:
            symbolic_object = self.graph.nodes[node_name]['obj']
        else:
            node_object = self.graph.nodes[node_name]['obj']
            attribute_string = '.'.join(splitted_attributes[1:])
            symbolic_object  = getattr(node_object, attribute_string)
        return node_name, symbolic_object
            
    
    def _decorate_relation_methods(self):
        sym = None
        node_type = None
        methods = ['Mirrored', 'Centered', 'Equal_to', 'Oriented', 'UserInput']
        self._relation_methods = self._decorate_components(node_type, sym, methods, CR)
    
    def _decorate_point_methods(self):
        sym = 'hp'
        node_type = vector
        methods = ['Mirrored', 'Centered', 'Equal_to', 'UserInput']
        self._point_methods = self._decorate_components(node_type, sym, methods, CR)
        
    def _decorate_vector_methods(self):
        sym = 'vc'
        node_type = vector
        methods = ['Mirrored', 'Oriented', 'Equal_to', 'UserInput']
        self._vector_methods = self._decorate_components(node_type, sym, methods, CR)

    def _decorate_scalar_methods(self):
        sym = ''
        node_type = sm.symbols
        methods = ['Equal_to', 'UserInput']
        self._scalar_methods = self._decorate_components(node_type, sym, methods, CR)
            
    def _decorate_geometry_methods(self):
        sym = 'gm'
        node_type = Geometry
        methods = ['Composite_Geometry', 'Cylinder_Geometry', 'Triangular_Prism']
        self._geometry_methods = self._decorate_components(node_type, sym, methods, Geometries)
            
    
    def _decorate_components(self, node_type, sym, methods_list, methods_class):   
        container_class = type('container', (object,), {})
        def dummy_init(dself): pass
        container_class.__init__ = dummy_init
        container_instance = container_class()

        for name in methods_list:
            method = getattr(methods_class, name)
            decorated_method = self._decorate_as_attr(node_type, sym, method)
            setattr(container_instance, name, decorated_method)
        
        return container_instance
    
    def _decorate_as_attr(self, node_type, sym, construction_method):
        
        if construction_method is None:
            def decorated(*args, **kwargs):
                self._add_node(node_type, args[0], **kwargs, sym=sym)
            decorated.__doc__ = ''
        
        elif node_type is None:
            def decorated(*args, **kwargs):
                self._add_relation(construction_method, *args, **kwargs)
            decorated.__doc__ = construction_method.__doc__
       
        else:
            def decorated(*args, **kwargs):
                node = self._add_node(node_type, args[0], **kwargs, sym=sym)
                self._add_relation(construction_method, node, *args[1:], **kwargs)
            decorated.__doc__ = construction_method.__doc__
        
        return decorated

###############################################################################
###############################################################################

class parametric_configuration(standalone_configuration):
    
    def assign_geometry_to_body(self, body, geo, eval_inertia=True, mirror=False):
        b1 = body
        g1 = geo
        b2 = self.bodies[body]['mirr']
        g2 = self.graph.nodes[geo]['mirr']
        super().assign_geometry_to_body(b1,g1,eval_inertia)
        if b1 != b2 : super().assign_geometry_to_body(b2,g2,eval_inertia)
            
    def _add_node(self, typ, name, sym='', mirror=False):
        Eq = self._set_base_equality
        graph = self.graph
        if mirror:
            node1 = '%sr_%s'%(sym, name)
            node2 = '%sl_%s'%(sym, name)
            super()._add_node(typ, node1)
            super()._add_node(typ, node2)
            graph.nodes[node1].update({'mirr':node2, 'align':'r'})
            graph.nodes[node2].update({'mirr':node1, 'align':'l'})
            obj1 = graph.nodes[node1]['obj']
            obj2 = graph.nodes[node2]['obj']
            graph.nodes[node2].update({'func': Eq(obj1, obj2)})
            graph.add_edge(node1, node2)
        else:
            node1 = node2 = '%ss_%s'%(sym, name)
            super()._add_node(typ, node1)
            graph.nodes[node1]['mirr'] = node2
        return node1
            
    def _add_relation(self, relation, node, args, mirror=False):
        if mirror:
            node1 = node
            node2 = self.graph.nodes[node1]['mirr']
            args1 = args
            args2 = [self.graph.nodes[i]['mirr'] for i in args1]
            super()._add_relation(relation, node1, args1)
            super()._add_relation(relation, node2, args2)
        else:
            super()._add_relation(relation, node, args)

###############################################################################
###############################################################################
