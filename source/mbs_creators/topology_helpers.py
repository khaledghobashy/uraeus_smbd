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

from source.symbolic_classes.abstract_matrices import (vector, quatrenion,
                                                       Mirrored, matrix_symbol,
                                                       Equal_to)

###############################################################################
###############################################################################

class geometry(sm.Symbol):
    """
    A symbolic geometry class.
    
    Parameters
    ----------
    name : str
        Name of the geometry object
    
    """
    def __new__(cls,name,*args):
        return super().__new__(cls,name)
    
    def __init__(self,name,*args):
        self.name = name
        self._args = args

        self.R = vector('%s.R'%name)
        self.P = quatrenion('%s.P'%name)
        self.m = sm.symbols('%s.m'%name)
        self.J = matrix_symbol('%s.J'%name,3,3)
        
    def __call__(self,*args):
        return geometry(self.name,*args)


class simple_geometry(sm.Function):
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
        return r'%s%s'%(name,(*self.args,))

class composite_geometry(simple_geometry):
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

class cylinder_geometry(simple_geometry):
    
    def __init__(self,arg1,arg2,ro=10,ri=0):
        self.arg1 = arg1
        self.arg2 = arg2
        self.ri = ri
        self.ro = ro

class triangular_prism(simple_geometry):
    
    def __init__(self,arg1,arg2,arg3,l=10):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3
        self.l = l

    
###############################################################################
###############################################################################

class parametric_configuration(object):
        
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
        
    def mid_equalities(self,graph,output_nodes):
        mid_layer = []
        for n in output_nodes:
            self._get_node_dependencies(n,mid_layer)
        return [graph.nodes[n]['func'] for n in mid_layer]
    
    @property
    def geometry_nodes(self):
        return set(i[0] for i in self.graph.nodes(data='obj') if isinstance(i[-1],geometry))


    def assemble_base_layer(self):
        self._get_nodes_arguments()
        self._get_edges_arguments()
        nx.set_node_attributes(self.graph,True,'primary')
        self.bodies = {n:self.topology.nodes[n] for n in self.topology.bodies}
        self._get_topology_args()
    
    def add_scalar(self,name):
        self._add_nodes(name,False,'',sm.symbols)
    
    def add_point(self,name,mirror=False):
        self._add_nodes(name,mirror,'hp')
    
    def add_vector(self,name,mirror=False):
       self._add_nodes(name,mirror,'vc')
   
    def add_geometry(self,name,mirror=False):
        self._add_nodes(name,mirror,'gm',geometry)
    
    def assign_geometry_to_body(self,body,geo,eval_inertia=True,mirror=False):
        b1 = body
        g1 = geo
        b2 = self.bodies[body]['mirr']
        g2 = self.graph.nodes[geo]['mirr']
        self._assign_geometry_to_body(b1,g1,eval_inertia)
        if b1 != b2 : self._assign_geometry_to_body(b2,g2,eval_inertia)
            
        
    def add_relation(self,node,relation,*nbunch,mirror=False):
        if mirror:
            node1 = node
            node2 = self.graph.nodes[node1]['mirr']
            nbunch1 = nbunch
            nbunch2 = [self.graph.nodes[i]['mirr'] for i in nbunch1]
            self._add_relation(node1,relation,*nbunch1)
            self._add_relation(node2,relation,*nbunch2)
        else:
            self._add_relation(node,relation,*nbunch)
    
    def add_sub_relation(self,node,relation,*nbunch,mirror=False):
        if mirror:
            node1 = node
            node2 = self.graph.nodes[node1]['mirr']
            nbunch1 = nbunch
            nbunch2 = [self.graph.nodes[i]['mirr'] for i in nbunch1]
            self._add_sub_relation(node1,relation,*nbunch1)
            self._add_sub_relation(node2,relation,*nbunch2)
        else:
            self._add_sub_relation(node,relation,*nbunch)
    
    
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
    
    def _get_node_dependencies(self,n,mid_layer):
        self.get_node_deps(self.graph,n,self.input_nodes,mid_layer)
    
    def _add_nodes(self,name,mirror=False,sym='hp',typ=vector):
        Eq = self._set_base_equality
        graph = self.graph
        if mirror:
            node1 = '%sr_%s'%(sym,name)
            node2 = '%sl_%s'%(sym,name)
            self._add_node(node1,typ)
            self._add_node(node2,typ)
            graph.nodes[node1].update({'mirr':node2,'align':'r'})
            graph.nodes[node2].update({'mirr':node1,'align':'l'})
            obj1 = graph.nodes[node1]['obj']
            obj2 = graph.nodes[node2]['obj']
            graph.nodes[node2].update({'func':Eq(obj1,obj2)})
            graph.add_edge(node1,node2)
        else:
            node1 = node2 = '%ss_%s'%(sym,name)
            self._add_node(node1,typ)
            graph.nodes[node1]['mirr'] = node2
    
    def _add_node(self,name,typ):
        obj = typ(name)
        attr_dict = self._obj_attr_dict(obj)
        self.graph.add_node(name,**attr_dict)
    
    def _add_relation(self,node,relation,*nbunch):
        graph = self.graph
        edges = [(i,node) for i in nbunch]
        obj = graph.nodes[node]['obj']
        nobj = [graph.nodes[n]['obj'] for n in nbunch]
        graph.nodes[node]['func'] = sm.Equality(obj,relation(*nobj))
        removed_edges = list(graph.in_edges(node))
        graph.remove_edges_from(removed_edges)
        graph.add_edges_from(edges)
    
    def _add_sub_relation(self,node,relation,*nbunch):
        graph = self.graph
        mod_nbunch  = []
        mod_objects = []
        obj = graph.nodes[node]['obj']
        for n in nbunch:
            splited = n.split('.')
            name = splited[0]
            attr = '.'.join(splited[1:])
            mod_nbunch.append(name)
            nobj = getattr(graph.nodes[name]['obj'],attr)
            mod_objects.append(nobj)
        edges = [(i,node) for i in mod_nbunch]
        graph.nodes[node]['func'] = sm.Equality(obj,relation(*mod_objects))
        removed_edges = list(graph.in_edges(node))
        graph.remove_edges_from(removed_edges)
        graph.add_edges_from(edges)
    
    def _assign_geometry_to_body(self,body,geo,eval_inertia):
        b = self.bodies[body]['obj']
        R, P, m, J = [str(getattr(b,i)) for i in 'R,P,m,Jbar'.split(',')]
        self.geometries_map[geo] = body
        if eval_inertia:
            self.add_sub_relation(R,Equal_to,'%s.R'%geo)
            self.add_sub_relation(P,Equal_to,'%s.P'%geo)
            self.add_sub_relation(J,Equal_to,'%s.J'%geo)
            self.add_sub_relation(m,Equal_to,'%s.m'%geo)


    def _obj_attr_dict(self,obj):
        Eq = self._set_base_equality
        attr_dict = {'obj':obj,'mirr':None,'align':'s','func':Eq(obj),
                     'primary':False}
        return attr_dict
    
    
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
                nx.set_node_attributes(self.graph,mirr,'mirr')
            else:
                args_m = t_nodes[m]['arguments_symbols']
                args_c = zip(args_n,args_m)
                nodes_args_m = [(str(m),{'func':Eq(n,m),'obj':m}) for n,m in args_c]
                graph.add_nodes_from(nodes_args_n+nodes_args_m)
                edges = [(str(n),str(m)) for n,m in zip(args_n,args_m)]
                graph.add_edges_from(edges)    
                mirr = {m:n for n,m in edges}
                nx.set_node_attributes(self.graph,mirr,'mirr')
                mirr = {n:m for n,m in edges}
                nx.set_node_attributes(self.graph,mirr,'mirr')
    
    
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
    
    
    
    def _set_base_equality(self,sym1,sym2=None):
        if sym1 and sym2:
            if isinstance(sym1,sm.MatrixSymbol):
                return sm.Eq(sym2,Mirrored(sym1))
            elif isinstance(sym1,sm.Symbol):
                return sm.Eq(sym2,sym1)
            elif issubclass(sym1,sm.Function):
                return sm.Eq(sym2,sym1)
        else:
            if isinstance(sym1,sm.MatrixSymbol):
                return sm.Eq(sym1,sm.zeros(*sym1.shape))
            elif isinstance(sym1,sm.Symbol):
                return sm.Eq(sym1,1)
            elif issubclass(sym1,sm.Function):
                t = sm.symbols('t')
                return sm.Eq(sym1,sm.Lambda(t,0))
    
#    @staticmethod
#    def _set_mirror_equality(arg1,arg2):
#         return sm.Eq(arg2,Mirrored(arg1),evaluate=False)
#    @staticmethod
#    def _set_direct_equality(arg1,arg2):
#        return sm.Eq(arg2,arg1,evaluate=False)
#    @staticmethod
#    def _set_array_equality(arg1):
#        default = sm.zeros(*arg1.shape)
#        return sm.Eq(arg1,default,evaluate=False)
#    @staticmethod
#    def _set_function_equality(arg1):
#        t = sm.symbols('t')
#        return sm.Eq(arg1,sm.Lambda(t,0),evaluate=False)
    
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
