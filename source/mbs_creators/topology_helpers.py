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

from source.symbolic_classes.abstract_matrices import vector, Mirrored

###############################################################################

class parametric_configuration(object):
    
    def __init__(self,mbs_instance):
        self.topology = mbs_instance
        self.name  = '%s_cfg'%self.topology.name
        self.graph = nx.DiGraph(name=self.name)
   
    @property
    def arguments_symbols(self):
        return set(nx.get_node_attributes(self.graph,'obj').values())
    
    @property
    def input_nodes(self):
        nodes = [i for i,d in self.graph.in_degree() if d == 0]
        return nodes
    @property
    def input_equalities(self):
        g = self.graph
        nodes = self.input_nodes
        equalities = [g.nodes[i]['func'] for i,d in g.in_degree(nodes) if d==0]
        return equalities
    
    @property
    def output_nodes(self):
        condition = lambda i,d : d==0 and self.graph.in_degree(i)!=0
        nodes = [i for i,d in self.graph.out_degree() if condition(i,d)]
        return nodes
    @property
    def output_equalities(self):
        g = self.graph
        nodes = self.output_nodes
        equalities = [g.nodes[i]['func'] for i,d in g.out_degree(nodes) if d==0]
        return self.mid_equalities + equalities
    
    @property
    def mid_equalities(self):
        mid_layer = []
        for n in self.output_nodes:
            self._get_node_dependencies(n,mid_layer)
        return [self.graph.nodes[n]['func'] for n in mid_layer]

    def assemble_base_layer(self):
        self._get_nodes_arguments()
        self._get_edges_arguments()
        nx.set_node_attributes(self.graph,True,'primary')
    
    
    def add_point(self,name,mirror=False):
        Eq = self._set_base_equality
        graph = self.graph
        if mirror:
            node1 = 'hpr_%s'%name
            node2 = 'hpl_%s'%name
            self._add_point(node1)
            self._add_point(node2)
            graph.nodes[node1].update({'mirr':node2,'align':'r'})
            graph.nodes[node2].update({'mirr':node1,'align':'l'})
            obj1 = graph.nodes[node1]['obj']
            obj2 = graph.nodes[node2]['obj']
            graph.nodes[node2].update({'func':Eq(obj1,obj2)})
            graph.add_edge(node1,node2)
        else:
            node1 = node2 = 'hps_%s'%name
            self._add_point(node1)
            graph.nodes[node1]['mirr'] = node2

    
    def add_relation(self,relation,node,nbunch,mirror=False):
        if mirror:
            node1 = node
            node2 = self.graph.nodes[node1]['mirr']
            nbunch1 = nbunch
            nbunch2 = [self.graph.nodes[i]['mirr'] for i in nbunch1]
            self._add_relation(relation,node1,nbunch1)
            self._add_relation(relation,node2,nbunch2)
        else:
            self._add_relation(relation,node,nbunch)
    
    def create_inputs_dataframe(self):
        equalities = self.input_equalities
        indecies = [str(i.lhs) for i in equalities 
                    if isinstance(i.lhs,sm.MatrixSymbol)]
        indecies.sort()
        shape = (len(indecies),4)
        dataframe = pd.DataFrame(np.zeros(shape),index=indecies,dtype=np.float64)
        return dataframe
    
    def draw_node_dependencies(self,n):
        edges = [e[:-1] for e in nx.edge_bfs(self.graph,n,'reverse')]
        g = self.graph.edge_subgraph(edges)
        plt.figure(figsize=(10,6))
        nx.draw_networkx(g,with_labels=True)
        plt.show() 
        
    def draw_graph(self):
        plt.figure(figsize=(10,6))
        nx.draw_circular(self.graph,with_labels=True)
        plt.show()     

            
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
            args_n = t_nodes[n]['arguments']
            nodes_args_n = [(str(i),{'func':Eq(i),'obj':i}) for i in args_n]
            if m == n:
                graph.add_nodes_from(nodes_args_n)
                mirr = {i[0]:i[0] for i in nodes_args_n}
                nx.set_node_attributes(self.graph,mirr,'mirr')
            else:
                args_m = t_nodes[m]['arguments']
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
        for e in t_edges:
            n = t_edges[e]['name']
            if t_edges[e]['align'] in 'sr':
                m = t_edges[e]['mirr']
                args_n = t_edges[e]['arguments']
                nodes_args_n = [(str(i),{'func':Eq(i),'obj':i}) for i in args_n]
                if m == n:
                    graph.add_nodes_from(nodes_args_n)
                    mirr = {i[0]:i[0] for i in nodes_args_n}
                    nx.set_node_attributes(self.graph,mirr,'mirr')
                else:
                    e2 = self.topology._edges_map[m]
                    args_m = t_edges[e2]['arguments']
                    args_c = zip(args_n,args_m)
                    nodes_args_m = [(str(m),{'func':Eq(n,m),'obj':m}) for n,m in args_c]
                    graph.add_nodes_from(nodes_args_n+nodes_args_m)
                    edges = [(str(n),str(m)) for n,m in zip(args_n,args_m)]
                    graph.add_edges_from(edges)    
                    mirr = {m:n for n,m in edges}
                    nx.set_node_attributes(self.graph,mirr,'mirr')
                    mirr = {n:m for n,m in edges}
                    nx.set_node_attributes(self.graph,mirr,'mirr')
    
    def _obj_attr_dict(self,obj):
        Eq = self._set_base_equality
        attr_dict = {'obj':obj,'mirr':None,'align':'s','func':Eq(obj),
                     'primary':False}
        return attr_dict
    
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

    def _add_point(self,name):
        v = vector(name)
        attr_dict = self._obj_attr_dict(v)
        self.graph.add_node(name,**attr_dict)

    def _add_relation(self,relation,node,nbunch):
        graph = self.graph
        edges = [(i,node) for i in nbunch]
        obj = graph.nodes[node]['obj']
        nobj = [graph.nodes[n]['obj'] for n in nbunch]
        graph.nodes[node]['func'] = sm.Equality(obj,relation(*nobj))
        removed_edges = list(graph.in_edges(node))
        graph.remove_edges_from(removed_edges)
        graph.add_edges_from(edges)
        
    def _get_node_dependencies1(self,n,mid_layer):
        edges = [e[:-1] for e in nx.edge_bfs(self.graph,n,'reverse')]
        print(edges)
        for e in edges:
            node = e[0]
            if node not in self.input_nodes and node not in mid_layer:
                mid_layer.append(node)
    
    def _get_node_dependencies(self,n,mid_layer):
        edges = reversed([e[:-1] for e in nx.edge_bfs(self.graph,n,'reverse')])
        for e in edges:
            node = e[0]
            if node not in self.input_nodes and node not in mid_layer:
                mid_layer.append(node)

    
###############################################################################
###############################################################################
