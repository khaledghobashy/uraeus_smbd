# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:31:35 2019

@author: khale
"""

import sympy as sm
import matplotlib.pyplot as plt
import networkx as nx
import copy

from source.symbolic_classes.abstract_matrices import global_frame, reference_frame, mbs_string
from source.symbolic_classes.bodies import body, ground, virtual_body
from source.symbolic_classes.spatial_joints import absolute_locator
from source.symbolic_classes.algebraic_constraints import joint_actuator


class abstract_topology(object):
    
    def __init__(self,name):
        self.graph = nx.MultiGraph(name=name)
        self.name = name
        self._edges_map = {}
        
        # adding ground node
        attr_dict = self._typ_attr_dict(ground)
        self.graph.add_node('ground',**attr_dict)
        
        self.input_args_graph = nx.MultiDiGraph(name=name)
    
    @property
    def selected_variant(self):
        return self.graph
    
    @property
    def nodes(self):
        return self.selected_variant.nodes
    @property
    def edges(self):
        return self.selected_variant.edges
        
    @property
    def n(self):
        n = sum([node[-1] for node in self.nodes(data='n')])
        return n
    @property
    def nc(self):
        nc_nodes = sum([node[-1] for node in self.nodes(data='nc')])
        nc_edges = sum([edge[-1] for edge in self.edges(data='nc')])
        return nc_nodes + nc_edges
    @property
    def nve(self):
        nve_nodes = sum([node[-1] for node in self.nodes(data='nve')])
        nve_edges = sum([edge[-1] for edge in self.edges(data='nve')])
        return nve_nodes + nve_edges
    
    @staticmethod
    def _typ_attr_dict(typ):
        attr_dict ={'n':typ.n,'nc':typ.nc,'nve':typ.nve,
                    'class':typ,'mirr':None,'align':'s'}
        return attr_dict
    @staticmethod
    def _obj_attr_dict(obj):
        attr_dict ={'obj':obj,'num_args':obj.numerical_arguments(),
                    'config_const':obj.configuration_constants()}
        return attr_dict
    
    
    def draw_topology(self):
        plt.figure(figsize=(10,6))
        nx.draw_spring(self.selected_variant,with_labels=True)
        plt.show()
    
    def _map_coordinates(self):
        q = []
        q_sym = sm.MatrixSymbol('q',self.n,1)
        qd = []
        qd_sym = sm.MatrixSymbol('qd',self.n,1)
        i = 0
        for n in self.nodes(data='obj'):
            if self._check_if_virtual(n[0]):
                continue
            q.append(sm.Eq(n[-1].R,q_sym[i:i+3,0]))
            q.append(sm.Eq(n[-1].P,q_sym[i+3:i+3+4,0]))
            qd.append(sm.Eq(n[-1].Rd,qd_sym[i:i+3,0]))
            qd.append(sm.Eq(n[-1].Pd,qd_sym[i+3:i+3+4,0]))
            i+=7
        self.q_maped = q
        self.qd_maped = qd
    
    def _set_configuration_constants(self):
        cons = nx.get_edge_attributes(self.graph,'config_const').values()
        self.configuration_constants = sum(cons,[])

    def _set_numerical_arguments(self):
        edge_args = nx.get_edge_attributes(self.graph,'num_args').values()
        node_args = nx.get_node_attributes(self.graph,'num_args').values()
        self.numerical_arguments = sum(edge_args,[])+sum(node_args,[])
    
    def _initialize_toplogy_reqs(self):
        self._map_coordinates()
        self._set_configuration_constants()
        self._set_numerical_arguments()

    def _check_if_virtual(self,n):
        body_type = self.nodes[n]['class']
        return issubclass(body_type,virtual_body)
    
    def _assemble_nodes(self):
        for n in self.nodes:
            if self._check_if_virtual(n):
                continue
            body_type = self.nodes[n]['class']
            body_instance = body_type(n)
            self.nodes[n].update(self._obj_attr_dict(body_instance))
            
    def _assemble_edge(self,e):
        edge_class = self.edges[e]['class']
        b1, b2, name = e
        if issubclass(edge_class,joint_actuator):
            joint_name = self.edges[e]['joint_name']
            joint_object = self.edges[(b1,b2,joint_name)]['obj']
            edge_instance = edge_class(name,joint_object)
        elif issubclass(edge_class,absolute_locator):
            coordinate = self.edges[e]['coordinate']
            edge_instance = edge_class(name,self.nodes[b1]['obj'],coordinate)
        else:
            edge_instance = edge_class(name,self.nodes[b1]['obj'],self.nodes[b2]['obj'])
        self.edges[e].update(self._obj_attr_dict(edge_instance))

    def _assemble_edges(self):
        for e in self.edges:
            b1, b2, name = e
            if self._check_if_virtual(b1) or self._check_if_virtual(b2):
                continue
            self._assemble_edge(e)
    
    
    def _assemble_equations(self):
        
        edgelist = self.edges
        nodelist = self.nodes
        node_index = dict([(n,i) for i,n in enumerate(nodelist)])

        n_nodes = len(nodelist)
        cols = 2*n_nodes
        
        nve = self.nve  # number of constraint vector equations
        
        equations = sm.MutableSparseMatrix(nve,1,None)
        vel_rhs   = sm.MutableSparseMatrix(nve,1,None)
        acc_rhs   = sm.MutableSparseMatrix(nve,1,None)
        jacobian  = sm.MutableSparseMatrix(nve,cols,None)
        
        row_ind = 0
        for e in edgelist:
            u,v = e[:2]
            if self._check_if_virtual(u) or self._check_if_virtual(v):
#                print('     bypassing edge: %s'%(e,))
                continue
            eo  = self.edges[e]['obj']
            # tracker of row index based on the current joint type and the history
            # of the loop
            eo_nve = eo.nve+row_ind 
            
            ui = node_index[u]
            vi = node_index[v]
            
            # assigning the joint jacobians to the propper index in the system jacobian
            # on the "constraint vector equations" level.
            jacobian[row_ind:eo_nve,ui*2:ui*2+2] = eo.jacobian_i.blocks
            jacobian[row_ind:eo_nve,vi*2:vi*2+2] = eo.jacobian_j.blocks
            
            equations[row_ind:eo_nve,0] = eo.pos_level_equations.blocks
            vel_rhs[row_ind:eo_nve,0] = eo.vel_level_equations.blocks
            acc_rhs[row_ind:eo_nve,0] = eo.acc_level_equations.blocks
           
            row_ind += eo.nve
        
        for i,n in enumerate(nodelist):
            if self._check_if_virtual(n):
                continue
            b = self.nodes[n]['obj']
            if isinstance(b,ground):
                jacobian[row_ind:row_ind+2,i*2:i*2+2]    = b.normalized_jacobian.blocks
                equations[row_ind:row_ind+2,0] = b.normalized_pos_equation.blocks
                vel_rhs[row_ind:row_ind+2,0]   = b.normalized_vel_equation.blocks
                acc_rhs[row_ind:row_ind+2,0]   = b.normalized_acc_equation.blocks
            else:
                jacobian[row_ind,i*2+1] = b.normalized_jacobian[1]
                equations[row_ind,0] = b.normalized_pos_equation
                vel_rhs[row_ind,0]   = b.normalized_vel_equation
                acc_rhs[row_ind,0]   = b.normalized_acc_equation
                
            row_ind += b.nve
        
        self.pos_equations = equations
        self.vel_equations = vel_rhs
        self.acc_equations = acc_rhs
        self.jac_equations = jacobian
    
    def assemble_model(self):
        self._assemble_nodes()
        self._assemble_edges()
        self._assemble_equations()
        self._initialize_toplogy_reqs()

###############################################################################
###############################################################################

class topology(abstract_topology):
    
    def add_body(self,name):
        variant = self.selected_variant
        if name not in variant.nodes():
            attr_dict = self._typ_attr_dict(body)
            variant.add_node(name,**attr_dict)
        
    
    def add_joint(self,typ,name,body_i,body_j):
        variant = self.selected_variant
        edge  = (body_i,body_j,name)
        if edge not in variant.edges:
            attr_dict = self._typ_attr_dict(typ)
            variant.add_edge(*edge,**attr_dict)
            self._edges_map[name] = edge
            
    def add_joint_actuator(self,typ,name,joint_name):
        variant = self.selected_variant
        joint_edge = self._edges_map[joint_name]
        act_edge = (*joint_edge[:2],name)
        if act_edge not in variant.edges:
            attr_dict = self._typ_attr_dict(typ)
            variant.add_edge(*act_edge,**attr_dict,joint_name=joint_name)
            self._edges_map[name] = act_edge
    
    def add_absolute_actuator(self,name,body,coordinate):
        variant = self.selected_variant
        edge  = (body,'ground',name)
        if edge not in variant.edges:
            attr_dict = self._typ_attr_dict(absolute_locator)
            variant.add_edge(*edge,**attr_dict,coordinate=coordinate)
            self._edges_map[name] = edge
    
    def remove_body(self,name):
        self.selected_variant.remove_node(name)
    
    def remove_joint(self,name):
        edge = self._edges_map[name]
        self.selected_variant.remove_edge(*edge)
    
###############################################################################
###############################################################################

class template_based_topology(topology):
    
    def __init__(self,name):
        super().__init__(name)
        self.selected_variant.nodes['ground']['mirr']='ground'
        
    @property
    def virtual_bodies(self):
        condition = self._check_if_virtual
        virtuals  = filter(condition,self.selected_variant.nodes)
        return set(virtuals)
    
    
    def add_body(self,name,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            node1 = mbs_string(name,id_='rbr')
            node2 = mbs_string(name,id_='rbl')
            super().add_body(node1)
            super().add_body(node2)
            variant.nodes[node1]['mirr'] = node2
            variant.nodes[node2]['mirr'] = node1
        else:
            node1 = node2 =  mbs_string(name,id_='rbs')
            super().add_body(node1)
            variant.nodes[node1]['mirr'] = node2
    
    def _add_virtual_body(self,name):
        variant = self.selected_variant
        if name not in variant.nodes():
            attr_dict = self._typ_attr_dict(virtual_body)
            variant.add_node(name,**attr_dict)
    
    def add_virtual_body(self,name,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            node1 = mbs_string(name,id_='vbr')
            node2 = mbs_string(name,id_='vbl')
            self._add_virtual_body(node1)
            self._add_virtual_body(node2)
            variant.nodes[node1]['mirr'] = node2
            variant.nodes[node2]['mirr'] = node1 
        else:
            node1 = node2 = mbs_string(name,id_='vbs')
            self._add_virtual_body(node1)
            variant.nodes[node1]['mirr'] = node2
    
    def add_joint(self,typ,name,body_i,body_j,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            body_i_mirr = variant.nodes[body_i]['mirr']
            body_j_mirr = variant.nodes[body_j]['mirr']
            key1 = mbs_string(name,id_='jcr')
            key2 = mbs_string(name,id_='jcl')
            super().add_joint(typ,key1,body_i,body_j)
            super().add_joint(typ,key2,body_i_mirr,body_j_mirr)
            joint_edge1 = self._edges_map[key1]
            joint_edge2 = self._edges_map[key2]
            variant.edges[joint_edge1]['mirr'] = key2
            variant.edges[joint_edge2]['mirr'] = key1
        else:
            key = mbs_string(name,id_='jcs')
            super().add_joint(typ,key,body_i,body_j)
            joint_edge = self._edges_map[key]
            variant.edges[joint_edge]['mirr'] = key
    
    def add_joint_actuator(self,typ,name,joint_name,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            joint_edge1 = self._edges_map[joint_name]
            joint_name2 = variant.edges[joint_edge1]['mirr']
            key1 = mbs_string(name,id_='mcr')
            key2 = mbs_string(name,id_='mcl')
            super().add_joint_actuator(typ,key1,joint_name)
            super().add_joint_actuator(typ,key2,joint_name2)
        else:
            key = mbs_string(name,id_='mcs')
            super().add_joint_actuator(typ,key,joint_name)
    
    def add_absolute_actuator(self,name,body_i,coordinate,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            body_i_mirr = variant.nodes[body_i]['mirr']
            key1 = mbs_string(name,id_='mcr')
            key2 = mbs_string(name,id_='mcl')
            super().add_absolute_actuator(key1,body_i,coordinate)
            super().add_absolute_actuator(key2,body_i_mirr,coordinate)
        else:
            key = mbs_string(name,id_='mcs')
            super().add_absolute_actuator(key,body_i,coordinate)

###############################################################################
###############################################################################

class subsystem(abstract_topology):
    
    def __init__(self,name,topology):
        self.global_instance = global_frame(name)
        reference_frame.set_global_frame(self.global_instance)

        self.name = name
        self.topology = topology
        
#        self.graph = copy.deepcopy(topology.graph)
        self.graph = topology.graph.copy()
        self._relable()
        self._virtual_bodies = []
    
    def _relable(self):
        def label(x):
            if x!='ground':
                x = mbs_string(x.name,self.name,x._id_)
            return x
    
        nx.relabel_nodes(self.graph,label,copy=False)
        maped = map(label,dict(self.nodes(data='mirr')).values())
        maped = dict(zip(self.nodes,maped))
        nx.set_node_attributes(self.graph,maped,'mirr')
    
    @property
    def virtual_bodies(self):
        condition = self._check_if_virtual
        virtuals  = filter(condition,self.selected_variant.nodes)
        return set(virtuals)
    
###############################################################################
###############################################################################

class assembly(subsystem):
    
    def __init__(self,name):
        self.global_instance = global_frame(name)
        reference_frame.set_global_frame(self.global_instance)
        self.name = name
        self.graph = nx.MultiGraph(name=name)
        attr_dict = self._typ_attr_dict(ground)
        self.graph.add_node('ground',**attr_dict)
        self.subsystems = {}
        self._virtual_bodies_map = {}
        self.interface_graph = nx.MultiGraph(name=name)
    
    @property
    def virtual_bodies_map(self):
        return self._virtual_bodies_map
    @property
    def nve_iterface(self):
        interface_graph = self.interface_graph
        nve_edges = sum([edge[-1] for edge in interface_graph.edges(data='nve')])
        return nve_edges
    
    def _update_virtual_bodies_map(self,subsystem):
        new_virtuals = subsystem.virtual_bodies
        new_virtuals = zip(new_virtuals,len(new_virtuals)*['ground'])
        self._virtual_bodies_map.update(new_virtuals)
            
    def add_subsystem(self,subsystem):
        self.subsystems[subsystem.name] = subsystem
        subsystem_graph = subsystem.selected_variant
        self.graph.add_nodes_from(subsystem_graph.nodes(data=True))
        self.graph.add_edges_from(subsystem_graph.edges(data=True,keys=True))
        self._update_virtual_bodies_map(subsystem)
        self.global_instance.merge_global(subsystem.global_instance)
    
    def assign_virtual_body(self,virtual_node,actual_node):
        virtual_node_1 = virtual_node
        virtual_node_2 = self.nodes[virtual_node]['mirr']
        actual_node_1 = actual_node
        actual_node_2 = self.nodes[actual_node]['mirr']
        self.virtual_bodies_map[virtual_node_1] = actual_node_1
        self.virtual_bodies_map[virtual_node_2] = actual_node_2
    
    def _contracted_nodes(self,u,v):
        from itertools import chain
        H = self.graph
        if H.is_directed():
            in_edges = ((w, u, d) for w, x, d in H.in_edges(v, data=True))
            out_edges = ((u, w, d) for x, w, d in H.out_edges(v, data=True))
            new_edges = chain(in_edges, out_edges)
        else:
            new_edges = [(u,w,k,d) for x,w,k,d in H.edges(v,keys=True,data=True) if w!=u]
        H.add_edges_from(new_edges)
        self.interface_graph.add_edges_from(new_edges)
        
    
    def _initialize_interface(self):
        self.graph.add_edges_from(self.virtual_bodies_map.items())
        for v,u in self.virtual_bodies_map.items():
#            print('replacing %s with %s'%(v,u))
            self._contracted_nodes(u,v)
        for v in self.virtual_bodies_map.keys():
            self.graph.remove_node(v)
    
    def _set_configuration_constants(self):
        graph = self.graph.edge_subgraph(self.interface_graph.edges)
        cons = nx.get_edge_attributes(graph,'config_const').values()
        self.configuration_constants = sum(cons,[])

    def _set_numerical_arguments(self):
        graph = self.graph.edge_subgraph(self.interface_graph.edges)
        edge_args = nx.get_edge_attributes(graph,'num_args').values()
        self.numerical_arguments = sum(edge_args,[])
    
    def _initialize_toplogy_reqs(self):
        self._set_configuration_constants()
        self._set_numerical_arguments()
    
    
    def _assemble_edges(self):
        for e in self.interface_graph.edges:
            self._assemble_edge(e)

    
    def _assemble_equations(self):
        interface = self.interface_graph
        node_index = dict([(n,i) for i,n in enumerate(self.nodes)])

        n_nodes = len(self.nodes)
        cols = 2*n_nodes
        
        # number of constraint vector equations
        nve = self.nve_iterface
        
        equations = sm.MutableSparseMatrix(nve,1,None)
        vel_rhs   = sm.MutableSparseMatrix(nve,1,None)
        acc_rhs   = sm.MutableSparseMatrix(nve,1,None)
        jacobian  = sm.MutableSparseMatrix(nve,cols,None)
        
        row_ind = 0
        for e in interface.edges:
            u,v = e[:2]
            eo  = self.edges[e]['obj']
            
            # tracker of row index based on the current joint type and the history
            # of the loop
            eo_nve = eo.nve+row_ind 
            
            ui = node_index[u]
            vi = node_index[v]
            
            # assigning the joint jacobians to the propper index in the system jacobian
            # on the "constraint vector equations" level.
            jacobian[row_ind:eo_nve,ui*2:ui*2+2] = eo.jacobian_i.blocks
            jacobian[row_ind:eo_nve,vi*2:vi*2+2] = eo.jacobian_j.blocks
            
            equations[row_ind:eo_nve,0] = eo.pos_level_equations.blocks
            vel_rhs[row_ind:eo_nve,0] = eo.vel_level_equations.blocks
            acc_rhs[row_ind:eo_nve,0] = eo.acc_level_equations.blocks
           
            row_ind += eo.nve
        
        self.pos_equations = equations
        self.vel_equations = vel_rhs
        self.acc_equations = acc_rhs
        self.jac_equations = jacobian
    
    
    def assemble_model(self,full=False):
        self._initialize_interface()
        self._assemble_edges()
        self._initialize_toplogy_reqs()
        if full:
            super()._assemble_equations()
        else:
            self._assemble_equations()
        
    def draw_interface_graph(self):
        plt.figure(figsize=(10,6))
        nx.draw_spring(self.interface_graph,with_labels=True)
        plt.show()

    
    


