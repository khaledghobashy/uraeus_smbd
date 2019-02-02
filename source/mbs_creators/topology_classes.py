# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:31:35 2019

@author: khale
"""

import sympy as sm
import matplotlib.pyplot as plt
import networkx as nx
import itertools

from source.symbolic_classes.abstract_matrices import (global_frame, vector,
                                                       reference_frame, Mirror)
from source.symbolic_classes.bodies import body, ground, virtual_body
from source.symbolic_classes.spatial_joints import absolute_locator
from source.symbolic_classes.algebraic_constraints import joint_actuator


class parametric_configuration(object):
    
    def __init__(self,mbs_instance):
        self.topology = mbs_instance
        self.name = self.topology.name
        self.graph = nx.DiGraph(name=self.name)
        self._arguments_sym = []
    
    
    def _obj_attr_dict(self,obj):
        Eq = self._set_equality
        attr_dict = {'obj':obj,'mirr':None,'align':'s','func':Eq(obj),
                     'primary':False}
        return attr_dict
    
    
    def _set_equality(self,sym1,sym2=None):
        t = sm.symbols('t')
        if sym1 and sym2:
            if isinstance(sym1,sm.MatrixSymbol):
                return sm.Eq(sym2,Mirror(sym1))
            elif issubclass(sym1,sm.Function):
                return sm.Eq(sym2,sym1)
        else:
            if isinstance(sym1,sm.MatrixSymbol):
                return sm.Eq(sym1,sm.zeros(*sym1.shape))
            elif issubclass(sym1,sm.Function):
                return sm.Eq(sym1,sm.Lambda(t,0))
    
    @property
    def arguments_symbols(self):
        return set(self._arguments_sym)
    
    def _get_nodes_arguments(self):
        Eq = self._set_equality
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
            nodes_args_n = [(str(i),{'func':Eq(i)}) for i in args_n]
            self._arguments_sym += args_n
            if m == n:
                graph.add_nodes_from(nodes_args_n)
            else:
                args_m = t_nodes[m]['arguments']
                args_c = zip(args_n,args_m)
                nodes_args_m = [(str(m),{'func':Eq(n,m)}) for n,m in args_c]
                graph.add_nodes_from(nodes_args_n+nodes_args_m)
                edges = [(str(n),str(m)) for n,m in zip(args_n,args_m)]
                graph.add_edges_from(edges)    
                self._arguments_sym += args_m

    def _get_edges_arguments(self):
        Eq = self._set_equality
        graph   = self.graph
        t_edges = self.topology.edges
        for e in t_edges:
            n = t_edges[e]['name']
            if t_edges[e]['align'] in 'sr':
                m = t_edges[e]['mirr']
                args_n = t_edges[e]['arguments']
                nodes_args_n = [(str(i),{'func':Eq(i)}) for i in args_n]
                self._arguments_sym += args_n
                if m == n:
                    graph.add_nodes_from(nodes_args_n)
                else:
                    e2 = self.topology._edges_map[m]
                    args_m = t_edges[e2]['arguments']
                    args_c = zip(args_n,args_m)
                    nodes_args_m = [(str(m),{'func':Eq(n,m)}) for n,m in args_c]
                    graph.add_nodes_from(nodes_args_n+nodes_args_m)
                    edges = [(str(n),str(m)) for n,m in zip(args_n,args_m)]
                    graph.add_edges_from(edges)    
                    self._arguments_sym += args_m
    
    def assemble_base_layer(self):
        self._get_nodes_arguments()
        self._get_edges_arguments()
        nx.set_node_attributes(self.graph,True,'primary')
    
    def _add_point(self,name):
        v = vector(name)
        attr_dict = self._obj_attr_dict(v)
        self.graph.add_node(name,**attr_dict)
    
    def add_point(self,name,mirror=False):
        Eq = self._set_equality
        graph = self.graph
        if mirror:
            node1 = 'hpr_%s'%name
            node2 = 'hpl_%s'%name
            self._add_point(node1)
            self._add_point(node2)
            graph.nodes[node1].update({'mirr':node2,'align':'r'})
            graph.nodes[node2].update({'mirr':node1,'align':'l'})
            obj1 = graph.nodes[node1]['obj']
            obj2 = graph.nodes[node1]['obj']
            graph.nodes[node2].update({'func':Eq(obj1,obj2)})
            graph.add_edge(node1,node2)
        else:
            node1 = node2 = 'hps_%s'%name
            self._add_point(node1)
            graph.nodes[node1]['mirr'] = node2
    
    def _add_relation(self,relation,node,nbunch):
        graph = self.graph
        edges = [(i,node) for i in nbunch]
        graph.nodes[node]['func'] = sm.Equality(node,relation(*nbunch))
        graph.add_edges_from(edges)
    
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
    
    def inputs_layer(self):
        g = self.graph
        inputs = [g.nodes[i]['func'] for i,d in g.in_degree() if d == 0]
        return inputs
    
    def outputs_layer(self):
        g = self.graph
        condition = lambda i,d : d==0 and g.in_degree(i)!=0
        inputs = [g.nodes[i]['func'] for i,d in g.out_degree() if condition(i,d)]
        return inputs
    
    def draw_graph(self):
        plt.figure(figsize=(10,6))
        nx.draw_spring(self.graph,with_labels=True)
        plt.show()        

###############################################################################
###############################################################################

class abstract_topology(object):
    
    def __init__(self,name):
        self.name = name
        self.graph = nx.MultiDiGraph(name=name)
        self._edges_map = {}
        self._edges_keys_map = {}
        self._insert_ground()
        self._param_config = parametric_configuration(self)
        
    def _set_global_frame(self):
        self.global_instance = global_frame(self.name)
        reference_frame.set_global_frame(self.global_instance)        
        
    def _insert_ground(self):
        typ_dict = self._typ_attr_dict(ground)
        self.grf = 'ground'
        self.graph.add_node(self.grf,**typ_dict)
    
    @property
    def param_config(self):
        return self._param_config
        
    @property
    def selected_variant(self):
        return self.graph
    @property
    def nodes(self):
        return self.selected_variant.nodes
    @property
    def bodies(self):
        bodies = itertools.filterfalse(self._check_virtual_node,self.nodes)
        return list(bodies)
    @property
    def edges(self):
        return self.selected_variant.edges
    
    def _check_virtual_node(self,n):
        body_type = self.nodes[n]['class']
        return issubclass(body_type,virtual_body)
    @property
    def virtual_bodies(self):
        condition = self._check_virtual_node
        virtuals  = filter(condition,self.nodes)
        return set(virtuals)
    
    def _check_virtual_edge(self,e):
       virtual_flag = self.edges[e].get('virtual',False)
       return virtual_flag
    @property
    def virtual_edges(self):
        condition = self._check_virtual_edge
        virtuals  = filter(condition,self.edges)
        return set(virtuals)
    
    @property
    def nodes_indicies(self):
        node_index = dict([(n,i) for i,n in enumerate(self.nodes)])
        return node_index
    @property
    def edges_indicies(self):
        edges_index = dict([(n,i) for i,n in enumerate(self.edges)])
        return edges_index

    @property
    def nodes_arguments(self):
        args = nx.get_node_attributes(self.graph,'arguments').values()
        return sum(args,[])
    @property
    def edges_arguments(self):
        args = nx.get_edge_attributes(self.graph,'arguments').values()
        return sum(args,[])
    
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
        attr_dict = {'n':typ.n,'nc':typ.nc,'nve':typ.nve,'class':typ,'mirr':None,
                    'align':'s'}
        return attr_dict
    @staticmethod
    def _obj_attr_dict(obj):
        attr_dict = {'obj':obj,'arguments':obj.arguments,'constants':obj.constants}
        return attr_dict
    
    
    def draw_topology(self):
        plt.figure(figsize=(10,6))
        nx.draw_spring(self.selected_variant,with_labels=True)
        plt.show()
    
    def _coordinates_mapper(self,sym):
        q_sym  = sm.MatrixSymbol(sym,self.n,1)
        q = []
        i = 0
        for b in itertools.filterfalse(self._check_virtual_node,self.nodes):
            q_block = getattr(self.nodes[b]['obj'],sym)
            for qi in q_block.blocks:
                s = qi.shape[0]
                q.append(sm.Eq(qi,q_sym[i:i+s,0]))
                i+=s
        return q
    
    @property
    def mapped_gen_coordinates(self):
        return self._coordinates_mapper('q')
    @property
    def mapped_gen_velocities(self):
        return self._coordinates_mapper('qd')
    @property
    def virtual_coordinates(self):
        q_virtuals = []
        for n in self.virtual_bodies:
            obj = self.nodes[n]['obj']
            q_virtuals += [obj.R,obj.P,obj.Rd,obj.Pd]
        return q_virtuals
    
    @property    
    def constants(self):
        cons = nx.get_edge_attributes(self.graph,'constants').values()
        return sum(cons,[])
    
    def _initialize_toplogy_reqs(self):
        self.param_config.assemble_base_layer()
    
    
    def _assemble_node(self,n):
        body_type = self.nodes[n]['class']
        body_instance = body_type(n)
        self.nodes[n].update(self._obj_attr_dict(body_instance))
    
    def _assemble_nodes(self):
        [self._assemble_node(n) for n in self.nodes]
    
    def _assemble_edge(self,e):
        edge_class = self.edges[e]['class']
        name = self.edges[e]['name']
        b1, b2, key = e
        b1_obj = self.nodes[b1]['obj'] 
        b2_obj = self.nodes[b2]['obj'] 
        if issubclass(edge_class,joint_actuator):
            joint_key     = self._edges_keys_map[self.edges[e]['joint_name']]
            joint_object  = self.edges[(b1,b2,joint_key)]['obj']
            edge_instance = edge_class(name,joint_object)
        elif issubclass(edge_class,absolute_locator):
            coordinate    = self.edges[e]['coordinate']
            edge_instance = edge_class(name,b1_obj,b2_obj,coordinate)
        else:
            edge_instance = edge_class(name,b1_obj,b2_obj)
        self.edges[e].update(self._obj_attr_dict(edge_instance))
    
    def _assemble_edges(self):
        [self._assemble_edge(e) for e in self.edges]
    
    def _remove_virtual_edges(self):
        self.graph.remove_edges_from(self.virtual_edges)


    def _assemble_equations(self):
        
        edgelist    = self.edges
        nodelist    = self.nodes
        node_index  = self.nodes_indicies

        cols = 2*len(nodelist)
        nve  = self.nve  # number of constraint vector equations
        
        equations = sm.MutableSparseMatrix(nve,1,None)
        vel_rhs   = sm.MutableSparseMatrix(nve,1,None)
        acc_rhs   = sm.MutableSparseMatrix(nve,1,None)
        jacobian  = sm.MutableSparseMatrix(nve,cols,None)
        
        row_ind = 0
        for e in edgelist:
            if self._check_virtual_edge(e):
                continue
            eo  = self.edges[e]['obj']
            u,v = e[:-1]
            
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
            if self._check_virtual_node(n):
                continue
            b = self.nodes[n]['obj']
            if isinstance(b,ground):
                jacobian[row_ind:row_ind+2,i*2:i*2+2] = b.normalized_jacobian.blocks
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
        
        ind_b    = {v:k for k,v in node_index.items()}
        cols_ind = [i[1] for i in self.jac_equations.row_list()]
        self.jac_cols = [(ind_b[i//2]+'*2' if i%2==0 else ind_b[i//2]+'*2+1') for i in cols_ind]

    
    def assemble_model(self):
        self._set_global_frame()
        self._assemble_nodes()
        self._assemble_edges()
        self._remove_virtual_edges()
        self._assemble_equations()
        self._initialize_toplogy_reqs()
    
#    def add_node(self,node,dicts):
#        variant = self.selected_variant
#        if node not in variant.nodes():
#            variant.add_node(node,**dicts)
#    
#    def add_edge(self,edge,dicts):
#        variant = self.selected_variant
#        if edge not in variant.edges:
#            variant.add_edge(*edge,**dicts)
#            self._edges_map[edge[-1]] = edge
        
###############################################################################
###############################################################################

class topology(abstract_topology):
    
    def add_body(self,name):
        variant = self.selected_variant
        if name not in variant.nodes():
            attr_dict = self._typ_attr_dict(body)
            variant.add_node(name,**attr_dict)
    
    def add_joint(self,typ,name,body_i,body_j):
        assert body_i and body_j in self.nodes , 'bodies do not exist!'
        variant = self.selected_variant
        edge  = (body_i,body_j)
        if edge not in variant.edges:
            attr_dict = self._typ_attr_dict(typ)
            attr_dict.update({'name':name})
            key = variant.add_edge(*edge,**attr_dict)
            self._edges_map[name] = (*edge,key)
            self._edges_keys_map[name] = key
            
    def add_joint_actuator(self,typ,name,joint_name):
        assert joint_name in self._edges_map, 'joint does not exist!'
        variant = self.selected_variant
        joint_edge = self._edges_map[joint_name]
        act_edge   = joint_edge[:2]
        if name not in self._edges_map:
            attr_dict = self._typ_attr_dict(typ)
            attr_dict.update({'joint_name':joint_name,'name':name})
            key = variant.add_edge(*act_edge,**attr_dict)
            self._edges_map[name] = (*act_edge,key)
            self._edges_keys_map[name] = key
    
    def add_absolute_actuator(self,name,body,coordinate):
        assert body in self.nodes , 'body does not exist!'
        variant = self.selected_variant
        edge  = (body,self.grf)
        if edge not in variant.edges:
            attr_dict = self._typ_attr_dict(absolute_locator)
            key = variant.add_edge(*edge,**attr_dict,coordinate=coordinate,name=name)
            self._edges_map[name] = (*edge,key)
            self._edges_keys_map[name] = key
    
    
###############################################################################
###############################################################################

class template_based_topology(topology):
    
    def __init__(self,name):
        super().__init__(name)
        self.grf = 'vbs_ground'
    
    def _insert_ground(self):
        self.add_virtual_body('ground')    
    
    def add_body(self,name,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            node1 = 'rbr_%s'%name
            node2 = 'rbl_%s'%name
            super().add_body(node1)
            super().add_body(node2)
            variant.nodes[node1].update({'mirr':node2,'align':'r'})
            variant.nodes[node2].update({'mirr':node1,'align':'l'})
        else:
            node1 = node2 = 'rbs_%s'%name
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
            node1 = 'vbr_%s'%name
            node2 = 'vbl_%s'%name
            self._add_virtual_body(node1)
            self._add_virtual_body(node2)
            variant.nodes[node1].update({'mirr':node2,'align':'r'})
            variant.nodes[node2].update({'mirr':node1,'align':'l'})
        else:
            node1 = node2 = 'vbs_%s'%name
            self._add_virtual_body(node1)
            variant.nodes[node1]['mirr'] = node2
    
    @staticmethod
    def _set_joint_as_virtual(variant,edge):
        d = {'nc':0,'nve':0,'virtual':True}
        variant.edges[edge].update(d)
    
    def add_joint(self,typ,name,body_i,body_j,mirrored=False,virtual=False):
        variant = self.selected_variant
        if mirrored:
            body_i_mirr = variant.nodes[body_i]['mirr']
            body_j_mirr = variant.nodes[body_j]['mirr']
            name1 = 'jcr_%s'%name
            name2 = 'jcl_%s'%name
            super().add_joint(typ,name1,body_i,body_j)
            super().add_joint(typ,name2,body_i_mirr,body_j_mirr)
            joint_edge1 = self._edges_map[name1]
            joint_edge2 = self._edges_map[name2]
            variant.edges[joint_edge1].update({'mirr':name2,'align':'r'})
            variant.edges[joint_edge2].update({'mirr':name1,'align':'l'})
            if virtual:
                self._set_joint_as_virtual(variant,joint_edge1)
                self._set_joint_as_virtual(variant,joint_edge2)
        else:
            name = 'jcs_%s'%name
            super().add_joint(typ,name,body_i,body_j)
            joint_edge = self._edges_map[name]
            variant.edges[joint_edge].update({'mirr':name})
            if virtual:
                self._set_joint_as_virtual(variant,joint_edge)
    
    def add_joint_actuator(self,typ,name,joint_name,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            joint_edge1 = self._edges_map[joint_name]
            joint_name2 = variant.edges[joint_edge1]['mirr']
            name1 = 'mcr_%s'%name
            name2 = 'mcl_%s'%name
            super().add_joint_actuator(typ,name1,joint_name)
            super().add_joint_actuator(typ,name2,joint_name2)
            act_edge1 = self._edges_map[name1]
            act_edge2 = self._edges_map[name2]
            variant.edges[act_edge1].update({'mirr':name2,'align':'r'})
            variant.edges[act_edge2].update({'mirr':name1,'align':'l'})
        else:
            name = 'mcs_%s'%name
            super().add_joint_actuator(typ,name,joint_name)
            act_edge = self._edges_map[name]
            variant.edges[act_edge].update({'mirr':name})
    
    def add_absolute_actuator(self,name,body_i,coordinate,mirrored=False):
        variant = self.selected_variant
        if mirrored:
            body_i_mirr = variant.nodes[body_i]['mirr']
            name1 = 'mcr_%s'%name
            name2 = 'mcl_%s'%name
            super().add_absolute_actuator(name1,body_i,coordinate)
            super().add_absolute_actuator(name2,body_i_mirr,coordinate)
            act_edge1 = self._edges_map[name1]
            act_edge2 = self._edges_map[name2]
            variant.edges[act_edge1].update({'mirr':name2,'align':'r'})
            variant.edges[act_edge2].update({'mirr':name1,'align':'l'})
        else:
            name = 'mcs_%s'%name
            super().add_absolute_actuator(name,body_i,coordinate)
            act_edge = self._edges_map[name]
            variant.edges[act_edge].update({'mirr':name})

###############################################################################
###############################################################################

class subsystem(abstract_topology):
    
    def __init__(self,name,topology):
        self.name = name
        self._set_global_frame()
        self.topology = topology
        self.graph = topology.graph.copy()
        self._relable()
        self._virtual_bodies = []
    
    def _relable(self):
        def label(x): return self.name+'.'+ x
        labels_map = {i:label(i) for i in self.topology.nodes}
        self.graph = nx.relabel_nodes(self.graph,labels_map)
        mirr_maped = {k:label(v) for k,v in self.nodes(data='mirr')}
        nx.set_node_attributes(self.graph,mirr_maped,'mirr')
    
    
###############################################################################
###############################################################################

class assembly(subsystem):
    
    def __init__(self,name):
        self.name = name
        self.grf = 'ground'
        self._set_global_frame()
        
        self.graph = nx.MultiGraph(name=name)
        self.interface_graph = nx.MultiGraph(name=name)
        
        self.subsystems = {}
        self._interface_map = {}

        typ_attr_dict = self._typ_attr_dict(ground)
        obj_attr_dict = self._obj_attr_dict(ground())
        self.graph.add_node(self.grf,**typ_attr_dict,**obj_attr_dict)
        
        
    @property
    def interface_map(self):
        return self._interface_map
    @property
    def nve_iterface(self):
        interface_graph = self.interface_graph
        nve_edges = sum([edge[-1] for edge in interface_graph.edges(data='nve')])
        return nve_edges
    
    def _set_virtual_equalities(self):
        self.mapped_vir_coordinates = []
        for v,a in self.interface_map.items():
            self._assemble_node(v)
            if a!= self.grf : self._assemble_node(a)
            virtual_coordinates = self.nodes[v]['obj'].q
            actual_coordinates  = self.nodes[a]['obj'].q
            equality = list(map(sm.Eq,virtual_coordinates.blocks,actual_coordinates.blocks))
            self.mapped_vir_coordinates += equality
        
        self.mapped_vir_velocities = []
        for v,a in self.interface_map.items():
            self._assemble_node(v)
            if a!= self.grf : self._assemble_node(a)
            virtual_coordinates = self.nodes[v]['obj'].qd
            actual_coordinates  = self.nodes[a]['obj'].qd
            equality = list(map(sm.Eq,virtual_coordinates.blocks,actual_coordinates.blocks))
            self.mapped_vir_velocities += equality
    
    def _update_interface_map(self,subsystem):
        new_virtuals = subsystem.virtual_bodies
        new_virtuals = {i: self.grf for i in new_virtuals}
        self._interface_map.update(new_virtuals)
            
    def add_subsystem(self,sub):
        assert isinstance(sub,subsystem), 'value should be instance of subsystem'
        self.subsystems[sub.name] = sub
        subsystem_graph = sub.selected_variant
        self.graph.add_nodes_from(subsystem_graph.nodes(data=True))
        self.graph.add_edges_from(subsystem_graph.edges(data=True,keys=True))
        self._update_interface_map(sub)
        self.global_instance.merge_global(sub.global_instance)
    
    def assign_virtual_body(self,virtual_node,actual_node):
        virtual_node_1 = virtual_node
        virtual_node_2 = self.nodes[virtual_node]['mirr']
        actual_node_1 = actual_node
        actual_node_2 = self.nodes[actual_node]['mirr']
        self.interface_map[virtual_node_1] = actual_node_1
        self.interface_map[virtual_node_2] = actual_node_2
        
    
    def _contracted_nodes(self,u,v):
        H = self.graph
        new_edges = [(u,w,k,d) for x,w,k,d in H.edges(v,keys=True,data=True) if w!=u]
        H.add_edges_from(new_edges)
        self.interface_graph.add_edges_from(new_edges)
        
    
    def _initialize_interface(self):
        self._set_virtual_equalities()
        self.graph.add_edges_from(self.interface_map.items())
        for v,u in self.interface_map.items():
            self._contracted_nodes(u,v)
        self.graph.remove_nodes_from(self.interface_map.keys())
        self.interface_graph.remove_nodes_from(self.interface_map.keys())
    
    def _set_constants(self):
        graph = self.graph.edge_subgraph(self.interface_graph.edges)
        cons = nx.get_edge_attributes(graph,'constants').values()
        self.constants = sum(cons,[])

    def _set_arguments(self):
        graph = self.graph.edge_subgraph(self.interface_graph.edges)
        edge_args = nx.get_edge_attributes(graph,'arguments').values()
        self.arguments = sum(edge_args,[])
    
    def _initialize_toplogy_reqs(self):
        self._set_constants()
        self._set_arguments()
    
    def _assemble_edges(self):
        for e in self.interface_graph.edges:
            self._assemble_edge(e)
    
    def _assemble_equations(self):
        
        nodelist    = self.nodes
        cols = 2*len(nodelist)
        nve  = 2
        
        equations = sm.MutableSparseMatrix(nve,1,None)
        vel_rhs   = sm.MutableSparseMatrix(nve,1,None)
        acc_rhs   = sm.MutableSparseMatrix(nve,1,None)
        jacobian  = sm.MutableSparseMatrix(nve,cols,None)
        
        row_ind = 0
        b = self.nodes['ground']['obj']
        i = self.nodes_indicies['ground']
        jacobian[row_ind:row_ind+2,i*2:i*2+2]    = b.normalized_jacobian.blocks
        equations[row_ind:row_ind+2,0] = b.normalized_pos_equation.blocks
        vel_rhs[row_ind:row_ind+2,0]   = b.normalized_vel_equation.blocks
        acc_rhs[row_ind:row_ind+2,0]   = b.normalized_acc_equation.blocks
            
        self.pos_equations = equations
        self.vel_equations = vel_rhs
        self.acc_equations = acc_rhs
        self.jac_equations = jacobian
    
    def assemble_model(self):
        self._initialize_interface()
        self._assemble_equations()
        
    def draw_interface_graph(self):
        plt.figure(figsize=(10,6))
        nx.draw_spring(self.interface_graph,with_labels=True)
        plt.show()

    
    


