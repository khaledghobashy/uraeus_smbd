# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:47:50 2019

@author: khale
"""

import sympy as sm

import networkx as nx
import matplotlib.pyplot as plt

sm.init_printing(pretty_print=False,use_latex=True,forecolor='White')
enclose = True

class mbs_string(object):
        
    def __init__(self,name,prefix='',id_=''):
        self._name  = name
        self._prefix = prefix
        self._id_ = id_
        
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        self._name = value
        
    @property
    def prefix(self):
        prefix = (self._prefix if self._prefix=='' else self._prefix+'.')
        return prefix
    @prefix.setter
    def prefix(self,value):
        self._prefix = value
    
    @property
    def id_(self):
        id_ = (self._id_ if self._id_=='' else self._id_+'_')
        return id_
    @id_.setter
    def id_(self,value):
        self._id_ = value
    
    def __str__(self):
        return '%s%s%s'%(self.prefix,self.id_,self.name)
    
    def __repr__(self):
        return '%s%s%s'%(self.prefix,self.id_,self.name)
    
    def __iter__(self):
        return iter((self.prefix,self.id_,self.name))
    
    def __eq__(self,other):
        return str(self)==str(other)
    
    def __hash__(self):
        return hash(str(self))
        
###############################################################################
###############################################################################

class AbstractMatrix(sm.MatrixExpr):
    
    is_commutative = False
    is_Matrix = True
    shape = (3,3)
    
    def __init__(self,sym):
        pass
    
    def doit(self):
        return self
    
class A(AbstractMatrix):
    
    def _latex(self,expr):
        return '{A(%s)}'%self.args[0]


class G(AbstractMatrix):
    shape = (3,4)
    

class E(AbstractMatrix):
    shape = (3,4)
    
class B(AbstractMatrix):
    shape = (3,4)
    def __init__(self,sym1,sym2):
        super().__init__(sym1)
    
    def _latex(self,expr):
        p,u = self.args
        return '{B(%s,%s)}'%(p,u.name)
    

class Triad(AbstractMatrix):
    def __init__(self,v1,v2=None):
        super().__init__(v1)

###############################################################################
###############################################################################

class base_vector(sm.MatrixSlice):
    
    shape = (3,1)
    
    def __new__(cls,frame,sym):
        slices = {'i':(0,1),'j':(1,2),'k':(2,3)}
        return super().__new__(cls,frame.A,(0,3),slices[sym])
    
    def __init__(self,frame,sym):
        slices = {'i':(0,1),'j':(1,2),'k':(2,3)}
        self.slice = slices[sym]
        self.frame = frame
        self._sym  = sym
        self._args = (frame,sym)
        self._formated = '{\hat{%s}_{%s}}'%(self._sym,self.frame.name)
    
    @property
    def name(self):
        return self._formated
    
    def express(self,frame=None):
        frame = (frame if frame is not None else self.frame.global_frame)
        A = self.frame.parent.express(frame)
        return A*self
    
    def doit(self):
        return self
    
    def _latex(self,expr):
        return self._formated
    
    def _ccode(self,expr,**kwargs):
        global enclose
        if enclose:
            return '%r[:,%s:%s]'%(self.frame.name,*self.slice)
        else:
            return '%s[:,%s:%s]'%(self.frame.name,*self.slice)
    
    def _sympystr (self,expr):
        return '%s[:,%s]'%(self.frame.name,self.slice)
        
    @property
    def func(self):
        return base_vector

###############################################################################
class dcm(sm.MatrixSymbol):
    shape = (3,3)
    
    is_commutative = False
    
    def __new__(cls,name, format_as=None):
        if format_as:
            name = format_as
        return super(dcm,cls).__new__(cls,name,3,3)
    
    def __init__(self,name, frame=None, format_as=None):
        self._args=(name,frame,format_as)
        self._raw_name = name
        self._formated_name = super().name
        
    @property
    def name(self):
        return self._formated_name
    
    def doit(self):
        return self
    
    @property
    def func(self):
        return dcm
        
###############################################################################
class zero_matrix(sm.MatrixSymbol):
    
    def __new__(cls,m,n):
        sym = '{Z_{%sx%s}}'%(m,n)
        return super().__new__(cls,sym,m,n)
    def __init__(self,m,n):
        self.sym = super().name
        self._args = (m,n)
    
    def _latex(self,expr):
        return self.sym
    def _ccode(self,expr,**kwargs):
        return 'np.zeros((%s,%s),dtype=np.float64)'%(self.shape)
    def _sympystr(self,expr):
        return 'zero_matrix(%s,%s)'%self.shape
    
    def doit(self):
        return self
    @property
    def func(self):
        return zero_matrix
    @property
    def shape(self):
        return self._args

###############################################################################
class global_frame(object):
        
    def __init__(self,name='global'):
        self.name = name
        self.references_tree = nx.DiGraph(name=name)
        self.references_tree.add_node(self.name)
    
    @property
    def nodes(self):
        return self.references_tree.nodes
    @property
    def edges(self):
        return self.references_tree.edges
    
    def merge_globals(self,globals_):
        for g in globals_:
            self.references_tree.add_nodes_from(g.nodes(data=True))
            self.references_tree.add_edges_from(g.edges(data=True))
    
    def draw_tree(self):
        plt.figure(figsize=(10,6))
        nx.draw(self.references_tree,with_labels=True)
        plt.show()

###############################################################################
class reference_frame(object):
    
    _is_global_set  = False
    global_frame = None
    
    @classmethod
    def set_global_frame(cls,global_instance):
        cls._is_global_set = True
        cls.global_frame = global_instance
    
    def __new__(cls, name, parent=None,format_as=None):
        if not cls._is_global_set:
            print('creating global')
            global_instance = global_frame()
            cls.set_global_frame(global_instance)
        return super(reference_frame,cls).__new__(cls)
    
    
    def __init__(self, name, parent=None,format_as=None):
        #print('Inside reference_frame()')
        self._raw_name = name
        self._formated_name = (format_as if format_as else name)
        self.parent = (parent if parent else self.global_frame)
        self.A = dcm(self._raw_name,self._formated_name)
             
    def update_tree(self):
        self.global_frame.references_tree.add_edge(self.parent.name, self.name, mat=self.A.T)
        self.global_frame.references_tree.add_edge(self.name, self.parent.name, mat=self.A)
    
    @property
    def A(self):
        return self._A
    @A.setter
    def A(self,value):
        self._A = value
        self.i = base_vector(self,'i')
        self.j = base_vector(self,'j')
        self.k = base_vector(self,'k')
        self.update_tree()
    
    @property
    def name(self):
        return self._formated_name
    
    def _ccode(self,expr,**kwargs):
        return self._raw_name
            
    def orient_along(self,v1,v2=None):
        if v2 is None:
            self.A = Triad(v1)
        else:
            self.A = Triad(v1,v2)
            
    def express(self,other):
        child_name  = self.name
        parent_name = other.name
        graph = self.global_frame.references_tree
        
        path_nodes  = nx.algorithms.shortest_path(graph, child_name, parent_name)
        path_matrices = []
        for i in range(len(path_nodes)-1):
            path_matrices.append(graph.edges[path_nodes[-i-2],path_nodes[-i-1]]['mat'])
        mat = sm.MatMul(*path_matrices)
        return mat
    
###############################################################################
###############################################################################

class vector(sm.MatrixSymbol):
    shape = (3,1)
    
    is_commutative = False
    
    def __new__(cls,name, frame=None, format_as=None):
        if format_as:
            name = format_as
        return super(vector,cls).__new__(cls,name,3,1)
    
    def __init__(self,name, frame=None, format_as=None):
        self._raw_name = name
        self._formated_name = super().name
        self.frame = (frame if frame is not None else reference_frame.global_frame)        
        self._args = (name,self.frame,self._formated_name)
        
    def express(self,frame=None):
        frame = (frame if frame is not None else reference_frame.global_frame)
        A = self.frame.express(frame)
        return A*self
    
    @property
    def name(self):
        return self._formated_name
    
    def doit(self):
        return self
    
    def rename(self,name,format_as=None):
        self._raw_name = name
        self._formated_name = (format_as if format_as else name)
        self._args = (name,self.frame,self._formated_name)
    
###############################################################################
class quatrenion(sm.MatrixSymbol):
    
    shape = (4,1)
    is_commutative = False
    
    def __new__(cls, name, format_as=None):
        if format_as:
            name = format_as
        return super(quatrenion,cls).__new__(cls,name,*cls.shape)
    
    def __init__(self,name, format_as=None):
        self._raw_name = name
        self._formated_name = super().name
        self._args = (name,format_as)
    
    @property
    def name(self):
        return self._formated_name 

    def doit(self):
        return self
    
    @property
    def func(self):
        return self.__class__
            
    def rename(self,name,format_as=None):
        self._raw_name = name
        self._formated_name = (format_as if format_as else name)
        self._args = (name,format_as)
    
###############################################################################
###############################################################################

class abstract_mbs(object):
    
    
    def configuration_constants(self):
        from source.symbolic_classes.bodies import body
        from source.symbolic_classes.algebraic_constraints import algebraic_constraints as algebraic_constraints
        if isinstance(self,body):
            return []
        elif isinstance(self,algebraic_constraints):
            loc  = vector('pt_%s'%self.name)
            axis = vector('ax_%s'%self.name)

            ui_bar_eq = sm.Eq(self.ui_bar, loc.express(self.body_i) - self.Ri.express(self.body_i))
            uj_bar_eq = sm.Eq(self.uj_bar, loc.express(self.body_j) - self.Rj.express(self.body_j))

            marker = reference_frame('M_%s'%self.name,format_as=r'{{M}_{%s}}'%self.name)
            marker.orient_along(axis)

            mi_bar      = marker.express(self.body_i)
            mi_bar_eq   = sm.Eq(self.mi_bar.A, mi_bar)

            mj_bar      = marker.express(self.body_j)
            mj_bar_eq   = sm.Eq(self.mj_bar.A, mj_bar)

            assignments = [ui_bar_eq,uj_bar_eq,mi_bar_eq,mj_bar_eq]
            return assignments
    
    def numerical_arguments(self):
        from source.symbolic_classes.bodies import body
        from source.symbolic_classes.algebraic_constraints import algebraic_constraints as algebraic_constraints
        if isinstance(self,body):
            R = sm.Eq(self.R,sm.MutableDenseMatrix([0,0,0]))
            P = sm.Eq(self.P,sm.MutableDenseMatrix([1,0,0,0]))
            Rd = sm.Eq(self.Rd,sm.MutableDenseMatrix([0,0,0]))
            Pd = sm.Eq(self.Pd,sm.MutableDenseMatrix([1,0,0,0]))
            return [R,P,Rd,Pd]
        
        elif isinstance(self,algebraic_constraints):
            loc  = vector('pt_%s'%self.name)
            axis = vector('ax_%s'%self.name)
            
            loc  = sm.Eq(loc,sm.MutableDenseMatrix([0,0,0]))
            axis = sm.Eq(axis,sm.MutableDenseMatrix([0,0,1]))
            return [loc,axis]



