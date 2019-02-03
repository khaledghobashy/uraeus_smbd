# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:47:50 2019

@author: khale
"""

import sympy as sm
import collections

import networkx as nx
import matplotlib.pyplot as plt

sm.init_printing(pretty_print=False,use_latex=True,forecolor='White')

###############################################################################
###############################################################################

class AbstractMatrix(sm.MatrixExpr):
    
    is_commutative = False
    is_Matrix = True
    shape = (3,3)
    
    def __init__(self,*args):
        pass
    def doit(self):
        return self
    
class A(AbstractMatrix):
    
    def _latex(self,expr):
        p = self.args[0]
        return r'{A(%s)}'%p.name


class G(AbstractMatrix):
    shape = (3,4)
    

class E(AbstractMatrix):
    shape = (3,4)
    
class B(AbstractMatrix):
    shape = (3,4)
    def __init__(self,sym1,sym2):
        pass
    def _latex(self,expr):
        p,u = self.args
        return r'{B(%s,%s)}'%(p.name,u.name)
    
###############################################################################
###############################################################################

class Triad(AbstractMatrix):
    def __init__(self,v1,v2=None):
        super().__init__(v1)
        
class Mirror(AbstractMatrix):
    def __init__(self,v1):
        super().__init__(v1)
        self.shape = v1.shape
    def _latex(self,expr):
        return r'{Mir(%s)}'%self.args[0].name

class Centered(AbstractMatrix):
    shape = (3,1)
    def __init__(self,*args):
        super().__init__(*args)
    def _latex(self,expr):
        return r'{Centered%s}'%(self.args,)

class Oriented(AbstractMatrix):
    shape = (3,1)
    def __init__(self,*args):
        super().__init__(*args)
    def _latex(self,expr):
        return r'{Oriented%s}'%(self.args,) 

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
        self._formated = r'{\hat{%s}_{%s}}'%(self._sym,self.frame.A)
    
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
        return '%r[:,%s:%s]'%(self.frame.name,*self.slice)
    
    def _sympystr (self,expr):
        return '%s[:,%s]'%(self.frame.name,self.slice)
        
    @property
    def func(self):
        return base_vector
    @property
    def free_symbols(self):
        return set()

###############################################################################
class dcm(sm.MatrixSymbol):
    shape = (3,3)
    
    is_commutative = False
    
    def __new__(cls,name, format_as=None):
        if format_as is not None:
            name = format_as
        return super(dcm,cls).__new__(cls,name,3,3)
    
    def __init__(self,name,format_as=None):
        self._raw_name = name
        self._formated_name = super().name
        self._args=(name,format_as)
        
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
        sym = r'{Z_{%sx%s}}'%(m,n)
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
        
    def __init__(self,name=''):
        self.name = name+'_grf'
        self.references_tree = nx.DiGraph(name=name)
        self.references_tree.add_node(self.name)
    
    @property
    def nodes(self):
        return self.references_tree.nodes
    @property
    def edges(self):
        return self.references_tree.edges
    
    def merge_global(self,g,orient=False):
        self.references_tree.add_nodes_from(g.nodes(data=True))
        self.references_tree.add_edges_from(g.edges(data=True))
        if not orient:
            self.references_tree.add_edge(self.name, g.name, mat=1)
            self.references_tree.add_edge(g.name, self.name, mat=1)
            
    def draw_tree(self):
        plt.figure(figsize=(10,6))
        nx.draw(self.references_tree,with_labels=True)
        plt.show()
    
    @staticmethod
    def express_func(frame1,frame2,tree):
        child_name  = frame1.name
        parent_name = frame2.name
        graph = tree
        
        path_nodes  = nx.algorithms.shortest_path(graph, child_name, parent_name)
        path_matrices = []
        for i in range(len(path_nodes)-1):
            path_matrices.append(graph.edges[path_nodes[-i-2],path_nodes[-i-1]]['mat'])
        mat = sm.MatMul(*path_matrices)
        return mat
    
    def express(self,other):
        return self.express_func(self,other,self.references_tree)

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
        self._formated_name = (format_as if format_as is not None else name)
        self.parent = (parent if parent else self.global_frame)
        self.A = dcm(str(name),format_as=format_as)
             
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
        return self._raw_name
    
    def _ccode(self,expr,**kwargs):
        return self._raw_name
            
    def orient_along(self,v1,v2=None):
        if v2 is None:
            self.A = Triad(v1)
        else:
            self.A = Triad(v1,v2)
            
    def express(self,other):
        tree = self.global_frame.references_tree
        return self.global_frame.express_func(self,other,tree)
    
    @property
    def free_symbols(self):
        return set()
    
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
    @property
    def raw_name(self):
        return self._raw_name
   
    def doit(self):
        return self
    
    def __str__(self):
        return self.raw_name

        
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
    @property
    def raw_name(self):
        return self._raw_name

    def doit(self):
        return self
    
    @property
    def func(self):
        return self.__class__
    
    def __str__(self):
        return self.raw_name
                
###############################################################################
###############################################################################



