# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:47:50 2019

@author: khale
"""

import sympy as sm
import networkx as nx
import matplotlib.pyplot as plt

sm.init_printing(pretty_print=False,use_latex=True,forecolor='White')

###############################################################################
###############################################################################

class AbstractMatrix(sm.MatrixExpr):
    """
    **Abstract Class**
    
    Base class of symbolic matrices which their values are evaluated based 
    on given parameters. Can be thought of as an undefind functions that
    returns a matrix

    Class Attributes
    ----------------
    is_commutative : False
        
    is_Matrix : True
    
    """
    
    is_commutative = False
    is_Matrix = True
    shape = (3,3)
    
    def __init__(self,*args):
        pass
    def doit(self):
        return self
    def _latex(self,expr):
        name = self.__class__.__name__
        return r'{%s%s}'%(name,self.args,)
    
    def _entry(self,i,j):
        v = sm.MatrixSlice(self,i,j)
        return v
    
class A(AbstractMatrix):
    """
    Representaion of symbolic transformation matrix which represents the 
    orientation of a rigid body in 3D space. The matrix is a function of 
    the orientation parameters used, e.g. Euler-Parameters.
    
    Parameters
    ----------
    P : quatrenion or vector
        Orientation parameters of the rigid body.
    """
    shape = (3,3)
    def _latex(self,expr):
        p = self.args[0]
        return r'{A(%s)}'%p.name


class G(AbstractMatrix):
    shape = (3,4)
    def _latex(self,expr):
        p = self.args[0]
        return r'{G(%s)}'%p.name
    

class E(AbstractMatrix):
    shape = (3,4)
    def _latex(self,expr):
        p = self.args[0]
        return r'{E(%s)}'%p.name
    
class B(AbstractMatrix):
    r"""
    Representaion of symbolic jacobian of the transformation
    :math:`u = A(P)\bar{u}` with respect to euler-parameters quatrenion :math:`P`, 
    where :math:`\bar{u}` is a body-local vector.
    
    Parameters
    ----------
    P : quatrenion or vector
        Orientation parameters of the rigid body.
    u : body-local vector
        A vector thet is defind relative to the body's local reference frame.
    """
    
    shape = (3,4)
    def __init__(self,P,u):
        pass
    def _latex(self,expr):
        p,u = self.args
        return r'{B(%s,%s)}'%(p.name,u.name)

class Triad(AbstractMatrix):
    """
    A symbolic matrix function that represents a triad of three orthonormal
    vectors in 3D that is oriented given one or two vetcors
    
    Parameters
    ----------
    v1 : vector
        Z-Axis of the triad.
    v2 : vector, optional
        X-Axis of the triad.
    """
    
    def __init__(self,v1,v2=None):
        super().__init__(v1)

class Skew(AbstractMatrix):
    shape = (3,3)
    def _latex(self,expr):
        p = self.args[0]
        return r'{Skew(%s)}'%p

class Force(AbstractMatrix):
    shape = (3,1)
    def _latex(self,expr):
        name = self.__class__.__name__
        return r'{%s%s}'%(name,self.args,)
class Moment(AbstractMatrix):
    shape = (3,1)
    def _latex(self,expr):
        name = self.__class__.__name__
        return r'{%s%s}'%(name,self.args,)


def matrix_function_constructor(cls_name,shape=(3,1),**kwargs):
    attrs = {'shape':shape}
    cls = type(cls_name,(AbstractMatrix,),attrs)
    return cls
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

###############################################################################
###############################################################################

class base_vector(sm.MatrixSlice):
    """
    A symbolic matrix function that represents the base-vectors (i,j,k) of 
    a reference frame.
    
    Parameters
    ----------
    frame : reference frame
        An instance of a reference frame.
    sym : str, {'i','j','k'}
        A string character representing the base-vector name.
        
    """
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
    
    def __getnewargs_ex__(self):
        args = (self.frame,self._sym)
        kwargs = {}
        return (args, kwargs)
    
    @property
    def name(self):
        """
        Returns the formated name of the instance which is suitable for Latex
        printings.
        """
        return self._formated
    
    def express(self,frame=None):
        """
        Transform the base-vector form its' current frame to the given frame.

        Args
        ----
        frame : reference frame, optional
            If frame = None, the default is to perform the transformation to the
            global frame.
        """
        frame = (frame if frame is not None else self.frame.global_frame)
        A = self.frame.parent.express(frame)
        return A*self
    
    def doit(self):
        return self
    
    def _latex(self,expr):
        return self._formated
        
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
    """
    A symbolic matrix that represents a 3x3 directional cosines matrix.
    Used to represent symbolic markers and as the default value of a reference
    frame orientation matrix.
    
    Parameters
    ----------
    name : str
        Should not contain any special characters.
    format_as : str, optional
        A formated version of the name that may contain special characters used
        for pretty latex printing.
    """
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
    """
    A symbolic matrix that represents a (m x n) zero matrix in a non-explicit
    form.
    
    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of cols.
    """
    is_ZeroMatrix = True
    
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
    """
    A representation of the global frame of reference of a given spatial 
    multi-body system.
    
    Parameters
    ----------
    name : str, (optional, Defaults to blank)
        Name of the global_frame instance.
    
    Methods
    -------
    draw_tree()
        Draw the directed graph.
        
    express_func(frame_1, frame_2, tree)
        Perform transformation from frame_1 to frame_2 using the ralational tree.
    
    express(other)
        Performe reference transformation between the global frame and a given
        reference frame.
    
    merge_global(g,orient=False)
        Add the content of a given global reference `g` to the current global
        frame scoope.
    
    Notes
    -----
    This class provides two main functionalities:
        
        1. Initialization of a Directed Graph that serves as the storage of all
        instantiated reference-frame objects in a given scoope, e.g. bodies and 
        joint markers, and their bi-directional transformations.
        
        2. Separation of multi-body systems' globals to prevent names collisions
        and unpredictable wrong transformations, and also provide the extisability
        of nested globals in future development.
    
    The use of directed graph serves a natural and convenient way to capture
    the relational information of the instantiated references.
    
    Every new reference is stored as a **node**, this node is connected to its' 
    parent reference by two opposite **edges** that represents the tranfromation 
    between the frame and its' parent in the two directions.
    
    Each **edge** holds an attribute called *mat* that represents the 
    transformation matrix that performs transformation from the **tail node** 
    to the **head node** of that edge.
    
    Transformations bwteen two given nodes are done by searching the graph for
    the shortest path between these two nodes, where the transformation is 
    simply the multiplication of all the matricies stored on the edges making
    up this path. The seacrching is done using `networkx.algorithms.shortest_path`
    function.
    
    It should be mentioned that any `express` method in the `abstract_matricies` 
    module makes use of this class's `express_func` static method. 
    Also it is worth mentioning that this operation is done symbolically on the 
    matrix level and not its elements.
    
    """
        
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
    
    @property
    def global_frame(self):
        return self
    
    def merge_global(self,g,orient=False):
        """
        Add the content of a given global reference `g` to the current global
        frame scoope.
        
        Parameters
        ----------
        g : global_frame
            An instance of the global_frame class.
        orient : bool, (optional, Defaults to False)
            Orient the given global_frame instance relative to the current 
            global instance
        """
        self.references_tree.add_nodes_from(g.nodes(data=True))
        self.references_tree.add_edges_from(g.edges(data=True))
        if not orient:
            self.references_tree.add_edge(self.name, g.name, mat=1)
            self.references_tree.add_edge(g.name, self.name, mat=1)
            
    def draw_tree(self):
        """
        Draw the directed graph tree using matplotlib.pyplot with the nodes
        labled.
        """
        plt.figure(figsize=(10,6))
        nx.draw(self.references_tree,with_labels=True)
        plt.show()
    
    @staticmethod
    def express_func(frame1,frame2,tree):
        """
        Static Method of the class the serves as a helper to the 
        self.express(other) method definition and the reference_frame class
        express method.
        
        Parameters
        ----------
        frame1, frame_2 : reference_frame_like
            An instance of  `global_frame` or `reference_frame` classes.
        tree : nx.DiGraph
            The directed reference tree that stores the information of the 
            relation between the given references.
        
        Returns
        -------
        mat : sympy.MatMul
            A sequence of matrix multiplications that represents the 
            transformation between the given frame.
        """
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
        """
        Evaluate the symbolic transformation from self to other reference frame.
        
        Parameters
        ----------
        other : reference_frame_like
            An instance of `global_frame` or `reference_frame` classes.
       
        Returns
        -------
        mat : sympy.MatMul
            A sequence of matrix multiplications that represents the symbolic
            transformation to the given other frame.
        """
        return self.express_func(self,other,self.references_tree)

###############################################################################
class reference_frame(object):
    """A representation of generic reference frames.
    
    Parameters
    ----------
    name : str
        Name of the reference_frame instance. Should mimic a valid python 
        variable name.
    parent : reference_frame_like, optional
        The reference frame that this frame is referred to initially. Defaults
        to the current globale_frame instance
    formate_as : str, optional
        A specially formated name for appropriate latex printing. defaults to
        None.
    
    Attributes
    ----------
    global_instance : global_frame
        The instance of `global_frame` that hold this reference frame in its'
        directed graph network.
    A : dcm or A
        The directional cosines matrix that represents the symbolic orientation
        of the given reference frame.
    parent : reference_frame_like
        The reference frame that this frame is referred to.
    
    Methods
    -------
    set_global_frame(global_instance)
        **Class Method**. 
        Specify the global_frame instance used where this reference frame 
        exists.
    
    draw_tree()
        Draw the directed graph of the global_frame instance used to add this
        reference frame.
    
    orient_along(v1,v2=None)
        Specify the reference_frame matrix :math:`A` to be a triad that is 
        oriented using the given vectors v1 and v2.
        
    express(other)
        Performe reference transformation between this frame and a given
        reference frame.
    """
    
    _is_global_set  = False
    global_frame = None
    
    @classmethod
    def set_global_frame(cls,global_instance):
        cls._is_global_set = True
        cls.global_frame = global_instance
    
    def __new__(cls, name, parent=None,format_as=None):
        """
        Overloading the original __new__ method to check whether a global_frame
        exists or not, and creat one if no global_frame exists.
        """
        if not cls._is_global_set:
            print('creating global',name)
            global_instance = global_frame()
            cls.set_global_frame(global_instance)
        return super(reference_frame,cls).__new__(cls)
    
    
    def __init__(self, name, parent=None,format_as=None):
#        print('Inside reference_frame()',name)
        self._raw_name = name
        self._formated_name = (format_as if format_as is not None else name)
        self.global_frame = reference_frame.global_frame
        self.parent = (parent if parent else self.global_frame)
        self._A = dcm(str(name),format_as=format_as)
        self._update_tree()
    
    def __getnewargs_ex__(self):
        args = (self.name,)
        kwargs = {'parent':self.parent,'format_as':self._formated_name}
        return (args, kwargs)
    
#    def __getstate__(self):
#        # Copy the object's state from self.__dict__ which contains
#        # all our instance attributes. Always use the dict.copy()
#        # method to avoid modifying the original state.
#        state = self.__dict__.copy()
#        # Remove the unpicklable entries.
#        state['global_frame_name'] = self.global_frame.name
#        return state
#    
#    def __setstate__(self,state):
#        self.__dict__.update(state)
#        self.global_frame = global_frame(state['global_frame_name'])
#        reference_frame.global_frame = self.global_frame
        
    def _update_tree(self):
        """
        Update the global_frame references_tree and add directed edges with
        their *mat* attribute.
        """
        self.global_frame.references_tree.add_edge(self.parent.name, self.name, mat=self.A.T)
        self.global_frame.references_tree.add_edge(self.name, self.parent.name, mat=self.A)
    
    @property
    def A(self):
        """
        The matrix that represents the orientation of the reference frame 
        relative to its' parent.
        """
        return self._A
    @A.setter
    def A(self,value):
        self._A = value
        self._update_tree()
    
    @property
    def name(self):
        return self._raw_name
    
    @property
    def i(self):
        return base_vector(self,'i')
    @property
    def j(self):
        return base_vector(self,'j')
    @property
    def k(self):
        return base_vector(self,'k')
                
    def orient_along(self,v1,v2=None):
        """
        Specify the reference_frame matrix :math:`A` to be a triad that is 
        oriented using the given vectors v1 and/or v2.
        
        Parameters
        ----------
        v1 : vector
            Z-Axis of the triad.
        v2 : vector, optional
            X-Axis of the triad.
        """
        if v2 is None:
            self.A = Triad(v1)
        else:
            self.A = Triad(v1,v2)
            
    def express(self,other):
        """
        Performe reference transformation between this frame and a given
        reference frame.
        
        Parameters
        ----------
        other : reference_frame_like
            The reference frame the where this frame is referred relative to.
        
        Returns
        -------
        mat : sympy.MatMul
            A sequence of matrix multiplications that represents the symbolic
            transformation to the given other frame. 
        """
        tree = self.global_frame.references_tree
        return self.global_frame.express_func(self,other,tree)
    
    @property
    def free_symbols(self):
        return set()
    
    def _ccode(self,expr,**kwargs):
        return self._raw_name

    def _set_base_vectors(self):
        self.i = base_vector(self,'i')
        self.j = base_vector(self,'j')
        self.k = base_vector(self,'k')
    
###############################################################################
###############################################################################
class matrix_symbol(sm.MatrixSymbol):
    
    def __new__(cls,name,m,n, format_as=None):
        # Oveloading the MatrixSymbol __new__ method to supply it with the 
        # appropriat arguments (name,m,n)
        if format_as:
            name = format_as
        return super(matrix_symbol,cls).__new__(cls,name,m,n)
    
    def __init__(self,name,m,n, format_as=None):
        self._raw_name = name
        self._formated_name = super().name
        self._args = (name,m,n,self._formated_name)
    
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

   
class vector(sm.MatrixSymbol):
    """A (3 x 1) symbolic matrix.
    
    Parameters
    ----------
    name : str
        Name of the vector instance. Should mimic a valid python 
        variable name.
    frame : reference_frame_like, optional
        The reference frame that this vector is referred to initially. Defaults
        to the current globale_frame instance
    formate_as : str, optional
        A specially formated name for appropriate latex printing. defaults to
        None.
    
    Attributes
    ----------
    frame : reference_frame_like
        The reference frame that this frame is referred to.
    
    Methods
    -------        
    express(other)
        Transform the vector from its' current frame to the given frame.
    """
    
    shape = (3,1)
    
    is_commutative = False
    
    def __new__(cls,name, frame=None, format_as=None):
        # Oveloading the MatrixSymbol __new__ method to supply it with the 
        # appropriat arguments (name,m,n)
        if format_as:
            name = format_as
        return super(vector,cls).__new__(cls,name,3,1)
    
    def __init__(self,name, frame=None, format_as=None):
        self._raw_name = name
        self._formated_name = super().name
        self.frame = (frame if frame is not None else reference_frame.global_frame)        
        self._args = (name,self.frame,self._formated_name)
        
    def express(self,frame=None):
        """
        Transform the vector from its' current frame to the given frame.
        
        Parameters
        ----------
        frame : reference_frame_like, optional
            The reference frame the where this frame is referred relative to.
            Defaults to the vector's global_frame instance.
        
        Returns
        -------
        v : sympy.MatMul
            The vector multiplied by sequence of matrix multiplications that 
            represents the symbolic transformation to the given other frame. 
        """
        frame = (frame if frame is not None else self.frame.global_frame)
        A = self.frame.express(frame)
        v = A*self
        return v
    
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
    """A (4 x 1) symbolic matrix.
    
    Parameters
    ----------
    name : str
        Name of the vector instance. Should mimic a valid python 
        variable name.
    formate_as : str, optional
        A specially formated name for appropriate latex printing. defaults to
        None.    
    """
    shape = (4,1)
    is_commutative = False
    
    def __new__(cls, name, format_as=None):
        # Oveloading the MatrixSymbol __new__ method to supply it with the 
        # appropriat arguments (name,m,n)
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



