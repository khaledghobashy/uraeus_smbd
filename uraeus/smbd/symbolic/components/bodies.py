# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 10:57:47 2019

@author: khale
"""

# Standard library imports
import itertools

# 3rd parties library imports
import sympy as sm

# Local application imports
from .matrices import (reference_frame, vector, quatrenion, zero_matrix, 
                       A, G, E, matrix_symbol)


__all__ = ['body','ground']

class body(reference_frame):
    r"""
    
    A class that represents an un-constrained rigid body object in 3D in a 
    symbolic form, where all the body parameters and equations are generated 
    automatically in a symbolic format.
    
    Parameters
    ----------
    name : str
        Name of the body instance. Should mimic a valid python variable name.
    
    Attributes
    ----------
    n : int
        Number of generalized coordinates used to define the body 
        configuration. Equals 7.
    nc : int
        Number of scalar constraint equations (euler-parameters normalization).
    nve : int
        Number of vetor constraint equations (euler-parameters normalization).
    
    A :  A
        The directional cosines matrix that represents the symbolic orientation
        of the body relative to the global_frame. This matrix is function of
        the body orientation parameters e.g. euler-parameters
    R : vector
        A symbolic matrix that represents the location of the body's reference 
        point relative to the global origin.
    Rd : vector
        A symbolic matrix that represents the translational velocity of the
        body's reference point relative to the global origin.
    P : quatrenion
        A symbolic matrix that represents the orientation of the body's 
        reference frame relative to the global frame in terms of 
        euler-parameters.
    Pd : quatrenion
        A symbolic matrix that represents the rotaional velocity of the body's 
        reference frame relative to the global frame in terms of
        euler_parameters time-derivatives.
    
    q : sympy.BlockMatrix
        Blockmatrix containing the position-level coordinates of the body
    qd : sympy.BlockMatrix
        Blockmatrix containing the velocity-level coordinates of the body
        
    normalized_pos_equation : sympy.MatrixExpr
        The normalization equation of the euler-parameters quatrenion at the
        position level.
    
    normalized_vel_equation : sympy.MatrixExpr
        The normalization equation of the euler-parameters quatrenion at the
        velocity level.
    
    normalized_acc_equation : sympy.MatrixExpr
        The normalization equation of the euler-parameters quatrenion at the
        acceleration level.
    
    normalized_jacobian : list (of sympy.MatrixExpr)
        The jacobian of the normalization equation of the euler-parameters 
        quatrenion relative to the vector of euler-parameters.
        
    arguments_symbols : list (of symbolic objects)
        A list containing the symbolic mathematical objects that should be 
        nuemrically defined by the user in a numerical simulation session.
    
    runtime_symbols : list (of symbolic objects)
        A list containing the symbolic mathematical objects that changes during
        the run-time of a nuemric simulation's "solve" method.
         
    constants_symbolic_expr : list (of sympy.Equality)
        A list containing sympy equalities representing the values of internal
        class symbolic constants that are evaluated from other symbolic 
        expressions.
    
    constants_numeric_expr : list (of sympy.Equality)
        A list containing sympy equalities representing the values of internal
        class symbolic constants that are evaluated directly from numerical 
        expressions.
    
    constants_symbols : list (of symbolic objects)
        A list containing all the symbolic mathematical objects that represent 
        constants for the given body instance.
      
    Notes
    -----
    An un-constraied body in space is typically defined using 6 generalized 
    coordinates representing its' location and orientation. In cartesian 
    coordinate system, body location is simply defined by the $$(x,y,z)$$ 
    coordinates of a reference point on the body -normally the center-of-mass -
    , where the body orientation can be defined in various ways, such as the 
    directional cosines matrix, euler-angles and euler-parameters.
    
    The package uses euler-parameters -which is a 4D unit quaternion- to 
    represents a given body orientation in space. This makes the generalized 
    coordinates used to fully define a body in space to be 7, instead of 6, 
    it also adds an algebraic equation to the constraints that ensures the 
    unity/normalization of the body quaternion.
        
    """
    
    n   = 7
    nc  = 1
    nve = 1
    
    def __init__(self, name):
        self._name = name
        super().__init__(name)
        
        splited_name = name.split('.')
        self.id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self.prefix  = (self.prefix+'.' if self.prefix!='' else self.prefix)
        
        format_ = (self.prefix, self.id_name)
        
        self.R  = vector('%sR_%s'%format_, format_as=r'{%sR_{%s}}'%format_)
        self.P  = quatrenion('%sP_%s'%format_, format_as=r'{%sP_{%s}}'%format_)
        
        self.Rd = vector('%sRd_%s'%format_, format_as=r'{%s\dot{R}_{%s}}'%format_)
        self.Pd = quatrenion('%sPd_%s'%format_, format_as=r'{%s\dot{P}_{%s}}'%format_)
        
        self.Rdd = vector('%sRdd_%s'%format_, format_as=r'{%s\ddot{R}_{%s}}'%format_)
        self.Pdd = quatrenion('%sPdd_%s'%format_, format_as=r'{%s\ddot{P}_{%s}}'%format_)
        
        self.A = A(self.P)
        self.G = G(self.P)
        self.E = E(self.P)
                
        self.normalized_pos_equation = (self.P.T*self.P) - sm.Identity(1)
        self.normalized_vel_equation = zero_matrix(1,1)
        self.normalized_acc_equation = 2*(self.Pd.T*self.Pd)
        self.normalized_jacobian = [zero_matrix(1, 3), 2*self.P.T]
                        
        self.m = sm.symbols('m_%s'%self.id_name)
        self.M = self.m*sm.Identity(3)
        self.Jbar = matrix_symbol('%sJbar_%s'%format_, 3, 3, r'{%s\bar{J}_{%s}}'%format_)
        self.J = 4*E(self.P).T * self.A * self.Jbar * self.A.T * E(self.P)
        
    
    @property
    def name(self):
        return self._name
    
    @property
    def q(self):
        return sm.BlockMatrix([[self.R], [self.P]])
    @property
    def qd(self):
        return sm.BlockMatrix([[self.Rd], [self.Pd]])
    @property
    def qdd(self):
        return sm.BlockMatrix([[self.Rdd], [self.Pdd]])
        
    @property
    def arguments_symbols(self):
        return [self.R, self.P, self.Rd, self.Pd, self.Rdd, self.Pdd,
                self.m, self.Jbar]
    @property
    def runtime_symbols(self):
        return [self.R, self.P, self.Rd, self.Pd, self.Rdd, self.Pdd]
    @property
    def constants_symbolic_expr(self):
        return []
    @property
    def constants_numeric_expr(self):
        return []
    @property
    def constants_symbols(self):
        constants_expr = itertools.chain(self.constants_symbolic_expr,
                                         self.constants_numeric_expr)
        return [expr.lhs for expr in constants_expr]
        


class ground(body):
    """
    A representation of the gorund as a special case of the body class. This 
    adds the needed constraint equations that makes the ground fixed in the
    global reference frame.
    
    """
    
    n   = 7
    nc  = 7
    nve = 2
    
    def __new__(cls, *args, **kwargs):
        name = 'ground'
        return super().__new__(cls, name)
    def __init__(self, *args, **kwargs):
        name = 'ground'
        super().__init__(name)
        self.P_ground = quatrenion('Pg_%s'%self.name, format_as=r'{Pg_{%s}}'%self.name)
        
        self.normalized_pos_equation = sm.BlockMatrix([[self.R], [self.P-self.P_ground]])
        self.normalized_vel_equation = sm.BlockMatrix([[zero_matrix(3,1)],[zero_matrix(4,1)]])
        self.normalized_acc_equation = sm.BlockMatrix([[zero_matrix(3,1)],[zero_matrix(4,1)]])
        self.normalized_jacobian = sm.BlockMatrix([[sm.Identity(3), zero_matrix(3,4)],
                                                   [zero_matrix(4,3), sm.Identity(4)]])
    
    @property
    def arguments_symbols(self):
        return []#[self.R, self.P, self.Rd, self.Pd, self.Rdd, self.Pdd]
    
    @property
    def constants_numeric_expr(self):
        position = sm.Eq(self.R, sm.Matrix([0,0,0]))
        expr1 = sm.Eq(self.P, sm.Matrix([1,0,0,0]))
        expr2 = sm.Eq(self.P_ground, sm.Matrix([1,0,0,0]))
        mass = sm.Eq(self.m, sm.Float(1))
        iner = sm.Eq(self.Jbar, sm.eye(3,3))
        return [position, expr1, expr2, mass, iner]


###############################################################################
###############################################################################


