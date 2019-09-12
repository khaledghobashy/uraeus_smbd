# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:09:05 2019

@author: khaledghobashy
"""


# 3rd parties library imports
import sympy as sm

# Local application imports
from .matrices import (zero_matrix, B)

# Commonly used variables
I  = sm.Identity(3)
I1 = sm.Identity(1)

###############################################################################

class spehrical_constraint(object):
    
    # Number of Scalar Constraint Equations
    nc  = 3
    
    def __init__(self):
        """
        A class that represents a primitive Spherical/At-Point Constraint.
        This constrains a given pair of bodies to be coincident at a given 
        point where no relative translation is allowed in any direction, 
        only rotations may take place at that point about any axis.
        
        The construction of this constraint does not require any symbolic 
        parameters.
        
        Notes
        -----
        TODO
        
        """
        pass
    
    def construct(self, obj):
        pos_level_equation = obj.dij
        vel_level_equation = zero_matrix(3,1)
        acc_level_equation = B(obj.Pdi,obj.ui_bar)*obj.Pdi\
                            -B(obj.Pdj,obj.uj_bar)*obj.Pdj
            
        jacobian = ([ I,  obj.Bui] , [-I, -obj.Buj])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
    
    
###############################################################################
class dot_product_1(object):
    
    nc  = 1
    
    def __init__(self, v1='i', v2='j'):
        """
        A class that represents a primitive dot-product Constraint.
        This constraint enforces two vectors (v1, v2) on two different bodies 
        (body_i, body_j) to be perpendicular at all time by setting their 
        dot-product to zero.
        
        Parameters
        ----------
        v1 : str (i, j, k)
            A string of 'i', j' or 'k' that represents a unit vector of the 
            local joint marker on the 1st body (body_i).
        
        v2 : str (i, j, k)
            A string of 'i', j' or 'k' that represents a unit vector of the 
            local joint marker on the 2nd body (body_j)
        
        Notes
        -----
        TODO
        
        """
        self.v1 = v1
        self.v2 = v2
    
    def construct(self,obj):
        v1 = self.v1
        v2 = self.v2
        
        v1_bar = getattr(obj.mi_bar,v1)
        v1     = v1_bar.express()
        v2_bar = getattr(obj.mj_bar,v2)
        v2     = v2_bar.express()
        
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        Z = zero_matrix(1,3)
        
        pos_level_equation = v1.T*v2
        vel_level_equation = zero_matrix(1, 1)
        acc_level_equation =   v1.T*B(Pdj,v2_bar)*Pdj \
                             + v2.T*B(Pdi,v1_bar)*Pdi \
                             + 2*(B(obj.Pi,v1_bar)*Pdi).T*(B(obj.Pj,v2_bar)*Pdj)
        
        jacobian = ([Z, v2.T*B(obj.Pi,v1_bar)], [Z, v1.T*B(obj.Pj,v2_bar)])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
    
###############################################################################

class dot_product_2(object):
    
    nc  = 1
    
    def __init__(self, v='i'):
        """
        A class that represents a primitive dot-product Constraint.
        This constraint enforces two vectors (v1, rij) to be perpendicular 
        at all time by setting their dot-product to zero. 
        The vector (v1) is assumed to be local to (body_i), where the vector 
        (rij) represents the relative position vector of joint definition 
        points on the bodies (body_i, body_j).
        
        Parameters
        ----------
        v1 : str (i, j, k)
            A string of 'i', j' or 'k' that represents a unit vector of the 
            local joint marker on the 1st body (body_i).
        
        Notes
        -----
        TODO
        
        """
        
        self.v = v
    
    def construct(self, obj):
        v = self.v
                
        v_bar = getattr(obj.mi_bar, v)
        v = v_bar.express()
        
        dij = obj.dij
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        dijd = obj.dijd
        
        pos_level_equation = v.T*dij
        vel_level_equation = zero_matrix(1, 1)
        acc_level_equation =   v.T*( B(Pdi, obj.ui_bar)*Pdi - B(Pdj, obj.uj_bar)*Pdj ) \
                             + dij.T*B(Pdi, v_bar)*Pdi \
                             + 2*(B(obj.Pi, v_bar)*Pdi).T*dijd
        
        jacobian = ([ v.T, dij.T*B(obj.Pi, v_bar) + v.T*obj.Bui], 
                    [-v.T, -v.T*obj.Buj])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
        
###############################################################################

class angle_constraint(object):
    
    nc = 1
    
    def __init__(self):
        """
        A class that represents an angle Constraint.
        This constraint enforces the angle between two vectors (i, j) to be of
        a specified value.
        
        The vector (i) represents a unit base vector of the joint marker that
        is local to the (body_i), where vector (j) represents a unit base 
        vector of the joint marker that is local to the (body_j)
        
        Notes
        -----
        TODO
        
        """
        pass
    
    def construct(self, obj):
        v1 = 'i'
        v2 = 'i'
        v3 = 'j'
        
        v1_bar = getattr(obj.mi_bar, v1)
        v1     = v1_bar.express()
        v2_bar = getattr(obj.mj_bar, v2)
        v2     = v2_bar.express()
        v3_bar = getattr(obj.mi_bar, v3)
        v3     = v3_bar.express()
        
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        Z = zero_matrix(1, 3)
        
        theta = obj.act_func(obj.t)
        c = sm.cos(theta)
        s = sm.sin(theta)
        
        pos_level_equation = (v3.T*v2)*c - (v1.T*v2)*s
        vel_level_equation = zero_matrix(1, 1)        
        acc_level_equation =   (c*v3.T - s*v1.T)*B(Pdj, v2_bar)*obj.Pdj \
                             + v2.T*(c*B(Pdi,v3_bar) - s*B(Pdi,v1_bar))*Pdi \
                             + 2*(c*B(obj.Pi,v3_bar)*Pdi - s*B(obj.Pi,v1_bar)*Pdi).T * (B(obj.Pj,v2_bar)*Pdj)
        
        jacobian = ([Z, v2.T*(c*B(obj.Pi,v3_bar) - s*B(obj.Pi,v1_bar))],
                    [Z, (c*v3.T - s*v1.T)*B(obj.Pj,v2_bar)])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])

###############################################################################

class distance_constraint(object):
    
    nc  = 1
    
    def __init__(self):
        pass
    
    def construct(self,obj):
        dij = obj.dij
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        dijd = obj.dijd
        
        distance = obj.act_func(obj.t)
        
        pos_level_equation = sm.sqrt(dij.T*dij) - I1*distance
        
        vel_level_equation = zero_matrix(1, 1)
        
        acc_level_equation =   2*dij.T * (B(Pdi, obj.ui_bar)*Pdi - B(Pdj, obj.uj_bar)*Pdj) \
                             + 2*dijd.T*dijd
        
        jacobian = ([ (2*dij.T) * I,  (2*dij.T) * obj.Bui] ,
                    [-(2*dij.T) * I, -(2*dij.T) * obj.Buj])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
            

###############################################################################

class coordinate_constraint1(object):
    
    """
    A class that represents a Coordinate Constraint.
    This constraint enforces the given coordinate of a given body to be of a 
    specified value.
    
    Notes
    -----
    TODO
    
    """
    
    nc  = 1
    
    def __init__(self):
        pass
    
    def construct(self, obj):
        i  = obj.i
        Ri = obj.Ri
        Ai = obj.Ai
        ui_bar = obj.ui_bar
        Pdi = obj.Pdi

        pos_level_equation = (Ri + Ai*ui_bar)[i,:]
        vel_level_equation = zero_matrix(1, 1)
        acc_level_equation = (B(Pdi,ui_bar)*Pdi)[i,:]
        
        J_R = I[i,:]
        J_P = obj.Bui[i,:]
            
        jacobian = ([J_R, J_P], 
                    [zero_matrix(1,3), zero_matrix(1,4)])
       
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
        
###############################################################################

class coordinate_constraint(object):
    
    """
    A class that represents a Coordinate Constraint.
    This constraint enforces the given coordinate of a given body (body_i) to 
    be of a specified value relative to another body (body_j) .
    
    Notes
    -----
    TODO
    
    """
    
    nc  = 1
    
    def __init__(self):
        pass
    
    def construct(self, obj):
        i  = obj.i        
        Ri = obj.Ri
        Ai = obj.Ai
        Rj = obj.Rj
        Aj = obj.Aj
        ui_bar = obj.ui_bar
        uj_bar = obj.uj_bar

        pos_level_equation = (Ri + Ai*ui_bar)[i,:] - (Rj + Aj*uj_bar)[i,:]
        vel_level_equation = zero_matrix(1, 1)
        acc_level_equation = (B(obj.Pdi,obj.ui_bar)*obj.Pdi\
                            -B(obj.Pdj,obj.uj_bar)*obj.Pdj)[i,:]
        
        J_R = I[i,:]
        J_P = obj.Bui[i,:]
            
        jacobian = ([J_R, J_P], 
                    [-I[i,:], -obj.Buj[i,:]])
       
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
        

###############################################################################
