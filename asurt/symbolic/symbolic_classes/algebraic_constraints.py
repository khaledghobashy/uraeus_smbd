# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:06:05 2019

@author: khale
"""
# Standard library imports
import itertools
from IPython.display import display

# 3rd parties library imports
import sympy as sm

# Local application imports
from .matrices import (reference_frame, vector, zero_matrix, B, E, 
                               matrix_symbol, Skew)
from .bodies import body
from .helpers import body_setter, name_setter


I = sm.Identity(3)

class algebraic_constraints(object):
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self,name,body_i=None,body_j=None):
        name_setter(self,name)
                
        for i in range(self.def_axis):
            self._create_joint_def_axis(i+1)
        for i in range(self.def_locs):
            self._create_joint_def_loc(i+1)
        
        self._create_joint_arguments()
                
        if body_i and body_j:
            self.body_i = body_i
            self.body_j = body_j
            self.construct()

    @property
    def name(self):
        return self._name
    @property
    def id_name(self):
        splited_name = self.name.split('.')
        return ''.join(splited_name[-1])
    
    @property
    def body_i(self):
        return self._body_i
    @body_i.setter
    def body_i(self,body_i):
        body_setter(self,body_i,'i')
            
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self,body_j):
        body_setter(self,body_j,'j')
    
    @property
    def dij(self):
        return self.Ri + self.ui - self.Rj - self.uj
    @property
    def dijd(self):
        return self.Rdi + self.Bui*self.Pdi - self.Rdj - self.Buj*self.Pdj
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix(self._pos_level_equations)
    @property
    def vel_level_equations(self):
        return sm.BlockMatrix(self._vel_level_equations)
    @property
    def acc_level_equations(self):
        return sm.BlockMatrix(self._acc_level_equations)
    @property
    def jacobian_i(self):
        return sm.BlockMatrix(self._jacobian_i)
    @property
    def jacobian_j(self):
        return sm.BlockMatrix(self._jacobian_j)
        
    @property
    def reactions_equalities(self):
        return self._reactions_equalities
    
    @property
    def reactions_symbols(self):
        return [self.Fi,self.Ti]
    
    @property
    def arguments_symbols(self):
        return self._arguments
    @property
    def runtime_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return self._sym_constants
    @property
    def constants_numeric_expr(self):
        return []
    @property
    def constants_symbols(self):
        constants_expr = itertools.chain(self.constants_symbolic_expr,
                                         self.constants_numeric_expr)
        return [expr.lhs for expr in constants_expr]

    
    
    def _create_joint_def_axis(self,i):
        format_ = (self.prefix,i,self.id_name)
        v = vector('%sax%s_%s'%format_)
        m = reference_frame('%sM%s_%s'%format_,format_as=r'{%s{M%s}_{%s}}'%format_)
        setattr(self,'axis_%s'%i,v)
        setattr(self,'marker_%s'%i,m)
    
    def _create_joint_def_loc(self,i):
        format_ = (self.prefix,i,self.id_name)
        u = vector('%spt%s_%s'%format_)
        setattr(self,'loc_%s'%i,u)
        
    def _create_joint_arguments(self):
        l = []
        for i in range(self.def_axis):
            n = i+1
            v = getattr(self,'axis_%s'%n)
            l.append(v)
        for i in range(self.def_locs):
            n = i+1
            u = getattr(self,'loc_%s'%n)
            l.append(u)
        self._arguments = l
    
    
    def _construct(self):
        self._create_local_equalities()
        self._create_reactions_args()
        self._create_reactions_equalities()
            
    def _create_equations_lists(self):
        self._pos_level_equations = []
        self._vel_level_equations = []
        self._acc_level_equations = []
        self._jacobian_i = []
        self._jacobian_j = []
    
    def _create_local_equalities(self):
        self._sym_constants = []
        
        if self.def_axis == 1:
            axis   = self.axis_1
            marker = self.marker_1
            marker.orient_along(axis)
            mi_bar    = marker.express(self.body_i)
            mi_bar_eq = sm.Eq(self.mi_bar.A, mi_bar)
            mj_bar    = marker.express(self.body_j)
            mj_bar_eq = sm.Eq(self.mj_bar.A, mj_bar)
            markers_equalities = [mi_bar_eq,mj_bar_eq]
            
        elif self.def_axis == 2:
            axis1  = self.axis_1
            axis2  = self.axis_2
            marker1 = self.marker_1
            marker2 = self.marker_2
            
            marker1.orient_along(axis1)
            mi_bar    = marker1.express(self.body_i)
            mi_bar_eq = sm.Eq(self.mi_bar.A, mi_bar)
            
            marker2.orient_along(axis2,marker1.A[:,1])
            mj_bar    = marker2.express(self.body_j)
            mj_bar_eq = sm.Eq(self.mj_bar.A, mj_bar)
            markers_equalities = [mi_bar_eq,mj_bar_eq]
        
        elif self.def_axis == 0:
            markers_equalities = []
        else: raise NotImplementedError
        self._sym_constants += markers_equalities

        if self.def_locs == 1:
            loc  = self.loc_1
            ui_bar_eq = sm.Eq(self.ui_bar, loc.express(self.body_i) - self.Ri.express(self.body_i))
            uj_bar_eq = sm.Eq(self.uj_bar, loc.express(self.body_j) - self.Rj.express(self.body_j))
            location_equalities = [ui_bar_eq,uj_bar_eq]
        elif self.def_locs == 0:
            location_equalities = []
        elif self.def_locs == 2: 
            loc1 = self.loc_1
            loc2 = self.loc_2
            ui_bar_eq = sm.Eq(self.ui_bar, loc1.express(self.body_i) - self.Ri.express(self.body_i))
            uj_bar_eq = sm.Eq(self.uj_bar, loc2.express(self.body_j) - self.Rj.express(self.body_j))
            location_equalities = [ui_bar_eq,uj_bar_eq]

        self._sym_constants += location_equalities
        
    def _construct_actuation_functions(self):
        pass
    
    def _create_reactions_args(self):
        body_i_name = self.body_i.id_name
        
        format_ = (self.prefix,self.id_name)
        L_raw_name = '%sL_%s'%format_
        L_frm_name = r'{%s\lambda_{%s}}'%format_
        self.L = matrix_symbol(L_raw_name,self.nc,1,L_frm_name)
        
        format_ = (self.prefix,body_i_name,self.id_name)
        
        #Joint Reaction Load acting on body_i.
        Qi_raw_name = '%sQ_%s_%s'%format_
        Qi_frm_name = r'{%sQ^{%s}_{%s}}'%format_
        self.Qi = matrix_symbol(Qi_raw_name,7,1,Qi_frm_name)
        
        #Joint Reaction Force acting on body_i.
        Fi_raw_name = '%sF_%s_%s'%format_
        Fi_frm_name = r'{%sF^{%s}_{%s}}'%format_
        self.Fi = matrix_symbol(Fi_raw_name,3,1,Fi_frm_name)
        
        #Joint Reaction Torque acting on body_i in terms of orientation parameters.
        Tie_raw_name = '%sTe_%s_%s'%format_
        Tie_frm_name = r'{%sTe^{%s}_{%s}}'%format_
        self.Ti_e = matrix_symbol(Tie_raw_name,4,1,Tie_frm_name)
        
        #Joint Reaction Torque acting on body_i in terms of cartesian coordinates.
        Ti_raw_name = '%sT_%s_%s'%format_
        Ti_frm_name = r'{%sT^{%s}_{%s}}'%format_
        self.Ti = matrix_symbol(Ti_raw_name,3,1,Ti_frm_name)
        self.Ti_eq = 0.5*E(self.Pi)*self.Ti_e - Skew(self.ui)*self.Fi
        
        
    def _create_reactions_equalities(self):
        jacobian_i = self.jacobian_i
        Qi_eq = sm.Eq(self.Qi,-jacobian_i.T*self.L)
        Fi_eq = sm.Eq(self.Fi,self.Qi[0:3,0])
        Ti_e_eq = sm.Eq(self.Ti_e,self.Qi[3:7,0])
        Ti_eq = sm.Eq(self.Ti,self.Ti_eq)
        self._reactions_equalities = [Qi_eq,Fi_eq,Ti_e_eq,Ti_eq]
        
    
    @classmethod
    def represent_equations(cls):
        a = body('1')
        b = body('2')
        j = cls('c',a,b)
        
        equations = [j.pos_level_equations,
                     j.vel_level_equations,
                     j.acc_level_equations,
                     sm.BlockMatrix([[j.jacobian_i,j.jacobian_j]])]
        
        for i in equations: display(i)
    
###############################################################################
###############################################################################

class spehrical_constraint(object):
    
    nc  = 3
    
    def __init__(self):
        pass
    
    def construct(self,obj):
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
    
    def __init__(self,v1='i',v2='j'):
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
        vel_level_equation = zero_matrix(1,1)
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
    
    def __init__(self,v='i'):
        self.v = v
    
    def construct(self,obj):
        v = self.v
                
        v_bar = getattr(obj.mi_bar,v)
        v = v_bar.express()
        
        dij = obj.dij
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        dijd = obj.dijd
        
        pos_level_equation = v.T*dij
        vel_level_equation = zero_matrix(1,1)
        acc_level_equation =   v.T*( B(Pdi,obj.ui_bar)*Pdi - B(Pdj,obj.uj_bar)*Pdj ) \
                             + dij.T*B(Pdi,v_bar)*Pdi \
                             + 2*(B(obj.Pi,v_bar)*Pdi).T*dijd
        jacobian = ([ v.T, dij.T*B(obj.Pi,v_bar) + v.T*obj.Bui], 
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
        pass
    
    def construct(self,obj):
        v1 = 'i'
        v2 = 'i'
        v3 = 'j'
        
        v1_bar = getattr(obj.mi_bar,v1)
        v1     = v1_bar.express()
        v2_bar = getattr(obj.mj_bar,v2)
        v2     = v2_bar.express()
        v3_bar = getattr(obj.mi_bar,v3)
        v3     = v3_bar.express()
        
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        Z = zero_matrix(1, 3)
        
        c = sm.cos(obj.act_func('t'))
        s = sm.sin(obj.act_func('t'))
        
        pos_level_equation = (v3.T*v2)*c - (v1.T*v2)*s
        vel_level_equation = zero_matrix(1, 1)        
        acc_level_equation =   (c*v3.T - s*v1.T)*B(Pdj,v2_bar)*obj.Pdj \
                             + v2.T*(c*B(Pdi,v3_bar) - s*B(Pdi,v1_bar))*Pdi \
                             + 2*(c*B(obj.Pi,v3_bar)*Pdi - s*B(obj.Pi,v1_bar)*Pdi).T*(B(obj.Pj,v2_bar)*Pdj)
        
        jacobian = ([Z, v2.T*(c*B(obj.Pi,v3_bar) - s*B(obj.Pi,v1_bar))],
                    [Z, (c*v3.T - s*v1.T)*B(obj.Pj,v2_bar)])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])

###############################################################################
class coordinate_constraint(object):
    
    nc  = 1
    
    def __init__(self):
        pass
    
    def construct(self,obj):
        i  = obj.i
        Ri = obj.Ri
        Ai = obj.Ai
        C  = obj.loc_1
        ui_bar = obj.ui_bar
        Pdi = obj.Pdi

        pos_level_equation = (Ri + Ai*ui_bar - C)[i,:]
        vel_level_equation = zero_matrix(1,1)
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
###############################################################################

class joint_constructor(type):
    
    def __new__(mcls, name, bases, attrs):
        
        vector_equations = attrs['vector_equations']
        nve = len(vector_equations)
        nc  = sum([e.nc for e in vector_equations])
        
        def construct(self):
            self._create_equations_lists()
            self._construct_actuation_functions()
            for e in vector_equations:
                e.construct(self)
            self._construct()
                    
            
        attrs['construct'] = construct
        attrs['nve'] = nve
        attrs['nc']  = nc
        attrs['n']  = 0
        
        return super(joint_constructor, mcls).__new__(mcls, name, tuple(bases), attrs)


###############################################################################
###############################################################################

class actuator(algebraic_constraints):
    
    def __init__(self,*args):
        super().__init__(*args)
        
    def _construct_actuation_functions(self):
        self.t = t = sm.symbols('t', integer=True)
        self.act_func = sm.Function('%sAF_%s'%(self.prefix,self.id_name), integer=True)
        self._pos_function = self.act_func(t)
        self._vel_function = sm.diff(self._pos_function,t)
        self._acc_function = sm.diff(self._pos_function,t,t)
            
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._pos_level_equations[0] - sm.Identity(1)*self._pos_function])
    @property
    def vel_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._vel_level_equations[0] - sm.Identity(1)*self._vel_function])
    @property
    def acc_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._acc_level_equations[0] - sm.Identity(1)*self._acc_function])
    
    @property
    def arguments_symbols(self):
        return super().arguments_symbols + [self.act_func]

###############################################################################
###############################################################################

class joint_actuator(actuator):
    
    def __init__(self,name,joint=None):
        if joint is not None:
            body_i = joint.body_i
            body_j = joint.body_j
            super().__init__(joint.name,body_i,body_j)
            self._name = name
        else:
            super().__init__(name)
    
    def _create_reactions_equalities(self):
        self.Ti_eq = 0.5*E(self.Pi)*self.Ti_e
        jacobian_i = self.jacobian_i
        Qi_eq = sm.Eq(self.Qi,-jacobian_i.T*self.L)
        Fi_eq = sm.Eq(self.Fi,self.Qi[0:3,0])
        Ti_e_eq = sm.Eq(self.Ti_e,self.Qi[3:7,0])
        Ti_eq = sm.Eq(self.Ti,self.Ti_eq)
        self._reactions_equalities = [Qi_eq,Fi_eq,Ti_e_eq,Ti_eq]
    
###############################################################################
###############################################################################

class absolute_actuator(actuator):
    
    coordinates_map = {'x':0,'y':1,'z':2}
    
    def __init__(self,name,body_i=None,body_j=None,coordinate='z'):
        self.coordinate = coordinate
        self.i = self.coordinates_map[self.coordinate]
        super().__init__(name,body_i,body_j)
        
