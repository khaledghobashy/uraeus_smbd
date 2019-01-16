# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:06:05 2019

@author: khale
"""

import sympy as sm
from source.symbolic_classes.abstract_matrices import (reference_frame, abstract_mbs, 
                               vector, zero_matrix, B, mbs_string)
from source.symbolic_classes.bodies import body
from IPython.display import display


I = sm.Identity(3)

class algebraic_constraints(object):
    
    def __init__(self,name,body_i=None,body_j=None):
        splited_name = name.split('.')
        self.id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self._name   = name
        
        if body_i and body_j:
            self.body_i = body_i
            self.body_j = body_j
            self.construct()

    @property
    def name(self):
        return self._name
    
    def _construct(self):
        self._create_equations_lists()
            
    def _create_equations_lists(self):
        self._pos_level_equations = []
        self._vel_level_equations = []
        self._acc_level_equations = []
        self._jacobian_i = []
        self._jacobian_j = []

    
    @property
    def body_i(self):
        return self._body_i
    @body_i.setter
    def body_i(self,body_i):
        self._body_i = body_i
        self.Ri  = body_i.R
        self.Rdi = body_i.Rd
        self.Pi  = body_i.P
        self.Pdi = body_i.Pd
        self.Ai  = body_i.A
        
        fromate_ = (self.prefix,body_i.id_name,self.id_name)
        v_raw_name = '%subar_%s_%s'%fromate_
        v_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromate_
        m_raw_name = '%sMbar_%s_%s'%fromate_
        m_frm_name = r'{%s\bar{M}^{%s}_{%s}}'%fromate_

        self.ui_bar = vector(v_raw_name,body_i,v_frm_name)        
        self.mi_bar = reference_frame(m_raw_name,body_i,m_frm_name)
        self.Bui = B(self.Pi,self.ui_bar)
        self.ui = self.ui_bar.express()
    
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self,body_j):
        self._body_j = body_j
        self.Rj  = body_j.R
        self.Rdj = body_j.Rd
        self.Pj  = body_j.P
        self.Pdj = body_j.Pd
        self.Aj  = body_j.A
        
        fromate_ = (self.prefix,body_j.id_name,self.id_name)
        v_raw_name = '%subar_%s_%s'%fromate_
        v_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromate_
        m_raw_name = '%sMbar_%s_%s'%fromate_
        m_frm_name = r'{%s\bar{M}^{%s}_{%s}}'%fromate_

        self.uj_bar = vector(v_raw_name,body_j,v_frm_name)        
        self.mj_bar = reference_frame(m_raw_name,body_j,m_frm_name)
        
        self.Buj = B(self.Pj,self.uj_bar)
        self.uj = self.uj_bar.express()
    
    @property
    def dij(self):
        return self.Ri + self.ui - self.Rj - self.uj
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
    

class dot_product_2(object):
    
    nc  = 1
    
    def __init__(self,v='i'):
        self.v = v
    
    def construct(self,obj):
        v = self.v
        
        k_bar = getattr(obj.mi_bar,'k')
        k = k_bar.express()
        
        v_bar = getattr(obj.mi_bar,v)
        v = v_bar.express()
        dij = obj.dij
        Rdi = obj.Rdi
        Rdj = obj.Rdj
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        dijd = Rdi + obj.Bui*Pdi - Rdj + obj.Buj*Pdj
        
        pos_level_equation = v.T*(dij+k)
        vel_level_equation = zero_matrix(1,1)
        acc_level_equation =   v.T*( B(Pdi,obj.ui_bar)*Pdi - B(Pdj,obj.uj_bar)*Pdj ) \
                             + (dij+k).T*B(Pdi,v_bar)*Pdi \
                             + 2*(B(obj.Pi,v_bar)*Pdi).T*dijd
        jacobian = ([v.T, (dij+k).T*B(obj.Pi,v_bar) + v.T*obj.Bui], [ -v.T, -v.T*obj.Buj])
        
        obj._pos_level_equations.append(pos_level_equation)
        obj._vel_level_equations.append(vel_level_equation)
        obj._acc_level_equations.append(acc_level_equation)
        
        obj._jacobian_i.append(jacobian[0])
        obj._jacobian_j.append(jacobian[1])
        

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
        Z = zero_matrix(1,3)
        
        c = sm.cos(obj.F('t'))
        s = sm.sin(obj.F('t'))
        
        pos_level_equation = (v3.T*v2)*c - (v1.T*v2)*s
        vel_level_equation = zero_matrix(1,1)        
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


class coordinate_constraint(object):
    
    nc  = 1
    
    def __init__(self):
        pass
    
    def construct(self,obj):
        pos_level_equation = obj.Ri[obj.i,0]
        vel_level_equation = zero_matrix(1,1)
        acc_level_equation = zero_matrix(1,1)
        
        jac = sm.MatrixSymbol('J_%s'%obj.name,1,3)
            
        jacobian = ([jac,zero_matrix(1,4)], 
                    [zero_matrix(1,3),zero_matrix(1,4)])
       
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
            self._construct()
            for e in vector_equations:
                e.construct(self)
            
        attrs['construct'] = construct
        attrs['nve'] = nve
        attrs['nc']  = nc
        attrs['n']  = 0
        
        bases = list(bases) + [abstract_mbs,]
        
        return super(joint_constructor, mcls).__new__(mcls, name, tuple(bases), attrs)


###############################################################################
###############################################################################

class actuator(algebraic_constraints):
    
    def __init__(self,*args):
        super().__init__(*args)
        
    def _construct_actuation_functions(self):
        self.t = t = sm.symbols('t')
        self.F = sm.Function('F_%s'%self.name)
        self._pos_function = self.F(t)
        self._vel_function = sm.diff(self._pos_function,t)
        self._acc_function = sm.diff(self._pos_function,t,t)
    
    def _construct(self):
        super()._construct()
        self._construct_actuation_functions()
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._pos_level_equations[0] - sm.Identity(1)*self._pos_function])
    @property
    def vel_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._vel_level_equations[0] - sm.Identity(1)*self._vel_function])
    @property
    def acc_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._acc_level_equations[0] - sm.Identity(1)*self._acc_function])
    
    def numerical_arguments(self):
        function  = sm.Eq(self.F,sm.Lambda(self.t,0))
        return [function]
    def configuration_constants(self):
        return []
    

class joint_actuator(actuator):
    
    def __init__(self,name,joint):
        body_i = joint.body_i
        body_j = joint.body_j
        super().__init__(joint.name,body_i,body_j)
        self._name = name


class absolute_actuator(actuator):
    
    coordinates_map = {'x':0,'y':1,'z':2}
    
    def __init__(self,name,body_i,coordinate):
        self.coordinate = coordinate
        self.i = self.coordinates_map[self.coordinate]
        super().__init__(name)
        self.body_i = body_i
        self.construct()
    
    def numerical_arguments(self):
        sym_jac = sm.MatrixSymbol('J_%s'%self.name,1,3)
        num_jac = sm.Matrix([[0,0,0]]) 
        num_jac[0,self.i] = 1
        eq = sm.Eq(sym_jac,num_jac)
        return super().numerical_arguments() + [eq]


    
    
