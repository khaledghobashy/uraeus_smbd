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
        self._name = name
        self._create_equations_lists()
                    
        try:
            self.body_i = body_i
            self.body_j = body_j
        except AttributeError:
            pass        
    
    def _create_equations_lists(self):
        self._pos_level_equations = []
        self._vel_level_equations = []
        self._acc_level_equations = []
        self._jacobian_i = []
        self._jacobian_j = []
    
    @property
    def name(self):
        return self._name

    @property
    def id_name(self):
        return self._name.id_name

    def construct(self):
        self._create_equations_lists()
            
    
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
        
        prefix, id_, name = self.name
        fromate_ = (prefix,body_i.id_name,self.id_name)
        local_id = (body_i.id_name,self.id_name)
        
        vector_name = mbs_string('ubar_%s_%s'%local_id,prefix[:-1])
        self.ui_bar = vector(vector_name,frame=body_i,format_as=r'{%s\bar{u}^{%s}_{%s}}'%fromate_)
        
        marker_name = mbs_string('Mbar_%s_%s'%local_id,prefix[:-1])
        self.mi_bar = reference_frame(marker_name,parent=body_i,format_as=r'{%s\bar{M}^{%s}_{%s}}'%fromate_)
        
        self.Bui = B(self.Pi,self.ui_bar)
        self.ui = self.ui_bar.express()
#        try:
#            self.construct()
#        except AttributeError:
#            pass
    
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
        
        prefix, id_, name = self.name
        fromate_ = (prefix,body_j.id_name,self.id_name)
        local_id = (body_j.id_name,self.id_name)
        
        vector_name = mbs_string('ubar_%s_%s'%local_id,prefix[:-1])
        self.uj_bar = vector(vector_name,frame=body_j,format_as=r'{%s\bar{u}^{%s}_{%s}}'%fromate_)
        
        marker_name = mbs_string('Mbar_%s_%s'%local_id,prefix[:-1])
        self.mj_bar = reference_frame(marker_name,parent=body_j,format_as=r'{%s\bar{M}^{%s}_{%s}}'%fromate_)
        
        self.Buj = B(self.Pj,self.uj_bar)
        self.uj = self.uj_bar.express()
        self.construct()
#        try:
#            self.construct()
#        except AttributeError:
#            pass
    
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
        
        for i in equations:
            display(i)

        
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
        self._construct_actuation_functions()
    
    def _construct_actuation_functions(self):
        self.t = t = sm.symbols('t')
        self.F = sm.Function('F_%s'%self.name)
        self._pos_function = self.F(t)
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
        self.construct()

class absolute_actuator(actuator):
    
    coordinates_map = {'x':0,'y':1,'z':2}
    
    def __init__(self,name,body_i,coordinate):
        self.coordinate = coordinate
        self.i = self.coordinates_map[self.coordinate]
        super().__init__(name,body_i)
    
    def numerical_arguments(self):
        sym_jac = sm.MatrixSymbol('J_%s'%self.name,1,3)
        num_jac = sm.Matrix([[0,0,0]]) 
        num_jac[0,self.i] = 1
        eq = sm.Eq(sym_jac,num_jac)
        return super().numerical_arguments() + [eq]


    
    
