# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 11:06:05 2019

@author: khale
"""

import sympy as sm
from source.symbolic_classes.abstract_matrices import (reference_frame,vector, zero_matrix, 
                                                       B, E, matrix_symbol, Skew)
from source.symbolic_classes.bodies import body
from IPython.display import display


I = sm.Identity(3)

class algebraic_constraints(object):
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self,name,body_i=None,body_j=None):
        splited_name = name.split('.')
        self._id_name = ''.join(splited_name[-1])
        self.prefix  = '.'.join(splited_name[:-1])
        self.prefix  = (self.prefix+'.' if self.prefix!='' else self.prefix)
        self._name   = name
        
                
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
    
    def _construct(self):
        self._create_equations_lists()
        self._create_bodies_locals()
            
    def _create_equations_lists(self):
        self._pos_level_equations = []
        self._vel_level_equations = []
        self._acc_level_equations = []
        self._jacobian_i = []
        self._jacobian_j = []
    
    def _create_reactions_args(self):
        body_i_name = self.body_i.id_name
        
        fromate_ = (self.prefix,self.id_name)
        L_raw_name = '%sL_%s'%fromate_
        L_frm_name = r'{%s\lambda_{%s}}'%fromate_
        self.L = matrix_symbol(L_raw_name,self.nc,1,L_frm_name)
        
        fromate_ = (self.prefix,body_i_name,self.id_name)
        
        #Joint Reaction Load acting on body_i.
        RLi_raw_name = '%sJL_%s_%s'%fromate_
        RLi_frm_name = r'{%sL^{%s}_{%s}}'%fromate_
        self.RLi = matrix_symbol(RLi_raw_name,7,1,RLi_frm_name)
        self.RLi = -self.jacobian_i.T*self.L
        
        #Joint Reaction Force acting on body_i.
        RFi_raw_name = '%sJF_%s_%s'%fromate_
        RFi_frm_name = r'{%sF^{%s}_{%s}}'%fromate_
        self.RFi = matrix_symbol(RFi_raw_name,3,self.nc,RFi_frm_name)
        
        #Joint Reaction Torque acting on body_i in terms of orientation parameters.
        RTie_raw_name = '%sJTe_%s_%s'%fromate_
        RTie_frm_name = r'{%sTe^{%s}_{%s}}'%fromate_
        self.RTi_e = matrix_symbol(RTie_raw_name,4,self.nc,RTie_frm_name)
        
        #Joint Reaction Torque acting on body_i in terms of cartesian coordinates.
        RTic_raw_name = '%sJTc_%s_%s'%fromate_
        RTic_frm_name = r'{%sTc^{%s}_{%s}}'%fromate_
        self.RTi_c = matrix_symbol(RTic_raw_name,self.nc,1,RTic_frm_name)
        
        self.RTi_c_eq = 0.5*E(self.Pi)*self.RTi_e - Skew(self.ui)*self.RFi
        

    
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

    def _create_bodies_locals(self):
        self._constants = []
        
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
        self._constants += markers_equalities

        if self.def_locs == 1:
            loc  = self.loc_1
            ui_bar_eq = sm.Eq(self.ui_bar, loc.express(self.body_i) - self.Ri.express(self.body_i))
            uj_bar_eq = sm.Eq(self.uj_bar, loc.express(self.body_j) - self.Rj.express(self.body_j))
            location_equalities = [ui_bar_eq,uj_bar_eq]
        elif self.def_locs == 0:
            location_equalities = []
        else: raise NotImplementedError
        self._constants += location_equalities
    
    
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
    
    @property
    def arguments(self):
        return self._arguments
    @property
    def constants(self):
        return self._constants
    
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
                
        v_bar = getattr(obj.mi_bar,v)
        v = v_bar.express()
        dij = obj.dij
        Rdi = obj.Rdi
        Rdj = obj.Rdj
        Pdi = obj.Pdi
        Pdj = obj.Pdj
        dijd = Rdi + obj.Bui*Pdi - Rdj + obj.Buj*Pdj
        
        pos_level_equation = v.T*dij
        vel_level_equation = zero_matrix(1,1)
        acc_level_equation =   v.T*( B(Pdi,obj.ui_bar)*Pdi - B(Pdj,obj.uj_bar)*Pdj ) \
                             + dij.T*B(Pdi,v_bar)*Pdi \
                             + 2*(B(obj.Pi,v_bar)*Pdi).T*dijd
        jacobian = ([v.T, dij.T*B(obj.Pi,v_bar) + v.T*obj.Bui], [ -v.T, -v.T*obj.Buj])
        
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
        
#        bases = list(bases) + [abstract_mbs,]
        return super(joint_constructor, mcls).__new__(mcls, name, tuple(bases), attrs)


###############################################################################
###############################################################################

class actuator(algebraic_constraints):
    
    def __init__(self,*args):
        super().__init__(*args)
        
    def _construct_actuation_functions(self):
        self.t = t = sm.symbols('t')
        self.F = sm.Function('%sF_%s'%(self.prefix,self.id_name))
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
    
    @property
    def arguments(self):
        return super().arguments + [self.F]
        

class joint_actuator(actuator):
    
    def __init__(self,name,joint=None):
        if joint is not None:
            body_i = joint.body_i
            body_j = joint.body_j
            super().__init__(joint.name,body_i,body_j)
            self._name = name
        else:
            super().__init__(name)
            self._construct_actuation_functions()


class absolute_actuator(actuator):
    
    coordinates_map = {'x':0,'y':1,'z':2}
    
    def __init__(self,name,body_i=None,body_j=None,coordinate='z'):
        self.coordinate = coordinate
        self.i = self.coordinates_map[self.coordinate]
        super().__init__(name,body_i,body_j)
        self._construct_actuation_functions()
#        if body_i is not None:
#            self.body = body_i
#    
#    @property
#    def body(self):
#        return self._body_j
#    
#    @body.setter
#    def body(self,value):
#        self.body_j = value
#        self.construct()
    
    def _construct(self):
        self._create_equations_lists()
    
    @property
    def arguments(self):
        sym_jac = sm.MatrixSymbol('%sJ_%s'%(self.prefix,self.id_name),1,3)
        return super().arguments + [sym_jac]

    @property
    def constants(self):
        return []

    
    
