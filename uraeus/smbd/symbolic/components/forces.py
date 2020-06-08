
# Standard library imports
import itertools

# 3rd parties libraries imports
import sympy as sm

# Local application imports
from .helpers import body_setter, name_setter
from .matrices import (A, vector, G, E, Skew, zero_matrix,
                       matrix_function_constructor, Force, Triad, 
                       reference_frame, matrix_symbol)


class abstract_force(object):
    r"""
    **Abstract Class**
    
    A class that acts as a base class for forces equations. The
    class is used to construct spatial action-only and action-reaction force 
    elements acting on single bodies and body pairs.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
        
    Attributes
    ----------
    n : int
        Class member. Number of generalized coordinates used to define the
        force configuration. Equals zero.
    nc : int
        Class member. Number of scalar constraint equations.Equals zero.
    nve : int
        Class member. Number of vetor constraint equations.Equals zero.
        
    def_axis : int
        Class member. Number of axes used to define the given force element.
    def_locs : int
        Class member. Number of location points used to define the given 
        force element.
        
    body_i : body
        The 1st body isntance.
    body_j : body
        The 2nd body isntance.
        
    reactions_equalities : list (of sympy.Equality)
        A list containg the reactions' equalities acting on body_i. These are
        sympy equalities containing lhs vactor symbols and rhs matrix 
        expressions. These are: 
            - Reaction Force Equality (Fi).
            - Reaction Torque Euality (Ti) in terms of cartesian coordinates.
    
    reactions_symbols : list (of sympy.MatrixExpr)
        A list contating the reaction force vector Fi and the reaction torque 
        vector Ti acting on body_i.
        
    arguments_symbols : list (of sympy.MatrixSymbol)
        A list containing the symbolic mathematical objects -location points 
        and orientation axes- that should be numerically defined by the user in
        a numerical simulation session.
        The number of arguments are given by the sum of `def_axis` and 
        `def_locs`.
        
    runtime_symbols : list (of sympy.MatrixSymbol)
        A list containing the symbolic mathematical objects that changes during
        the run-time of a nuemric simulation's "solve" method. Here this is 
        mostly an empty list.
    
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
        constants for the given joint/actuator instance.
        
    dij : sympy.MatrixExpr
        A symbolic matrix expression representing the relative position vector
        between the joint location point on body_i and the location point on
        body_j. 
        $$ R_i + A(P_i) \bar{u}_i - R_j - A(P_j) \bar{u}_j $$
    
    dijd : sympy.MatrixExpr
        A symbolic matrix expression representing the relative velocity vector
        between the joint location point on body_i and the location point on
        body_j.
        $$ d(R_i + A(P_i) \bar{u}_i - R_j - A(P_j) \bar{u}_j) / dt $$
    
    """
    n   = 0
    nc  = 0
    nve = 0
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self, name, body_i=None, body_j=None):
        name_setter(self, name)
        if body_i : self.body_i = body_i 
        if body_j : self.body_j = body_j
        
        self._create_arguments()
        self._create_local_equalities()

        self._reactions_equalities = []
        self._reactions_symbols = []
        
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
    def body_i(self, body_i):
        body_setter(self, body_i, 'i')
        self.Gi  = body_i.G
        self._construct_force_i()
    
    @property
    def body_j(self):
        return self._body_j
    @body_j.setter
    def body_j(self, body_j):
        body_setter(self, body_j, 'j')
        self.Gj  = body_j.G
        self._construct_force_j()
        
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [self.Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[self.Fj], [self.Tj_e]])
    
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
    
    @property
    def reactions_equalities(self):
        """
        A list containg the reactions' equalities acting on body_i. These are
        sympy equalities containing lhs vactor symbols and rhs matrix 
        expressions.
        """
        return self._reactions_equalities
    
    @property
    def reactions_symbols(self):
        """
        A list contating the reaction force vector Fi and the reaction torque 
        vector Ti acting on body_i.
        """
        return self._reactions_symbols

    
    
    def _create_def_axis(self, i):
        format_ = (self.prefix, i, self.id_name)
        v = vector('%sax%s_%s'%format_)
        setattr(self, 'axis_%s'%i, v)
    
    def _create_def_loc(self, i):
        format_ = (self.prefix, i, self.id_name)
        u = vector('%spt%s_%s'%format_)
        setattr(self, 'loc_%s'%i, u)
        
    def _create_arguments(self):
        
        for i in range(self.def_axis):
            self._create_def_axis(i+1)
        for i in range(self.def_locs):
            self._create_def_loc(i+1)
            
        l = []
        for i in range(self.def_axis):
            n = i+1
            v = getattr(self, 'axis_%s'%n)
            l.append(v)
        for i in range(self.def_locs):
            n = i+1
            u = getattr(self, 'loc_%s'%n)
            l.append(u)
        self._arguments = l


    def _create_local_equalities(self):
        self._sym_constants = []
        
        if self.def_axis == 0:
            axis_equalities = []
        
        elif self.def_axis == 1:
            axis   = self.axis_1
            format_ = (self.prefix, 1, self.id_name)
            marker = reference_frame('%sM%s_%s'%format_, format_as=r'{%s{M%s}_{%s}}'%format_)

            axis_bar  = axis.express(self.body_i)
            axis_bar_eq = sm.Eq(self.vi_bar, axis_bar/sm.sqrt(axis_bar.T*axis_bar))
            
            # Creating a global marker/triad oriented along the definition 
            # axis, where Z-axis of the triad is parallel to the axis.
            marker.orient_along(axis)
            
            # Expressing the created marker/triad in terms of the 1st body 
            # local reference frame resulting in matrix transformation 
            # expression
            mi_bar    = marker.express(self.body_i)
            # Creating a symbolic equality that equates the symbolic dcm of the
            # marker to the matrix transformation expression created.
            mi_bar_eq = sm.Eq(self.mi_bar.A, mi_bar)
            
            # Expressing the created marker/triad in terms of the 2nd body 
            # local reference frame resulting in matrix transformation 
            # expression
            mj_bar    = marker.express(self.body_j)
            # Creating a symbolic equality that equates the symbolic dcm of the
            # marker to the matrix transformation expression created.
            mj_bar_eq = sm.Eq(self.mj_bar.A, mj_bar)
            
            # Storing the equalities in the markers list.
            axis_equalities = [axis_bar_eq, mi_bar_eq, mj_bar_eq]
        else: 
            raise NotImplementedError
        self._sym_constants += axis_equalities
        
        if self.def_locs == 0:
            location_equalities = []

        elif self.def_locs == 1:
            loc  = self.loc_1
            ui_bar_eq = sm.Eq(self.ui_bar, loc.express(self.body_i) - self.Ri.express(self.body_i))
            uj_bar_eq = sm.Eq(self.uj_bar, loc.express(self.body_j) - self.Rj.express(self.body_j))
            location_equalities = [ui_bar_eq, uj_bar_eq]
        
        elif self.def_locs == 2: 
            loc1 = self.loc_1
            loc2 = self.loc_2
            
            # Relative position vector of 1st joint location relative to the 1st 
            # body reference point, in the body-local reference frame
            ui_bar = loc1.express(self.body_i) - self.Ri.express(self.body_i)
            # Creating a symbolic equality that equates the symbolic vector of
            # the local position to the matrix transformation expression created.
            ui_bar_eq = sm.Eq(self.ui_bar, ui_bar)

            # Relative position vector of 2nd joint location relative to the 2nd 
            # body reference point, in the body-local reference frame
            uj_bar = loc2.express(self.body_j) - self.Rj.express(self.body_j)
            # Creating a symbolic equality that equates the symbolic vector of
            # the local position to the matrix transformation expression created.
            uj_bar_eq = sm.Eq(self.uj_bar, uj_bar)
            
            # Storing the equalities in the locations list.
            location_equalities = [ui_bar_eq, uj_bar_eq]

        else: 
            raise NotImplementedError
        self._sym_constants += location_equalities

        
    def _construct_force_i(self):
        bname = self.body_i.id_name
        F_format = (self.prefix, 'F', bname, self.id_name)
        T_format = (self.prefix, 'T', bname, self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        T_raw_name, T_frm_name = self._formatter(*T_format)
        self.Fi   = vector(F_raw_name, format_as=F_frm_name)
        self.Ti   = vector(T_raw_name, format_as=T_frm_name)
        self.Ti_e = 2*E(self.Pi).T * (self.Ti + Skew(self.ui)*self.Fi)
    
    def _construct_force_j(self):
        bname = self.body_j.id_name
        F_format = (self.prefix, 'F', bname, self.id_name)
        T_format = (self.prefix, 'T', bname, self.id_name)
        F_raw_name, F_frm_name = self._formatter(*F_format)
        T_raw_name, T_frm_name = self._formatter(*T_format)
        self.Fj   = vector(F_raw_name, format_as=F_frm_name)
        self.Tj   = vector(T_raw_name, format_as=T_frm_name)
        self.Tj_e = 2*E(self.Pj).T * (self.Tj + Skew(self.uj)*self.Fj)
    
    @staticmethod
    def _formatter(*args):
        raw_name = '%s%s_%s_%s'%(*args,)
        frm_name = r'{%s{%s}^{%s}_{%s}}'%(*args,)
        return (raw_name, frm_name)

###############################################################################
###############################################################################

class gravity_force(abstract_force):
    
    def_axis = 0
    def_locs = 0
    
    def __init__(self, name, body, *args):
        name = 'gravity'
        super().__init__(name, body, *args)
    
    @property
    def Qi(self):
        return sm.BlockMatrix([[self.Fi], [zero_matrix(4, 1)]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return []
    @property
    def constants_numeric_expr(self):
        gravity = sm.Eq(self.Fi, self.body_i.m*sm.Matrix([0, 0, -9.81e3]))
        return [gravity]

###############################################################################
###############################################################################

class centrifugal_force(abstract_force):
    
    def_axis = 0
    def_locs = 0
    
    def __init__(self, name, body ,*args):
        name = 'centrifugal'
        super().__init__(name, body, *args)
    
    @property
    def Qi(self):
        Ti_e = 8*G(self.Pdi).T*self.body_i.Jbar*G(self.Pdi)*self.Pi
        return sm.BlockMatrix([[zero_matrix(3,1)], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3,1)], [zero_matrix(4,1)]])
    
    @property
    def arguments_symbols(self):
        return []
    @property
    def constants_symbolic_expr(self):
        return []

###############################################################################
###############################################################################

class generic_load(abstract_force):
    
    def_axis = 0
    def_locs = 1
    
    def __init__(self, name, body, *args):
        super().__init__(name, body, *args)

        self.t  = sm.MatrixSymbol('t', 3, 1)
        
        self.Fi = matrix_function_constructor('UF_%s_F'%name, (3, 1))
        self.Ti = matrix_function_constructor('UF_%s_T'%name, (3, 1))
        
        self._Fi_alias = sm.Function('UF_%s_F'%name, is_Vector=True)
        self._Ti_alias = sm.Function('UF_%s_T'%name, is_Vector=True)

        self._construct_force_vector()
        
    @property
    def Qi(self):
        return self._Qi
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        forces_args = [self._Fi_alias, self._Ti_alias, self.loc_1]
        return forces_args
    
    def _construct_force_vector(self):
        # Ti_e = 2E(Pi).T * (M + (ui x Fi))
        Fi = self.Fi(self.t)
        Ti = self.Ti(self.t)
        Ti_e = 2*E(self.Pi).T * (Ti + Skew(self.ui)*Fi)
        self._Qi = sm.BlockMatrix([[Fi], [Ti_e]])
        
        Fj = -Fi
        Tj = -Ti
        Tj_e = 2*E(self.Pj).T * (Ti + Skew(self.uj)*Fj)
        self._Qj = sm.BlockMatrix([[Fj], [Tj_e]])

###############################################################################
###############################################################################

class local_force(abstract_force):
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self, name, body, *args):
        super().__init__(name, body, *args)
        self.t  = sm.symbols('t', real=True)
        self.Fi = sm.Function('UF_%s'%name)
    
    @property
    def Qi(self):
        force = A(self.Pi) * (self.Fi(self.t) * self.vi_bar)
        Ti_e = 2*E(self.Pi).T * Skew(self.ui)*force
        return sm.BlockMatrix([[force], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        forces_args = [self.Fi, self.axis_1, self.loc_1]
        return forces_args
    @property
    def constants_numeric_expr(self):
        eq1 = sm.Eq(self.Ti, zero_matrix(3, 1), evaluate=False)
        return [eq1]

###############################################################################
###############################################################################

class local_torque(abstract_force):
    
    def_axis = 1
    def_locs = 0
    
    def __init__(self, name, body, *args):
        super().__init__(name, body, *args)
        self.t  = sm.symbols('t', real=True)
        self.Ti = sm.Function('UF_%s'%name)

    @property
    def Qi(self):
        local_torque = self.Ti(self.t) * self.vi_bar
        Ti_e = 2*G(self.Pi).T * local_torque 
        return sm.BlockMatrix([[zero_matrix(3, 1)], [Ti_e]])
    @property
    def Qj(self):
        return sm.BlockMatrix([[zero_matrix(3, 1)], [zero_matrix(4, 1)]])
    
    @property
    def arguments_symbols(self):
        forces_args = [self.Ti, self.axis_1]
        return forces_args
    @property
    def constants_numeric_expr(self):
        eq1 = sm.Eq(self.Fi, zero_matrix(3, 1), evaluate=False)
        return [eq1]

###############################################################################
###############################################################################

class TSDA(abstract_force):
    
    def_axis = 0
    def_locs = 2
    
    def __init__(self, name, body_i=None, body_j=None):
        super().__init__(name, body_i, body_j)
        format_ = (self.prefix, self.id_name)
        self.LF = sm.symbols('%s%s_FL'%format_, real=True)

        self.Fs = sm.Function('UF_%s_Fs'%name)
        self.Fd = sm.Function('UF_%s_Fd'%name, real=True)
        self.Fa = sm.Function('UF_%s_Fa'%name)
                
        self._construct_force_vector()
        self._construct_reactions()
        
    @property
    def Qi(self):
        return self._Qi
    
    @property
    def Qj(self):
        return self._Qj
    
    @property
    def arguments_symbols(self):
        configuration_args = [self.loc_1, self.loc_2]
        forces_args = [self.Fs, self.Fd, self.LF]
        return configuration_args + forces_args

    @property
    def constants_numeric_expr(self):
        eq1 = sm.Eq(self.Ti, zero_matrix(3, 1), evaluate=False)
        eq2 = sm.Eq(self.Tj, zero_matrix(3, 1), evaluate=False)
        return [eq1, eq2]
    
    def _construct_force_vector(self):
        
        dij  = (self.Ri + self.ui - self.Rj - self.uj)
        dijd = (self.Rdi + self.Bui*self.Pdi - self.Rdj - self.Buj*self.Pdj)

        # Fs = K(l - l0) + C*ld + Fa
        l  = sm.sqrt(dij.T*dij)[0,0]
        l0 = self.LF
        unit_vector = dij/l
        ld = -((unit_vector).T * dijd)

        defflection = l0 - l

        Fs = -self.Fs(defflection) - self.Fd(ld)

        self.Fi = -Fs * unit_vector
        Ti_e = Fs * 2*E(self.Pi).T * Skew(self.ui).T * unit_vector        
        
        self.Fj = -self.Fi
        Tj_e = -Fs * 2*E(self.Pj).T * Skew(self.uj).T * unit_vector

        self._Qi = sm.BlockMatrix([[self.Fi], [Ti_e]])
        self._Qj = sm.BlockMatrix([[self.Fj], [Tj_e]])
    
    def _construct_reactions(self):
        body_i_name = self.body_i.id_name
        format_ = (self.prefix, body_i_name, self.id_name)
        Fi_raw_name = '%sF_%s_%s'%format_
        Fi_frm_name = r'{%sF^{%s}_{%s}}'%format_
        Ti_raw_name = '%sT_%s_%s'%format_
        Ti_frm_name = r'{%sT^{%s}_{%s}}'%format_

        Fi = matrix_symbol(Fi_raw_name, 3, 1, Fi_frm_name)
        Ti = matrix_symbol(Ti_raw_name, 3, 1, Ti_frm_name)
        self._reactions_symbols = [Fi, Ti]
        self._reactions_equalities = [sm.Eq(Fi, self.Fi), 
                                      sm.Eq(Ti, zero_matrix(3,1))]


###############################################################################
###############################################################################

class generic_bushing(abstract_force):
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self, name, body_i=None, body_j=None):
        super().__init__(name, body_i, body_j)

        self.t  = sm.MatrixSymbol('t', 3, 1)
        
        # Local Bush stiffness-loading functions
        self.Fs = matrix_function_constructor('UF_%s_Fs'%name, (3, 1))
        self.Ts = matrix_function_constructor('UF_%s_Ts'%name, (3, 1))
        
        # Local Bush damping-loading functions
        self.Fd = matrix_function_constructor('UF_%s_Fd'%name, (3, 1))
        self.Td = matrix_function_constructor('UF_%s_Td'%name, (3, 1))
        
        self._Fs_alias = sm.Function('UF_%s_Fs'%name, is_Vector=True)
        self._Ts_alias = sm.Function('UF_%s_Ts'%name, is_Vector=True)

        self._Fd_alias = sm.Function('UF_%s_Fd'%name, is_Vector=True)
        self._Td_alias = sm.Function('UF_%s_Td'%name, is_Vector=True)

        self._construct_force_vector()
        self._construct_reactions()
        
    @property
    def Qi(self):
        return self._Qi
    
    @property
    def Qj(self):
        return self._Qj

    @property
    def arguments_symbols(self):
        configuration_args = [self.axis_1, self.loc_1]
        forces_args = [self._Fs_alias, self._Ts_alias, self._Fd_alias, self._Td_alias]
        return configuration_args + forces_args
    
    def _construct_force_vector(self):
        
        dij  = (self.Ri + self.ui - self.Rj - self.uj)
        dijd = (self.Rdi + self.Bui*self.Pdi - self.Rdj - self.Buj*self.Pdj)

        bush_trasformation = self.mi_bar.A.T * self.Ai.T

        F_bush_i = self.Fs(bush_trasformation, dij) \
                 + self.Fd(bush_trasformation, dijd)

        self.Fi = self.Ai * self.mi_bar.A * -F_bush_i
        Ti_e = - 2*E(self.Pi).T * Skew(self.ui).T * self.Fi
        #Ti_e = -(self.Ai * Skew(self.ui_bar) * 2*G(self.Pi)).T * self.Fi
        self._Qi = sm.BlockMatrix([[self.Fi], [Ti_e]])
        
        self.Fj = -self.Fi
        Tj_e = - 2*E(self.Pj).T * Skew(self.uj).T * self.Fj
        #Tj_e = -(self.Aj * Skew(self.uj_bar) * 2*G(self.Pj)).T * self.Fj
        self._Qj = sm.BlockMatrix([[self.Fj], [Tj_e]])
    
    def _construct_reactions(self):
        body_i_name = self.body_i.id_name
        format_ = (self.prefix, body_i_name, self.id_name)
        Fi_raw_name = '%sF_%s_%s'%format_
        Fi_frm_name = r'{%sF^{%s}_{%s}}'%format_
        Ti_raw_name = '%sT_%s_%s'%format_
        Ti_frm_name = r'{%sT^{%s}_{%s}}'%format_

        Fi = matrix_symbol(Fi_raw_name, 3, 1, Fi_frm_name)
        Ti = matrix_symbol(Ti_raw_name, 3, 1, Ti_frm_name)
        self._reactions_symbols = [Fi, Ti]
        self._reactions_equalities = [sm.Eq(Fi, self.Fi), sm.Eq(Ti, zero_matrix(3,1))]

###############################################################################
###############################################################################

class isotropic_bushing(abstract_force):
    
    def_axis = 1
    def_locs = 1
    
    def __init__(self, name, body_i=None, body_j=None):
        super().__init__(name, body_i, body_j)
        
        self.Kt = sm.symbols('Kt_%s'%self.id_name, real=True)
        self.Ct = sm.symbols('Ct_%s'%self.id_name, real=True)
        
        self.Kr = sm.symbols('Kr_%s'%self.id_name, real=True)
        self.Cr = sm.symbols('Cr_%s'%self.id_name, real=True)

        self._construct_force_vector()
        self._construct_reactions()
            
    @property
    def Qi(self):
        return self._Qi
    
    @property
    def Qj(self):
        return self._Qj

    @property
    def arguments_symbols(self):
        configuration_args = [self.axis_1, self.loc_1]
        forces_args = [self.Kt, self.Ct, self.Kr, self.Cr]
        return configuration_args + forces_args
    
    def _construct_force_vector(self):
        
        dij  = (self.Ri + self.ui - self.Rj - self.uj)
        dijd = (self.Rdi + self.Bui*self.Pdi - self.Rdj - self.Buj*self.Pdj)

        dij_bush_i  = self.mi_bar.A.T * self.Ai.T * dij
        dijd_bush_i = self.mi_bar.A.T * self.Ai.T * dijd
        F_bush_i = (self.Kt*sm.Identity(3) * dij_bush_i) \
                 + (self.Ct*sm.Identity(3) * dijd_bush_i)

        self.Fi = self.Ai * self.mi_bar.A * -F_bush_i
        Ti_e = - 2*E(self.Pi).T * Skew(self.ui).T * self.Fi
        self._Qi = sm.BlockMatrix([[self.Fi], [Ti_e]])
        
        dij_bush_j  = self.mj_bar.A.T * self.Aj.T * dij
        dijd_bush_j = self.mj_bar.A.T * self.Aj.T * dijd
        F_bush_j = (self.Kt*sm.Identity(3) * dij_bush_j) \
                 + (self.Ct*sm.Identity(3) * dijd_bush_j)

        self.Fj = self.Aj * self.mj_bar.A * F_bush_j
        Tj_e = - 2*E(self.Pj).T * Skew(self.uj).T * self.Fj
        self._Qj = sm.BlockMatrix([[self.Fj], [Tj_e]])
    
    def _construct_reactions(self):
        body_i_name = self.body_i.id_name
        format_ = (self.prefix, body_i_name, self.id_name)
        Fi_raw_name = '%sF_%s_%s'%format_
        Fi_frm_name = r'{%sF^{%s}_{%s}}'%format_
        Ti_raw_name = '%sT_%s_%s'%format_
        Ti_frm_name = r'{%sT^{%s}_{%s}}'%format_

        Fi = matrix_symbol(Fi_raw_name, 3, 1, Fi_frm_name)
        Ti = matrix_symbol(Ti_raw_name, 3, 1, Ti_frm_name)
        self._reactions_symbols = [Fi, Ti]
        self._reactions_equalities = [sm.Eq(Fi, self.Fi), 
                                      sm.Eq(Ti, zero_matrix(3,1))]

###############################################################################
###############################################################################

        