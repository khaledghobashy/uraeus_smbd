
# Standard library imports
import itertools

# 3rd parties library imports
import sympy as sm

# Local application imports
from .matrices import (reference_frame, vector, E, Skew,
                       matrix_symbol)
from .helpers import body_setter, name_setter


# Commonly used variables
I  = sm.Identity(3)
I1 = sm.Identity(1)


class abstract_joint(object):
    r"""
    **Abstract Class**
    
    A class that acts as a base class for algebraic constraints equations. The
    class is used to construct spatial joints connecting body pairs as well as
    actuators that acts on joints or bodies.
    
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
        joint configuration. Equals zero.
    nc : int
        Class member. Number of scalar constraint equations. This is specified 
        automatically based on the type and the quantity of base-constraints 
        used.
    nve : int
        Class member. Number of vetor constraint equations.This is specified 
        automatically based on the type and the quantity of base-constraints 
        used.
        
    def_axis : int
        Class member. Number of axes used to define the given constraint.
    def_locs : int
        Class member. Number of location points used to define the given 
        constraint.
        
    
    body_i : body
        The 1st body isntance.
    body_j : body
        The 2nd body isntance.
    
    
    pos_level_equations : sympy.BlockMatrix
        A block matrix that stores the set of vector equations that represents 
        the position constraints equations. The shape of the blocks is (nve, 1),
        where the scalar shape is (nc, 1).
        
    vel_level_equations : sympy.BlockMatrix
        A block matrix that stores the set of vector equations that
        represents the right hand-side of the velocity level of the constraints
        equations. The shape of the blocks is (nve, 1), where the scalar shape 
        is (nc, 1).
        
    acc_level_equations : sympy.BlockMatrix
        A block matrix that stores the set of vector equations that
        represents the right hand-side of the acceleration level of the 
        constraints equations. The shape of the blocks is (nve, 1), where the 
        scalar shape is (nc, 1).
        
    jacobian_i : sympy.BlockMatrix
        A block matrix that stores the jacobian of the constraint equations
        with respect to the body_i generalized coordinates. The shape of the 
        blocks is (nve, 2), where the scalar shape is (nc, 7).
    
    jacobian_j : sympy.BlockMatrix
        A block matrix that stores the jacobian of the constraint equations
        with respect to the body_j generalized coordinates. The shape of the 
        blocks is (nve, 2), where the scalar shape is (nc, 7).
        
    reactions_equalities : list (of sympy.Equality)
        A list containg the reactions' equalities acting on body_i. These are
        sympy equalities containing lhs vactor symbols and rhs matrix 
        expressions. These are: 
            - Total Reaction Load Equality (Qi).
            - Reaction Force Equality (Fi).
            - Reaction Torque Euality (Ti) in terms of cartesian coordinates.
            - Reaction Torque Euality (Ti_e) in terms of euler parameters.
    
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
    
    # default number of defintion axes
    def_axis = 1

    # default number of defintion locations
    def_locs = 1
    
    def __init__(self, name, body_i=None, body_j=None):
        
        # Setting the joint object names [_name, _id_name, prefix]
        name_setter(self, name)

        # construct joint axes and locations vectors and markers
        self._create_joint_arguments()
        
        # constructing the algabraic constraints equations between the given 
        # bodies. The construct method is provided by the `joint_constructor` 
        # meta-class
        if body_i and body_j:
            self.body_i = body_i
            self.body_j = body_j
            self.construct()

    @property
    def name(self):
        """
        Joint/Actuator full name
        """
        return self._name
    
    @property
    def id_name(self):
        """
        Joint/Actuator name without the perfixed initials
        """
        splited_name = self.name.split('.')
        _id_name = ''.join(splited_name[-1])
        return _id_name
    
    @property
    def body_i(self):
        """
        Joint/Actuator 1st body
        """
        return self._body_i
    @body_i.setter
    def body_i(self, body_i):
        body_setter(self, body_i, 'i')
            
    @property
    def body_j(self):
        """
        Joint/Actuator 2nd body
        """
        return self._body_j
    @body_j.setter
    def body_j(self, body_j):
        body_setter(self, body_j, 'j')
        
    @property
    def pos_level_equations(self):
        """
        A block matrix that stores the set of matrix/vector equations that
        represents the position constraints equations. The shape of the blocks
        is (nve, 1), where the scalar shape is (nc, 1).
        """
        return sm.BlockMatrix(self._pos_level_equations)
    
    @property
    def vel_level_equations(self):
        """
        A block matrix that stores the set of matrix/vector equations that
        represents the right hand-side of the velocity level of the constraints
        equations. The shape of the blocks is (nve, 1), where the scalar shape 
        is (nc, 1).
        """
        return sm.BlockMatrix(self._vel_level_equations)
    
    @property
    def acc_level_equations(self):
        """
        A block matrix that stores the set of matrix/vector equations that
        represents the right hand-side of the acceleration level of the 
        constraints equations. The shape of the blocks is (nve, 1), where the 
        scalar shape is (nc, 1).
        """
        return sm.BlockMatrix(self._acc_level_equations)
    
    @property
    def jacobian_i(self):
        """
        A block matrix that stores the jacobian of the constraint equations
        with respect to the body_i generalized coordinates. The shape of the 
        blocks is (nve, 2), where the scalar shape is (nc, 7).
        """
        return sm.BlockMatrix(self._jacobian_i)
    
    @property
    def jacobian_j(self):
        """
        A block matrix that stores the jacobian of the constraint equations
        with respect to the body_j generalized coordinates. The shape of the 
        blocks is (nve, 2), where the scalar shape is (nc, 7).
        """
        return sm.BlockMatrix(self._jacobian_j)
        
    @property
    def reactions_equalities(self):
        """
        A list containg the reactions' equalities acting on body_i. These are
        sympy equalities containing lhs vactor symbols and rhs matrix 
        expressions. These are: 
            - Total Reaction Load Equality (Qi).
            - Reaction Force Equality (Fi).
            - Reaction Torque Euality (Ti) in terms of cartesian coordinates.
            - Reaction Torque Euality (Ti_e) in terms of euler parameters.
        """
        return self._reactions_equalities
    
    @property
    def reactions_symbols(self):
        """
        A list contating the reaction force vector Fi and the reaction torque 
        vector Ti acting on body_i.
        """
        return [self.Fi, self.Ti]
    
    @property
    def arguments_symbols(self):
        """
        A list containing the symbolic mathematical objects -location points 
        and orientation axes- that should be nuemrically defined by the user in
        a numerical simulation session.
        The number of arguments are given by the sum of `def_axis` and 
        `def_locs`.
        """
        return self._arguments
    
    @property
    def runtime_symbols(self):
        """
        A list containing the symbolic mathematical objects that changes during
        the run-time of a nuemric simulation's "solve" method. Here this is 
        mostly an empty list.
        """
        return []
    
    @property
    def constants_symbolic_expr(self):
        """
        A list containing sympy equalities representing the values of internal
        class symbolic constants that are evaluated from other symbolic 
        expressions.
        """
        return self._sym_constants
    
    @property
    def constants_numeric_expr(self):
        """
        A list containing sympy equalities representing the values of internal
        class symbolic constants that are evaluated directly from numerical 
        expressions.
        """
        return []
    
    @property
    def constants_symbols(self):
        """
        A list containing all the symbolic mathematical objects that represent
        constants for the given joint/actuator instance.
        """
        constants_expr = itertools.chain(self.constants_symbolic_expr,
                                         self.constants_numeric_expr)
        return [expr.lhs for expr in constants_expr]

    @property
    def dij(self):
        equation = self.Ri + self.ui - self.Rj - self.uj
        return equation
    @property
    def dijd(self):
        equation = self.Rdi + self.Bui*self.Pdi - self.Rdj - self.Buj*self.Pdj
        return equation

            
    def _create_joint_arguments(self):
        """
        Private method used to create the required symbolic mathematical 
        objects that should be nuemrically defined by the user in a numerical 
        simulation session.
        """
        
        # Creating the joint definition axes
        for i in range(self.def_axis):
            self._create_joint_def_axis(i+1)
            
        # Creating the joint definition locations
        for i in range(self.def_locs):
            self._create_joint_def_loc(i+1)
        
        # Empty list to store the created symbolic vectors
        l = []
        
        # Storing the created symbolic vetors in the list
        for i in range(self.def_axis):
            n = i+1
            v = getattr(self, 'axis_%s'%n)
            l.append(v)
            
        for i in range(self.def_locs):
            n = i+1
            u = getattr(self, 'loc_%s'%n)
            l.append(u)
        
        self._arguments = l
    
    
    def _create_joint_def_axis(self, i):
        """
        Private method used to create a symbolic definition axis.
        
        Parameters
        ----------
        i : int
            Axis number ID
        
        Returns
        -------
        None
        
        Notes
        -----
        The method constructs a vector instance with the name `axis_i_name` 
        where `i` is the ID number and `name` is the joint/actuator name. 
        The method then sets two memebrs to the instance that represents the 
        `axis_i` and its corresponding marker `marker_i`.
        
        """
        format_ = (self.prefix, i, self.id_name)
        v = vector('%sax%s_%s'%format_)
        m = reference_frame('%sM%s_%s'%format_, format_as=r'{%s{M%s}_{%s}}'%format_)
        setattr(self, 'axis_%s'%i, v)
        setattr(self, 'marker_%s'%i, m)
    
    
    def _create_joint_def_loc(self, i):
        """
        Private method used to create a symbolic definition location.
        
        Parameters
        ----------
        i : int
            Location ID number.
        
        Returns
        -------
        None
        
        Notes
        -----
        The method constructs a vector instance with the name `pt_i_name` 
        where `i` is the ID number and `name` is the joint/actuator name.
        The method then sets a memebr variable to the instance that represents 
        the `loc_i`.
        
        """
        format_ = (self.prefix, i, self.id_name)
        u = vector('%spt%s_%s'%format_)
        setattr(self, 'loc_%s'%i, u)
    
    def _create_equations_lists(self):
        """
        Creats empty lists to hold the created symbolic equations.
        """
        self._pos_level_equations = []
        self._vel_level_equations = []
        self._acc_level_equations = []
        self._jacobian_i = []
        self._jacobian_j = []

    def _construct(self):
        """
        A method that calls the other methods resposible for constructing the
        symbolic objects/equalities of the joint/actuator.
        """
        self._create_local_equalities()
        self._create_reactions_args()
        self._create_reactions_equalities()
            
    def _create_local_equalities(self):
        """
        A private method to create the joint/actuator local symbolic equalities
        that represents constants to the joint/acutator.
        
        Notes
        -----
        When a joint/actuator gets created by the user, the class checks the 
        number of arguments used to fully define the instance, 
        e.g; defintion axes and locations.
        It then creates local version of these arguments that are defined in 
        the bodies local reference frame. 
        These local arguments do not change relative to the given body 
        reference point and orientation.
        TODO....
        """
        
        # Empty list to store symbolic equalities
        self._sym_constants = []
        
        # Creating Local Constant Markers based on the joint/actuator 
        # definition axes.
        # ===========================================================
        if self.def_axis == 0:
            markers_equalities = []
        
        elif self.def_axis == 1:
            axis   = self.axis_1
            marker = self.marker_1
            
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
            markers_equalities = [mi_bar_eq, mj_bar_eq]
          
        
        # The case of two definition axes for universal/hook/tripod joints.
        # Where two global triads/markers created along two different axis.
        elif self.def_axis == 2:

            axis1   = self.axis_1
            marker1 = self.marker_1

            axis2   = self.axis_2
            marker2 = self.marker_2
            
            # Orienting 1st marker along 1st axis
            marker1.orient_along(axis1)
            
            # Expressing the created marker/triad in terms of the 1st body 
            # local reference frame resulting in matrix transformation 
            # expression
            mi_bar    = marker1.express(self.body_i)
            
            # Creating a symbolic equality that equates the symbolic dcm of the
            # marker to the matrix transformation expression created.
            mi_bar_eq = sm.Eq(self.mi_bar.A, mi_bar)
            
            # Orienting the 2nd marker along the 2nd axis, where the 2nd marker
            # x-axis is parallel to the 1st marker's y-axis.
            marker2.orient_along(axis2, marker1.A[:, 1])
            
            # Expressing the created marker/triad in terms of the 2nd body 
            # local reference frame resulting in matrix transformation 
            # expression
            mj_bar    = marker2.express(self.body_j)
            
            # Creating a symbolic equality that equates the symbolic dcm of the
            # marker to the matrix transformation expression created.
            mj_bar_eq = sm.Eq(self.mj_bar.A, mj_bar)
            
            # Storing the equalities in the markers list.
            markers_equalities = [mi_bar_eq, mj_bar_eq]
        
        else:
            # Other cases not considered yet.
            raise NotImplementedError
        
        # Increment/Update the symbolic_constants list.
        self._sym_constants += markers_equalities
        
        
        # Creating Local Constant Position Vectors based on the joint/actuator 
        # definition locations.
        # ====================================================================
        if self.def_locs == 0:
            location_equalities = []

        elif self.def_locs == 1:
            loc  = self.loc_1
            
            # Relative position vector of joint location relative to the 1st 
            # body reference point, in the body-local reference frame
            ui_bar = loc.express(self.body_i) - self.Ri.express(self.body_i)
            
            # Creating a symbolic equality that equates the symbolic vector of
            # the local position to the matrix transformation expression created.
            ui_bar_eq = sm.Eq(self.ui_bar, ui_bar)
            
            # Relative position vector of joint location relative to the 2nd 
            # body reference point, in the body-local reference frame
            uj_bar = loc.express(self.body_j) - self.Rj.express(self.body_j)
            
            # Creating a symbolic equality that equates the symbolic vector of
            # the local position to the matrix transformation expression created.
            uj_bar_eq = sm.Eq(self.uj_bar, uj_bar)

            # Storing the equalities in the locations list.
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
    
    
    def _construct_actuation_functions(self):
        """
        A private method to create actuation functions in actuator classes.
        """
        pass
    
    def _create_reactions_args(self):
        """
        TODO
        """
        
        body_i_name = self.body_i.id_name
        
        format_ = (self.prefix, self.id_name)
        L_raw_name = '%sL_%s'%format_
        L_frm_name = r'{%s\lambda_{%s}}'%format_
        self.L = matrix_symbol(L_raw_name, self.nc, 1, L_frm_name)
        
        format_ = (self.prefix, body_i_name, self.id_name)
        
        # Symbol of constraint reaction load acting on body_i.
        Qi_raw_name = '%sQ_%s_%s'%format_
        Qi_frm_name = r'{%sQ^{%s}_{%s}}'%format_
        self.Qi = matrix_symbol(Qi_raw_name, 7, 1, Qi_frm_name)
        
        # Symbol of constraint reaction force acting on body_i.
        Fi_raw_name = '%sF_%s_%s'%format_
        Fi_frm_name = r'{%sF^{%s}_{%s}}'%format_
        self.Fi = matrix_symbol(Fi_raw_name, 3, 1, Fi_frm_name)
        
        # Symbol of constraint reaction acting on body_i in terms of 
        # orientation parameters.
        Tie_raw_name = '%sTe_%s_%s'%format_
        Tie_frm_name = r'{%sTe^{%s}_{%s}}'%format_
        self.Ti_e = matrix_symbol(Tie_raw_name, 4, 1, Tie_frm_name)
        
        # Symbol of constraint reaction acting body_i in terms of cartesian 
        # coordinates.
        Ti_raw_name = '%sT_%s_%s'%format_
        Ti_frm_name = r'{%sT^{%s}_{%s}}'%format_
        self.Ti = matrix_symbol(Ti_raw_name, 3, 1, Ti_frm_name)
        
        if self.def_locs > 0:
            self.Ti_eq = 0.5*E(self.Pi)*self.Ti_e - Skew(self.ui)*self.Fi
        else:
            self.Ti_eq = 0.5*E(self.Pi)*self.Ti_e
        
        
    def _create_reactions_equalities(self):
        """
        TODO
        """
        
        jacobian_i = self.jacobian_i
        Qi_eq = sm.Eq(self.Qi, -jacobian_i.T*self.L)
        Fi_eq = sm.Eq(self.Fi, self.Qi[0:3,0])
        Ti_e_eq = sm.Eq(self.Ti_e, self.Qi[3:7,0])
        Ti_eq = sm.Eq(self.Ti, self.Ti_eq)
        self._reactions_equalities = [Qi_eq, Fi_eq, Ti_e_eq, Ti_eq]
        
        
###############################################################################
###############################################################################

class abstract_actuator(abstract_joint):
        
    """
    **Abstract Class**
    
    An abstract class that acts as a base class for motion actuators imposed on
    joints and/or bodies.
    The class implements the ```_construct_actuation_functions()``` method that
    adds undefined symbolic functions representing generic equations of time 
    (t) as an input variable.
    
    Notes
    -----
    TODO
    
    """

    
    def __init__(self, *args):
        super().__init__(*args)
        
    def _construct_actuation_functions(self):
        """
        A private method to create actuation functions in actuator classes.
        This method creats the following private members:
            - t: sympy.symbol
                A symbol that represents the time `t` variable as a real number

            - act_func: sympy.UndefinedFunction
                An undefined function object that represents the actuation 
                functionality.
            
            - _pos_function: sympy.Function
                The actuation function as a function in `t`
            
            - _vel_function: sympy.diff
                The 1st time derivative of the _pos_function
            
            - _acc_function: sympy.diff
                The 1st time derivative of the _vel_function
        """
        self.t = t = sm.symbols('t', real=True)
        self.act_func = sm.Function('%sUF_%s'%(self.prefix, self.id_name))
        self._pos_function = self.act_func(t)
        self._vel_function = sm.diff(self._pos_function, t)
        self._acc_function = sm.diff(self._vel_function, t)
            
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([  self._pos_level_equations[0] \
                               - I1 * self._pos_function])
    @property
    def vel_level_equations(self):
        return sm.BlockMatrix([  self._vel_level_equations[0] \
                               - I1 * self._vel_function])
    @property
    def acc_level_equations(self):
        return sm.BlockMatrix([  self._acc_level_equations[0] \
                               - I1 * self._acc_function])
    
    @property
    def arguments_symbols(self):
        """
        A list containing the symbolic mathematical objects -location points,
        orientation axes and actuation functionality- that should be 
        nuemrically defined by the user in a numerical simulation session.
        The number of arguments are given by the sum of `def_axis`, `def_locs` 
        in addition to `act_func`.
        """
        return super().arguments_symbols + [self.act_func]

###############################################################################
###############################################################################

class joint_actuator(abstract_actuator):

    """
    **Abstract Class**
    
    An abstract class that acts as a base class for joint actuators imposed on
    joints.

    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    joint : joint instance
        A joint instance of a sub-type of abstract_joint class.        
    
    Notes
    -----
    TODO
    """
    
    def __init__(self, name, joint=None):
        """
        
        """
        if joint is not None:
            self.joint = joint
            body_i = joint.body_i
            body_j = joint.body_j
            super().__init__(joint.name, body_i, body_j)
            self._name = name
            self.construct()
        else:
            super().__init__(name)
    
    def _create_reactions_equalities(self):
        self.Ti_eq = 0.5*E(self.Pi)*self.Ti_e
        jacobian_i = self.jacobian_i
        Qi_eq = sm.Eq(self.Qi, -jacobian_i.T*self.L)
        Fi_eq = sm.Eq(self.Fi, self.Qi[0:3,0])
        Ti_e_eq = sm.Eq(self.Ti_e, self.Qi[3:7,0])
        Ti_eq = sm.Eq(self.Ti, self.Ti_eq)
        self._reactions_equalities = [Qi_eq, Fi_eq, Ti_e_eq, Ti_eq]
        
    
###############################################################################
###############################################################################

class absolute_actuator(abstract_actuator):
    
    _coordinates_map = {'x':0, 'y':1, 'z':2}
    
    def __init__(self, name, body_i=None, body_j=None, coordinate='z'):
        self.coordinate = coordinate
        self.i = self._coordinates_map[self.coordinate]
        super().__init__(name, body_i, body_j)

        
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


