
# 3rd parties library imports
import sympy as sm

# Local application imports
from .algebraic_constraints import (abstract_joint, abstract_actuator,
                                    absolute_actuator, joint_actuator, 
                                    joint_constructor)
                                    
from .constraints_equations import (spehrical_constraint, 
                                    dot_product_1, dot_product_2,
                                    angle_constraint, coordinate_constraint)


###############################################################################

class fixed(abstract_joint, metaclass=joint_constructor):
    """    
    A joint that constraints two bodies to be fixed relative to each-other, by
    imposing six algebraic constraints equations to deminish the relative six
    degrees-of-freedom between the constrained bodies.
    
    The joint definition requires one defintion point and one defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    
    """
    
    def_axis = 1
    def_locs = 1
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_1('i', 'j')]

###############################################################################

class fixed_orientation(abstract_joint, metaclass=joint_constructor):
    """    
    A joint that constraints two bodies to have fixed relative orientation 
    w.r.t each-other, by imposing three algebraic constraints equations to 
    deminish the relative three relative orientation degrees-of-freedom between
    the constrained bodies.
    
    The joint definition requires only one defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    
    """
    def_axis = 1
    def_locs = 0
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_1('i', 'j')]

###############################################################################

class spherical(abstract_joint, metaclass=joint_constructor):
    """    
    The spherical joint prevents the relative translational movement between 
    the two connected bodies at a given common location, where the two bodies 
    are free to rotate relative to each-other in all directions.
    
    The joint definition requires one defintion point and one defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    
    """
    def_axis = 1
    def_locs = 1
    vector_equations = [spehrical_constraint()]

###############################################################################

class revolute(abstract_joint, metaclass=joint_constructor):
    """    
    The revolute joint allows only one rotation freedome between the connected 
    bodies around a common axis, thus it fully prevents the relative 
    translation between the bodies at the joint definition location, as well as
    any rotation other-than around the joint definition axis.
    
    The joint definition requires one defintion point and one defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    
    """
    def_axis = 1
    def_locs = 1
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i', 'k'),
                        dot_product_1('j', 'k')]

###############################################################################

class cylinderical(abstract_joint, metaclass=joint_constructor):
    """    
    The cylinderical joint allows only one relative rotation freedome and one
    relative translation freedome between the connected bodies along a common 
    axis, thus it prevents any relative translation and rotation along any 
    other direction, other-than around the joint definition axis.
    
    The joint definition requires one defintion point and one defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    
    """
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_2('i'),
                        dot_product_2('j')]
    
###############################################################################

class translational(abstract_joint, metaclass=joint_constructor):
    """    
    The translational joint allows only one relative translation freedome 
    between the connected bodies along a common axis, thus it prevents all 
    relative rotations between the connected bodies, and any relative 
    translation along any other direction, other-than around the joint 
    definition axis.
    
    The joint definition requires one defintion point and one defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    """
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_2('i'),
                        dot_product_2('j'),
                        dot_product_1('i', 'j')]

###############################################################################

class universal(abstract_joint, metaclass=joint_constructor):
    """    
    The universal joint prevents the relative translational movements between 
    the connected bodies just like the spherical joint, but it also prevents 
    the relative rotation/spin too, so, the connected body pair is only allowed
    to rotate around two common axes.
    
    The joint definition requires one defintion point and two defintion axis.
    
    Parameters
    ----------
    name : str
        Name of the joint instance. Should mimic a valid python variable name.
    body_i : body
        The 1st body isntance. Should be an instance of the `body` class.
    body_j : body
        The 2nd body isntance. Should be an instance of the `body` class.
    """
    def_axis = 2
    def_locs = 1
    vector_equations = [spehrical_constraint(),
                        dot_product_1('i', 'i')]
    
###############################################################################

class tripod(abstract_joint, metaclass=joint_constructor):
    def_axis = 2
    def_locs = 1
    vector_equations = [dot_product_1('i', 'i'),
                        dot_product_2('i'),
                        dot_product_2('j')]

###############################################################################

class rotational_actuator(joint_actuator, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 0
    vector_equations = [angle_constraint()]
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._pos_level_equations[0]])


###############################################################################

class translational_actuator(joint_actuator, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 1
    vector_equations = [dot_product_2('k')]
    
###############################################################################

class absolute_locator(absolute_actuator, metaclass=joint_constructor):
    def_axis = 0
    def_locs = 2
    vector_equations = [coordinate_constraint()]

###############################################################################

class absolute_rotator(abstract_actuator, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 0
    vector_equations = [angle_constraint()]
    
    @property
    def pos_level_equations(self):
        return sm.BlockMatrix([sm.Identity(1)*self._pos_level_equations[0]])

###############################################################################

class inline(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 2
    vector_equations = [dot_product_2('i'),
                        dot_product_2('j')]

###############################################################################

class dummy_cylinderical(abstract_joint, metaclass=joint_constructor):
    def_axis = 1
    def_locs = 2
    vector_equations = [dot_product_1('i', 'k'),
                        dot_product_1('j', 'k'),
                        dot_product_2('i'),
                        dot_product_2('j')]
    
###############################################################################
