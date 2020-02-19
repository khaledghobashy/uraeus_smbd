# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
# Standard library imports
import os

# 3rd party library imports
import cloudpickle
import sympy as sm

# Local applicataion imports
from ...symbolic.components import joints as joints
from ...symbolic.components import forces as forces
from ...symbolic.systems import topology_classes as topology_classes
from ...symbolic.systems import configuration_classes as cfg_cls
from ...symbolic.components.matrices import vector

###############################################################################

class abstract_decorator(object):
    def __init__(self, sym_system):
        self._sym_system = sym_system
        self._decorate_items()
    
    @property
    def _items(self):
        members = {i:getattr(self, i) for i in dir(self) if not i.startswith('_') and not i.startswith("__")}
        return members

    def _decorate_items(self):
        for attr, obj in self._items.items():
            setattr(self, attr, self._decorate(obj))
    
    def _decorate(self, constructor):
        raise NotImplementedError


class joints_container(abstract_decorator):
    
    def __init__(self, topology):
        self.spherical = joints.spherical
        self.revolute  = joints.revolute
        self.universal = joints.universal
        self.translational = joints.translational
        self.cylinderical  = joints.cylinderical
        self.tripod = joints.tripod
        self.fixed  = joints.fixed
        self.fixed_orientation = joints.fixed_orientation
        self.inline = joints.inline
        
        super().__init__(topology)
        self._topology = self._sym_system
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
            self._topology.add_joint(edge_component, *args, **kwargs)
        return decorated

    
class actuators_container(abstract_decorator):
    
    def __init__(self, topology):
        self.rotational_actuator = joints.rotational_actuator
        self.absolute_locator = joints.absolute_locator
        self.translational_actuator = joints.translational_actuator
        self.absolute_rotator = joints.absolute_rotator
        super().__init__(topology)
        self._topology = self._sym_system
    
    def _decorate(self, edge_component):
        if issubclass(edge_component, joints.absolute_locator):
            def decorated(*args, **kwargs):
                self._topology.add_absolute_actuator(edge_component, *args, **kwargs)
        elif issubclass(edge_component, joints.absolute_rotator):
            def decorated(*args, **kwargs):
                self._topology.add_absolute_actuator(edge_component, *args, **kwargs)
        else:
            def decorated(*args, **kwargs):
                self._topology.add_joint_actuator(edge_component, *args, **kwargs)
        return decorated


class forces_container(abstract_decorator):
    
    def __init__(self, topology):
        self.internal_force = forces.internal_force
        self.force = forces.force
        self.torque = forces.torque
        self.generic_force = forces.generic_force
        self.bushing = forces.bushing
        self.generic_bushing = forces.generic_bushing
        
        super().__init__(topology)
        self._topology = self._sym_system
    
    def _decorate(self, edge_component):
        def decorated(*args, **kwargs):
            self._topology.add_force(edge_component, *args, **kwargs)
        return decorated


class geometries_nodes(abstract_decorator):
    def __init__(self, config_instance):
        self._sym = 'gm'
        self._symbolic_type = cfg_cls.Geometry

        # Geometry Constructors
        self.Composite_Geometry = cfg_cls.Composite_Geometry
        self.Cylinder_Geometry = cfg_cls.Cylinder_Geometry
        self.Triangular_Prism = cfg_cls.Triangular_Prism
        self.Sphere_Geometry = cfg_cls.Sphere_Geometry

        super().__init__(config_instance)
        self._config = self._sym_system

    def _decorate(self, constructor):
        def decorated(name, args, mirror=False):
            node = self._config.add_node(name, self._symbolic_type, self._sym, mirror)
            self._config.add_relation(constructor, node, args, mirror)
        return decorated

class scalar_nodes(abstract_decorator):
    def __init__(self, config_instance):
        self._sym = ''
        self._symbolic_type = sm.symbols

        # Constructors
        self.Equal_to = cfg_cls.Equal_to
        self.UserInput = None

        super().__init__(config_instance)
        self._config = self._sym_system

    def _decorate(self, constructor):
        def decorated(name, args=None):
            node = self._config.add_node(name, self._symbolic_type, self._sym)
            if constructor:
                self._config.add_relation(constructor, node, args)
        return decorated

class vector_nodes(abstract_decorator):
    def __init__(self, config_instance):
        self._sym = 'vc'
        self._symbolic_type = vector

        # Constructors
        self.Mirrored = cfg_cls.Mirrored
        self.Oriented = cfg_cls.Oriented
        self.Equal_to = cfg_cls.Equal_to
        self.UserInput = None

        super().__init__(config_instance)
        self._config = self._sym_system

    def _decorate(self, constructor):
        def decorated(name, args=None, mirror=False):
            node = self._config.add_node(name, self._symbolic_type, self._sym, mirror)
            if constructor:
                self._config.add_relation(constructor, node, args, mirror)
        return decorated
    
class points_nodes(abstract_decorator):
    def __init__(self, config_instance):
        self._sym = 'hp'
        self._symbolic_type = vector

        # Constructors
        self.Mirrored = cfg_cls.Mirrored
        self.Centered = cfg_cls.Centered
        self.Equal_to = cfg_cls.Equal_to
        self.UserInput = None

        super().__init__(config_instance)
        self._config = self._sym_system

    def _decorate(self, constructor):
        def decorated(name, args=None, mirror=False):
            node = self._config.add_node(name, self._symbolic_type, self._sym, mirror)
            if constructor:
                self._config.add_relation(constructor, node, args, mirror)
        return decorated

class relations_methods(abstract_decorator):
    def __init__(self, config_instance):
        # Constructors
        self.Mirrored = cfg_cls.Mirrored
        self.Centered = cfg_cls.Centered
        self.Oriented = cfg_cls.Oriented
        self.Equal_to = cfg_cls.Equal_to

        super().__init__(config_instance)
        self._config = self._sym_system

    def _decorate(self, constructor):
        def decorated(node, args, mirror=False):
            self._config.add_relation(constructor, node, args, mirror)
        return decorated
