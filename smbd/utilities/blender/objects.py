# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:23:04 2019

@author: khaled.ghobashy
"""

import bpy
import bmesh
import numpy as np

from smbd.utilities.numerics.spatial_alg import triad

###############################################################################
###############################################################################

class generic_geometry(object):
    
    def __init__(self,*args):
        raise NotImplementedError
    
    @property
    def R(self):
        location = np.array(self.obj.location)
        location = np.reshape(location, (3,1))
        return location
        
    def create(self):
        self._create_mesh()
        self._assemble()
    
    def _create_mesh(self):
        verts = self._create_verts()
        faces = self._create_faces()
        self.mesh = bpy.data.meshes.new('%s_mesh'%self.name)
        self.mesh.from_pydata(verts, [], faces)
        self.mesh.update()

    def _assemble(self):
        self.obj = obj  = bpy.data.objects.new(self.name, self.mesh)
        bpy.context.scene.objects.link(self.obj)
        bpy.context.scene.objects.active = obj
        obj.select = True
        obj.rotation_mode = 'QUATERNION'
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        obj.select = False

    def _create_verts(self):
        raise NotImplementedError
    
    def _create_faces(self):
        raise NotImplementedError
    
###############################################################################
###############################################################################

class triangular_prism(generic_geometry):
    
    def __init__(self, p1, p2, p3, l=10):
        self.name = 'prism'
        self.p1 = p1 
        self.p2 = p2 
        self.p3 = p3
        self.l = l
        self.n = 3
        self.create()
        
    def _create_verts(self):
        base_verts = [self.p1, self.p2, self.p3, self.p1]        
        axis = self._get_normal()
        v = axis * self.l/2
        
        top_verts = [i+v for i in base_verts]
        top_verts = [tuple(i[:,0]) for i in top_verts]

        bot_verts = [i-v for i in base_verts]
        bot_verts = [tuple(i[:,0]) for i in bot_verts]
        
        side_verts = top_verts + bot_verts
        cape_verts = top_verts + bot_verts
        verts = cape_verts + side_verts
        return verts

    def _create_faces(self):
        n_verts = self.n+1
        offset  = n_verts
        top_face = tuple(range(n_verts))
        bot_face = tuple(range(n_verts, 2*n_verts))
        side_faces = [(i, i+1, i+offset+1, i+offset) for i in range(n_verts)]
        self.faces = [top_face, bot_face] + side_faces
        return self.faces
        
    def _get_normal(self):
        a1 = self.p2 - self.p1
        a2 = self.p3 - self.p1
        v = np.cross(a2, a1, axisa=0, axisb=0)
        v = np.reshape(v, (3, 1))
        v = v/np.linalg.norm(v)
        return abs(v)

###############################################################################
###############################################################################

class cylinder_geometry(generic_geometry):
    
    def __init__(self, arg1, arg2, ro=10, ri=0):
        self.name = 'cyl'
        self.p1 = arg1
        self.p2 = arg2
        self.ro = ro
        self.ri = ri
        self.n = 30
        self.axis = self.p2 - self.p1
        self.norm = self.axis/np.linalg.norm(self.axis)
        self.create()
        
    def _create_verts(self):
        top_verts = self._create_circle_verts(self.p1, self.norm, self.n, self.ro)
        bot_verts = self._create_circle_verts(self.p2, self.norm, self.n, self.ro)
        return top_verts + bot_verts

    def _create_faces(self):
        n = self.n
        faces = [(i, i+1, i+n+1, i+n) for i in range(self.n)]
        return faces
    
    @staticmethod
    def _create_circle_verts(point, axis, n, radius):
        p = point[:,0]
        frame = triad(axis)
        v1 = frame[:,0]
        v2 = frame[:,1]
        angle = np.linspace(0, 2*np.pi, n)
        verts = [tuple(p + radius*np.cos(i)*v1 + radius*np.sin(i)*v2) for i in angle]
        verts.append(verts[0])
        return verts

###############################################################################
###############################################################################

class sphere_geometry(generic_geometry):
    
    def __init__(self, p1, radius):
        self.name = 'sphere'
        self.p1 = p1
        self.radius = radius
        self.create()
        
    def _assemble(self):
        location = self.p1[:,0]        
        super()._assemble()
        self.obj.location = location        
        
    def _create_mesh(self):
        self.mesh = bpy.data.meshes.new('%s_mesh'%self.name)
        empty_mesh = bmesh.new()
        bmesh.ops.create_uvsphere(empty_mesh, u_segments=32, v_segments=16, 
                                  diameter=self.radius)
        empty_mesh.to_mesh(self.mesh)
        empty_mesh.free()

###############################################################################
###############################################################################
        
class composite_geometry(object):
    pass




