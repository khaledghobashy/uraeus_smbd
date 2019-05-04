# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:23:04 2019

@author: khaled.ghobashy
"""

import bpy
import numpy as np

from asurt.utilities.numerics.spatial_alg import triad

class cylinder_geometry(object):
    
    def __init__(self,arg1,arg2,ro=10,ri=0):
        self.name = 'cyl'
        self.p1 = arg1
        self.p2 = arg2
        self.ro = ro
        self.ri = ri
        self.n = 30
        self.axis = self.p2 - self.p1
        self.norm = self.axis/np.linalg.norm(self.axis)
        self.create_obj()
    
    @property
    def R(self):
        arr = np.array(self.obj.location)
        arr = np.reshape(arr, (3,1))
        return arr
            
    def create_obj(self):
        self._create_data()
        verts = self.data['verts']
        faces = self.data['faces']
        mesh = bpy.data.meshes.new('%s_mesh'%self.name)
        self.mesh = mesh
        obj  = bpy.data.objects.new(self.name, mesh)
        self.obj = obj
        mesh.from_pydata(verts, [], faces)
        mesh.update()
        bpy.context.scene.objects.link(obj)
        obj.rotation_mode = 'QUATERNION'
        obj.select = True
        bpy.context.scene.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        obj.select = False
        
    def _create_data(self):
        c1_verts = self._create_circle_verts(self.p1)
        c2_verts = self._create_circle_verts(self.p2)
        faces = self._create_cylinder_faces()
        self.data = {'verts':c1_verts+c2_verts,'faces':faces}

    def _create_circle_verts(self,p):
        p = p[:,0]
        frame = triad(self.norm)
        v1 = frame[:,0]
        v2 = frame[:,1]
        angle = np.linspace(0,2*np.pi,self.n)
        verts = [tuple(p + self.ro*np.cos(i)*v1 + self.ro*np.sin(i)*v2) for i in angle]
        verts.append(verts[0])
        return verts

    def _create_cylinder_faces(self):
        n = self.n + 1
        faces = [(i, i+1, i+n+1, i+n) for i in range(self.n)]
        return faces

class triangular_prism(object):
    
    def __init__(self,p1,p2,p3,l=10):
        self.name = 'prism'
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.l = l        
        self.create()
    
    @property
    def R(self):
        arr = np.array(self.obj.location)
        arr = np.reshape(arr,(3,1))
        return arr

    def create(self):
        self._create_verts()
        self._create_faces()
        self._extrude()
        self._create_mesh()
        self._create_obj()
        
    def _create_verts(self):
        self._verts = [self.p1,self.p2,self.p3]
        self.n = len(self._verts)+1
        self._verts.append(self.p1)

    def _create_faces(self):
        top_face = tuple(range(self.n-1))
        bot_face = tuple(range(self.n,2*self.n,1))
        side_faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n-1)]
        self.faces = [top_face,bot_face] + side_faces

    def _extrude(self):
        v = self._get_normal() * self.l/2
        self.top_verts = [i+v for i in self._verts]
        self.bot_verts = [i-v for i in self._verts]
        self.top_verts = [tuple(i[:,0]) for i in self.top_verts]
        self.bot_verts = [tuple(i[:,0]) for i in self.bot_verts]
        
    def _create_mesh(self):
        self.mesh = bpy.data.meshes.new('%s_mesh'%self.name)
        side_verts = self.top_verts + self.bot_verts
        cape_verts = self.top_verts[:-1] + self.bot_verts[:-1]
        verts = side_verts + cape_verts
        self.mesh.from_pydata(verts, [], self.faces)
        self.mesh.update()
    
    def _create_obj(self):
        self.obj = obj  = bpy.data.objects.new(self.name, self.mesh)
        bpy.context.scene.objects.link(self.obj)
        obj.rotation_mode = 'QUATERNION'
        obj.select = True
        bpy.context.scene.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
        obj.select = False

    def _get_normal(self):
        verts = self._verts
        a1 = verts[1] - verts[0]
        a2 = verts[2] - verts[0]
        v = np.cross(a2,a1,axisa=0,axisb=0)
        v = np.reshape(v,(3,1))
        v = v/np.linalg.norm(v)
        return abs(v)

class composite_geometry(object):
    pass




