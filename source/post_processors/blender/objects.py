# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:23:04 2019

@author: khaled.ghobashy
"""

import bpy  
import numpy as np

def skew_matrix(v):
    vs = np.array([[0,-v[2,0],v[1,0]],
                 [v[2,0],0,-v[0,0]],
                 [-v[1,0],v[0,0],0]])
    return vs

def orthogonal_vector(v):
    dummy = np.ones((3,1))
    i = np.argmax(np.abs(v))
    dummy[i] = 0
    m = np.linalg.multi_dot([skew_matrix(v),dummy])
    return m

def triad(v1):
    k = v1/np.linalg.norm(v1)
    i = orthogonal_vector(k)
    i = i/np.linalg.norm(i)
    j = np.linalg.multi_dot([skew_matrix(k),i])
    j = j/np.linalg.norm(j)
    m = np.concatenate([i,j,k],axis=1)
    return m


class asbtract_geometry(object):
    
    def __init__(self,*args):
        self.mesh = bpy.data.meshes.new('%s_mesh'%self.name)
        self._create_obj()
            
    def add_verts(self,*verts):
        self._verts = list(verts)
        self._verts.append(verts[0])
        self.n = len(verts)+1
    
    def extrude(self,l):
        v = self._get_normal() * l/2
        self.top_verts = [i+v for i in self._verts]
        self.bot_verts = [i-v for i in self._verts]
        self.top_verts = [tuple(i[:,0]) for i in self.top_verts]
        self.bot_verts = [tuple(i[:,0]) for i in self.bot_verts]
    
    def create(self):
        bpy.context.scene.objects.link(self.obj)
        self._set_faces()
        self._update_mesh()

    def _set_faces(self):
        top_face = tuple(range(self.n-1))
        bot_face = tuple(range(self.n,2*self.n,1))
        side_faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n-1)]
        self.faces = [top_face,bot_face] + side_faces
        
    def _update_mesh(self):
        side_verts = self.top_verts + self.bot_verts
        cape_verts = self.top_verts[:-1] + self.bot_verts[:-1]
        verts = side_verts + cape_verts
        self.mesh.from_pydata(verts, [], self.faces)
        self.mesh.update()
    
    def _create_obj(self):
        self.obj = obj  = bpy.data.objects.new(self.name, self.mesh)
        obj.rotation_mode = 'QUATERNION'
        obj.select = True
        bpy.context.scene.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

    def _get_normal(self):
        verts = self._verts
        a1 = verts[1] - verts[0]
        a2 = verts[2] - verts[0]
        v = np.cross(a2,a1,axisa=0,axisb=0)
        v = np.reshape(v,(3,1))
        v = v/np.linalg.norm(v)
        return abs(v)
    
    
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
        faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n)]
        return faces

class triangular_prism(object):
    
    def __init__(self,p1,p2,p3,l=10):
        self.name = 'prism'
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.l = l        
        self.mesh = bpy.data.meshes.new('%s_mesh'%self.name)
        self._create_obj()
        self.add_verts()
        self.extrude()
        self.create()
            
    def add_verts(self):
        self._verts = [self.p1,self.p2,self.p3]
        self.n = len(self._verts)+1
        self._verts.append(self.p1)
    
    def extrude(self):
        v = self._get_normal() * self.l/2
        self.top_verts = [i+v for i in self._verts]
        self.bot_verts = [i-v for i in self._verts]
        self.top_verts = [tuple(i[:,0]) for i in self.top_verts]
        self.bot_verts = [tuple(i[:,0]) for i in self.bot_verts]
    
    def create(self):
        bpy.context.scene.objects.link(self.obj)
        self._set_faces()
        self._update_mesh()

    def _set_faces(self):
        top_face = tuple(range(self.n-1))
        bot_face = tuple(range(self.n,2*self.n,1))
        side_faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n-1)]
        self.faces = [top_face,bot_face] + side_faces
        
    def _update_mesh(self):
        side_verts = self.top_verts + self.bot_verts
        cape_verts = self.top_verts[:-1] + self.bot_verts[:-1]
        verts = side_verts + cape_verts
        self.mesh.from_pydata(verts, [], self.faces)
        self.mesh.update()
    
    def _create_obj(self):
        self.obj = obj  = bpy.data.objects.new(self.name, self.mesh)
        obj.rotation_mode = 'QUATERNION'
        obj.select = True
        bpy.context.scene.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')

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




