# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:23:04 2019

@author: khaled.ghobashy
"""
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

    
class cylinder_mesh(object):
    
    def __init__(self,loc1,loc2,r,n=30):
        
        self.r = r
        self.n = n
        self.loc1 = loc1
        self.loc2 = loc2
        self.axis = loc2 - loc1
        self.norm = self.axis/np.linalg.norm(self.axis)
            
    def create_cylider(self):
        c1_verts = self._create_circle_verts(self.loc1)
        c2_verts = self._create_circle_verts(self.loc2)
        faces = self._create_cylinder_faces()
        self.data = {'verts':c1_verts+c2_verts,'faces':faces}
    
    def _create_circle_verts(self,p):
        p = p[:,0]
        frame = triad(self.norm)
        v1 = frame[:,0]
        v2 = frame[:,1]
        angle = np.linspace(0,2*np.pi,self.n)
        verts = [tuple(p + self.r*np.cos(i)*v1 + self.r*np.sin(i)*v2) for i in angle]
        verts.append((verts[0]))
        return verts

    def _create_cylinder_faces(self):
        faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n)]
        return faces
        
        

b1 = np.array([[ 0  ],[-430],[0]])
b2 = np.array([[ 139],[ 215],[0]])
b3 = np.array([[-139],[ 215],[0]])

s1 = np.array([[-100],[-430],[0]])
s2 = np.array([[ 39],[215],[0]])
s3 = np.array([[-39],[215],[0]])

p1 = np.array([[0   ],[-430],[430]])
p2 = np.array([[ 139],[215],[430]])
p3 = np.array([[-139],[215],[430]])

r = 5
n = 30

rocker_1 = cylinder_mesh(b1,s1,r,n)
rocker_1.create_cylider()
rocker_1_verts = rocker_1.data['verts']
rocker_1_faces = rocker_1.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
obj = bpy.data.objects.new("rocker_1", mesh_data)


rocker_2 = cylinder_mesh(b2,s2,r,n)
rocker_2.create_cylider()
rocker_2_verts = rocker_2.data['verts']
rocker_2_faces = rocker_2.data['faces']
mesh_data = bpy.data.meshes.new("rocker_2_mesh")
mesh_data.from_pydata(rocker_2_verts, [], rocker_2_faces)
mesh_data.update()
obj = bpy.data.objects.new("rocker_2", mesh_data)


rocker_3 = cylinder_mesh(b3,s3,r,n)
rocker_3.create_cylider()
rocker_3_verts = rocker_3.data['verts']
rocker_3_faces = rocker_3.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
obj = bpy.data.objects.new("rocker_1", mesh_data)


conn_1 = cylinder_mesh(s1,b1,r,n)
conn_1.create_cylider()
conn_1_verts = conn_1.data['verts']
conn_1_faces = conn_1.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
obj = bpy.data.objects.new("rocker_1", mesh_data)


conn_2 = cylinder_mesh(s2,b2,r,n)
conn_2.create_cylider()
conn_2_verts = conn_2.data['verts']
conn_2_faces = conn_2.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
obj = bpy.data.objects.new("rocker_1", mesh_data)


conn_3 = cylinder_mesh(s3,b3,r,n)
conn_3.create_cylider()
conn_3_verts = conn_3.data['verts']
conn_3_faces = conn_3.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
obj = bpy.data.objects.new("rocker_1", mesh_data)


scene = bpy.context.scene
scene.objects.link(obj)
obj.select = True


r1.rotation_mode = 'QUATERNION'
r2.rotation_mode = 'QUATERNION'
r3.rotation_mode = 'QUATERNION'

c1.rotation_mode = 'QUATERNION'
c2.rotation_mode = 'QUATERNION'
c3.rotation_mode = 'QUATERNION'


with open('stewart_sim_data_v1.csv') as f:
    for i,line in enumerate(f):
        if i == 0: continue
        data = line.split(",")[8+7:]
        for k,ob in enumerate([c1,c2,c3,r1,r2,r3]):
            ob.location = [float(n) for n in data[k:k+3]]
            ob.rotation_quaternion = [float(n) for n in data[k+3:k+7]]
            ob.keyframe_insert('location', frame=i)
            ob.keyframe_insert('rotation_quaternion', frame=i)


# obj.select = True
# scene.objects.active = obj  # make the selection effective


import bpy  
import numpy as np
scene = bpy.context.scene

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

    
class cylinder_mesh(object):
    
    def __init__(self,loc1,loc2,r,n=30):
        
        self.r = r
        self.n = n
        self.loc1 = loc1
        self.loc2 = loc2        
        self.axis = loc2 - loc1
        self.norm = self.axis/np.linalg.norm(self.axis)
            
    def create_cylider(self):
        c1_verts = self._create_circle_verts(self.loc1)
        c2_verts = self._create_circle_verts(self.loc2)
        faces = self._create_cylinder_faces()
        self.data = {'verts':c1_verts+c2_verts,'faces':faces}
    
    def _create_circle_verts(self,p):
        p = p[:,0]
        frame = triad(self.norm)
        v1 = frame[:,0]
        v2 = frame[:,1]
        angle = np.linspace(0,2*np.pi,self.n)
        verts = [tuple(p + self.r*np.cos(i)*v1 + self.r*np.sin(i)*v2) for i in angle]
        verts.append((verts[0]))
        return verts

    def _create_cylinder_faces(self):
        faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n)]
        return faces
        
        
b1 = np.array([[ 0  ],[-430],[0]])/20
b2 = np.array([[ 139],[ 215],[0]])/20
b3 = np.array([[-139],[ 215],[0]])/20

s1 = np.array([[-100],[-430],[0]])/20
s2 = np.array([[ 39],[215],[0]])/20
s3 = np.array([[-39],[215],[0]])/20

p1 = np.array([[0   ],[-430],[430]])/20
p2 = np.array([[ 139],[215],[430]])/20
p3 = np.array([[-139],[215],[430]])/20

r = 0.5
n = 30


rocker_1 = cylinder_mesh(b1,s1,r,n)
rocker_1.create_cylider()
rocker_1_verts = rocker_1.data['verts']
rocker_1_faces = rocker_1.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
r1 = bpy.data.objects.new("rocker_1", mesh_data)
scene.objects.link(r1)


rocker_2 = cylinder_mesh(b2,s2,r,n)
rocker_2.create_cylider()
rocker_2_verts = rocker_2.data['verts']
rocker_2_faces = rocker_2.data['faces']
mesh_data = bpy.data.meshes.new("rocker_2_mesh")
mesh_data.from_pydata(rocker_2_verts, [], rocker_2_faces)
mesh_data.update()
r2 = bpy.data.objects.new("rocker_2", mesh_data)
scene.objects.link(r2)

rocker_3 = cylinder_mesh(b3,s3,r,n)
rocker_3.create_cylider()
rocker_3_verts = rocker_3.data['verts']
rocker_3_faces = rocker_3.data['faces']
mesh_data = bpy.data.meshes.new("rocker_3_mesh")
mesh_data.from_pydata(rocker_3_verts, [], rocker_3_faces)
mesh_data.update()
r3 = bpy.data.objects.new("rocker_3", mesh_data)
scene.objects.link(r3)


conn_1 = cylinder_mesh(s1,p1,r,n)
conn_1.create_cylider()
conn_1_verts = conn_1.data['verts']
conn_1_faces = conn_1.data['faces']
mesh_data = bpy.data.meshes.new("conn_1_mesh")
mesh_data.from_pydata(conn_1_verts, [], conn_1_faces)
mesh_data.update()
c1 = bpy.data.objects.new("conn_1", mesh_data)
scene.objects.link(c1)


conn_2 = cylinder_mesh(s2,p2,r,n)
conn_2.create_cylider()
conn_2_verts = conn_2.data['verts']
conn_2_faces = conn_2.data['faces']
mesh_data = bpy.data.meshes.new("conn_2_mesh")
mesh_data.from_pydata(conn_2_verts, [], conn_2_faces)
mesh_data.update()
c2 = bpy.data.objects.new("conn_2", mesh_data)
scene.objects.link(c2)

conn_3 = cylinder_mesh(s3,p3,r,n)
conn_3.create_cylider()
conn_3_verts = conn_3.data['verts']
conn_3_faces = conn_3.data['faces']
mesh_data = bpy.data.meshes.new("conn_3_mesh")
mesh_data.from_pydata(conn_3_verts, [], conn_3_faces)
mesh_data.update()
c3 = bpy.data.objects.new("conn_3", mesh_data)
scene.objects.link(c3)



r1.rotation_mode = 'QUATERNION'
r2.rotation_mode = 'QUATERNION'
r3.rotation_mode = 'QUATERNION'

c1.rotation_mode = 'QUATERNION'
c2.rotation_mode = 'QUATERNION'
c3.rotation_mode = 'QUATERNION'


with open('stewart_sim_data_v1.csv') as f:
    for i,line in enumerate(f):
        if i == 0: continue
        data = line.split(",")[15:]
        k = 0
        for ob in [c1,c2,c3,r1,r2,r3]:
            ob.location = [float(n)/20 for n in data[k:k+3]]
            ob.rotation_quaternion = [float(n) for n in data[k+3:k+7]]
            ob.keyframe_insert('location', frame=i)
            ob.keyframe_insert('rotation_quaternion', frame=i)
            k+=7


# obj.select = True
# scene.objects.active = obj  # make the selection effective


import bpy  
import numpy as np
scene = bpy.context.scene

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

    
class cylinder_mesh(object):
    
    def __init__(self,loc1,loc2,r,n=30):
        
        self.r = r
        self.n = n
        self.loc1 = loc1
        self.loc2 = loc2        
        self.axis = loc2 - loc1
        self.norm = self.axis/np.linalg.norm(self.axis)
            
    def create_cylider(self):
        c1_verts = self._create_circle_verts(self.loc1)
        c2_verts = self._create_circle_verts(self.loc2)
        faces = self._create_cylinder_faces()
        self.data = {'verts':c1_verts+c2_verts,'faces':faces}
    
    def _create_circle_verts(self,p):
        p = p[:,0]
        frame = triad(self.norm)
        v1 = frame[:,0]
        v2 = frame[:,1]
        angle = np.linspace(0,2*np.pi,self.n)
        verts = [tuple(p + self.r*np.cos(i)*v1 + self.r*np.sin(i)*v2) for i in angle]
        verts.append((verts[0]))
        return verts

    def _create_cylinder_faces(self):
        faces = [(i,i+1,i+self.n+1,i+self.n) for i in range(self.n)]
        return faces
        
        
b1 = np.array([[ 0  ],[-430],[0]])/20
b2 = np.array([[ 139],[ 215],[0]])/20
b3 = np.array([[-139],[ 215],[0]])/20

s1 = np.array([[-100],[-430],[0]])/20
s2 = np.array([[ 39],[215],[0]])/20
s3 = np.array([[-39],[215],[0]])/20

p1 = np.array([[0   ],[-430],[430]])/20
p2 = np.array([[ 139],[215],[430]])/20
p3 = np.array([[-139],[215],[430]])/20

r = 0.5
n = 30


rocker_1 = cylinder_mesh(b1,s1,r,n)
rocker_1.create_cylider()
rocker_1_verts = rocker_1.data['verts']
rocker_1_faces = rocker_1.data['faces']
mesh_data = bpy.data.meshes.new("rocker_1_mesh")
mesh_data.from_pydata(rocker_1_verts, [], rocker_1_faces)
mesh_data.update()
r1 = bpy.data.objects.new("rocker_1", mesh_data)
r1.location = (-50,-430,0)
scene.objects.link(r1)


rocker_2 = cylinder_mesh(b2,s2,r,n)
rocker_2.create_cylider()
rocker_2_verts = rocker_2.data['verts']
rocker_2_faces = rocker_2.data['faces']
mesh_data = bpy.data.meshes.new("rocker_2_mesh")
mesh_data.from_pydata(rocker_2_verts, [], rocker_2_faces)
mesh_data.update()
r2 = bpy.data.objects.new("rocker_2", mesh_data)
r2.location = (88.5,215,0)
scene.objects.link(r2)

rocker_3 = cylinder_mesh(b3,s3,r,n)
rocker_3.create_cylider()
rocker_3_verts = rocker_3.data['verts']
rocker_3_faces = rocker_3.data['faces']
mesh_data = bpy.data.meshes.new("rocker_3_mesh")
mesh_data.from_pydata(rocker_3_verts, [], rocker_3_faces)
mesh_data.update()
r3 = bpy.data.objects.new("rocker_3", mesh_data)
r3.location = (-88.5,215,0)
scene.objects.link(r3)


conn_1 = cylinder_mesh(s1,p1,r,n)
conn_1.create_cylider()
conn_1_verts = conn_1.data['verts']
conn_1_faces = conn_1.data['faces']
mesh_data = bpy.data.meshes.new("conn_1_mesh")
mesh_data.from_pydata(conn_1_verts, [], conn_1_faces)
mesh_data.update()
c1 = bpy.data.objects.new("conn_1", mesh_data)
c1.location = (-50,-430,215)
scene.objects.link(c1)


conn_2 = cylinder_mesh(s2,p2,r,n)
conn_2.create_cylider()
conn_2_verts = conn_2.data['verts']
conn_2_faces = conn_2.data['faces']
mesh_data = bpy.data.meshes.new("conn_2_mesh")
mesh_data.from_pydata(conn_2_verts, [], conn_2_faces)
mesh_data.update()
c2 = bpy.data.objects.new("conn_2", mesh_data)
c2.location = (88.5,215,215)
scene.objects.link(c2)

conn_3 = cylinder_mesh(s3,p3,r,n)
conn_3.create_cylider()
conn_3_verts = conn_3.data['verts']
conn_3_faces = conn_3.data['faces']
mesh_data = bpy.data.meshes.new("conn_3_mesh")
mesh_data.from_pydata(conn_3_verts, [], conn_3_faces)
mesh_data.update()
c3 = bpy.data.objects.new("conn_3", mesh_data)
c3.location = (-88.5,215,215)
scene.objects.link(c3)



r1.rotation_mode = 'QUATERNION'
r2.rotation_mode = 'QUATERNION'
r3.rotation_mode = 'QUATERNION'

c1.rotation_mode = 'QUATERNION'
c2.rotation_mode = 'QUATERNION'
c3.rotation_mode = 'QUATERNION'


with open('stewart_sim_data_v1.csv') as f:
    for i,line in enumerate(f):
        if i == 0: continue
        data = line.split(",")[15:]
        k = 0
        for ob in [c1,c2,c3,r1,r2,r3]:
            ob.location = [float(n)/20 for n in data[k:k+3]]
            ob.rotation_quaternion = [float(n) for n in data[k+3:k+7]]
            ob.keyframe_insert('location', frame=i)
            ob.keyframe_insert('rotation_quaternion', frame=i)
            k+=7


# obj.select = True
# scene.objects.active = obj  # make the selection effective
