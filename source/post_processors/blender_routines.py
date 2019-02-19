# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:23:04 2019

@author: khaled.ghobashy
"""
import itertools
import numpy as np
from source.cython_definitions.matrix_funcs import triad


class cylinder_mesh(object):
    
    def __init__(self,loc1,loc2,r,n=30):
        
        self.r = r
        self.n = n
        self.axis = loc2 - loc1
        self.norm = self.axis/np.linalg.norm(self.axis)
            
    def create_cylider(self):
        c1_verts = self._create_circle_verts(self.loc1)
        c2_verts = self._create_circle_verts(self.loc2)
        faces = self._create_cylinder_faces()
        self.data = {'verts':itertools.chain(c1_verts,c2_verts),'faces':faces}
    
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
        
        

p1 = np.array([[0],[0],[0]])
p2 = np.array([[10],[10],[10]])        
r = 5
n = 30

cyl = cylinder_mesh(p1,p2,r,n)
c1_verts = cyl._create_circle_verts(p1)
c2_verts = cyl._create_circle_verts(p2)

faces = cyl._create_cylinder_faces()

import matplotlib.pyplot as plt
x = [i[0] for i in c1_verts]
y = [i[1] for i in c1_verts]

plt.plot(x,y)
plt.grid()
plt.show()
