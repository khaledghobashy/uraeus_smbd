import bpy  
import numpy as np
scene = bpy.context.scene

def clear_scene():
    for obj in bpy.context.scene.objects:
        if obj.name != 'Sun' or obj.name!='Camera':
            bpy.context.scene.objects.active = obj
            bpy.ops.object.delete()

clear_scene()

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
    
    def __init__(self,name,loc1,loc2,r,n=30):       
        self.name = name
        self.r = r
        self.n = n
        self.loc1 = loc1
        self.loc2 = loc2        
        self.axis = loc2 - loc1
        self.norm = self.axis/np.linalg.norm(self.axis)
            
    def create_data(self):
        c1_verts = self._create_circle_verts(self.loc1)
        c2_verts = self._create_circle_verts(self.loc2)
        faces = self._create_cylinder_faces()
        self.data = {'verts':c1_verts+c2_verts,'faces':faces}
    
    def show(self):
        self.create_data()
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
        
scale = 1/20        
b1 = np.array([[ 0  ],[-430],[0]])*scale
b2 = np.array([[ 139],[ 215],[0]])*scale
b3 = np.array([[-139],[ 215],[0]])*scale

s1 = np.array([[-100],[-430],[0]])*scale
s2 = np.array([[ 39],[215],[0]])*scale
s3 = np.array([[-39],[215],[0]])*scale

p1 = np.array([[0   ],[-430],[430]])*scale
p2 = np.array([[ 139],[215],[430]])*scale
p3 = np.array([[-139],[215],[430]])*scale

r = 0.5
n = 30

rocker_1 = cylinder_mesh('rocker_1',b1,s1,r,n)
rocker_1.show()
link_1 = cylinder_mesh('link_1',s1,p1,r,n)
link_1.show()

rocker_2 = cylinder_mesh('rocker_2',b2,s2,r,n)
rocker_2.show()
link_2 = cylinder_mesh('link_2',s2,p2,r,n)
link_2.show()

rocker_3 = cylinder_mesh('rocker_3',b3,s3,r,n)
rocker_3.show()
link_3 = cylinder_mesh('link_3',s3,p3,r,n)
link_3.show()


objects = [link_1.obj,link_2.obj,link_3.obj,rocker_1.obj,rocker_2.obj,rocker_3.obj]
with open('stw_gou_sin_inputs.csv') as f:
    for i,line in enumerate(f):
        if i == 0: continue
        data = line.split(",")[15:]
        k = 0
        for ob in objects:
            ob.location = [float(n)*scale for n in data[k:k+3]]
            ob.rotation_quaternion = [float(n) for n in data[k+3:k+7]]
            ob.keyframe_insert('location', frame=i)
            ob.keyframe_insert('rotation_quaternion', frame=i)
            k+=7
#bpy.context.scene.render.frame_map_old = 250
#bpy.context.scene.render.frame_map_new = 50

