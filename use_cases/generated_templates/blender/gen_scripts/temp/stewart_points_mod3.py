
import os
import csv
import numpy as np
import bpy
from source.post_processors.blender.objects import (cylinder_geometry,
                                                    composite_geometry,
                                                    triangular_prism)



#full_path = os.path.dirname(__file__).split('/')
#path2pkg = []
#d = full_path.index('asurt_cdt_symbolic')
#path2pkg = full_path[0:d+1]

try:
    bpy.context.scene.objects.active = bpy.data.objects['Cube']
    bpy.ops.object.delete()
except KeyError:
    pass

class blender_scene(object):

    def __init__(self,prefix=''):
        self.prefix = prefix
        scale = 1/20
        self.scale = scale

        self.hps_upper_3 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_bottom_3 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_middle_3 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_upper_2 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.s_rockers_ro = 0.5
        self.hps_middle_2 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_middle_1 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_bottom_2 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_upper_1 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hps_bottom_1 = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.s_links_ro = 0.5

        self._inputs = ['hps_upper_3', 'hps_bottom_3', 'hps_middle_3', 'hps_upper_2', 's_rockers_ro', 'hps_middle_2', 'hps_middle_1', 'hps_bottom_2', 'hps_upper_1', 'hps_bottom_1', 's_links_ro']
        self.geometries = {'gms_table': 'rbs_table', 'gms_rocker_1': 'rbs_rocker_1', 'gms_rocker_2': 'rbs_rocker_2', 'gms_rocker_3': 'rbs_rocker_3', 'gms_link_1': 'rbs_link_1', 'gms_link_2': 'rbs_link_2', 'gms_link_3': 'rbs_link_3'}

    
    def get_data(self,csv_file):
        with open(csv_file, newline='') as csvfile:
            content = csv.reader(csvfile)
            next(content)
            for row in content:
                attr = row[0]
                if attr in self._inputs:
                    value = np.array(row[1:],dtype=np.float64)
                    value = np.resize(value,(3,1))*self.scale
                    setattr(self,attr,value)


    def load_anim_data(self,csv_file):
        with open(csv_file, newline='') as csvfile:
            content = csv.reader(csvfile)
            keys = {k:i for i,k in enumerate(next(content)[1:])}
            arr = np.array(list(content))[:,1:]
            arr = np.array(arr,dtype=np.float64)

        scale = self.scale
        for i,row in enumerate(arr):
            for g,b in self.geometries.items():
                k = keys['%s%s.x'%(self.prefix,b)]
                obj = getattr(self,g).obj
                obj.location = [float(n)*scale for n in row[k:k+3]]
                obj.rotation_quaternion = [float(n) for n in row[k+3:k+7]]
                obj.keyframe_insert('location', frame=i)
                obj.keyframe_insert('rotation_quaternion', frame=i)

        bpy.context.scene.render.frame_map_old = i+1
        bpy.context.scene.render.frame_map_new = 50
        bpy.context.scene.frame_end = bpy.context.scene.render.frame_map_new

    def create_scene(self):
        self.gms_rocker_2 = cylinder_geometry(self.hps_bottom_2,self.hps_middle_2,self.s_rockers_ro)
        self.gms_link_1 = cylinder_geometry(self.hps_upper_1,self.hps_middle_1,self.s_links_ro)
        self.gms_link_3 = cylinder_geometry(self.hps_upper_3,self.hps_middle_3,self.s_links_ro)
        self.gms_link_2 = cylinder_geometry(self.hps_upper_2,self.hps_middle_2,self.s_links_ro)
        self.gms_rocker_3 = cylinder_geometry(self.hps_bottom_3,self.hps_middle_3,self.s_rockers_ro)
        self.gms_rocker_1 = cylinder_geometry(self.hps_bottom_1,self.hps_middle_1,self.s_rockers_ro)
        self.gms_table = triangular_prism(self.hps_upper_1,self.hps_upper_2,self.hps_upper_3,self.s_rockers_ro)

blend = blender_scene('SG.')
blend.get_data(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\stewart_points_v3.csv')
blend.create_scene()
blend.load_anim_data(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\simulations\stewart_sim_data_v1.csv')

