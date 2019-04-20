
import os
import csv
import numpy as np
import bpy
from source.post_processors.blender.objects import (cylinder_geometry,
                                                    composite_geometry,
                                                    triangular_prism)


path = os.path.dirname(__file__)

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

        self.s_links_ro = 0.5
        self.s_rockers_ro = 0.5
        self.hps_upper_1 = np.array([[0   ],[-430],[430]],dtype=np.float64)*scale
        self.hps_upper_3 = np.array([[-139],[215],[430]],dtype=np.float64)*scale
        self.hps_upper_2 = np.array([[ 139],[215],[430]],dtype=np.float64)*scale
        self.hps_middle_2 = np.array([[ 39],[215],[0]],dtype=np.float64)*scale
        self.hps_middle_1 = np.array([[-100],[-430],[0]],dtype=np.float64)*scale
        self.hps_middle_3 = np.array([[-39],[215],[0]],dtype=np.float64)*scale
        self.hps_bottom_1 = np.array([[ 0  ],[-430],[0]],dtype=np.float64)*scale
        self.hps_bottom_2 = np.array([[ 139],[ 215],[0]],dtype=np.float64)*scale
        self.hps_bottom_3 = np.array([[-139],[ 215],[0]],dtype=np.float64)*scale

        self.geometries = ['gms_rocker_2', 'gms_link_1', 'gms_link_3', 'gms_link_2', 'gms_rocker_3', 'gms_rocker_1', 'gms_table']

    
    def load_anim_data(self,csv_file):
        with open('%s'%csv_file, newline='') as csvfile:
            content = csv.reader(csvfile)
            keys = {k:i for i,k in enumerate(next(content)[1:])}
            arr = np.array(list(content))[:,1:]
            arr = np.array(arr,dtype=np.float64)
        
        scale = self.scale
        for i,row in enumerate(arr):
            for geo in self.geometries:
                k = keys['%s%s.x'%(self.prefix,geo.replace('gms_','rbs_'))]
                obj = getattr(self,geo).obj
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
blend.create_scene()
blend.load_anim_data(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\simulations\stewart_sim_data_v1.csv')
