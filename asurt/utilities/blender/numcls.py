#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:18:39 2019

@author: khaledghobashy
"""

import csv
import numpy as np
import bpy

class bpy_scene(object):
    
    def __init__(self, prefix='', scale=1/20):
        self.prefix = prefix
        self.scale = scale
        self._inputs = []
        self.geometries = {}

    def create_scene(self):
        raise NotImplementedError
    
    def get_data(self, csv_file):
        self.cfg_file = csv_file
        with open(csv_file, newline='') as csvfile:
            content = csv.reader(csvfile)
            next(content)
            for row in content:
                attr = row[0]
                if attr in self._inputs:
                    value = getattr(self,attr)
                    setattr(self,attr,value)
                    if isinstance(value, np.ndarray):
                        value = np.array(row[1:],dtype=np.float64)
                        value = np.resize(value,(3,1))*self.scale
                        setattr(self,attr,value)
                    else:
                        value = float(row[1])*self.scale
                        setattr(self,attr,value)

    def load_anim_data(self,csv_file):
        with open(csv_file, newline='') as csvfile:
            content = csv.reader(csvfile)
            keys = {k:i for i,k in enumerate(next(content)[1:])}
            arr = np.array(list(content))[:,1:]
            arr = np.array(arr,dtype=np.float64)

        scale = self.scale
        for i,row in enumerate(arr,1):
            for g,b in self.geometries.items():
                k = keys['%s%s.x'%(self.prefix,b)]
                obj = getattr(self,g).obj
                obj.location = [float(n)*scale for n in row[k:k+3]]
                obj.rotation_quaternion = [float(n) for n in row[k+3:k+7]]
                obj.keyframe_insert('location', frame=i)
                obj.keyframe_insert('rotation_quaternion', frame=i)

        fps = 24
        total_time = arr[-1,-1]
        bpy.context.scene.render.frame_map_old = (i/total_time)/fps
        bpy.context.scene.render.frame_map_new = 1
        bpy.context.scene.frame_end = fps*total_time

    
    @staticmethod
    def setup_VIEW_3D():
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {'area': area, 'region': region, 'edit_object': bpy.context.edit_object}
                        bpy.ops.view3d.view_all(override)


