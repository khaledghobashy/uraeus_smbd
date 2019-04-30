#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:31:46 2019

@author: khaledghobashy
"""

import os
import pickle
import importlib.util

import bpy

loaded_models = {}
loaded_instances = {}

class blmbs_saver(bpy.types.Operator):
    """Save Assembled Model From a Script"""
    bl_idname = "object.save_blmbs"
    bl_label = "Save Scene"

    @classmethod
    def poll(cls, context):
        file_name = os.path.basename(context.scene.blmbs_path)
        cond = file_name.isidentifier() and len(loaded_instances)!=0
        return cond

    def execute(self, context):
        data = {}
        for name, b in loaded_instances.items():
            sub_data = {}
            instance, script = b
            sub_data['prefix'] = instance.prefix
            sub_data['config'] = instance.cfg_file
            sub_data['script'] = script
            data[name] = sub_data
            
        self.save_model(data)
        print('\nSAVING DATA')
        print(data)
        return {'FINISHED'}
    
    @staticmethod
    def save_model(data):
        name = bpy.data.scenes["Scene"].blmbs_path
        print(name)
        with open('%s.blmbs'%name,'wb') as f:
            pickle.dump(data,f)

###############################################################################
class blmbs_opener(bpy.types.Operator):
    """Save Assembled Model From a Script"""
    bl_idname = "object.open_blmbs"
    bl_label = "Open Scene"

    @classmethod
    def poll(cls, context):
        file = os.path.splitext(context.scene.blmbs_path)[-1]
        cond = '.blmbs' == file
        return cond

    def execute(self, context):
        file = bpy.data.scenes["Scene"].blmbs_path
        data = self.open_model(file)
        print(data)
        for sub in data:
            sub_data = data[sub]
            bpy.data.scenes["Scene"].cfg_path  = sub_data['config']
            bpy.data.scenes["Scene"].subsys_id = sub_data['prefix']
            bpy.data.scenes["Scene"].scpt_path = sub_data['script']
            bpy.ops.object.load_mbsmodel()
        return {'FINISHED'}
    
    @staticmethod
    def open_model(name):
        with open(name,'rb') as f:
            data = pickle.load(f)
        return data
        
###############################################################################
class model_loader(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.load_mbsmodel"
    bl_label = "Insert Model"

    @classmethod
    def poll(cls, context):
        scene_vars = ['scpt_path','cfg_path']
        files = [os.path.splitext(getattr(context.scene,i))[-1] for i in scene_vars]
        cond = ['.py','.csv'] == files
        return cond


    def execute(self, context):
        print("Reading Blender Script...")
        script_path = bpy.data.scenes["Scene"].scpt_path
        config_data = bpy.data.scenes["Scene"].cfg_path
        prefix = bpy.data.scenes["Scene"].subsys_id
        
        script_name = os.path.basename(script_path)
        script_name = os.path.splitext(script_name)[0]
        
        spec  = importlib.util.spec_from_file_location(script_name, script_path)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        
        self.create_scene(model, config_data, prefix)
        print('\nLoaded Model : %s, with Subsystem_ID (%s)'%(script_name, prefix))
        bpy.ops.scene.reset_fields()
        return {'FINISHED'}
    
    @staticmethod
    def create_scene(model, config_data, prefix=''):
        blend = model.blender_scene(prefix)
        blend.get_data(config_data)
        blend.create_scene()
        script_path = bpy.data.scenes["Scene"].scpt_path
        loaded_instances[prefix] = (blend, script_path)

###############################################################################
class sim_loader(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.load_mbssim"
    bl_label = "Load Data"

    @classmethod
    def poll(cls, context):
        scene_vars = ['sim_path']
        files = [os.path.splitext(getattr(context.scene, i))[-1] for i in scene_vars]
        cond  = ['.csv'] == files
        return cond

    def execute(self, context):
        sim_data = bpy.data.scenes["Scene"].sim_path
        for b in loaded_instances.values():
            b[0].load_anim_data(sim_data)
        return {'FINISHED'}
    
###############################################################################
class clear_scene(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "scene.clear"
    bl_label = "Clear Scene"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        objects = [o.name for o in context.scene.objects]
        for obj in objects:
            try:
                bpy.data.objects[obj].select = True
                context.scene.objects.active = bpy.data.objects[obj]
                bpy.ops.object.delete()
            except KeyError:
                pass
        return {'FINISHED'}

###############################################################################
class reset_fields(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "scene.reset_fields"
    bl_label = "Clear Fields"
    
    fields = ['scpt_path','cfg_path','subsys_id']

    @classmethod
    def poll(cls, context):
        scene_vars = cls.fields[:-1]
        files = [os.path.isfile(getattr(context.scene,i)) for i in scene_vars]
        files.append(scene_vars[-1] !='')
        cond = False not in files
        return cond

    def execute(self, context):
        scene_vars = self.fields
        for n in scene_vars:
            path = getattr(context.scene,n)
            path = os.path.dirname(path)
            path = os.path.join(path,'')
            setattr(context.scene,n,path)
        return {'FINISHED'}
    
######################################################################

