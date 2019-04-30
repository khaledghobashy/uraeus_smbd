#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:29:41 2019

@author: khaledghobashy
"""

import sys
from bpy.types import Panel

pkg_path = '/home/khaledghobashy/Documents/asurt_cdt_symbolic'
if pkg_path not in sys.path:
    sys.path.append(pkg_path)
    

loaded_models = {}
loaded_instances = {}

###############################################################################
class ASURT_Panel(Panel):
    bl_space_type =  'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_context = "objectmode"
    bl_category = "ASURT"
    
###############################################################################
class SaveOpen(ASURT_Panel):
    bl_label = "Tools"

    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Open")
        split = layout.split()
        col = split.column(align=True)
        col.operator("bpy.ops.buttons.file_browse")
        col.prop(context.scene, 'blmbs_path')
        col.operator("object.open_blmbs")
        
        layout.label(text="SAVE")
        split = layout.split()
        col = split.column(align=True)
        col.operator("bpy.ops.buttons.file_browse")
        col.prop(context.scene, 'blmbs_path')
        col.operator("object.save_blmbs")

###############################################################################
class SceneControls(ASURT_Panel):
    bl_label = "Scene Controls"

    def draw(self, context):
        layout = self.layout
        
        split = layout.split()
        col = split.column(align=True)
        col.label(text="Load Simulation Data:")
        col.operator("bpy.ops.buttons.file_browse")
        col.prop(context.scene, 'sim_path')
        col.operator("object.load_mbssim")
        
        layout.label(text="Animation Controls")
        screen = context.screen
        row = layout.row(align=True)
        row.operator("screen.frame_jump", text="", icon='REW').end = False
        row.operator("screen.keyframe_jump", text="", icon='PREV_KEYFRAME').next = False
        if not screen.is_animation_playing:
            row.operator("screen.animation_play", text="", icon='PLAY_REVERSE').reverse = True
            row.operator("screen.animation_play", text="", icon='PLAY')
        else:
            sub = row.row(align=True)
            sub.scale_x = 2.0
            sub.operator("screen.animation_play", text="", icon='PAUSE')
        row.operator("screen.keyframe_jump", text="", icon='NEXT_KEYFRAME').next = True
        row.operator("screen.frame_jump", text="", icon='FF').end = True
        
        row = layout.row()
        row.operator("scene.clear")

###############################################################################

class ConstructNew(ASURT_Panel):
    bl_label = "Construct New Model"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        split = layout.split()        
        col = split.column(align=True)
        col.label(text="Blender Script:")
        col.operator("bpy.ops.buttons.file_browse")
        col.prop(context.scene, 'scpt_path')
        
        split = layout.split()
        col = split.column(align=True)
        col.label(text="Configuration Data:")
        col.operator("bpy.ops.buttons.file_browse")
        col.prop(context.scene, 'cfg_path')
        
        row = layout.row()
        row.prop(context.scene, 'subsys_id')
        
        row = layout.row()
        row.operator("object.load_mbsmodel")
        
