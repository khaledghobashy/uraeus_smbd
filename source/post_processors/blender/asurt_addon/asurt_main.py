bl_info = {
    "name": "ASURTCDT",
    "author": "Khaled Ghobashy",
    "version": (1, 0, 0),
    "blender": (2, 7, 9),
    "description": "Visualize Animations Generated by ASURTCDT",
    "category": "3D View"
    }

import sys
import os
import bpy
from bpy.types import Panel


#asurt_path = r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic'
asurt_path = 'E:\\Main\\asurt_cdt_symbolic'
if asurt_path not in sys.path:
    sys.path.append(asurt_path)


class ASURT_Panel(Panel):
    bl_space_type =  'VIEW_3D'
    bl_region_type = 'TOOLS'
    bl_label = "System Files"  
    bl_context = "objectmode"
    bl_category = "ASURT"

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
        
        split = layout.split()
        col = split.column(align=True)
        col.label(text="Simulation Data:")
        col.operator("bpy.ops.buttons.file_browse")
        col.prop(context.scene, 'sim_path')
        
        layout.label(text="")
        row = layout.row()
        row.operator("object.run_script")
        
        row = layout.row()
        row.operator("scene.clear")
        
        row = layout.row()
        row.operator("scene.reset_fields")
        

class runScript(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.run_script"
    bl_label = "Load Files"

    @classmethod
    def poll(cls, context):
        scene_vars = ['scpt_path','cfg_path','sim_path']
        files = [os.path.splitext(getattr(context.scene,i))[-1] for i in scene_vars]
        print(files)
        cond = ['.py','.csv','.csv'] == files
        return cond

    def execute(self, context):
        script = bpy.data.scenes["Scene"].scpt_path
        bpy.ops.script.python_file_run(filepath=script)
        return {'FINISHED'}

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

class reset_fields(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "scene.reset_fields"
    bl_label = "Clear Fields"

    @classmethod
    def poll(cls, context):
        scene_vars = ['scpt_path','cfg_path','sim_path']
        files = [os.path.isfile(getattr(context.scene,i)) for i in scene_vars]
        print(files)
        cond = False not in files
        return cond

    def execute(self, context):
        scene_vars = ['scpt_path','cfg_path','sim_path']
        for n in scene_vars:
            path = getattr(context.scene,n)
            path = os.path.dirname(path)
            path = os.path.join(path,'')
            setattr(context.scene,n,path)
        return {'FINISHED'}

def register():
    bpy.utils.register_module(__name__)
    
    bpy.types.Scene.scpt_path = bpy.props.StringProperty \
      (name = "",
      default = "%s//use_cases//generated_templates//blender//gen_scripts//"%asurt_path,
      description = "Define the script file of the system.",
      subtype = 'FILE_PATH')
    
    bpy.types.Scene.cfg_path = bpy.props.StringProperty \
      (name = "",
      default = "%s//use_cases//generated_templates//configurations//"%asurt_path,
      description = "Define the config. file of the system.",
      subtype = 'FILE_PATH')
      
    bpy.types.Scene.sim_path = bpy.props.StringProperty \
      (name = "",
      default = "%s//use_cases//simulations//"%asurt_path,
      description = "Define the simulation results file of the system.",
      subtype = 'FILE_PATH')
      
      
      
def unregister():
    bpy.utils.unregister_module(__name__)
    del bpy.types.Scene.scpt_path
    del bpy.types.Scene.cfg_path
    del bpy.types.Scene.sim_path
    

if __name__ == "__main__":
    register()

print("Loaded ASURT-CDT")
