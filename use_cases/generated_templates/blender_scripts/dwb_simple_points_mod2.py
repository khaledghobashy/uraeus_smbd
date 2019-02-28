
import csv
import numpy as np
import bpy
from source.solvers.py_numerical_functions import centered, mirrored
from source.post_processors.blender.objects import (cylinder_geometry,
                                                    composite_geometry,
                                                    triangular_prism)



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

        self.s_strut_inner = 1*scale
        self.hpr_ucao = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.s_strut_outer = 1*scale
        self.hpr_strut_lca = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hpr_lcaf = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.s_tire_radius = 1*scale
        self.hpr_lcao = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hpr_tri = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hpr_wc = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hpr_ucar = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hpr_strut_chassis = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.hpr_lcar = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.s_thickness = 1*scale
        self.hpr_tro = np.array([[0], [0], [0]],dtype=np.float64)*scale
        self.s_links_ro = 1*scale
        self.hpr_ucaf = np.array([[0], [0], [0]],dtype=np.float64)

        self._inputs = ['s_strut_inner', 'hpr_ucao', 's_strut_outer', 'hpr_strut_lca', 'hpr_lcaf', 's_tire_radius', 'hpr_lcao', 'hpr_tri', 'hpr_wc', 'hpr_ucar', 'hpr_strut_chassis', 'hpr_lcar', 's_thickness', 'hpr_tro', 's_links_ro', 'hpr_ucaf']
        self.geometries = {'gmr_uca': 'rbr_uca', 'gml_uca': 'rbl_uca', 'gmr_lca': 'rbr_lca', 'gml_lca': 'rbl_lca', 'gmr_upright': 'rbr_upright', 'gml_upright': 'rbl_upright', 'gmr_upper_strut': 'rbr_upper_strut', 'gml_upper_strut': 'rbl_upper_strut', 'gmr_lower_strut': 'rbr_lower_strut', 'gml_lower_strut': 'rbl_lower_strut', 'gmr_tie_rod': 'rbr_tie_rod', 'gml_tie_rod': 'rbl_tie_rod', 'gmr_tire': 'rbr_hub', 'gml_tire': 'rbl_hub'}

    
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
        bpy.context.scene.render.frame_map_new = 24*2
        bpy.context.scene.frame_end = bpy.context.scene.render.frame_map_new

    def create_scene(self):
        self.hpl_ucao = mirrored(self.hpr_ucao)
        self.hpl_ucar = mirrored(self.hpr_ucar)
        self.hpl_ucaf = mirrored(self.hpr_ucaf)
        self.hpl_tro = mirrored(self.hpr_tro)
        self.hpl_tri = mirrored(self.hpr_tri)
        self.hpl_lcao = mirrored(self.hpr_lcao)
        self.hpl_lcar = mirrored(self.hpr_lcar)
        self.hpl_lcaf = mirrored(self.hpr_lcaf)
        self.hpr_strut_mid = centered(self.hpr_strut_chassis,self.hpr_strut_lca)
        self.hpl_strut_lca = mirrored(self.hpr_strut_lca)
        self.hpl_strut_chassis = mirrored(self.hpr_strut_chassis)
        self.hpl_strut_mid = centered(self.hpl_strut_chassis,self.hpl_strut_lca)
        self.gmr_upright = triangular_prism(self.hpr_ucao,self.hpr_wc,self.hpr_lcao,self.s_thickness)
        self.R_rbr_upright = centered(self.hpr_ucao,self.hpr_wc,self.hpr_lcao)
        self.hpl_wc = mirrored(self.hpr_wc)
        self.gml_upright = triangular_prism(self.hpl_ucao,self.hpl_wc,self.hpl_lcao,self.s_thickness)
        self.R_rbl_upright = mirrored(self.R_rbr_upright)
        self.gml_uca = triangular_prism(self.hpl_ucaf,self.hpl_ucar,self.hpl_ucao,self.s_thickness)
        self.gml_tie_rod = cylinder_geometry(self.hpl_tri,self.hpl_tro,self.s_links_ro)
        self.gmr_uca = triangular_prism(self.hpr_ucaf,self.hpr_ucar,self.hpr_ucao,self.s_thickness)
        self.gml_lca = triangular_prism(self.hpl_lcaf,self.hpl_lcar,self.hpl_lcao,self.s_thickness)
        self.gmr_lca = triangular_prism(self.hpr_lcaf,self.hpr_lcar,self.hpr_lcao,self.s_thickness)
        self.gmr_upper_strut = cylinder_geometry(self.hpr_strut_chassis,self.hpr_strut_mid,self.s_strut_outer)
        self.gmr_tie_rod = cylinder_geometry(self.hpr_tri,self.hpr_tro,self.s_links_ro)
        self.gmr_lower_strut = cylinder_geometry(self.hpr_strut_mid,self.hpr_strut_lca,self.s_strut_inner)
        self.gml_upper_strut = cylinder_geometry(self.hpl_strut_chassis,self.hpl_strut_mid,self.s_strut_outer)
        self.gmr_tire = cylinder_geometry(self.hpr_wc,self.R_rbr_upright,self.s_tire_radius)
        self.gml_lower_strut = cylinder_geometry(self.hpl_strut_mid,self.hpl_strut_lca,self.s_strut_inner)
        self.gml_tire = cylinder_geometry(self.hpl_wc,self.R_rbl_upright,self.s_tire_radius)

blend = blender_scene('SU.')
blend.get_data(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\generated_templates\configurations\dwb_st500_axletech_2.csv')
blend.create_scene()
blend.load_anim_data(r'C:\Users\khaled.ghobashy\Desktop\Khaled Ghobashy\Mathematical Models\asurt_cdt_symbolic\use_cases\simulations\sim_dwb_st500_axletech_temp.csv')
