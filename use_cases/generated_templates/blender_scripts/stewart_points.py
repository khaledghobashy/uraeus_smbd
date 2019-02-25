
import os
import numpy as np
from source.post_processors.blender.objects import (cylinder_geometry,
                                                    composite_geometry,
                                                    triangular_prism)



path = os.path.dirname(__file__)

class blender_scene(object):

    def __init__(self):
        self.hps_middle_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_middle_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_upper_2 = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_middle_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_upper_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_links_ro = 1
        self.hps_bottom_1 = np.array([[0], [0], [0]],dtype=np.float64)
        self.s_rockers_ro = 1
        self.hps_bottom_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_upper_3 = np.array([[0], [0], [0]],dtype=np.float64)
        self.hps_bottom_2 = np.array([[0], [0], [0]],dtype=np.float64)                       

    

    def load_from_csv(self,csv_file):
        file_path = os.path.join(path,csv_file)
        dataframe = pd.read_csv(file_path,index_col=0)
        for ind in dataframe.index:
            shape = getattr(self,ind).shape
            v = np.array(dataframe.loc[ind],dtype=np.float64)
            v = np.resize(v,shape)
            setattr(self,ind,v)
        self._set_arguments()

    def _set_arguments(self):
        gms_rocker_1 = cylinder_geometry(self.hps_bottom_1,self.hps_middle_1,self.s_rockers_ro)
        gms_rocker_2 = cylinder_geometry(self.hps_bottom_2,self.hps_middle_2,self.s_rockers_ro)
        gms_link_1 = cylinder_geometry(self.hps_upper_1,self.hps_middle_1,self.s_links_ro)
        gms_link_2 = cylinder_geometry(self.hps_upper_2,self.hps_middle_2,self.s_links_ro)
        gms_rocker_3 = cylinder_geometry(self.hps_bottom_3,self.hps_middle_3,self.s_rockers_ro)
        gms_link_3 = cylinder_geometry(self.hps_upper_3,self.hps_middle_3,self.s_links_ro)
    

