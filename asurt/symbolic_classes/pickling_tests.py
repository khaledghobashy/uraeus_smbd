# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:01:46 2019

@author: khale
"""

import pickle
import cloudpickle
from IPython.display import display

import source.symbolic_classes.abstract_matrices as am
from source.symbolic_classes.bodies import body, ground

global_instance = am.global_frame('G')
am.reference_frame.set_global_frame(global_instance)

m = am.reference_frame('m')
a = body('a')

i = am.base_vector(m,'i')

obj = a

pickled = cloudpickle.dumps(obj)
unpickled = pickle.loads(pickled) 

display(unpickled)
