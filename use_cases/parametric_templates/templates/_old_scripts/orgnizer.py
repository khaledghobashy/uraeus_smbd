# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:11:26 2019

@author: khaled.ghobashy
"""

import os
import shutil
files = os.listdir()

for file in files:
    if not file.startswith('__') and not file.endswith('.py') and not os.path.isdir(file):
        folder = os.path.splitext(file)[0]
        shutil.copy(file, folder)
        
