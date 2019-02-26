# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:09:53 2019

@author: khale
"""

import os
import sys

def get_pkg_path():
    fullpath = os.path.dirname(__file__)
    print(fullpath)
#    path2pkg = []
#    imports = []
#    d = full_path.index('asurt_cdt_symbolic')
#    path2pkg = full_path[0:d+1]
#    imports  = full_path[d+1:]
#    file = imports[-1].split('.')[0]
#    del imports[-1]
#    path2pkg_str = os.path.join(*path2pkg)
#    imports_str  = '.'.join(imports)
#    sys.path.append(r'%s'%path2pkg_str)
#    print('from %s import %s'%(imports_str,file))
    
get_pkg_path()
