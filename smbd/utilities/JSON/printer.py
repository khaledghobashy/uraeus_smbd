# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:22:44 2019

@author: khale
"""

# Third party imports
import sympy as sm
import numpy as np
from sympy.printing.codeprinter import CodePrinter

class printer(CodePrinter):
    
    
    def _print_Simple_geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        return '%s(%s)'%(expr_lowerd,(self._print(expr.args)))
    
    """ def _print_Cylinder_Geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        return '%s%s'%(expr_lowerd, (self._print(expr.args))) """

    def _print_Cylinder_Geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()

        geo_type = '"type" : %s'%expr_lowerd
        geo_args = '"args" : %s'%([self._print(arg) for arg in expr.args])

        json_obj = '%s,\n%s'%(geo_type, geo_args)
        return json_obj
    
    def _print_Triangular_Prism(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        return '%s%s'%(expr_lowerd, (self._print(expr.args)))
    
    def _print_Sphere_Geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        args = [self._print(i) for i in expr.args]
        return '%s(%s, %s)'%(expr_lowerd, *args)
    
    def _print_Equal_to(self, expr):
        return '%s'%self._print(expr.args[0])
    
    


        