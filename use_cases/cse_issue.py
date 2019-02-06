# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 08:45:20 2019

@author: khaled.ghobashy
"""

import sympy as sm

x,y,t = sm.symbols('x,y,t')
F = sm.Function('F')

eq = F(t) + 2*x + y**3 + 2*F(t).diff(t,1,evaluate=False) + 5*F(t).diff(t,2,evaluate=False)

cse_vars, cse_expr = sm.cse(eq,ignore=(t,))
