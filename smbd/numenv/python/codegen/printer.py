# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:22:44 2019

@author: khale
"""

# Third party imports
import sympy as sm
import numpy as np
from sympy.printing.ccode import C99CodePrinter

class npsc_printer(C99CodePrinter):

        
    def _print_ZeroMatrix(self,expr):
        return 'np.zeros(%s, dtype=np.float64)'%(expr.shape)
    
    def _print_zero_matrix(self,expr):
        return 'np.zeros(%s, dtype=np.float64)'%(expr.shape)
        
    def _print_AbstractMatrix(self,expr):
        args = ','.join([self._print(i) for i in expr.args])
        name = expr.__class__.__name__
        name = (name.lower() if len(name)>1 else name)
        return '%s(%s)'%(name,args)
    
    
    def _print_Simple_geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        return '%s(%s)'%(expr_lowerd,(self._print(expr.args)))
    
    def _print_Cylinder_Geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        return '%s%s'%(expr_lowerd, (self._print(expr.args)))
    
    def _print_Triangular_Prism(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        return '%s%s'%(expr_lowerd, (self._print(expr.args)))
    
    def _print_Sphere_Geometry(self, expr):
        expr_lowerd = expr.__class__.__name__.lower()
        args = [self._print(i) for i in expr.args]
        return '%s(%s, %s)'%(expr_lowerd, *args)
    
    def _print_Equal_to(self,expr):
        return '%s'%self._print(expr.args[0])
    
    def _print_matrix_symbol(self,expr):
        return '%s'%expr._raw_name
    
    def _print_dcm(self,expr):
        return '%s'%expr._raw_name
    
    def _print_base_vector(self,expr,**kwargs):
        return '%s[:,%s:%s]'%(expr.frame.name,*expr.slice)
    
    def _print_vector(self,expr):
        return '%s'%expr._raw_name
    
    def _print_quatrenion(self,expr):
        return '%s'%expr._raw_name
    
    def _print_Mul(self,expr):
        return '*'.join([self._print(i) for i in expr.args])
    
    def _print_MatMul(self,expr):
        scalars = []
        vectors = []
        express = []
        args = expr.args
        
        for i in args:
            #print(i)
            if isinstance(i,sm.MatrixExpr) and not isinstance(i,sm.MatMul):
                #print('vector: %s'%i)
                #print(type(i))
                vectors.append(self._print(i))
            elif isinstance(i,sm.MatMul):
                #print('MatMul: %s'%i)
                vectors.append(self._print_MatMul(i))
            elif isinstance(i,sm.Number):
                #print('Scalar: %s'%i)
                scalars.append(self._print(i))
            else:
                #print('Expression: %s'%i)
                express.append(self._print(i))
        
        if len(scalars)==0:
            s = ''
        else:
            s = '*'.join(scalars)+'*'
            
        if len(vectors)>1:
            v = 'multi_dot([%s])'%','.join(vectors)
        else:
            v = str(vectors[0])
            
        if len(express)==0:
            e = ''
        else:
            e = '*'.join(express)+'*'
        #print('end \n')
        return e + s + v 
        
    
    def _print_Identity(self,expr):
        shape = expr.args[0]
        return 'np.eye(%s, dtype=np.float64)'%shape
    
    def _print_MatExpr(self,expr):
        return '%s'%expr._ccode()
    
    
    def _print_tuple(self,expr):
        if len(expr)>2:
            return super()._print_tuple(expr)
        else:
            return str(expr[0]) + ' = ' + self._print(expr[1])
    
    def _print_Equality(self,expr):
        args = expr.args
        return self._print(args[0]) + ' = ' + self._print(args[1])
    
    def _print_MatAdd(self,expr):
        nested_operations = [self._print(i) for i in expr.args]
        value = ' + '.join(nested_operations)
        return '(%s)'%value
    
    def _print_BlockMatrix(self,expr):
        blocks = []
        rows,cols = expr.blockshape
        for r in np.arange(rows):
            row = []
            for c in np.arange(cols):
                row.append(self._print(expr.blocks[r,c]))
            string = ','.join(row)
            blocks.append('[%s]'%string)
        return 'np.bmat([%s])'%(','.join(blocks))
    
    def _print_MatrixSlice(self,expr):
        m, row_slice, col_slice = expr.args
        row_slice = '%s:%s'%(*row_slice[:-1],)
        col_slice = '%s:%s'%(*col_slice[:-1],)
        return '%s[%s,%s]'%(self._print(m),row_slice,col_slice)
    
    def _print_MutableDenseMatrix(self,expr):
        elements = expr.tolist()
        return 'np.array(%s, dtype=np.float64)'%(elements)
    
    def _print_ImmutableDenseMatrix(self,expr):
        elements = expr.tolist()
        return 'np.array(%s, dtype=np.float64)'%(elements)
    
    def _print_MutableSparseMatrix(self,expr):
        rows = []
        cols = []
        data = []
        for i,j,v in expr.row_list():
            rows.append(i)
            cols.append(j)
            data.append(v)
        
        rows_print = 'np.array([' + ','.join([self._print(i) for i in rows]) + '])'
        cols_print = 'np.array([' + ','.join([self._print(i) for i in cols]) + '])'
        data_print = '[' + ','.join([self._print(i) for i in data]) + ']'
        code_block = '\n'.join([rows_print,cols_print,data_print])
        return code_block
    
    def _print_UndefinedFunction(self,expr):
        return '%s'%expr
    
    def _print_Function(self,expr):
        func = expr.__class__.__name__
        args = ','.join([self._print(arg) for arg in expr.args])
        return '%s(%s)'%(func,args)
        
    def _print_transpose(self,expr):
        return '%s'%(*[self._print(i) for i in expr.args],)

    def _print_Derivative(self, expr):
        func = expr.args[0].__class__.__name__
#        func = self._print(func).strip(str(func.args))
        return 'derivative(%s, t, 0.1, %s)'%(func, expr.args[1][1])
    
    def _print_Lambda(self, obj):
        args, expr = obj.args
        if len(args) == 1:
            return "lambda %s : %s" % (self._print(args.args[0]), self._print(expr))
        else:
            arg_string = ", ".join(self._print(arg) for arg in args)
            return "lambda (%s) : %s" % (arg_string, self._print(expr))


        