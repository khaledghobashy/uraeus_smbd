import json
import textwrap
import sympy as sm


class Encoder(object):

    def __init__(self):
        pass
    
    def encode(self, obj, nest=0):
        try:
            cls_name = obj.__class__.__name__
            method = getattr(self, '_encode_%s'%cls_name)
            return method(obj, nest)
        except AttributeError:
            raise NotImplementedError
    
    def _encode_dict(self, obj, nest=0):
        indent = (nest * 4 * ' ') if nest >0 else ''
        text = str(obj)
        text = ',\n'.join(text.split(','))
        #text.expandtabs()
        #text = textwrap.dedent(text)
        text = textwrap.indent(text, indent)
        text = '\n' + text + '\n'
        return text
    
    def _encode_Cylinder_Geometry(self, obj, nest=0):
        
        indent = (nest * 4 * ' ') if nest >0 else ''
        p1, p2, radius = obj.args
        
        args = {'p1': p1, 'p2': p2, 'radius': radius}
        
        text = \
        '''
        type: '{object_type}',
        args: {object_args}
        '''
        text.expandtabs()
        text = textwrap.dedent(text)
        text = textwrap.indent(text, indent)
        
        text = text.format(object_type=obj.__class__.__name__.lower(), object_args=self.encode(args, nest+1))
        text = '{' + text + '}'

        return text

class SMBDEncoder(json.JSONEncoder):
    
    def default(self, obj):
        
        try:
            cls_name = obj.__class__.__name__
            method = getattr(self, '_dump_%s'%cls_name)
            return method(obj)
        except AttributeError:
            pass
        
        else:
            return super().default(obj)
    
    
    def _dump_Cylinder_Geometry(self, obj):
        expr_lowerd = obj.__class__.__name__.lower()
        data = {'type': expr_lowerd,
                'args': [str(arg) for arg in obj.args]}
        return data
    
    def _dump_Symbol(self, obj):
        return str(obj)
    
    def _dump_Equality(self, obj):
        data = {self.default(obj.lhs): self.default(obj.rhs)}
        return data
    
    def _dump_MutableDenseMatrix(self, obj):
        data = [float(i) for i in obj]
        return data
    
    def _dump_ImmutableDenseMatrix(self, obj):
        data = [float(i) for i in obj]
        return data
    
    def _dump_One(self, obj):
        return float(obj)
    
    def _dump_Float(self, obj):
        return float(obj)
    
    def _dump_Integer(self, obj):
        return float(obj)
    
    def _dump_Zero(self, obj):
        return float(obj)
    
    