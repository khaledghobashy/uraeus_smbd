import json
import sympy as sm

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
    
    