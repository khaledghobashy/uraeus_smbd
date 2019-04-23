# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 10:28:45 2019

@author: khaled.ghobashy
"""

from source.symbolic_classes.abstract_matrices import (reference_frame, 
                                                       vector, B)

def name_setter(obj,name):
    splited_name = name.split('.')
    obj._id_name = ''.join(splited_name[-1])
    obj.prefix  = '.'.join(splited_name[:-1])
    obj.prefix  = (obj.prefix+'.' if obj.prefix!='' else obj.prefix)
    obj._name   = name


def body_setter(obj,body,sym):
    setattr(obj,'_body_%s'%sym,body)
    
    attrs = ['R','Rd','P','Pd','A']
    for attr in attrs:
        v = getattr(body,attr)
        setattr(obj,'%s%s'%(attr,sym),v)
    
    fromat_ = (obj.prefix,body.id_name,obj.id_name)
    v_raw_name = '%subar_%s_%s'%fromat_
    v_frm_name = r'{%s\bar{u}^{%s}_{%s}}'%fromat_
    m_raw_name = '%sMbar_%s_%s'%fromat_
    m_frm_name = r'{%s\bar{M}^{%s}_{%s}}'%fromat_
    
    u_bar = vector(v_raw_name,body,v_frm_name)
    m_bar = reference_frame(m_raw_name,body,m_frm_name)
    
    setattr(obj,'u%s_bar'%sym,u_bar)
    setattr(obj,'m%s_bar'%sym,m_bar)
    
    Bu = B(body.P,u_bar)
    u = u_bar.express()
    
    setattr(obj,'Bu%s'%sym,Bu)
    setattr(obj,'u%s'%sym,u)
    
    
