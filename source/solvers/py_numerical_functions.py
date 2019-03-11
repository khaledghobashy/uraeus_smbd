# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:24:10 2019

@author: khaled.ghobashy
"""
from collections import namedtuple
import numpy as np

###############################################################################
###############################################################################
#class lookup_table(object):
#    import pandas as pd
#    from scipy.interpolate import interp1d
#    import matplotlib.pyplot as plt
#
#    def __init__(self,name):
#        self.name = name
#        
#    def read_csv(self,csv_file):
#        self.data = pd.read_csv(csv_file,header=None,names=['x','y'])
#        self.x = self.data.x
#        self.y = self.data.y
#    
#    def draw(self):
#        plt.figure(figsize=(10,6))
#        plt.plot(self.x,self.y)
#        plt.legend()
#        plt.grid()
#        plt.show()
#    
#    def get_interpolator(self):
#        self.interp = f = interp1d(self.x,self.y,kind='cubic')
#        return f
        

###############################################################################
###############################################################################

def skew_matrix(v):
    vs = np.array([[0,-v[2,0],v[1,0]],
                 [v[2,0],0,-v[0,0]],
                 [-v[1,0],v[0,0],0]])
    return vs

def orthogonal_vector(v):
    dummy = np.ones((3,1))
    i = np.argmax(np.abs(v))
    dummy[i] = 0
    m = np.linalg.multi_dot([skew_matrix(v),dummy])
    return m

def triad(v1,v2=None):
    k = v1/np.linalg.norm(v1)
    
    if v2 is not None:
        i = v2/np.linalg.norm(v2)
    else:
        i = orthogonal_vector(k)
        i = i/np.linalg.norm(i)
    
    j = np.linalg.multi_dot([skew_matrix(k),i])
    j = j/np.linalg.norm(j)
    
    m = np.concatenate([i,j,k],axis=1)
    return m

###############################################################################
###############################################################################

def mirrored(v):
    if v.shape != (3,1):
        return v
    else:
        m = np.array([[1,0,0],[0,-1,0],[0,0,1]],dtype=np.float64)
        return m.dot(v)

def centered(*args):
    p = np.sum(args,0)
    p = p/len(args)
    return p

def oriented(*args):
    if len(args) == 2:
        v = args[1] - args[0]
    elif len(args) == 3:
        a1 = args[1] - args[0]
        a2 = args[2] - args[0]
        v = np.cross(a2,a1,axisa=0,axisb=0)
        v = np.reshape(v,(3,1))
    v = v/np.linalg.norm(v)
    return v

###############################################################################
###############################################################################

geometry = namedtuple('geometry',['R','P','m','J'])

def cylinder_geometry(arg1,arg2,ro=10,ri=0):
    v = arg2-arg1
    l = np.linalg.norm(v)
    frame = triad(v)
    
    vol = (np.pi/4)*(ro-ri)**2*l*10**-3
    m   = 7.9*vol
    Jzz = (m/2)*(ro**2+ri**2)
    Jxx = Jyy = (m/12)*(3*(ro**2+ri**2)+(l**2))
    
    
    R = centered(arg1,arg2)
    P = dcm2ep(frame)
    J = np.diag([Jxx,Jyy,Jzz])
    
    J = A(P).dot(J).dot(A(P).T)
    P = np.array([[1],[0],[0],[0]],dtype=np.float64)
    
    return geometry(R,P,m,J)


def triangular_prism(p1,p2,p3,thickness=10):
    v1 = p1-p2
    v2 = p1-p3
    v3 = p2-p3
    
    l1 = np.linalg.norm(v1) # assuming this is the base, where p3 is the vertix
    l2 = np.linalg.norm(v2)
    l3 = np.linalg.norm(v3)
    pr = (l1+l2+l3)/2 # half the premiter
    
    # The normal height of the vertix from the base.
    theta = np.arccos((v1.T.dot(v2))/(l1*l2))
    height = l2*np.sin(theta)
    area   = np.sqrt(pr*(pr-l1)*(pr-l2)*(pr-l3))
    volume = area*thickness
    density = 7.9
        
    # Creating a centroidal reference frame with z-axis normal to triangle
    # plane and x-axis oriented with the selected base vector v1.
    n = oriented(p1,p2,p3)
    frame = triad(n,v1)
    
    # Calculating the principle inertia properties "moment of areas" at the
    # geometry centroid.
    a   = v2.T.dot(v1) # Offset of p3 from p1 projected on v1.
    Ixc = (l1*height**3)/36
    Iyc = ((l1**3*height)-(l1**2*height*a)+(l1*height*a**2))/36
    Izc = ((l1**3*height)-(l1**2*height*a)+(l1*height*a**2)+(l1*height**3))/36
    
    # Calculating moment of inertia from the moment of area
    m = density*volume*1e-3
    J = m*np.diag([float(i) for i in [Ixc,Iyc,Izc]])
    R = centered(p1,p2,p3)
    P = dcm2ep(frame)
    
    J = A(P).dot(J).dot(A(P).T)
    P = np.array([[1],[0],[0],[0]],dtype=np.float64)
    
    return geometry(R,P,m,J)


def composite_geometry(*geometries):
    
    # composite body total mass as the sum of it's subcomponents
    m = sum([i.m for i in geometries])
    # center of mass vector relative to the origin
    R = (1/m) * sum([g.m*g.R for g in geometries])
    
    J = sum([ A(g.P).dot(g.J).dot(A(g.P).T) 
            + g.m*(np.linalg.norm(g.R-R)**2*np.eye(3)-(g.R-R).dot((g.R-R).T)) 
            for g in geometries])
    
    P = np.array([[1],[0],[0],[0]],dtype=np.float64)
    
    return geometry(R,P,m,J)

###############################################################################
###############################################################################

def dcm2ep(dcm):
    ''' 
    extracting euler parameters from a transformation matrix
    Note: this is not fully developed.
    ===========================================================================
    inputs  : 
        A   : The transofrmation matrix
    ===========================================================================
    outputs : 
        p   : set containing the four parameters
    ===========================================================================
    '''
    trA = dcm.trace()
    e0s = (trA+1)/4
    
    e1s = (2*dcm[0,0]-trA+1)/4
    e2s = (2*dcm[1,1]-trA+1)/4
    e3s = (2*dcm[2,2]-trA+1)/4
    
    # Case 1 : e0 != zero
    if e0s == max([e0s,e1s,e2s,e2s,e3s]):
        e0 = abs(np.sqrt(e0s))
        e1 = (dcm[2,1]-dcm[1,2])/(4*e0)
        e2 = (dcm[0,2]-dcm[2,0])/(4*e0)
        e3 = (dcm[0,1]-dcm[1,0])/(4*e0)
        
    elif e1s == max([e0s,e1s,e2s,e2s,e3s]):
        e1 = abs(np.sqrt(e1s))
        e0 = (dcm[2,1]-dcm[1,2])/(4*e1)
        e2 = (dcm[0,1]+dcm[1,0])/(4*e1)
        e3 = (dcm[0,2]+dcm[2,0])/(4*e1)
    
    elif e2s == max([e0s,e1s,e2s,e2s,e3s]):
        e2 = abs(np.sqrt(e2s))
        e0 = (dcm[0,2]-dcm[2,0])/(4*e2)
        e1 = (dcm[0,1]+dcm[1,0])/(4*e2)
        e3 = (dcm[2,1]+dcm[1,2])/(4*e2)
    
    elif e3s == max([e0s,e1s,e2s,e2s,e3s]):
        e3 = abs(np.sqrt(e3s))
        e0 = (dcm[0,1]-dcm[1,0])/(4*e3)
        e1 = (dcm[0,2]+dcm[2,0])/(4*e3)
        e2 = (dcm[2,1]+dcm[1,2])/(4*e3)
    
    p = np.array([[e0],[e1],[e2],[e3]])
    return p


def E(p):
    ''' 
    A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the global coordinate system.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
    ===========================================================================
    outputs : 
        E   : 3x4 ndarray
    ===========================================================================
    '''
    e0,e1,e2,e3 = p.flatten()
    m = np.array([[-e1, e0,-e3, e2],
                [-e2, e3, e0,-e1],
                [-e3,-e2, e1, e0]])
    return m

def G(p):
    # Note: This is half the G_bar given in shabana's book
    ''' 
    A property matrix of euler parameters. Mostly used to transform between the
    cartesian angular velocity of body and the euler-parameters time derivative
    in the body coordinate system.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
    ===========================================================================
    outputs : 
        G   : 3x4 ndarray
    ===========================================================================
    '''
    e0,e1,e2,e3 = p.flatten()
    m = np.array([[-e1, e0, e3,-e2],
                [-e2,-e3, e0, e1],
                [-e3, e2,-e1, e0]])
    return m

def A(p):
    '''
    evaluating the transformation matrix as a function of euler parameters
    Note: The matrix is defined as a prodcut of the two special matrices 
    of euler parameters, the E and G matrices. This function is faster.
    ===========================================================================
    inputs  : 
        p   : set "any list-like object" containing the four parameters
    ===========================================================================
    outputs : 
        A   : The transofrmation matrix 
    ===========================================================================
    '''
    return E(p).dot(G(p).T)



def B(p,a):
    ''' 
    This matrix represents the variation of the body orientation with respect
    to the change in euler parameters. This can be thought as the jacobian of
    the A.dot(a), where A is the transformation matrix in terms of euler 
    parameters.
    ===========================================================================
    inputs  : 
        p   : set containing the four parameters
        a   : vector 
    ===========================================================================
    outputs : 
        B   : 3x4 ndarray representing the jacobian of A.dot(a)
    ===========================================================================
    '''
    e0,e1,e2,e3=p.flatten()
    e = np.array([[e1],[e2],[e3]])
    a = np.array(a).reshape((3,1))
    a_s = skew_matrix(a)
    I = np.eye(3)
    e_s = skew_matrix(e)
    
    m = 2*np.bmat([[(e0*I+e_s).dot(a), e.dot(a.T)-(e0*I+e_s).dot(a_s)]])
    
    return m


