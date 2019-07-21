#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 09:38:20 2019

@author: khaledghobashy
"""

import numpy as np
import scipy as sc
from scipy.integrate import ode

from smbd.numenv.python.numerics.misc import A, E, G, skew_matrix


def normalize(v):
    normalized = v/np.linalg.norm(v)
    return normalized


class abstract_tire(object):
    
    def __init__(self, P_carrier):
        self._set_SAE_LF(P_carrier)
        self._set_CPM_integrator(0, 0)
    
    def _process_wheel_kinematics(self, R_hub, P_hub, Rd_hub, Pd_hub, P_carrier):
        
        # Wheel Center Translational Velocity in Global Frame
        V_wc = Rd_hub
        
        # DCM of wheel carrier in Global Frame
#        A_carrier = A(P_carrier)

        # Penetration Length assuming flat horizontal ground
        self.ru = max(self.R_ul - R_hub[2,0], 0)
        
        # Position of contact point C relative to the wheel center
        R_C = np.array([[R_hub[0,0]], [R_hub[1,0]], [0]]) - R_hub
        
        # Tire Effective Radius as a ratio of loaded radius
        delta_r_eff = (self.R_ul - self.ru) * 0.025
        R_S = R_C + np.array([[0], [0], [-delta_r_eff]])

        
        # Rotational velocity vector of wheel in Global and Local frames.
        AngVel_Hub_GF = 2*G(P_hub)@Pd_hub # Global
        AngVel_Hub_LF = 2*E(P_hub)@Pd_hub # Local
        
        
        # Velocity of Slip point S relative to the wheel center
#        V_SW = skew_matrix(AngVel_Hub_GF).dot(R_S)
        
        # Velocity of Slip point S relative to the global origin in global frame
#        V_SG = V_wc + V_SW
        
        self._set_SAE_LF(P_carrier)
        
        # Circumferential Velocity
        Omega = AngVel_Hub_LF[1,0]
        Omega = self.SAE_LF.T.dot(A(P_carrier).T).dot(AngVel_Hub_GF)[1,0]
        V_C = Omega * np.linalg.norm(R_S)
        
        
        V_wc_SAE = self.SAE_LF.T.dot(A(P_carrier).T).dot(V_wc)
        
        V_sx = (V_wc_SAE[0,0] - V_C)
        V_sy =  V_wc_SAE[1,0]
        V_x  = abs(V_wc_SAE[0,0])
        
        self.V_sx = V_sx
        self.V_sy = V_sy
        self.V_x  = V_x 
        self.V_C  = V_C
        
#        vth = 3000
#        if  V_x <= vth:
#            print('SLOW !!')
#            self.S_lng = (2*V_sx)/(vth + (V_x**2/vth))
#            self.S_ltr = (2*V_sy)/(vth + (V_x**2/vth))
#        else:
#            self.S_lng = -V_sx/V_C
#            self.S_ltr = V_sy/V_x
        
        
        print('Omega_LF = %s'%Omega)
#        print('AngVel_Hub_LF = %s'%AngVel_Hub_LF.T)
#        print('V_WC = %s'%V_wc_SAE.T)
#        print('V_C = %s'%V_C)
#        print('V_sx = %s'%V_sx.T)
#        print('V_sy = %s'%V_sy.T)
#        print('V_x  = %s'%V_x)
#        print('S_lng = %s'%self.S_lng)
#
#        print('S_ltr = %s'%self.S_ltr)
#        print('SlipAngle = %s'%np.arctan(self.S_ltr))
    
    def _set_SAE_LF(self, P_carrier):
        
        A_carrier = A(P_carrier)
        
        x_vec = A_carrier[:,0:1]
        x_vec[2,0] = 0
        x_vec = -1* normalize(x_vec)
        y_vec = A_carrier[:,1:2]
        y_vec[2,0] = 0
        y_vec = normalize(y_vec)
        z_vec = np.array([[0],[0],[-1]])
        
        SAE_frame = np.concatenate([x_vec, y_vec, z_vec], axis=1)
    
        self.SAE_LF = A(P_carrier).T.dot(SAE_frame)
        self.X_SAE_LF = self.SAE_LF[:,0:1]
        self.Y_SAE_LF = self.SAE_LF[:,1:2]
        self.Z_SAE_LF = self.SAE_LF[:,2:3]
        
    
    def _integrate_CPM(self, t, V_sx, V_sy, Vx):
        self.u_intg.set_f_params(V_sx, self.sigma_k, Vx)
        u = self.u_intg.integrate(t)

        self.v_intg.set_f_params(V_sy, self.sigma_a, Vx)
        v = self.v_intg.integrate(t)
        
        self.ui = u
        self.vi = v

    def _set_CPM_integrator(self, u0, v0):
        self.u_intg = ode(self._CPM_ODE)
        self.u_intg.set_integrator('dopri5')
        self.u_intg.set_initial_value(u0)
        self.u_intg.set_f_params([0, self.sigma_k, 0])
        
        self.v_intg = ode(self._CPM_ODE)
        self.v_intg.set_integrator('dopri5')
        self.v_intg.set_initial_value(v0)
        self.v_intg.set_f_params([0, self.sigma_a, 0])
    
    def _CPM_ODE(self, t, y, slip_vel, relx, Vx):
        Vx = abs(Vx)
        dudt = -(1/relx)*Vx*y - slip_vel
        return dudt
    
    def _get_transient_slips(self, t, V_sx, V_sy, Vx):
        self._integrate_CPM(t, V_sx, V_sy, Vx)
        k = -self.ui/self.sigma_k
        a = self.vi/self.sigma_a
        
        print('k = %s'%k)
        print('a = %s'%a)
        
        return k, a
        

###############################################################################
###############################################################################

class brush_model(abstract_tire):
    
    def __init__(self, P_carrier):
        
        self.R_ul = 546
        self.mu = 0.85
        self.cp = 2000*1e3
        self.a  = 210
        
        self.kz = 650*1e6
        self.cz = 6*1e6
        
        C_fk = (2375*9.81*1e6*0.2)/(3*1e-2)
        C_fa = (150*1e6*1e2)/(np.deg2rad(2))
        C_fx = C_fy = self.cp
        
        self.sigma_k = (C_fk/C_fx)*1e-3
        self.sigma_a = (C_fa/C_fy)*1e-6
        
        super().__init__(P_carrier)
        
    
    def Eval_Forces(self, R_hub, P_hub, Rd_hub, Pd_hub, P_carrier, t):
        
        self._process_wheel_kinematics(R_hub, P_hub, Rd_hub, Pd_hub, P_carrier)
        
        self.Fz = self.kz * self.ru

        k, alpha = self._get_transient_slips(t, self.V_sx, self.V_sy, self.V_x)
        Theta   = (2/3)*((self.cp * self.a**2)/(self.mu*self.Fz))
        sigma_x = k/(1+k)# if abs(k)-1 >1e-3 else k
        sigma_y = alpha/(1+k) #if abs(k)-1 >1e-3 else alpha
        sigma   = np.sqrt(sigma_x**2 + sigma_y**2)
        TG = Theta*sigma

        sigma_vec = np.array([[sigma_x], [sigma_y]])
        
        if sigma <=1e-3:
            self.Fx = 0
            self.Fy = 0
            self.Mz = 0
            self._eval_GF_forces(R_hub, P_carrier)
            print('Tire_Ground Force = %s'%([self.Fx, self.Fy]))
        else:
            pneumatic_trail = (1/3)*self.a * ((1 - 3*abs(TG) + 3*TG**2 - abs(TG)**3) /
                                         (1 - abs(TG) + (1/3)*TG**2 ))
            if sigma <= 1/Theta:
                factor = (3*(TG) - 3*(TG)**2 + (TG)**3)
                force = self.mu * self.Fz * factor
            else:
                print('SLIDING !!')
                force = self.mu*self.Fz
            
            F  = force * normalize(sigma_vec)
            print('Tire_Ground Force = %s'%F.T)
            self.Fx = F[0,0] #- (0.02 * self.Fz)
            self.Fy = F[1,0]
            self.Mz = (- pneumatic_trail * self.Fy) * np.array([[0],[0],[1]])
            self._eval_GF_forces(R_hub, P_carrier)


    def _eval_GF_forces(self, R_hub, P_carrier):
        A_carrier = A(P_carrier)
        
        # SAE Frame Axes relative to global frame
        X_SAE_GF = A_carrier.dot(self.X_SAE_LF)
        X_SAE_GF[2,0] = 0
        X_SAE_GF = normalize(X_SAE_GF)
        
        Y_SAE_GF = A_carrier.dot(self.Y_SAE_LF)
        Y_SAE_GF[2,0] = 0
        Y_SAE_GF = normalize(Y_SAE_GF)
        
        R_pw = np.array([[R_hub[0,0]], [R_hub[1,0]], [0]]) - R_hub
        delta_r_eff = (self.R_ul - self.ru) * 0.025
        R_pw_eff = R_pw + np.array([[0], [0], [-delta_r_eff]])

        Force = self.Fx*X_SAE_GF + self.Fy*Y_SAE_GF + self.Fz*np.array([[0],[0],[1]])
        self.F = Force
        self.M = skew_matrix(R_pw_eff).dot(self.F) + self.Mz
        print('M = %s'%(self.M.T))
        
    
    def _rolling_resistance(self):
        pass


###############################################################################
###############################################################################

class interolator(abstract_tire):

    def __init__(self, P_carrier):
        self.R_ul = 546
        self.mu = 1.0
        
        self.kz = 650*1e6
        self.cz = 6*1e6
        
        self._set_SAE_LF(P_carrier)
        self._format_data()
    
    
    def Eval_Forces(self, R_hub, P_hub, Rd_hub, Pd_hub, P_carrier):
        
        self._process_wheel_kinematics(R_hub, P_hub, Rd_hub, Pd_hub, P_carrier)
        
        R_pw = np.array([[R_hub[0,0]], [R_hub[1,0]], [0]]) - R_hub
        vertical_force = self.kz * self.ru
        self.Fz = vertical_force
        
        k = abs(self.S_lng*100)
        print('k = %s'%k)
        if k <= 24.90091463:
            self.Fx = self.Fx_data(k) * max(self.Fz, 0) * 0.13
        else:
            print('SLIDING !!')
#            self.Fx = 0.06 * max(self.Fz, 0)
            self.Fx = self.Fx_data(24.90091463) * max(self.Fz, 0) * 0.13 *0.6
        
        self.Fy = 0
        
#        SAE_GF = A(P_carrier).dot(self.SAE_LF)
        self.Force_SAE_LF = np.array([[self.Fx], [self.Fy], [-self.Fz]])
#        self.F = SAE_GF.dot(self.Force_SAE_LF)
        self.F = np.array([[-self.Fx], [self.Fy], [self.Fz]])
        self.M = skew_matrix(R_pw).dot(self.F)
        print('M = %s'%(self.M.T))

        

    def _format_data(self):
        slip_ratio = np.array([0,0.316129501, 1.37542517, 2.342220425, 3.170783557, 
                               4.137371412, 5.011251452, 5.700960262, 6.390669073, 
                               7.263926912, 8.274587274, 9.193162021, 10.01985855, 
                               11.12094533, 12.1765078, 13.36885059, 15.06527916, 
                               16.30210926, 17.44726854, 18.54607392, 20.05625726, 
                               21.33695246, 22.93673262, 24.90091463])
    
        Fx_Fz = np.array([0,0.029478458, 0.09478458, 0.151020408, 0.198185941, 
                          0.25260771, 0.296145125, 0.328798186, 0.361451247, 
                          0.399546485, 0.439455782, 0.473922902, 0.504761905,
                          0.535600907, 0.568253968, 0.597278912, 0.63537415, 
                          0.653514739, 0.66984127, 0.680725624, 0.689795918, 
                          0.691609977, 0.684353741, 0.664399093])

        self.Fx_data = sc.interpolate.interp1d(slip_ratio, Fx_Fz)

    
       

