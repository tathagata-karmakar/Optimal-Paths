#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:04:18 2025

@author: tatha_k
"""

import os,sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
from scipy.integrate import simpson as intg
#from google.colab import files
#from google.colab import drive
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)

from numba import njit
import numba as nb

@njit(inline = 'always')
def Fidelity_PS(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  fid = np.trace(np.dot(rho_f_simul, rho_f).real)
  return fid

@njit(inline = 'always')
def Tr_Distance(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  #dist = jnp.sqrt(jnp.trace(delrho2).real)
  return -Fidelity_PS(rho_f_simul, rho_f)

@njit(inline = 'always')
def G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1):
    G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
    G011 = G01+dt*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*tau))
    k101 = k10+dt*k01
    k011 = k01-dt*(1+2*l1)*k10
    G201 = G20+dt*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*tau))
    G111 = G11+dt*(G02-(1+2*l1)*G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*tau))
    G021 = G02+dt*(-2*(1+2*l1)*G11+csth*(snth*k02+csth*k11-r*k01)/(2*tau))
    k201 = k20+dt*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/tau)
    k111 = k11+dt*(-(1+2*l1)*k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/tau)
    k021 = k02+dt*(-2*(1+2*l1)*k11+2*csth*(-snth*G02-csth*G11+r*G01)/tau)  
    return G101, G011, k101, k011, G201, G111, G021, k201, k111, k021

@njit(inline = 'always')
def G_k_updates_first_order(G10, G01, k10, k01, csth, snth, dt, tau, l1):
    G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
    G011 = G01+dt*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*tau))
    k101 = k10+dt*k01
    k011 = k01-dt*(1+2*l1)*k10
    return G101, G011, k101, k011

@njit(inline = 'always')
def Fac2_Strato(Id, X, P, X2, CXP, P2, csth, snth, tau, dt, r):
    return Id-dt*(X2*csth**2+csth*snth*CXP+P2*snth**2)/(4*tau)+dt*r*(csth*X+snth*P)/(2*tau)

@njit(inline = 'always')
def rho_update_control_NB(X, P, H, X2, CXP, P2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, ts, tau, dt, Id): #Optimal control integration with \lambda_1=0
  #X, P, H, X2, CXP, P2, rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, j, Id = Input_Initials
  AGamma = (G10**2-G01**2-G20+G02)/2.0
  BGamma = G10*G01-G11
  theta = np.arctan2(BGamma, AGamma)/2.0
  csth, snth = np.cos(theta), np.sin(theta)
  r = csth*G10+snth*G01
  l1 = -l1max*np.sign(k20)
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  #t = ts[j]
  
  G101, G011, k101, k011, G201, G111, G021, k201, k111, k021 = G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1)
  '''
  G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
  G011 = G01+dt*(-G10+csth*(csth*k10+snth*k01)/(4*tau))
  k101 = k10+dt*k01
  k011 = k01-dt*k10
  G201 = G20+dt*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*tau))
  G111 = G11+dt*(G02-G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*tau))
  G021 = G02+dt*(-2*G11+csth*(snth*k02+csth*k11-r*k01)/(2*tau))
  k201 = k20+dt*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/tau)
  k111 = k11+dt*(-k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/tau)
  k021 = k02+dt*(-2*k11+2*csth*(-snth*G02-csth*G11+r*G01)/tau)  
  '''
  
  #H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  #Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  #read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  #rho_update =H_update+Lind_update+read_update
  H1 = H+l1*X2
  #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
  Fac2 = Fac2_Strato(Id, X, P, X2, CXP, P2, csth, snth, tau, dt, r)#Id-dt*(X2*csth**2+csth*snth*CXP+P2*snth**2)/(4*tau)+dt*r*(csth*X+snth*P)/(2*tau)
  rho1 = np.dot(np.dot(Fac2-dt*1j*H1,rho),Fac2+dt*1j*H1)
  tmptr = np.trace(rho1)
  rho1 = rho1/tmptr
  #rho1 = rho + rho_update*dt
  #Idth1 = Idth#+1e0*dt*(r**2-2*r*expL)/(2*tau)
  return rho1, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021


@njit(inline = 'always')
def OPsoln_control_NB(Initials, X, P, H, X2, CXP, P2, rho_i, l1max, ts, dt,  tau, Idmat,  Id):
  #I_tR = jnp.array([0.0])
  '''
  G10 = jnp.matmul(Idmat[0], Initials)
  G01 = jnp.matmul(Idmat[1], Initials)
  k10 = jnp.matmul(Idmat[2], Initials)
  k01 = jnp.matmul(Idmat[3], Initials)
  G20 = jnp.matmul(Idmat[4], Initials)
  G11 = jnp.matmul(Idmat[5], Initials)
  G02 = jnp.matmul(Idmat[6], Initials)
  k20 = jnp.matmul(Idmat[7], Initials)
  k11 = jnp.matmul(Idmat[8], Initials)
  k02 = jnp.matmul(Idmat[9], Initials)
  '''
  G10 = Initials[0]
  G01 = Initials[1]
  k10 = Initials[2]
  k01 = Initials[3]
  G20 = Initials[4]
  G11 = Initials[5]
  G02 = Initials[6]
  k20 = Initials[7]
  k11 = Initials[8]
  k02 = Initials[9] 
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  #Idth = 0.0
  while (k1<len(ts)):
      rho, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, = rho_update_control_NB(X, P, H, X2, CXP, P2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, ts, tau, dt, Id)
      k1+=1
  #X, P, H, X2, CXP, P2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_control_l10,(X, P, H, X2, CXP, P2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,  ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho#, Idth

@njit(inline = 'always')
def CostF_control_NB(Initials, X, P, H, X2, CXP, P2,  rho_i, rho_f, l1max, ts, dt, tau, Idmat, Id):
  rho_f_simul = OPsoln_control_NB(Initials, X, P, H, X2, CXP, P2, rho_i, l1max, ts, dt, tau, Idmat, Id)
  return Tr_Distance(rho_f_simul, rho_f), rho_f_simul

'''
@njit(inline = 'always')
def SA(tempi, tempf, nsteps, Initials_c, stepsize, ):
    n=0
    while temp>tempf and (n<nsteps):
      stime = time.time()
      Initials_n = Initials_c+step_size*(np.random.rand(10)-0.5)
      cost_n, rhotmp = CostF_control_NB(Initials_n, jnpX, jnpP, jnpH, jnpX2, jnpCXP, jnpP2, jnp_rho_i, jnp_rho_f, l1max, ts, dt, tau, Idmat, jnpId)
      if (cost_n<cost_b):
          #if cost_n<=2.0 and  (J_n<=J_b):
              #Initials, cost_b, J_b = Initials_n, cost_n, J_n
          #elif cost_n>2.0:
          #print(Initials_n-Initials)
          Initials, cost_b = Initials_n, cost_n
          nb = n
      #print (nb, n,  -cost_b, temp) #Cost is the negative of fidelity
      diff = cost_n-cost_c
      metropolis = jnp.exp(-100*diff/temp)
      if (diff<0) or (jnp.array(np.random.rand())<metropolis):
          Initials_c, cost_c = Initials_n, cost_n
          temp = temp/(1+0.02*temp)
      else:
          temp = temp/(1-0.0002*temp)    
      #Initials = update_control2_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
      #cost_b, J_b = CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
      n+=1
      step_size = step_size/(1+0.0001*step_size)
      #print (CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId))
      #Initials = update_control_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
      #if (n>nsteps/4):
      #Initials = update_control2_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
      #print (Initials)
      print (nb, n,  -cost_b, temp, metropolis)
'''
