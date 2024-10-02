#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:23:11 2024

@author: t_karmakar
"""

import os,sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
from scipy.integrate import simps as intg
#from google.colab import files
#from google.colab import drive
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
from OP_Functions import *
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax
from jax import device_put
from jax import make_jaxpr
from jax.scipy.special import logsumexp
from jax._src.nn.functions import relu,gelu
from functools import partial
import collections
from typing import Iterable
from jaxopt import OptaxSolver
import optax

def Fourier_fns(Ncos, t_i, t_f, t):
    tmp = np.zeros(10+2*Ncos)
    for i in range(2*Ncos):
        #if i==0:
            #tmp[i]=1.0
        if (i<=Ncos-1):
            tmp[i+10]=np.cos(2*np.pi*(i)*(t-t_i)/(t_f-t_i))#-1
        else:
            tmp[i+10]=np.sin(2*np.pi*(i+1-Ncos)*(t-t_i)/(t_f-t_i))
    return tmp

def Fourier_mat(Ncos, t_i, t_f, ts):
    tmp = np.zeros((len(ts),10+2*Ncos))
    for i in range(len(ts)):
        tmp[i,:]=Fourier_fns(Ncos,t_i, t_f, ts[i])
    return tmp
             
def Multiply_Mat(nvars, Ncos):
    tmp1 = np.zeros((nvars,nvars+2*Ncos))
    #tmp2 = np.zeros((4,10+2*Ncos), dtype=complex)
    zerovec = np.zeros(2*Ncos) 
    idmat = np.identity(nvars)
    for i in range(nvars):
        tmp1[i] = np.concatenate((idmat[i],zerovec))
    return tmp1

def rho_update_control_l10_2input(i, Input_Initials): #Optimal control integration with \lambda_1=0
  Initials, X, P, H, rho, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,   Fmat, ts, tau, dt, j, Ncos, Id = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  #theta = theta_t[j]
  theta = (np.pi/2.0)*jnp.tanh(2*jnp.matmul(Fmat[j],Initials)/np.pi)
  
  AGamma = (G10**2-G01**2-G20+G02)/2.0
  BGamma = G10*G01-G11
  theta_star = jnp.arctan2(BGamma, AGamma)/2.0
  #phi = theta+t
  
  #dphi  = jnp.array([1.0])#+0.1*jnp.tanh(10.0*(dphi -1))
  #dphi = jnp.array([1.0])
  
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  #csph, snph = jnp.cos(phi), jnp.sin(phi)
  #cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  #Mjump = -snth*X+csth*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expV = jnp.trace(jnp.matmul(Ljump2, rho)).real
  expL = csth*expX + snth*expP
  #exphi = 
  #expM = -snph*expX + csph*expP
  delL = Ljump - expL*Id
  delV = Ljump2-expV*Id
  #e_jphi = jnp.exp(-1j*phi)
  #delh_t_Mat = e_jphi*(delh_t_mat1+t*delh_t_mat2)
  #ht = jnp.matmul(delh_t_Mat, Initials) + e_jphi*I_t
  r = csth*G10+snth*G01
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

  
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho_update =H_update+Lind_update+read_update
  rho1 = rho + rho_update*dt
  
  Idth1 = Idth+dt*(theta_star-theta)**2
  #Idth1 = -(-(GLL-w**2/4.0)/(2*tau)-(kappaLL+2*r*w)/2.0-(kappaMM+2*v*z)/2.0)
  return (Initials, X, P, H, rho1, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021, Idth1, Fmat,   ts, tau, dt, j+1, Ncos, Id)


def OPsoln_control_l10_JAX_2input(Initials, X, P, H, rho_i, Fmat, ts, dt,  tau, Ncos, MMat1, Id):
  
  Idth =  0.0
  G10 = jnp.matmul(MMat1[0],Initials)#+jnp.array([0])
  G01 = jnp.matmul(MMat1[1],Initials)#+jnp.array([0])
  k10 = jnp.matmul(MMat1[2],Initials)#+jnp.array([0])
  k01 = jnp.matmul(MMat1[3],Initials)#+jnp.array([0])
  G20 = jnp.matmul(MMat1[4],Initials)#+jnp.array([0])
  G11 = jnp.matmul(MMat1[5],Initials)#+jnp.array([0])
  G02 = jnp.matmul(MMat1[6],Initials)#+jnp.array([0])
  k20 = jnp.matmul(MMat1[7],Initials)#+jnp.array([0])
  k11 = jnp.matmul(MMat1[8],Initials)#+jnp.array([0])
  k02 = jnp.matmul(MMat1[9],Initials)#+jnp.array([0])
  rho = rho_i
  k1=0
  Idth=0
  Initials, X, P, H,  rho, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,  Fmat, ts, tau, dt, k1, Ncos, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_control_l10_2input,(Initials, X, P, H,  rho, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,  Fmat, ts, tau, dt, k1, Ncos, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho, Idth

def CostF1_control_l10_2input(Initials,  X, P, H,  rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX_2input(Initials,  X, P, H, rho_i, Fmat, ts, dt, tau, Ncos, MMat1, Id)
  #print (Idth)
  return Tr_Distance(rho_f_simul, rho_f)+Idth

@jit
def update_control1_l10_2input(Initials, X, P, H, rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, Id,  step_size):
    grads=grad(CostF1_control_l10_2input)(Initials,  X, P, H, rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, Id)
    gg = np.ones(14+10)
    gg[10:]=0
    grads = grads*gg
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

def CostF2_control_l10_2input(Initials, X, P, H,  rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX_2input(Initials, X, P, H, rho_i, Fmat, ts, dt, tau, Ncos, MMat1, Id)
  #print (Idth)
  return Idth
 
@jit
def update_control2_l10_2input(Initials,  X, P, H, rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, Id,  step_size):
    grads=grad(CostF2_control_l10_2input)(Initials, X, P, H, rho_i, rho_f, Fmat,  ts, dt, tau, Ncos, MMat1, Id)
    gg = np.ones(14+10)
    gg[:10]=0
    grads = grads*gg
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

def CostFt_control_l10_2input(Initials, X, P, H,  rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX_2input(Initials,  X, P, H, rho_i, Fmat, ts, dt, tau, Ncos, MMat1, Id)
  #print (Idth)
  return Tr_Distance(rho_f_simul, rho_f), Idth

def OPintegrate_strat_2inputs(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Ncos, MMat1, Id):
  G10 = jnp.matmul(MMat1[0],Initials)#+jnp.array([0])
  G01 = jnp.matmul(MMat1[1],Initials)#+jnp.array([0])
  k10 = jnp.matmul(MMat1[2],Initials)#+jnp.array([0])
  k01 = jnp.matmul(MMat1[3],Initials)#+jnp.array([0])
  G20 = jnp.matmul(MMat1[4],Initials)#+jnp.array([0])
  G11 = jnp.matmul(MMat1[5],Initials)#+jnp.array([0])
  G02 = jnp.matmul(MMat1[6],Initials)#+jnp.array([0])
  k20 = jnp.matmul(MMat1[7],Initials)#+jnp.array([0])
  k11 = jnp.matmul(MMat1[8],Initials)#+jnp.array([0])
  k02 = jnp.matmul(MMat1[9],Initials)
  rho = rho_i
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  diff = np.zeros(len(ts))
  rtheta = np.zeros(len(ts))
  Hs = np.zeros(len(ts))
  #zerovec = np.zeros(2*Ncos)
  while (j<len(ts)):
      t = ts[j]
      theta = theta_t[j]
      csth, snth = np.cos(theta), np.sin(theta)
      AGamma = (G10**2-G01**2-G20+G02)/2.0
      BGamma = G10*G01-G11
      theta_star = jnp.arctan2(BGamma, AGamma)/2.0
      r = csth*G10+snth*G01
      
      rtheta[j] = r
      Ljump = csth*X+snth*P
      Ljump2 = np.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
      Mjump = -snth*X+csth*P
      Mjump2 = np.matmul(Mjump, Mjump)
      expX = np.trace(np.matmul(X, rho)).real
      expP = np.trace(np.matmul(P, rho)).real
      expV = np.trace(np.matmul(Ljump2, rho)).real
      expL = csth*expX + snth*expP
      #exphi = 
      expM = -snth*expX + csth*expP
      #print (expL)
      Q1[j]=expX
      Q2[j]=expP
      delL = Ljump - expL*Id
      #print (delL)
      Q4[j] = np.trace(np.matmul(np.matmul(X, P)+np.matmul(P,X),rho)).real/2.0-expX*expP
      Q3[j] = np.trace(np.matmul(np.matmul(X,X),rho)).real-expX**2
      Q5[j] = np.trace(np.matmul(np.matmul(P,P),rho)).real-expM**2
      delV = Ljump2-expV*Id
      H_update = -1j*(np.matmul(H, rho)-np.matmul(rho, H))
      Lind_update = (-np.matmul(delV, rho)-np.matmul(rho, delV))/(4*tau)
      read_update = r*(np.matmul(delL, rho)+ np.matmul(rho, delL))*(1.0/(2*tau))
      drho = H_update+Lind_update+read_update
      #print (Lind_update)
      #print (np.trace(drho).real)
      rho+=drho*dt
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
      G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = G101, G011, k101, k011, G201, G111, G021, k201, k111, k021
      #rho = rho/np.trace(rho)
      #rho=rho/np.trace(rho)
      diff[j]=theta_star-theta
      Hs[j] = 0# -(GLL-w**2/4.0)/(2*tau)-(kappaLL+2*r*w)/2.0-(kappaMM+2*v*z)/2.0
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, rho, rtheta, diff, Hs
