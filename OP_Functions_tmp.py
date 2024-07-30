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
            tmp[i+10]=np.cos(2*np.pi*(i+1)*(t-t_i)/(t_f-t_i))-1
        else:
            tmp[i+10]=np.sin(2*np.pi*(i+1-Ncos)*(t-t_i)/(t_f-t_i))
    return tmp

def Fourier_mat(Ncos, t_i, t_f, ts):
    tmp = np.zeros((len(ts),10+2*Ncos))
    for i in range(len(ts)):
        tmp[i,:]=Fourier_fns(Ncos,t_i, t_f, ts[i])
    return tmp
             
def Multiply_Mat(tau, Ncos):
    tmp1 = np.zeros((10,10+2*Ncos))
    tmp2 = np.zeros((4,10+2*Ncos), dtype=complex)
    zerovec = np.zeros(2*Ncos) 
    tmp1[0] = np.concatenate((np.array([1.0,0,0,0,0,0,0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[1] = np.concatenate((np.array([0,1.0,0,0,0,0,0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[2] = np.concatenate((np.array([0,0,1.0,0,0,0,0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[3] = np.concatenate((np.array([0,0,0,1.0,0,0,0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[4] = np.concatenate((np.array([0,0,0,0,1,-1.0,0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[5] = np.concatenate((np.array([0,0,0,0,0,0,1.0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[6] = np.concatenate((np.array([0,0,0,0,1,1.0,0,0,0,0]),zerovec))#+jnp.array([0])
    tmp1[7] = np.concatenate((np.array([0,0,0,0,0,0,0,1,-1.0,0]),zerovec))#+jnp.array([0])
    tmp1[8] = np.concatenate((np.array([0,0,0,0,0,0,0,1,1.0,0]),zerovec))#+jnp.array([0])
    tmp1[9] = np.concatenate((np.array([0,0,0,0,0,0,0,0,0,1]),zerovec))#+jnp.array([0])
    tmp2[0] = np.concatenate((np.array([1,1j,0, 0.0,0,0,0,0,0,0]),zerovec))
    tmp2[1] = np.concatenate((np.array([0.0,0.0,1j/(8.0*tau), -1/(8.0*tau),0,0,0,0,0,0]),zerovec))
    tmp2[2] = np.concatenate((np.array([0,0,1,1j,0,0,0,0,0,0]),zerovec))
    tmp2[3] = np.concatenate((np.array([0.0 ,0.,1j/(8.0*tau), 1.0/(8.0*tau),0.0,0.0,0.0,0.0,0.0,0 ]),zerovec))
    return tmp1, tmp2

def rho_update_control_l10_2input(i, Input_Initials): #Optimal control integration with \lambda_1=0
  Initials, X, P, H, rho, I_t, I_kvp_t, I_k_t, I_Gvp_t, I_G_t, delh_t_mat1, delh_t_mat2, wzmat, delI_t_mat_coeff, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0,  GMM0, Idth,   Fmat, ts, tau, dt, j, Ncos, Id = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  #theta = theta_t[j]
  theta = 0.1*jnp.tanh(10*jnp.matmul(Fmat[j],Initials))
  phi = theta+t
  
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
  e_jphi = jnp.exp(-1j*phi)
  delh_t_Mat = e_jphi*(delh_t_mat1+t*delh_t_mat2)
  ht = jnp.matmul(delh_t_Mat, Initials) + e_jphi*I_t
  r = ht.real 
  v = ht.imag
  #wzmat = jnp.array([0,0,1,1j,0,0,0,0,0,0])
  wz = jnp.matmul(wzmat, Initials)*e_jphi
  w = wz.real
  z = wz.imag
  kappavp0 = (kappaLL0+kappaMM0)/2.0
  kappavp = I_kvp_t+kappavp0
  kappa0 = (kappaMM0-kappaLL0)/2.0+1j*kappaLM0
  kappa = (kappa0+I_k_t)/(e_jphi**2)
  kappar = kappa.real
  kappaMM = kappar+kappavp
  kappaLL = kappavp-kappar
  kappaLM = kappa.imag
  Gvp0 = (GLL0+GMM0)/2.0
  Gvp = Gvp0+I_Gvp_t
  G0 = (GMM0-GLL0)/2.0+1j*GLM0
  G = (G0+I_G_t)/(e_jphi**2)
  Gr = G.real
  GLL = Gvp-Gr
  GLM = G.imag

  
  #GLL=(C1+(z**2-w**2)/4.0)/2.0
  #dphi = -kappaLL/((z**2-w**2+8*Gr)*tau)
  #dtheta = dphi-1.0
  #delk_t_Mat=jnp.exp(1j*2*phi)*jnp.array([0,0,0,0,0,1,1j,0,0])
  #kappa = jnp.matmul(delk_t_Mat,Initials)+jnp.exp(1j*2*phi)*I_k_t
  #kappaLM = kappa.imag
  #llcoeff = jnp.array([0,0,0,0,1+1j*0,0,0,0,0])
  #kappaLL=jnp.matmul(llcoeff, Initials).real-kappa.real
  #prodmat = np.zeros((9,9),dtype=np.complex_)
  #prodmat[0,1]=1j
  #prodmat = jnp.array(prodmat)
  #GLM = r*v
  #Gr = (GMM-GLL)/2.0
  #dtheta = (2*r*w-kappaLL)/((4*r**2-4*v**2+8*Gr)*tau)-1
  #dphi = jnp.array([1])+0.1*jnp.tanh(10*dtheta)
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho_update =H_update+Lind_update+read_update
  rho1 = rho + rho_update*dt
  #rho1 = rho1/jnp.trace(rho1)
  delI_t_Mat = delI_t_mat_coeff/(e_jphi**2)
  I_t1 = I_t + jnp.matmul(delI_t_Mat, Initials)*dt
  I_kvp_t1 = I_kvp_t-dt*GLM/tau
  I_k_t1 = I_k_t-dt*(GLM+1j*GLL)*e_jphi**2/tau
  I_Gvp_t1 = I_Gvp_t+dt*kappaLM/(4*tau)
  I_G_t1 = I_G_t+dt*(kappaLM+1j*kappaLL)*e_jphi**2/(4*tau)
  
  #GLL1 = GLL+2*dphi*(GLM)*dt
  #GMM1 = GMM-2*dphi*(GLM)*dt+dt*kappaLM/(2*tau)
  #kappaLL1 = kappaLL+dt*(2*dphi*kappaLM)
  #kappaLM1 = kappaLM+dt*(dphi*(kappaMM-kappaLL)+GLL/tau)
  #kappaMM1 = kappaMM+dt*(-2*dphi*kappaLM + 2*(GLM)/tau)
  #GLM1 = GLM+dt*(dphi*(GMM-GLL)+kappaLL/(4*tau))
  #r1 = r+dt*dphi*v
  #v1 = v+dt*(-dphi*r+w/(4*tau))
  #w1 = w+dt*dphi*z
  #z1 = z-dt*w*dphi
  #theta1 = theta + dt*dtheta
  #Idth1 = Idth+ dt*(GLM-w*z/4)**2
  Idth1 = Idth+dt*(r**2-2*r*expL+expV)/(2*tau)
  #Idth1 = -(-(GLL-w**2/4.0)/(2*tau)-(kappaLL+2*r*w)/2.0-(kappaMM+2*v*z)/2.0)
  return (Initials, X, P, H, rho1, I_t1, I_kvp_t1, I_k_t1, I_Gvp_t1, I_G_t1, delh_t_mat1, delh_t_mat2, wzmat, delI_t_mat_coeff, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0,  GMM0, Idth1, Fmat,   ts, tau, dt, j+1, Ncos, Id)


def OPsoln_control_l10_JAX_2input(Initials, X, P, H, rho_i, Fmat, ts, dt,  tau, Ncos, MMat1, MMat2, Id):
  #I_tR = jnp.array([0.0])
  I_t = jnp.array(0.0+1j*0.0)
  I_kvp_t = jnp.array(0.0)
  I_k_t = jnp.array(0.0+1j*0.0)
  I_Gvp_t = jnp.array(0.0)
  I_G_t = jnp.array(0.0+1j*0.0)
  #zerovec = jnp.zeros(2*Ncos)
  #print (I_t)
  #I_k_t = jnp.array([0.0 + 1j*0.0])
  Idth =  0.0
  r0 = jnp.matmul(MMat1[0],Initials)#+jnp.array([0])
  v0 = jnp.matmul(MMat1[1],Initials)#+jnp.array([0])
  w0 = jnp.matmul(MMat1[2],Initials)#+jnp.array([0])
  z0 = jnp.matmul(MMat1[3],Initials)#+jnp.array([0])
  #GLL = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,-1.0]),Initials)#+jnp.array([0])
  kappaLL = jnp.matmul(MMat1[4],Initials)#+jnp.array([0])
  kappaLM = jnp.matmul(MMat1[5],Initials)#+jnp.array([0])
  kappaMM = jnp.matmul(MMat1[6],Initials)#+jnp.array([0])
  GLL = jnp.matmul(MMat1[7],Initials)#+jnp.array([0])
  GMM = jnp.matmul(MMat1[8],Initials)#+jnp.array([0])
  GLM = jnp.matmul(MMat1[9],Initials)#+jnp.array([0])
  #kappaLL = kappaLL-2*r0*w0 
  #kappaLM = kappaLM-v0*w0-r0*z0
  #kappaMM = kappaMM-2*v0*z0
  #GLL+= -r0**2+w0**2/4.0
  #C1 = 2*(GLL-r**2-w**2/4.0)-(z**2-w**2)/4.0
  #GLM+= -r0*v0+w0*z0/4.0
  #GMM+= -v0**2+z0**2/4.0
  GLL0 = GLL
  GLM0 = GLM
  GMM0 = GMM
  kappaLL0 = kappaLL
  kappaLM0 = kappaLM
  kappaMM0 = kappaMM
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth=0
  delh_t_mat1 = MMat2[0]
  delh_t_mat2 = MMat2[1]
  wzmat = MMat2[2]
  delI_t_mat_coeff = MMat2[3]
  Initials, X, P, H,  rho, I_t, I_kvp_t, I_k_t, I_Gvp_t, I_G_t, delh_t_mat1, delh_t_mat2, wzmat, delI_t_mat_coeff, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0, GMM0, Idth,  Fmat, ts, tau, dt, k1, Ncos, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_control_l10_2input,(Initials, X, P, H,  rho, I_t, I_kvp_t, I_k_t, I_Gvp_t, I_G_t, delh_t_mat1, delh_t_mat2, wzmat, delI_t_mat_coeff, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0,  GMM0, Idth,  Fmat, ts, tau, dt, k1, Ncos, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho, Idth

def CostF1_control_l10_2input(Initials,  X, P, H,  rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX_2input(Initials,  X, P, H, rho_i, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id)
  #print (Idth)
  return 1e2*Tr_Distance(rho_f_simul, rho_f)

@jit
def update_control1_l10_2input(Initials, X, P, H, rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id,  step_size):
    grads=grad(CostF1_control_l10_2input)(Initials,  X, P, H, rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

def CostF2_control_l10_2input(Initials, X, P, H,  rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX_2input(Initials, X, P, H, rho_i, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id)
  #print (Idth)
  return Idth
 
@jit
def update_control2_l10_2input(Initials,  X, P, H, rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id,  step_size):
    grads=grad(CostF2_control_l10_2input)(Initials, X, P, H, rho_i, rho_f, Fmat,  ts, dt, tau, Ncos, MMat1, MMat2, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

def CostFt_control_l10_2input(Initials, X, P, H,  rho_i, rho_f, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX_2input(Initials,  X, P, H, rho_i, Fmat, ts, dt, tau, Ncos, MMat1, MMat2, Id)
  #print (Idth)
  return 1e2*Tr_Distance(rho_f_simul, rho_f), Idth

def OPintegrate_strat_2inputs(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Ncos,MMat1, MMat2, Id):
  r0 = np.matmul(MMat1[0],Initials)#+jnp.array([0])
  v0 = np.matmul(MMat1[1],Initials)#+jnp.array([0])
  w0 = np.matmul(MMat1[2],Initials)#+jnp.array([0])
  z0 = np.matmul(MMat1[3],Initials)#+jnp.array([0])
  GLL0 = np.matmul(MMat1[7],Initials)#+jnp.array([0])
  kappaLL0 = np.matmul(MMat1[4],Initials)#+jnp.array([0])
  kappaLM0 = np.matmul(MMat1[5],Initials)#+jnp.array([0])
  kappaMM0 = np.matmul(MMat1[6],Initials)#+jnp.array([0])
  GMM0 = np.matmul(MMat1[8],Initials)#+jnp.array([0])
  GLM0 = np.matmul(MMat1[9],Initials)
   #GLM0 = r0*v0
  #kappaLL0 = kappaLL0-2*r0*w0
  #kappaLM0 = kappaLM0-v0*w0-r0*z0
  #kappaMM0 = kappaMM0-2*v0*z0
  #GLL0+= -r0**2+w0**2/4.0
  #GLM0+= -r0*v0+w0*z0/4.0
  #GMM0+= -v0**2+z0**2/4.0
  kappavp0 = (kappaLL0+kappaMM0)/2.0
  kappa0 = (kappaMM0-kappaLL0)/2.0+1j*kappaLM0
  Gvp0 = (GLL0+GMM0)/2.0
  G0 = (GMM0-GLL0)/2.0+1j*GLM0
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
  phi = 0
  I_t=0
  #zerovec = np.zeros(2*Ncos)
  delIt_mat_coeff = MMat2[3]
  delh_t_mat1 = MMat2[0]
  delh_t_mat2 = MMat2[1]
  wzmat = MMat2[2]
  I_kvp_t = 0
  I_k_t = 0
  I_Gvp_t = 0
  I_G_t = 0
  while (j<len(ts)):
      t = ts[j]
      phi = theta_t[j]+t
      e_jphi = np.exp(-1j*phi)
      delh_t_Mat = e_jphi*(delh_t_mat1+t*delh_t_mat2)
      ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = ht.real
      v = ht.imag
      wz = np.matmul(wzmat, Initials)*e_jphi
      w = wz.real
      z = wz.imag
      kappavp = I_kvp_t+kappavp0
      kappa = (kappa0+I_k_t)/(e_jphi**2)
      kappar = kappa.real
      kappaMM = kappar+kappavp
      kappaLL = kappavp-kappar
      kappaLM = kappa.imag
      Gvp = Gvp0+I_Gvp_t
      G = (G0+I_G_t)/(e_jphi**2)
      Gr = G.real
      GLL = Gvp-Gr
      GLM = G.imag
      
      rtheta[j]=r
      theta = phi-t
      csth, snth = np.cos(theta), np.sin(theta)
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
      Q1[j]=expL
      Q2[j]=expM
      delL = Ljump - expL*Id
      #print (delL)
      Q4[j] = np.trace(np.matmul(np.matmul(Ljump, Mjump)+np.matmul(Mjump,Ljump),rho)).real/2.0-expL*expM
      Q3[j] = np.trace(np.matmul(Ljump2,rho)).real-expL**2
      Q5[j] = np.trace(np.matmul(Mjump2,rho)).real-expM**2
      delV = Ljump2-expV*Id
      H_update = -1j*(np.matmul(H, rho)-np.matmul(rho, H))
      Lind_update = (-np.matmul(delV, rho)-np.matmul(rho, delV))/(4*tau)
      read_update = r*(np.matmul(delL, rho)+ np.matmul(rho, delL))*(1.0/(2*tau))
      drho = H_update+Lind_update+read_update
      #print (Lind_update)
      #print (np.trace(drho).real)
      rho+=drho*dt
      #rho = rho/np.trace(rho)
      #rho=rho/np.trace(rho)
      delI_t_Mat =delIt_mat_coeff/(e_jphi**2)
      I_t+= np.matmul(delI_t_Mat, Initials)*dt
      I_kvp_t+=-dt*GLM/tau
      I_k_t+=-dt*(GLM+1j*GLL)*e_jphi**2/tau
      I_Gvp_t+= dt*kappaLM/(4*tau)
      I_G_t+= dt*(kappaLM+1j*kappaLL)*e_jphi**2/(4*tau)
      diff[j]=GLM-w*z/4.0
      Hs[j] =  -(GLL-w**2/4.0)/(2*tau)-(kappaLL+2*r*w)/2.0-(kappaMM+2*v*z)/2.0
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, rho, rtheta, diff, Hs
