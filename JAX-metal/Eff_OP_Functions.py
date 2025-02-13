#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:14:39 2025

@author: tatha_k
"""

import os,sys
os.environ['JAX_PLATFORMS'] = 'cpu'
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
from numba import njit, prange
import numba as nb

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)


def CompMultR(Ar, Ai, Br, Bi):
    return Ar @ Br - Ai @ Bi#jnp.matmul(Ar, Br)-jnp.matmul(Ai, Bi)
def CompMultI(Ar, Ai, Br, Bi):
    return Ar @ Bi + Ai @ Br#jnp.matmul(Ar, Bi)+jnp.matmul(Ai, Br)
def CompMult(Ar, Ai, Br, Bi):
    return CompMultR(Ar, Ai, Br, Bi), CompMultI(Ar, Ai, Br, Bi)

def ExpVal(Xr, Xi, rhor, rhoi): #Expectation value of a hermitian operator wrt state rho
    return jnp.trace(CompMultR(Xr, Xi, rhor, rhoi))


def Fidelity_PS(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  tmpr  = CompMultR(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi)
  fid = jnp.trace(tmpr)
  return fid

def Tr_Distance(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  #dist = jnp.sqrt(jnp.trace(delrho2).real)
  
  return -Fidelity_PS(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi)#+1e2*dist

def Del_G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, l1,  r, params):
    G101 = params[2]*(G01-snth*(csth*k10+snth*k01)/(4*params[3]))
    G011 = params[2]*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*params[3]))
    k101 = params[2]*k01
    k011 = -params[2]*(1+2*l1)*k10
    G201 = +params[2]*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*params[3]))
    G111 = +params[2]*(G02-(1+2*l1)*G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*params[3]))
    G021 = +params[2]*(-2*(1+2*l1)*G11+csth*(snth*k02+csth*k11-r*k01)/(2*params[3]))
    k201 = +params[2]*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/params[3])
    k111 = +params[2]*(-(1+2*l1)*k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/params[3])
    k021 = +params[2]*(-2*(1+2*l1)*k11+2*csth*(-snth*G02-csth*G11+r*G01)/params[3])  
    return G101, G011, k101, k011, G201, G111, G021, k201, k111, k021

def G_k_updates_first_order(G10, G01, k10, k01, csth, snth, l1, params):
    G101 = G10+params[2]*(G01-snth*(csth*k10+snth*k01)/(4*params[3]))
    G011 = G01+params[2]*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*params[3]))
    k101 = k10+params[2]*k01
    k011 = k01-params[2]*(1+2*l1)*k10
    return G101, G011, k101, k011


def G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, l1, r, params):
    G101, G011, k101, k011, G201, G111, G021, k201, k111, k021 = Del_G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, l1, r, params)
    return G10+G101, G01+G011, k10+k101, k01+k011, G20+G201, G11+G111, G02+G021, k20+k201, k11+k111, k02+k021


def rho_kraus_update(rhor, rhoi, Fr, Fi):
    tmp1r, tmp1i = CompMult(rhor, rhoi, Fr.T, -Fi.T)
    rho1r, rho1i = CompMult(Fr, Fi, tmp1r, tmp1i)
    Nr = jnp.trace(rho1r)
    Ni = jnp.trace(rho1i)
    rho2r = (Nr*rho1r+Ni*rho1i)/(Nr**2+Ni**2)
    rho2i = (Nr*rho1i-Ni*rho1r)/(Nr**2+Ni**2)
    return rho2r, rho2i

def Optimal_theta_l1(G10, G01, G20, G11, G02, k20, params):
    AGamma = (G10**2-G01**2-G20+G02)/2.0
    BGamma = G10*G01-G11
    theta = jnp.arctan2(BGamma, AGamma)/2.0
    #csth, snth = jnp.cos(theta), jnp.sin(theta)
    #r = csth*G10+snth*G01
    l1 = -params[0]*jnp.sign(k20)
    return theta, l1

def integrator_step(Ops, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params): #Optimal control integration with \lambda_1=0
  #Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rhor, rhoi, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, j, Id = Input_Initials
  theta, l1 = Optimal_theta_l1(G10, G01, G20, G11, G02, k20, params)
  #AGamma = (G10**2-G01**2-G20+G02)/2.0
  #BGamma = G10*G01-G11
  #theta = jnp.arctan2(BGamma, AGamma)/2.0
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  r = csth*G10+snth*G01
  #l1 = -l1max*jnp.sign(k20)
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  #t = ts[j]
  
  G101, G011, k101, k011, G201, G111, G021, k201, k111, k021 = Del_G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, l1, r, params)
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
  H1r = Ops[4]+l1*Ops[6]
  H1i = Ops[5]+l1*Ops[7]
  #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
  Fac2r = -params[2]*(Ops[6]*csth**2+csth*snth*Ops[8]+Ops[10]*snth**2)/(4*params[3])+params[2]*r*(csth*Ops[0]+snth*Ops[2])/(2*params[3])+params[2]*H1i
  Fac2i = -params[2]*(Ops[7]*csth**2+csth*snth*Ops[9]+Ops[11]*snth**2)/(4*params[3])+params[2]*r*(csth*Ops[1]+snth*Ops[3])/(2*params[3])-params[2]*H1r
  #tmp1r, tmp1i = CompMult(rhor, rhoi, Fac2r-dt*H1i, Fac2i +dt*H1r)
  #rho1r, rho1i = CompMult(Fac2r+dt*H1i, Fac2i-dt*H1r, tmp1r, tmp1i)
  #Nr = jnp.trace(rho1r)
 # Ni = jnp.trace(rho1i)
  #rho2r = (Nr*rho1r+Ni*rho1i)/(Nr**2+Ni**2)
  #rho2i = (Nr*rho1i-Ni*rho1r)/(Nr**2+Ni**2)
  #rho1 = jnp.matmul(jnp.matmul(Fac2-dt*1j*H1,rho),Fac2+dt*1j*H1)
  #tmptr = jnp.trace(rho1)
  #rho1r = rho1/tmptr
  #rho1 = rho + rho_update*dt
  #Idth1 = Idth#+1e0*dt*(r**2-2*r*expL)/(2*tau)
  return Fac2r, Fac2i, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021


def RK4_delyn(k1, k2, k3, k4):
    return k1/6.0+k2/3.0+k3/3.0+k4/6.0

#@njit(inline='always')
@jit
def RK4_step(Ops,  rhor, rhoi, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params): 
  #= Input_Initials
  Fk1r, Fk1i, G10k1, G01k1, k10k1, k01k1, G20k1, G11k1, G02k1, k20k1, k11k1, k02k1 = integrator_step(Ops, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params)
  #rhok1r = rhor+(rho1r-rhor)/2.0
  #rhok1i = rhoi+(rho1i-rhoi)/2.0
  Fk2r, Fk2i, G10k2, G01k2, k10k2, k01k2, G20k2, G11k2, G02k2, k20k2, k11k2, k02k2 = integrator_step(Ops, G10+G10k1/2.0, G01+G01k1/2.0, k10+k10k1/2.0, k01+k01k1/2.0, G20+G20k1/2.0, G11+G11k1/2.0, G02+G02k1/2.0, k20+k20k1/2.0, k11+k11k1/2.0, k02+k02k1/2.0, params)
  Fk3r, Fk3i, G10k3, G01k3, k10k3, k01k3, G20k3, G11k3, G02k3, k20k3, k11k3, k02k3 = integrator_step(Ops, G10+G10k2/2.0, G01+G01k2/2.0, k10+k10k2/2.0, k01+k01k2/2.0, G20+G20k2/2.0, G11+G11k2/2.0, G02+G02k2/2.0, k20+k20k2/2.0, k11+k11k2/2.0, k02+k02k2/2.0, params)
  Fk4r, Fk4i, G10k4, G01k4, k10k4, k01k4, G20k4, G11k4, G02k4, k20k4, k11k4, k02k4 = integrator_step(Ops, G10+G10k3, G01+G01k3, k10+k10k3, k01+k01k3, G20+G20k3, G11+G11k3, G02+G02k3, k20+k20k3, k11+k11k3, k02+k02k3, params)
  #rho1r = rhor+rhok1r/6.0+rhok2r/3.0+rhok3r/3.0+rhok4r/6.0
  #rho1i = rhoi+rhok1i/6.0+rhok2i/3.0+rhok3i/3.0+rhok4i/6.0
  G101 = G10+RK4_delyn(G10k1, G10k2, G10k3, G10k4)
  G011 = G01+RK4_delyn(G01k1, G01k2, G01k3, G01k4)
  k101 = k10+RK4_delyn(k10k1, k10k2, k10k3, k10k4)
  k011 = k01+RK4_delyn(k01k1, k01k2, k01k3, k01k4)
  G201 = G20+RK4_delyn(G20k1, G20k2, G20k3, G20k4)
  G111 = G11+RK4_delyn(G11k1, G11k2, G11k3, G11k4)
  G021 = G02+RK4_delyn(G02k1, G02k2, G02k3, G02k4)
  k201 = k20+RK4_delyn(k20k1, k20k2, k20k3, k20k4)
  k111 = k11+RK4_delyn(k11k1, k11k2, k11k3, k11k4)
  k021 = k02+RK4_delyn(k02k1, k02k2, k02k3, k02k4)
  Fr = Ops[12]+RK4_delyn(Fk1r, Fk2r, Fk3r, Fk4r)
  Fi = RK4_delyn(Fk1i, Fk2i, Fk3i, Fk4i)
  rho1r, rho1i = rho_kraus_update(rhor, rhoi, Fr, Fi)
  
  #Idth1 = Idth
  return  rho1r, rho1i, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021


def RK4_stepJAX(i, Input_Initials): #Optimal control integration with \lambda_1=0
  Ops,  rhor, rhoi, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params, j, Idth = Input_Initials
  Idth1 = Idth
  
  rho1r, rho1i, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021 = RK4_step(Ops,  rhor, rhoi, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params)
  return (Ops, rho1r, rho1i, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021, params, j+1, Idth1)


def OPsoln_control_l10_JAX(Initials, Ops,  rho_ir, rho_ii, params):
  #I_tR = jnp.array([0.0])
  G10 = jnp.matmul(params[4][0], Initials)
  G01 = jnp.matmul(params[4][1], Initials)
  k10 = jnp.matmul(params[4][2], Initials)
  k01 = jnp.matmul(params[4][3], Initials)
  G20 = jnp.matmul(params[4][4], Initials)
  G11 = jnp.matmul(params[4][5], Initials)
  G02 = jnp.matmul(params[4][6], Initials)
  k20 = jnp.matmul(params[4][7], Initials)
  k11 = jnp.matmul(params[4][8], Initials)
  k02 = jnp.matmul(params[4][9], Initials)
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rhor = rho_ir
  rhoi = rho_ii
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = 0.0
  Ops, rhor, rhoi, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params, k1, Idth = jax.lax.fori_loop(0, len(params[1])-1, RK4_stepJAX,(Ops,  rho_ir, rho_ii, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params, k1, Idth))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rhor, rhoi

@jit
def CostF_control_l101(Initials, Ops ,  rho_ir, rho_ii, rho_fr, rho_fi, params):
  rho_f_simulr, rho_f_simuli = OPsoln_control_l10_JAX(Initials, Ops,  rho_ir, rho_ii, params)
  return Tr_Distance(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi), rho_f_simulr, rho_f_simuli


def OPintegrate_strat(Initials, Ops, rho_ir, rho_ii, params):
  #I_tR = jnp.array([0.0])
  G10 = params[4][0] @ Initials
  G01 = params[4][1] @ Initials
  k10 = params[4][2] @ Initials
  k01 = params[4][3] @ Initials
  G20 = params[4][4] @ Initials
  G11 = params[4][5] @ Initials
  G02 = params[4][6] @ Initials
  k20 = params[4][7] @ Initials
  k11 = params[4][8] @ Initials
  k02 = params[4][9] @ Initials
  
  rhor = rho_ir
  rhoi = rho_ii
  j=0
  npoints = len(params[1])
  Q1 = np.zeros(npoints)
  Q2 = np.zeros(npoints)
  Q3 = np.zeros(npoints)
  Q4 = np.zeros(npoints)
  Q5 = np.zeros(npoints)
  theta_t = np.zeros(npoints)
  l1_t = np.zeros(npoints)
  diff = np.zeros(npoints)
  readout = np.zeros(npoints)
  #k1 = 0
  while (j<npoints):
      
      #print (j,r)
      #Initials, X, P, H, rho, I_t, I_k_t, I_Gp_t, I_G_t,   phi,  ts, tau, dt, j, Id, Q1, Q2, Q3, Q4, Q5 = Input_Initials
      #I_t = I_tR + 1j*I_tI
      #print (tau)
      
      diff[j]=G10
      t = params[1][j]
      #print (jnp.shape(params[0]), jnp.shape(params[1]))
      expX = ExpVal(Ops[0], Ops[1], rhor, rhoi).item()
      expP = ExpVal(Ops[2], Ops[3], rhor, rhoi).item()
      Q1[j]=expX
      Q2[j]=expP
      #delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = ExpVal(Ops[6], Ops[7], rhor, rhoi).item()-expX**2
      Q5[j] = ExpVal(Ops[10], Ops[11], rhor, rhoi).item()-expP**2
      Q4[j] = ExpVal(Ops[8], Ops[9], rhor, rhoi).item()/2.0-expX*expP
      
      theta, l1 = Optimal_theta_l1(G10, G01, G20, G11, G02, k20, params)
      #AGamma = (G10**2-G01**2-G20+G02)/2.0
      #BGamma = G10*G01-G11
      #theta = np.arctan2(BGamma, AGamma)/2.0
      theta_t[j] = theta.item()
      csth, snth = jnp.cos(theta), jnp.sin(theta)
      #l1 = -l1max*np.sign(k20)
      l1_t[j] = l1.item()
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = csth*G10+snth*G01
      readout[j] = r.item()
      Idth = 0.0
      rhor, rhoi, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = RK4_step(Ops,  rhor, rhoi, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, params)
      
      '''
      H1 = H+l1*X2
      #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
      Fac2 = Id-dt*(X2*csth**2+csth*snth*CXP+P2*snth**2)/(4*tau)+dt*r*(csth*X+snth*P)/(2*tau)
      rho1 = np.matmul(np.matmul(Fac2-dt*1j*H1,rho),Fac2+dt*1j*H1)
      tmptr = np.trace(rho1)
      rho = rho1/tmptr
      
      G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1)
      
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
      '''
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, theta_t, l1_t, rhor, rhoi, readout, diff