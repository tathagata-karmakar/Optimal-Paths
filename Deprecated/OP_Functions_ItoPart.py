#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 12:04:30 2024

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

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
#torch.backends.cuda.cufft_plan_cache[0].max_size = 32
torch.autograd.set_detect_anomaly(True)

def OPsoln_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, varReturn = 0):
  rho = rho_i
  t=ts[0]
  t_f = ts[-1]
  dt = ts[1]-ts[0]
  I_t = 0
  expLs = np.zeros(len(ts))
  expMs = np.zeros(len(ts))
  varLs = np.zeros(len(ts))
  varMs = np.zeros(len(ts))
  covLMs = np.zeros(len(ts))
  i=0
  while (i<len(ts)):
    theta = theta_t[i]
    phi = theta + t
    csth, snth = np.cos(theta), np.sin(theta)
    csph, snph = np.cos(phi), np.sin(phi)
    cs2ph, sn2ph = np.cos(2*phi), np.sin(2*phi)
    Ljump = csth*X+snth*P
    Ljump2 = Ljump*Ljump
    Mjump = -snth*X+csth*P
    expL = expect(Ljump,rho).real
    expM = expect(Mjump,rho).real
    expLs[i] = expL
    expMs[i] = expM
    delL = Ljump - expL
    delL2 = delL*delL
    varL = expect(delL2,rho).real
    if (varReturn ==1):
      delM = Mjump-expM
      varLs[i] = varL
      varMs[i] = expect(delM*delM,rho).real
      covLMs[i] = (expect(Ljump*Mjump+Mjump*Ljump,rho).real-2*expL*expM)/2.0
    addL = (delL2-varL)/(2*tau)
    ft = expect(commutator(addL,Ljump,kind='anti')/2.0,rho).real
    #print (ft)
    gt = expect(commutator(addL,Mjump,kind='anti')/2.0,rho).real
    ht = np.exp(-1j*phi)*(alr+1j*ali+1j*t*(A+1j*B)/(8*tau))+np.exp(-1j*phi)*I_t
    u =  ht.real
    rho1 = rho+(-1j*commutator(H,rho)+(Ljump*rho*Ljump-Ljump*Ljump*rho/2.0-rho*Ljump*Ljump/2.0)/(4*tau)+u*commutator(delL, rho,kind='anti')/(2*tau))*dt
    rho = rho1#/rho1.tr()
    I_t+= ((ft+1j*gt)*np.exp(1j*phi)+1j*(A-1j*B)*np.exp(1j*2*phi)/(8*tau))*dt
    t = t+dt
    i+=1
    #print (expM)
  if (varReturn == 1):
    return rho, expLs, expMs, varLs, covLMs, varMs
  else:
    return rho, expLs, expMs

def rho_update(i, Input_Initials):
  Initials, X, P, H, rho, I_t, theta_t, ts, tau, dt, j, Id = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  theta = theta_t[j]
  phi = theta+t
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  csph, snph = jnp.cos(phi), jnp.sin(phi)
  cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  Mjump = -snth*X+csth*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expL = csth*expX + snth*expP
  expM = -snth*expX + csth*expP
  delL = Ljump - expL*Id
  delL2 =jnp.matmul(delL, delL) #Ljump2+expL**2-2*expL*Ljump
  #delL2 = Ljump2+expL**2-2*expL*Ljump
  varL = jnp.trace(jnp.matmul(delL2,rho)).real
  delM = Mjump-expM*Id
  addL = (delL2-varL*Id)/(2*tau)
  ft = jnp.trace(jnp.matmul((jnp.matmul(addL,Ljump)+ jnp.matmul(Ljump,addL)) /2.0,rho).real)
  gt = jnp.trace(jnp.matmul((jnp.matmul(addL,Mjump)+ jnp.matmul(Mjump,addL)) /2.0,rho).real)
  #delh_tR_Mat = jnp.array([csph,snph,t*snph/(8.0*tau), -t*csph/(8.0*tau)])
  #delh_tI_Mat = jnp.array([-snph,csph,t*csph/(8.0*tau), t*snph/(8.0*tau)])
  delh_t_Mat = jnp.exp(-1j*phi)*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau)])
  #htR = jnp.matmul(delh_tR_Mat, Initials) + csph*I_tR + snph*I_tI
  #htI = jnp.matmul(delh_tI_Mat, Initials) - snph*I_tR + csph*I_tI
  ht = jnp.matmul(delh_t_Mat, Initials) + jnp.exp(-1j*phi)*I_t
  u = ht.real
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (jnp.matmul(jnp.matmul(Ljump,rho),Ljump)-jnp.matmul(Ljump2, rho)/2.0-jnp.matmul(rho, Ljump2)/2.0)/(4*tau)
  Back_update = u*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho1 = rho + (H_update+Lind_update+Back_update)*dt
  #rho1 = rho1/jnp.trace(rho1).real
  #delI_tR_Mat = jnp.array([0.0,0.0,-sn2ph/(8*tau), cs2ph/(8*tau)])
  #delI_tI_Mat = jnp.array([0.0,0.0,cs2ph/(8*tau), sn2ph/(8*tau)])
  delI_t_Mat = jnp.array([0.0  ,0. , 1j*jnp.exp(1j*2*phi)/(8.0*tau), jnp.exp(1j*2*phi)/(8.0*tau) ])
  #delI_tR_b = jnp.array([ft*csph-gt*snph])
  #delI_tI_b = jnp.array([ft*snph+gt*csph])
  delI_t_b = (ft+1j*gt)*jnp.exp(1j*phi) 
  #I_tR = I_tR + (jnp.matmul(delI_tR_Mat, Initials) + delI_tR_b)*dt
  #I_tI = I_tI + (jnp.matmul(delI_tI_Mat, Initials) + delI_tI_b)*dt
  I_t1 = I_t + (delI_t_b + jnp.matmul(delI_t_Mat, Initials))*dt
  return (Initials, X, P, H, rho1, I_t1, theta_t, ts, tau, dt, j+1, Id)

def OPsoln_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Id):
  #I_tR = jnp.array([0.0])
  I_t = jnp.array([0.0 + 1j*0.0])
  rho = rho_i
  k1=0
  Initials, X, P, H,  rho, I_t, theta_t, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts), rho_update,(Initials, X, P, H,  rho, I_t,  theta_t, ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho

def CostF(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul = OPsoln_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  return 1e2*Tr_Distance(rho_f_simul, rho_f)

@jit
def update(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id,  step_size):
    grads=grad(CostF)(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])