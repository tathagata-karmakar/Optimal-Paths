#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 20:36:52 2025

@author: tatha_k
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



def Neg_Fid(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  #dist = jnp.sqrt(jnp.trace(delrho2).real)
  return -expect(rho_f_simul, rho_f)


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

def G_k_updates_first_order(G10, G01, k10, k01, csth, snth, dt, tau, l1):
    G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
    G011 = G01+dt*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*tau))
    k101 = k10+dt*k01
    k011 = k01-dt*(1+2*l1)*k10
    return G101, G011, k101, k011

def OC_update_QT(X, P, H, X2, CXP, P2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, tau, dt, Id): #Optimal control integration with \lambda_1=0
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
  
  #H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  #Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  #read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  #rho_update =H_update+Lind_update+read_update
  H1 = H+l1*X2
  #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
  Fac2 = Id-dt*(X2*csth**2+csth*snth*CXP+P2*snth**2)/(4*tau)+dt*r*(csth*X+snth*P)/(2*tau)
  rho1 = (Fac2-dt*1j*H1)*rho*(Fac2+dt*1j*H1)
  tmptr = rho1.tr()
  rho1 = rho1/tmptr
  return rho1, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021

def OC_soln_QT(Initials, X, P, H, X2, CXP, P2, rho_i, l1max, ts, dt,  tau, Idmat, Id):
  #I_tR = jnp.array([0.0])
  G10 = np.matmul(Idmat[0], Initials)
  G01 = np.matmul(Idmat[1], Initials)
  k10 = np.matmul(Idmat[2], Initials)
  k01 = np.matmul(Idmat[3], Initials)
  G20 = np.matmul(Idmat[4], Initials)
  G11 = np.matmul(Idmat[5], Initials)
  G02 = np.matmul(Idmat[6], Initials)
  k20 = np.matmul(Idmat[7], Initials)
  k11 = np.matmul(Idmat[8], Initials)
  k02 = np.matmul(Idmat[9], Initials)
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  #k1=0
  #Idth = 0.0
  k=0
  while (k<len(ts)):
      rho, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = OC_update_QT(X, P, H, X2, CXP, P2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, tau, dt, Id)
      k+=1
  return rho#,

def CostF_control_QT(Initials, X, P, H, X2, CXP, P2,  rho_i, rho_f, l1max, ts, dt, tau, Idmat, Id):
  rho_f_simul = OC_soln_QT(Initials, X, P, H, X2, CXP, P2, rho_i, l1max, ts, dt, tau, Idmat, Id)
  return Neg_Fid(rho_f_simul, rho_f)