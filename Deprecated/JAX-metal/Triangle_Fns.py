#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:52:59 2025

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
from JM_OP_Functions import *
import h5py
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
script_dir = os.path.dirname(__file__)

def T_tr(X, Id):
    return jnp.sum(X*Id)

def rho_update_T_l10(i, Input_Initials): #Optimal control integration with \lambda_1=0
  Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rhor, rhoi, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, j, Id = Input_Initials
  AGamma = (G10**2-G01**2-G20+G02)/2.0
  BGamma = G10*G01-G11
  theta = jnp.arctan2(BGamma, AGamma)/2.0
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  r = csth*G10+snth*G01
  l1 = -l1max*jnp.sign(k20)
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  
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
  H1r = Hr+l1*X2r
  H1i = Hi+l1*X2i
  #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
  Fac2r = Id-dt*(X2r*csth**2+csth*snth*CXPr+P2r*snth**2)/(4*tau)+dt*r*(csth*Xr+snth*Pr)/(2*tau)
  Fac2i = -dt*(X2i*csth**2+csth*snth*CXPi+P2i*snth**2)/(4*tau)+dt*r*(csth*Xi+snth*Pi)/(2*tau)
  tmp1r, tmp1i = CompMult(rhor, rhoi, Fac2r-dt*H1i, Fac2i +dt*H1r)
  rho1r, rho1i = CompMult(Fac2r+dt*H1i, Fac2i-dt*H1r, tmp1r, tmp1i)
  Nr = T_tr(rho1r, Id)
  Ni = T_tr(rho1i, Id)
  rho2r = (Nr*rho1r+Ni*rho1i)/(Nr**2+Ni**2)
  rho2i = (Nr*rho1i-Ni*rho1r)/(Nr**2+Ni**2)
  #rho1 = jnp.matmul(jnp.matmul(Fac2-dt*1j*H1,rho),Fac2+dt*1j*H1)
  #tmptr = jnp.trace(rho1)
  #rho1r = rho1/tmptr
  #rho1 = rho + rho_update*dt
  Idth1 = Idth#+1e0*dt*(r**2-2*r*expL)/(2*tau)
  return (Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i, rho2r, rho2i, l1max, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021, Idth1, ts, tau, dt, j+1, Id)


def OPsoln_control_l10_T(Initials, Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rho_ir, rho_ii, l1max, ts, dt,  tau, Idmat,  Id):
  #I_tR = jnp.array([0.0])
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
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rhor = rho_ir
  rhoi = rho_ii
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = 0.0
  Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rhor, rhoi, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_T_l10,(Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rho_ir, rho_ii, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,  ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rhor, rhoi#, Idth

@jit
def CostT_control_l101(Initials, Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rho_ir, rho_ii, rho_fr, rho_fi, l1max, ts, dt, tau, Idmat, Id):
  rho_f_simulr, rho_f_simuli = OPsoln_control_l10_T(Initials, Xr, Xi, Pr, Pi, Hr, Hi, X2r, X2i, CXPr, CXPi, P2r, P2i,  rho_ir, rho_ii, l1max, ts, dt, tau, Idmat, Id)
  return Tr_Dist_T(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi), rho_f_simulr, rho_f_simuli
