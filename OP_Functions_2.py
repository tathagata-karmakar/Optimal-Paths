#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:15:35 2024

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
from OP_Functions import *

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)

'''
Functions needed to Simulate Optimal Path for a Continuously Monitored
Quantum Harmonic Oscillator with Controlled Quadrature Measurement
'''


def rho_update_control_parametric(i, Input_Initials): #Optimal control integration with \lambda_1=0
  X, P, H, rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, j, Id = Input_Initials
  AGamma = (G10**2-G01**2-G20+G02)/2.0
  BGamma = G10*G01-G11
  theta = jnp.arctan2(BGamma, AGamma)/2.0
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  l1 = -l1max*jnp.sign(k20)
  r = csth*G10+snth*G01
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  #theta = theta_t[j]
  #phi = theta+t
  
  #dphi  = jnp.array([1.0])#+0.1*jnp.tanh(10.0*(dphi -1))
  #dphi = jnp.array([1.0])
  
  
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
  G101, G011, k101, k011, G201, G111, G021, k201, k111, k021 = G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1)
  
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho_update =H_update+Lind_update+read_update
  rho1 = rho + rho_update*dt
  Idth1 = Idth+1e0*dt*(r**2-2*r*expL+expV)/(2*tau)
  return (X, P, H, rho1, l1max, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021, Idth1, ts, tau, dt, j+1, Id)


def OPsoln_control_l10_JAX(Initials, X, P, H, rho_i, l1max, ts, dt,  tau, Idmat,  Id):
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
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = 0.0
  X, P, H,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_control_parametric,(X, P, H,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,  ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho, Idth