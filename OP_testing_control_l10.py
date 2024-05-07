#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:45:15 2024

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

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
#torch.backends.cuda.cufft_plan_cache[0].max_size = 32
torch.autograd.set_detect_anomaly(True)

nlevels = 25

#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 3
ts = np.linspace(t_i, t_f, 500)
dt = ts[1]-ts[0]
tau = 15.0
q4f = np.sqrt(1+4*tau*tau)-2*tau
q3f = np.sqrt(4*tau*q4f)
q5f = q3f*(1+q4f/(2*tau))

snh2r = -np.sqrt(q4f**2+(q3f-q5f)**2/4.0)
csh2r = (q3f+q5f)/2.0
rparam = snh2r+csh2r
r_sq = np.log(rparam)/2
xiR = r_sq*(q5f-q3f)/(2*snh2r)
xiI = r_sq*(-q4f)/snh2r
in_alr = 0.0
in_ali = 0.0
fin_alr = 1.7
fin_ali = 0.8

rho_i = squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
#rho_f = basis(nlevels,4)
rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))

nsteps = 2000
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
#Ljump = X
#Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()
rho_f_int = rho_f_int*rho_f_int.dag()
sigma_i = rand_herm(nlevels)
sigma_i = sigma_i-expect(sigma_i, rho_i)

XItf = X*np.cos(t_f)+P*np.sin(t_f)
PItf = P*np.cos(t_f)-X*np.sin(t_f)

Q1 = expect(XItf,rho_f_int)
Q2 = expect(PItf,rho_f_int)
V1 = expect(XItf*XItf,rho_f_int)-Q1**2
V3 = expect(PItf*PItf,rho_f_int)-Q2**2
V2 = (expect(XItf*PItf+PItf*XItf,rho_f_int)-2*Q1*Q2)/2.0

lrate = 1e-1/2.0
q3, q4, q5, alr, ali, A, B, q1t, q2t,rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)

#alr = np.random.rand()
#ali = np.random.rand()
#A = np.random.rand()
#B = np.random.rand()
#Initials = jnp.array([alr, ali, A, B])
Initials = jnp.array(np.random.rand(9))
theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)
tb = 0
jnpId = jnp.identity(nlevels)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f_int.full())

for n in range(nsteps):
  stime = time.time()
  print (CostF_control_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId))
  Initials = update_control_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId, lrate)
  print (n, time.time()-stime)
  
  
Initvals = np.array(Initials)

PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig_control')
