#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:44:46 2024

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

nlevels = 8
rho_i = basis(nlevels,0)
#rho_f = basis(nlevels,4)
rho_f = coherent(nlevels, 0.5+1j*1.0)
#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 3
ts = np.linspace(t_i, t_f, 100)
dt = ts[1]-ts[0]
tau = 15.0
nsteps = 1500
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
#Ljump = X
#Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()
sigma_i = rand_herm(nlevels)
sigma_i = sigma_i-expect(sigma_i, rho_i)
xvec = np.linspace(-5,5,200)
pvec = np.linspace(-5,5,200)
W_i = wigner(rho_i,xvec,pvec)

lrate = 1e-2
q3, q4, q5, alr, ali, A, B, q1t, q2t = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)
#alr = np.random.rand()
#ali = np.random.rand()
#A = np.random.rand()
#B = np.random.rand()
#Initials = jnp.array([alr, ali, A, B])
Initials = jnp.array(np.random.rand(4))
theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)
tb = 0
jnpId = jnp.identity(nlevels)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnpX2 = jnp.matmul(jnpX, jnpX)
jnpP2 = jnp.matmul(jnpP, jnpP)
jnpXP = jnp.matmul(jnpX, jnpP)
jnpPX = jnp.matmul(jnpP, jnpX)
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f.full())

I_tR = jnp.array([0.0])
I_tI = jnp.array([0.0])
#Initials1, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho, I_tR, I_tI, theta_t, ts, tau, dt, l1 = rho_update(0,(Initials, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho_i, I_tR, I_tI,  theta_t, ts, tau, dt, 0))
#jnp_rho_simul = OPsoln_JAX1(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp.array(theta_t), jnp.array(ts), dt, tau, jnpId)
rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul = OPsoln_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)

for n in range(nsteps):
  stime = time.time()
  print (CostF(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId))
  Initials = update(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId, lrate)
  print (n, time.time()-stime)

Initvals = np.array(Initials)

PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig')


