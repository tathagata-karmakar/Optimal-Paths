#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:39:09 2024

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
from OP_Functions import *
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

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)


hf = h5py.File('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Data/Optimal_control_Ex9.hdf5', 'r')

nlevels = int(np.array(hf['nlevels']))
a = destroy(nlevels)
tau = np.array(hf['tau']).item()
theta_t = np.array(hf['theta_t'])
theta0 = np.zeros(len(theta_t))
l1_t = np.array(hf['l1_t'])
l10 = np.zeros(len(l1_t))
ts = np.array(hf['ts'])
ropt = np.array(hf['r_t'])
rho_i =Qobj(np.array(hf['rho_i']))
rho_f = Qobj(np.array(hf['rho_f_target']))
Initvals = np.array(hf['Initvals'])
dt = ts[1]-ts[0]
np_Idmat=np.identity(10)
Idmat = jnp.array(np_Idmat)

X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
X2 = X*X
#Ljump = X
#Mjump = P
#rho_i = rho_i*rho_i.dag()
#rho_f = rho_f*rho_f.dag()

jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f.full())
jnpX2= jnp.matmul(jnpX, jnpX)

Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simul, rs= OP_stochastic_trajectory(X.full(), P.full(), H.full(), X2.full(), rho_i.full(), l1_t, theta_t, ts, dt,  tau,  np.identity(nlevels))
Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, rho_f_simul2, rop_stratj = OP_wcontrol(Initvals, X.full(), P.full(), H.full(), X2.full(), rho_i.full(), l1_t, theta_t, ts, dt,  tau,  np_Idmat, np.identity(nlevels))

fig, axs = plt.subplots(6,1,figsize=(6,14),sharex='all')
axs[0].tick_params(labelsize=14)
axs[1].tick_params(labelsize=14)
axs[2].tick_params(labelsize=14)
axs[3].tick_params(labelsize=14)
axs[4].tick_params(labelsize=14)
axs[5].tick_params(labelsize=14)
axs[0].plot(ts, Q1j, linewidth =3, color='blue', label = 'Trajectory')
axs[0].plot(ts, Q1j1, linewidth =3, color='g', linestyle='dashed', label = 'MLP')
axs[0].set_ylabel(r'$\left\langle\hat{X}\right\rangle$',fontsize=15)
axs[1].plot(ts, Q2j, linewidth =3, color='blue')
axs[1].plot(ts, Q2j1, linewidth =3, color='g', linestyle='dashed')
axs[1].set_ylabel(r'$\left\langle\hat{P}\right\rangle$',fontsize=15)
axs[2].plot(ts, Q3j, linewidth =3, color='blue')
axs[2].plot(ts, Q3j1, linewidth =3, color='g', linestyle='dashed')
axs[2].set_ylabel(r'$\textrm{var}\hat{X}$',fontsize=15)
axs[3].plot(ts, Q4j, linewidth =3, color='blue')
axs[3].plot(ts, Q4j1, linewidth =3, color='g', linestyle='dashed')
axs[3].set_ylabel(r'$\textrm{cov}\left(\hat{X},\hat{P}\right)$',fontsize=15)
axs[4].plot(ts, Q5j, linewidth =3, color='blue')
axs[4].plot(ts, Q5j1, linewidth =3, color='g', linestyle='dashed')
axs[4].set_ylabel(r'$\textrm{var}\hat{P}$',fontsize=15)
axs[5].plot(ts, rs,linewidth =3, color='blue')
axs[5].plot(ts, rop_stratj,linewidth =3, color='g', linestyle='dashed')
axs[5].set_ylabel(r'$r$',fontsize=15)
axs[5].set_xlabel(r'$t$',fontsize=15)
#axs[6].plot(ts, theta_t,linewidth =3, color='g', linestyle='dashed')
#axs[7].plot(ts, l1_t, linewidth =3, color='g', linestyle='dashed')
plt.subplots_adjust(wspace=0.22, hspace=0.08)
axs[0].legend(loc=1,fontsize=15)

#plt.savefig('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Plots/trajectory_binomial_code2.pdf',bbox_inches='tight')
hf.close()

