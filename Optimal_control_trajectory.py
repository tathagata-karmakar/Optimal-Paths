#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:39:09 2024

@author: tatha_k
"""

'''

This script plots the optimal path and control found through
Optimal_control_l1_theta_anneal.py and also plots a trajectory. 

The plot is saved the Plots folder with the title sample_trajectory.

'''

import os,sys
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
from Eff_OP_Functions import *
#from Initialization import *
import h5py
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)
script_dir = os.path.dirname(__file__)


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
Dirname = script_dir+"/Data/testing2"
with h5py.File(Dirname+'/Optimal_control_solution.hdf5', 'r') as f:
    #Initvals = np.array(f['Initvals'])
    l1_t = np.array(f['l1_t'])
    theta_t = np.array(f['theta_t'])
    Q1t = np.array(f['Q1t'])
    Q2t = np.array(f['Q2t'])
    Q3t = np.array(f['Q3t'])
    Q4t = np.array(f['Q4t'])
    Q5t = np.array(f['Q5t'])
    rop_stratj = np.array(f['ML_readouts'])

Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)

ts = params[1]

tsi = np.linspace(params[1][0], params[1][-1], 5*len(params[1]))
l1_ti = np.interp(tsi, params[1], l1_t)
#rOCi = np.interp(tsi, params[1], rOC)
theta_ti = np.interp(tsi, params[1], theta_t)
newparams = (params[0], tsi, tsi[1]-tsi[0], params[3], params[4])
Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simulr, rho_f_simuli, rs= OP_stochastic_trajectory(Ops, rho_ir, rho_ii, l1_ti, theta_ti, newparams)
#Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, rho_f_simul2, rop_stratj = OP_wcontrol(Initvals, X.full(), P.full(), H.full(), X2.full(), CXP.full(), P2.full(), rho_i.full(), l1_t, theta_t, ts, dt,  tau,  np_Idmat, np.identity(nlevels))

fig, axs = plt.subplots(6,1,figsize=(6,14),sharex='all')
axs[0].tick_params(labelsize=18)
axs[1].tick_params(labelsize=18)
axs[2].tick_params(labelsize=18)
axs[3].tick_params(labelsize=18)
axs[4].tick_params(labelsize=18)
axs[5].tick_params(labelsize=18)
axs[0].plot(tsi, Q1j, linewidth =3, color='blue', label = 'Trajectory')
axs[0].plot(ts, Q1t, linewidth =3, color='g', linestyle='dashed', label = 'MLP')
axs[0].set_ylabel(r'$\left\langle\hat{X}\right\rangle$',fontsize=18)
axs[1].plot(tsi, Q2j, linewidth =3, color='blue')
axs[1].plot(ts, Q2t, linewidth =3, color='g', linestyle='dashed')
axs[1].set_ylabel(r'$\left\langle\hat{P}\right\rangle$',fontsize=18)
axs[2].plot(tsi, Q3j, linewidth =3, color='blue')
axs[2].plot(ts, Q3t, linewidth =3, color='g', linestyle='dashed')
axs[2].set_ylabel(r'$\textrm{var}\hat{X}$',fontsize=18)
axs[3].plot(tsi, Q4j, linewidth =3, color='blue')
axs[3].plot(ts, Q4t, linewidth =3, color='g', linestyle='dashed')
axs[3].set_ylabel(r'$\textrm{cov}\left(\hat{X},\hat{P}\right)$',fontsize=18)
axs[4].plot(tsi, Q5j, linewidth =3, color='blue')
axs[4].plot(ts, Q5t, linewidth =3, color='g', linestyle='dashed')
axs[4].set_ylabel(r'$\textrm{var}\hat{P}$',fontsize=18)
axs[5].plot(tsi, rs,linewidth =3, color='blue')
axs[5].plot(ts, rop_stratj,linewidth =3, color='g', linestyle='dashed')
axs[5].set_ylabel(r'$r$',fontsize=18)
axs[5].set_xlabel(r'$t$',fontsize=18)
axs[5].set_xlim(0,3)
#axs[6].plot(ts, theta_t,linewidth =3, color='g', linestyle='dashed')
#axs[7].plot(ts, l1_t, linewidth =3, color='g', linestyle='dashed')
plt.subplots_adjust(wspace=0.22, hspace=0.08)
axs[0].legend(loc=1,fontsize=15)

#plt.savefig(Dirname+'/Plots/sample_trajectory.pdf',bbox_inches='tight')