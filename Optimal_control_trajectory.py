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
Dirname = script_dir+"/Data/Cat_to_ground"
with h5py.File(Dirname+'/Optimal_control_solution.hdf5', 'r') as f:
    #Initvals = np.array(f['Initvals'])
    #l1_t = np.array(f['l1_t'])
    #theta_t = np.array(f['theta_t'])
    Q1t = np.array(f['Q1t'])
    Q2t = np.array(f['Q2t'])
    Q3t = np.array(f['Q3t'])
    Q4t = np.array(f['Q4t'])
    Q5t = np.array(f['Q5t'])
    rop_stratj = np.array(f['ML_readouts'])

Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)

ts = params[1]
#Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simul, rs= OP_stochastic_trajectory(Ops, rho_ir, rho_ii, l1_t, theta_t, ts, dt,  tau,  np.identity(nlevels))
#Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, rho_f_simul2, rop_stratj = OP_wcontrol(Initvals, X.full(), P.full(), H.full(), X2.full(), CXP.full(), P2.full(), rho_i.full(), l1_t, theta_t, ts, dt,  tau,  np_Idmat, np.identity(nlevels))

fig, axs = plt.subplots(6,1,figsize=(6,14),sharex='all')
axs[0].tick_params(labelsize=14)
axs[1].tick_params(labelsize=14)
axs[2].tick_params(labelsize=14)
axs[3].tick_params(labelsize=14)
axs[4].tick_params(labelsize=14)
axs[5].tick_params(labelsize=14)
#axs[0].plot(ts, Q1j, linewidth =3, color='blue', label = 'Trajectory')
axs[0].plot(ts, Q1t, linewidth =3, color='g', linestyle='dashed', label = 'MLP')
axs[0].set_ylabel(r'$\left\langle\hat{X}\right\rangle$',fontsize=15)
#axs[1].plot(ts, Q2j, linewidth =3, color='blue')
axs[1].plot(ts, Q2t, linewidth =3, color='g', linestyle='dashed')
axs[1].set_ylabel(r'$\left\langle\hat{P}\right\rangle$',fontsize=15)
#axs[2].plot(ts, Q3j, linewidth =3, color='blue')
axs[2].plot(ts, Q3t, linewidth =3, color='g', linestyle='dashed')
axs[2].set_ylabel(r'$\textrm{var}\hat{X}$',fontsize=15)
#axs[3].plot(ts, Q4j, linewidth =3, color='blue')
axs[3].plot(ts, Q4t, linewidth =3, color='g', linestyle='dashed')
axs[3].set_ylabel(r'$\textrm{cov}\left(\hat{X},\hat{P}\right)$',fontsize=15)
#axs[4].plot(ts, Q5j, linewidth =3, color='blue')
axs[4].plot(ts, Q5t, linewidth =3, color='g', linestyle='dashed')
axs[4].set_ylabel(r'$\textrm{var}\hat{P}$',fontsize=15)
#axs[5].plot(ts, rs,linewidth =3, color='blue')
axs[5].plot(ts, rop_stratj,linewidth =3, color='g', linestyle='dashed')
axs[5].set_ylabel(r'$r$',fontsize=15)
axs[5].set_xlabel(r'$t$',fontsize=15)
#axs[6].plot(ts, theta_t,linewidth =3, color='g', linestyle='dashed')
#axs[7].plot(ts, l1_t, linewidth =3, color='g', linestyle='dashed')
plt.subplots_adjust(wspace=0.22, hspace=0.08)
axs[0].legend(loc=1,fontsize=15)

#plt.savefig(script_dir+'/Plots/trajectory_binomial_code2.pdf',bbox_inches='tight')


