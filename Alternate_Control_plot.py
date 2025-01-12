#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:19:07 2024

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


hf = h5py.File('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Data/Optimal_control_Ex4_new.hdf5', 'r')

nlevels = int(np.array(hf['nlevels']))
a = destroy(nlevels)
tau = np.array(hf['tau']).item()
theta_t = np.array(hf['theta_t'])
theta0 =np.array(hf['theta_t_sample'])  #Control parameter \theta = 0
l1_t = np.array(hf['l1_t'])
l10 = np.array(hf['l1_t_sample'])  #Control parameter \lambda_1 = 0
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

hf.close()


fig, axs = plt.subplots(2,1,figsize=(6,4),sharex='all')
axs[0].tick_params(labelsize=14)
axs[1].tick_params(labelsize=14)
axs[1].plot(ts, l10, linewidth =3, color='orange')
axs[0].plot(ts, theta0, linewidth =3, color='orange')
axs[0].set_ylabel(r'$\theta(t)$',fontsize=15)
axs[1].set_ylabel(r'$\lambda_1(t)$',fontsize=15)
axs[1].set_xlabel(r'$t$',fontsize=15)
plt.subplots_adjust(wspace=0.22, hspace=0.08)

#plt.savefig('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Plots/sample_control_ex4.pdf',bbox_inches='tight')


