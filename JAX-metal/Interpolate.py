#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:48:24 2025

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


#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)


hf = h5py.File(script_dir+'/Optimal_control_Extmp.hdf5', 'r')

nlevels = int(np.array(hf['nlevels']))
a = destroy(nlevels)
tau = np.array(hf['tau']).item()
l1max = np.array(hf['l1max']).item()
theta_t = np.array(hf['theta_t'])
l1_t = np.array(hf['l1_t'])
ts = np.array(hf['ts'])
ropt = np.array(hf['r_t'])
rho_i =Qobj(np.array(hf['rho_i']))
rho_f = Qobj(np.array(hf['rho_f_target']))
Initvals = np.array(hf['Initvals'])
dt = ts[1]-ts[0]
np_Idmat=np.identity(10)
Idmat = jnp.array(np_Idmat)
hf.close()

tsi = np.linspace(ts[0], ts[-1], 5*len(ts))
l1_ti = np.interp(tsi, ts, l1_t)
theta_ti = np.interp(tsi, ts, theta_t)

fig, axs = plt.subplots(2,1,figsize=(10,8),sharex='all')

axs[0].plot(ts, l1_t, linewidth =3.5, color='g', linestyle='-')
axs[0].plot(tsi, l1_ti, linewidth =3.5, color='r', linestyle='--')
axs[0].set_ylabel(r'$\lambda_1(t)$',fontsize=15)
axs[0].tick_params(labelsize=14)

#axs[1].plot(ts, Q2j, linewidth =3, color='blue')
axs[1].plot(ts, theta_t, linewidth =3.5, color='g', linestyle='-')
axs[1].plot(tsi, theta_ti, linewidth =3.5, color='r', linestyle='--')
axs[1].set_ylabel(r'$\theta(t)$',fontsize=15)
axs[1].set_xlabel(r'$t$',fontsize=15)
axs[1].tick_params(labelsize=14)
