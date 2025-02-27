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
from scipy.integrate import simpson as intg
#from google.colab import files
#from google.colab import drive
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
from Eff_OP_Functions import *
from Initialization import *
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

Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)
ts = params[1]

with h5py.File(Dirname+'/Alternate_control.hdf5', 'r') as f:
    Initvals = np.array(f['Initials_sample'])
    l10 = np.array(f['l1_t_sample'])
    theta0 = np.array(f['theta_t_sample'])



fig, axs = plt.subplots(2,1,figsize=(6,4),sharex='all')
axs[0].tick_params(labelsize=14)
axs[1].tick_params(labelsize=14)
axs[1].plot(ts, l10, linewidth =3, color='orange')
axs[0].plot(ts, theta0, linewidth =3, color='orange')
axs[0].set_ylabel(r'$\theta(t)$',fontsize=15)
axs[1].set_ylabel(r'$\lambda_1(t)$',fontsize=15)
axs[1].set_xlabel(r'$t$',fontsize=15)
plt.subplots_adjust(wspace=0.22, hspace=0.08)

plt.savefig(Dirname+'/Plots/sample_control.pdf',bbox_inches='tight')


