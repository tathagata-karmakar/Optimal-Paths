#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:22:59 2024

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


hf = h5py.File('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Data/Histogram_Ex4_new.hdf5', 'r')

nlevels = int(np.array(hf['nlevels']))
#a = destroy(nlevels)
tau = np.array(hf['tau']).item()
theta_t = np.array(hf['theta_t'])
theta0 = np.zeros(len(theta_t))  #Control parameter \theta = 0
l1_t = np.array(hf['l1_t'])
l10 = np.zeros(len(l1_t))   #Control parameter \lambda_1 = 0
ts = np.array(hf['ts'])
#ropt = np.array(hf['r_t'])
rho_i =Qobj(np.array(hf['rho_i']))
rho_f = Qobj(np.array(hf['rho_f_target']))
Initvals = np.array(hf['Initvals'])
dt = ts[1]-ts[0]
samplesize =np.array(hf['Sample_size']).item()
fidelities0 = np.array(hf['Fidelities_wo_control'])
fidelities_OP = np.array(hf['Fidelities_w_control'])
#np_Idmat=np.identity(10)
#Idmat = jnp.array(np_Idmat)


fig, ax = plt.subplots(figsize=(6,4))


ax.hist(fidelities_OP, label = 'Optimal control', hatch ='|')
ax.hist(fidelities0, label="Sample control", alpha = 0.6, hatch ='\\')
ax.set_xlabel(r'$\mathcal{F}\left(\hat{\rho}_f,\hat{\rho}(t_f)\right)$', fontsize=18)
ax.set_ylabel('Number of Trajectories', fontsize=18)
ax.tick_params(labelsize=15)
ax.legend(loc=2,fontsize=15)
ax.set_xlim(0,1)
#plt.savefig('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Plots/histogramtmp.pdf',bbox_inches='tight')

hf.close()
