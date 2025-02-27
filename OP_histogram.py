#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:08:23 2024

@author: tatha_k
"""

import os,sys
os.environ['JAX_PLATFORMS'] = 'cpu'
#os.environ['JAX_DISABLE_JIT'] = '1'

os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
os.environ['CHECKPOINT_PATH']='${path_to_checkpoints}'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

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


with h5py.File(Dirname+'/Optimal_control_solution.hdf5', 'r') as f:
    Initvals = np.array(f['Initvals'])
    l1_t = np.array(f['l1_t'])
    theta_t = np.array(f['theta_t'])
    

with h5py.File(Dirname+'/Alternate_control.hdf5', 'r') as f:
    Initvals_s = np.array(f['Initials_sample'])
    l10 = np.array(f['l1_t_sample'])
    theta0 = np.array(f['theta_t_sample'])
    

tsi = np.linspace(ts[0], ts[-1], 5*len(ts))
l1_ti = np.interp(tsi, ts, l1_t)
theta_ti = np.interp(tsi, ts, theta_t)
l10i = np.interp(tsi, ts, l10)
theta0i = np.interp(tsi, ts, theta0)
newparams = (l1max, tsi, tsi[1]-tsi[0], tau, Idmat)

samplesize = 4

fidelitiesC = np.zeros(samplesize)
fidelities_OC = np.zeros(samplesize)
stime = time.time()


#print (nsample)
#Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simul, rs= OP_stochastic_trajectory(X.full(), P.full(), H.full(), X2.full(), rho_i.full(), l10, theta0, ts, dt,  tau,  np.identity(nlevels))
dWt = jnp.array(np.random.normal(scale=np.sqrt(dt), size = (samplesize,len(theta0i))))
batch_stochastic_trajectory = jax.vmap(OP_stochastic_trajectory_JAX, in_axes=[0,  None, None, None], out_axes = 0)
fidelitiesC = batch_stochastic_trajectory(dWt, l10i, theta0i, newparams)
#fidelitiesC[nsample] = Fidelity_PS(rho_f_simul0, jnp_rho_f).item()

dWt = jnp.array(np.random.normal(scale=np.sqrt(dt), size = (samplesize,len(theta_ti))))
fidelities_OC = batch_stochastic_trajectory(dWt,   l1_ti, theta_ti, newparams)

fidelitiesC = np.array(fidelitiesC)
fidelities_OC = np.array(fidelities_OC)
'''
for nsample in range(samplesize):
    print (nsample)
    #Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simul, rs= OP_stochastic_trajectory(X.full(), P.full(), H.full(), X2.full(), rho_i.full(), l10, theta0, ts, dt,  tau,  np.identity(nlevels))
    dWt = np.random.normal(scale=np.sqrt(dt), size = len(theta0i))
    fidelitiesC[nsample] = OP_stochastic_trajectory_JAX(dWt, Ops, jnp_rho_ir, jnp_rho_ii, jnp_rho_fr, jnp_rho_fi, l10i, theta0i, params).item()
    #fidelitiesC[nsample] = Fidelity_PS(rho_f_simul0, jnp_rho_f).item()
    
    dWt = np.random.normal(scale=np.sqrt(dt), size = len(theta_ti))
    fidelities_OC[nsample] = OP_stochastic_trajectory_JAX(dWt, Ops, jnp_rho_ir, jnp_rho_ii, jnp_rho_fr, jnp_rho_fi, l1_ti, theta_ti, params).item()
    #fidelities_OC[nsample] = Fidelity_PS(rho_f_simul, jnp_rho_f).item()
    #print (fidelities0[nsample])
    
#for nsample in range(samplesize):
   # print (nsample)
    #dWt = np.random.normal(scale=np.sqrt(dt), size = len(theta_t))
   # rho_f_simul = OP_stochastic_trajectory_JAX(jnpX, jnpP, jnpH, jnpX2, jnpCXP, jnpP2, jnp_rho_i, l1_t, theta_t, dWt, ts, dt,  tau,  jnpId)
    #fidelities_OC[nsample] = Fidelity_PS(rho_f_simul, jnp_rho_f).item()
    #print (fidelities0[nsample])
'''
print ('End time', time.time()-stime)
    




with h5py.File(Dirname+"/Histogram.hdf5", "w") as f:
    #dset1 = f.create_dataset("nlevels", data = nlevels, dtype ='int')
    #dset2 = f.create_dataset("rho_i", data = rho_i.full())
    #dset3 = f.create_dataset("rho_f_target", data = rho_f.full())
    #dset4 = f.create_dataset("ts", data = ts)
    #dset5 = f.create_dataset("tau", data = tau)
    #dset6 = f.create_dataset("theta_t", data = theta_t)
    #dset7 = f.create_dataset("l1_t", data = l1_t)
    dset1 = f.create_dataset("Fidelities_wo_control", data = fidelitiesC)
    dset2 = f.create_dataset("Fidelities_w_control", data = fidelities_OC)
    #dset10 = f.create_dataset("Initvals", data = Initvals)   
    dset3 = f.create_dataset("Sample_size", data = samplesize)   


fig, ax = plt.subplots()

ax.hist(fidelitiesC)
ax.hist(fidelities_OC)
ax.set_xlabel(r'$\mathcal{F}\left(\hat{\rho}_f,\hat{\rho}(t_f)\right)$')
