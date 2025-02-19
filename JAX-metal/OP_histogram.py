#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 10:08:23 2024

@author: tatha_k
"""

import os,sys
#os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_DISABLE_JIT'] = '1'
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


hf = h5py.File(script_dir+'/Optimal_control_Extmp1.hdf5', 'r')

nlevels = int(np.array(hf['nlevels']))
a = destroy(nlevels)
tau = np.array(hf['tau']).item()
l1max = np.array(hf['l1max']).item()
theta_t = np.array(hf['theta_t'])
theta0 =np.array(hf['theta_t_sample'])  #Control parameter \theta = 0
l1_t = np.array(hf['l1_t'])
l10 = np.array(hf['l1_t_sample'])  #Control parameter \lambda_1 = 0
ts = np.array(hf['ts'])
ropt = np.array(hf['r_t'])
rho_i =Qobj(np.array(hf['rho_i']))
rho_f = Qobj(np.array(hf['rho_f_target']))
Initvals = np.array(hf['Initvals'])
hf.close()
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


jnpId = jnp.identity(nlevels)
jnpXr = jnp.array(X.full().real)
jnpXi = jnp.array(X.full().imag)
jnpPr = jnp.array(P.full().real)
jnpPi = jnp.array(P.full().imag)
jnpHr = jnp.array(H.full().real)
jnpHi = jnp.array(H.full().imag)
CXP = X*P+P*X
jnpCXPr = jnp.array(CXP.full().real)
jnpCXPi = jnp.array(CXP.full().imag)
jnp_rho_ir = jnp.array(rho_i.full().real)
jnp_rho_ii= jnp.array(rho_i.full().imag)
jnp_rho_fr = jnp.array(rho_f.full().real)
jnp_rho_fi = jnp.array(rho_f.full().imag)
jnpX2r= jnp.matmul(jnpXr, jnpXr)-jnp.matmul(jnpXi, jnpXi)
jnpX2i= jnp.matmul(jnpXr, jnpXi)+jnp.matmul(jnpXi, jnpXr)
jnpP2r= jnp.matmul(jnpPr, jnpPr)-jnp.matmul(jnpPi, jnpPi)
jnpP2i= jnp.matmul(jnpPr, jnpPi)+jnp.matmul(jnpPi, jnpPr)
P2 = P*P

Ops = (jnpXr, jnpXi, jnpPr, jnpPi,  jnpHr, jnpHi, jnpX2r, jnpX2i, jnpCXPr, jnpCXPi,  jnpP2r, jnpP2i, jnpId)
tsi = np.linspace(ts[0], ts[-1], len(ts))
l1_ti = np.interp(tsi, ts, l1_t)
theta_ti = np.interp(tsi, ts, theta_t)
l10i = np.interp(tsi, ts, l10)
theta0i = np.interp(tsi, ts, theta0)
params = (l1max, tsi, tsi[1]-tsi[0], tau, Idmat)

samplesize =2

fidelitiesC = np.zeros(samplesize)
fidelities_OC = np.zeros(samplesize)
stime = time.time()


#print (nsample)
#Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simul, rs= OP_stochastic_trajectory(X.full(), P.full(), H.full(), X2.full(), rho_i.full(), l10, theta0, ts, dt,  tau,  np.identity(nlevels))
dWt = jnp.array(np.random.normal(scale=np.sqrt(dt), size = (samplesize,len(theta0i))))
batch_stochastic_trajectory = jax.vmap(OP_stochastic_trajectory_JAX, in_axes=[0, None, None, None, None, None, None, None, None], out_axes = 0)
fidelitiesC = batch_stochastic_trajectory(dWt, Ops, jnp_rho_ir, jnp_rho_ii, jnp_rho_fr, jnp_rho_fi, l10i, theta0i, params)
#fidelitiesC[nsample] = Fidelity_PS(rho_f_simul0, jnp_rho_f).item()

dWt = jnp.array(np.random.normal(scale=np.sqrt(dt), size = (samplesize,len(theta_ti))))
fidelities_OC = batch_stochastic_trajectory(dWt, Ops, jnp_rho_ir, jnp_rho_ii, jnp_rho_fr, jnp_rho_fi, l1_ti, theta_ti, params)

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
    




with h5py.File(script_dir+"/Histogram_Extmp1.hdf5", "w") as f:
    dset1 = f.create_dataset("nlevels", data = nlevels, dtype ='int')
    dset2 = f.create_dataset("rho_i", data = rho_i.full())
    dset3 = f.create_dataset("rho_f_target", data = rho_f.full())
    dset4 = f.create_dataset("ts", data = ts)
    dset5 = f.create_dataset("tau", data = tau)
    dset6 = f.create_dataset("theta_t", data = theta_t)
    dset7 = f.create_dataset("l1_t", data = l1_t)
    dset8 = f.create_dataset("Fidelities_wo_control", data = fidelitiesC)
    dset9 = f.create_dataset("Fidelities_w_control", data = fidelities_OC)
    dset10 = f.create_dataset("Initvals", data = Initvals)   
    dset11 = f.create_dataset("Sample_size", data = samplesize)   
    
f.close()

fig, ax = plt.subplots()

ax.hist(fidelitiesC)
ax.hist(fidelities_OC)
ax.set_xlabel(r'$\mathcal{F}\left(\hat{\rho}_f,\hat{\rho}(t_f)\right)$')
