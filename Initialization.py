#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:19:18 2025

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
#from Triangle_Fns import *
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
from numba import njit, prange
import numba as nb
from pathlib import Path

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)


Dirname = script_dir+"/Data/Coherent_to_ground"
Path(Dirname).mkdir(parents=True, exist_ok=True)
Path(Dirname+'/Plots').mkdir(parents=True, exist_ok=True)

nlevels = 35

#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 3.0
ts = np.linspace(t_i, t_f, int(t_f/0.0005))
dt = ts[1]-ts[0]
tau = 15.0
q4f = np.sqrt(1+4*tau*tau)-2*tau
q3f = np.sqrt(4*tau*q4f)
q5f = q3f*(1+q4f/(2*tau))

snh2r = -np.sqrt(q4f**2+(q3f-q5f)**2/4.0)
csh2r = (q3f+q5f)/2.0
rparam = snh2r+csh2r
r_sq = np.log(rparam)/2
xiR = r_sq*(q5f-q3f)/(2*snh2r)
xiI = r_sq*(-q4f)/snh2r
fin_alr = 1.35
fin_ali = -0.75
in_alr = -0.15#-0.1#in_alr*np.cos(t_f)+in_ali*np.sin(t_f)
in_ali = 0.25#0.5#in_ali*np.cos(t_f)-in_alr*np.sin(t_f)

'''
Initial and final states
'''
eps =0.1
#rho_f= coherent(nlevels, fin_alr+1j*fin_ali)+coherent(nlevels, -fin_alr-1j*fin_ali)#basis(nlevels, 0)#squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
#rho_i = (basis(nlevels, 0)-basis(nlevels,4))/np.sqrt(2)
rho_f = basis(nlevels, 0)#+basis(nlevels,4))/np.sqrt(2)
#rho_f = (basis(nlevels, 0)+basis(nlevels,4))/np.sqrt(2)
#rho_i = (basis(nlevels, 0)+np.sqrt(3)*basis(nlevels,4))/np.sqrt(4)
#rho_i=coherent(nlevels, in_alr+1j*in_ali)+coherent(nlevels, -in_alr-1j*in_ali)#(coherent(nlevels, in_alr+1j*in_ali)+coherent(nlevels, -in_alr-1j*in_ali))/np.sqrt(2)
#rho_f = basis(nlevels, 0)#+coherent(nlevels, -in_alr-1j*in_ali))#coherent(nlevels, fin_alr+1j*fin_ali)
#rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
rho_i=coherent(nlevels, in_alr+1j*in_ali)
rho_2 = (basis(nlevels,0)+basis(nlevels,4))/np.sqrt(2)
rho_1 =(basis(nlevels,0)-basis(nlevels,4))/np.sqrt(2)

'''
Operator definitions
'''
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
X2 = X*X
#Ljump =
#Mjump = P
rho_i = rho_i*rho_i.dag()
#rho_i = (1-eps)*rho_1*rho_1.dag()+eps*rho_2*rho_2.dag()
rho_f = rho_f*rho_f.dag()
rho_i = rho_i/rho_i.tr()   
rho_f = rho_f/rho_f.tr()


Q1i = expect(X,rho_i)
Q2i = expect(P,rho_i)
Q3i = (expect(X*X,rho_i)-Q1i**2)
Q5i = (expect(P*P,rho_i)-Q2i**2)
Q4i = (expect(P*X+X*P,rho_i)/2.0-Q2i*Q1i)

np_Idmat=np.identity(10)
Idmat = jnp.array(np_Idmat)

l1max = 0.2
tb = 0
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
rho_ir = jnp_rho_ir
rho_ii = jnp_rho_ii
rho_fr = jnp_rho_fr
rho_fi = jnp_rho_fi

params = (l1max, ts, dt, tau, Idmat)

Q1i1, Q2i1, Q3i1, Q4i1, Q5i1 = vars_calc(Ops, rho_ir, rho_ii)

print ('Moments: ', Q1i, Q2i, Q3i, Q4i, Q5i)
print ('Moment check: ', Q1i1, Q2i1, Q3i1, Q4i1, Q5i1)

with h5py.File(Dirname+"/Parameters.hdf5", "w") as f:
    dset1 = f.create_dataset("nlevels", data = nlevels, dtype ='int')
    dset2 = f.create_dataset("rho_i", data = rho_i.full())
    dset3 = f.create_dataset("rho_f_target", data = rho_f.full())
    dset4 = f.create_dataset("ts", data = ts)
    dset5 = f.create_dataset("tau", data = tau)
    dset6 = f.create_dataset("l1max", data = l1max) 
    dset7 = f.create_dataset("Ops", data = Ops) 
    dset8 = f.create_dataset("Idmat", data = np_Idmat) 
    
    
#f.close()