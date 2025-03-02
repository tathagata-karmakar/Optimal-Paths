#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:36:13 2024

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
Dirname = script_dir+"/Data/Cat_to_ground"
Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)
Q1i, Q2i, Q3i, Q4i, Q5i = vars_calc(Ops, rho_ir, rho_ii)
Q1f, Q2f, Q3f, Q4f, Q5f = vars_calc(Ops, rho_fr, rho_fi)
with h5py.File(Dirname+'/Optimal_control_solution.hdf5', 'r') as f:
    Initvals = np.array(f['Initvals'])
    l1_t = np.array(f['l1_t'])
    theta_t = np.array(f['theta_t'])
    Q1t = np.array(f['Q1t'])
    Q2t = np.array(f['Q2t'])
    Q3t = np.array(f['Q3t'])
    Q4t = np.array(f['Q4t'])
    Q5t = np.array(f['Q5t'])
    rop_stratj = np.array(f['ML_readouts'])

'''
G100 = np.matmul(np_Idmat[0], Initvals)
G010 = np.matmul(np_Idmat[1], Initvals)
k100 = np.matmul(np_Idmat[2], Initvals)
k010 = np.matmul(np_Idmat[3], Initvals)
G200 = np.matmul(np_Idmat[4], Initvals)
G110 = np.matmul(np_Idmat[5], Initvals)
G020 = np.matmul(np_Idmat[6], Initvals)
k200 = np.matmul(np_Idmat[7], Initvals)
k110 = np.matmul(np_Idmat[8], Initvals)
k020 = np.matmul(np_Idmat[9], Initvals)
#rho_f_simul1, X_simul1, P_simul1, varX_simul1, covXP_simul1, varP_simul1, rop_strat,nbar, theta_t = OPsoln_control_l10(X, P, H, rho_i, alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm, ts,   tau, 1)
#rho_f_simuld, X_simuld, P_simuld, varX_simuld, covXP_simuld, varP_simuld, rop_stratd,nbard = OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)# OPsoln_control_l10(X, P, H, rho_i, alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm, ts,   tau, 1)
'''


#Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, rho_f_simul2r, rho_f_simul2i, rop_stratj, Jval = OP_wcontrol(jnp.array(Initvals)[:10], Ops, rho_ir, rho_ii,  l1_t, theta_t, params)

#Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, theta_tj, l1_tj, rho_f_simul2r, rho_f_simul2i, rop_stratj, Jval = OPintegrate_strat(jnp.array(Initvals), Ops, rho_ir, rho_ii, params)
fig, axs = plt.subplots(4,2,figsize=(12,8),sharex='all')

ts = params[1]

Q1i = ExpVal(Ops[0], Ops[1], rho_ir, rho_ii).item()#expect(X,rho_i)
Q2i = ExpVal(Ops[2], Ops[3], rho_ir, rho_ii).item()
Q3i = ExpVal(Ops[6], Ops[7], rho_ir, rho_ii).item()-Q1i**2
Q5i = ExpVal(Ops[10], Ops[11], rho_ir, rho_ii).item()-Q2i**2
Q4i = ExpVal(Ops[8], Ops[9], rho_ir, rho_ii).item()/2.0-Q2i*Q1i

Q1f = ExpVal(Ops[0], Ops[1], rho_fr, rho_fi).item()#expect(X,rho_i)
Q2f = ExpVal(Ops[2], Ops[3], rho_fr, rho_fi).item()
Q3f = ExpVal(Ops[6], Ops[7], rho_fr, rho_fi).item()-Q1f**2
Q5f = ExpVal(Ops[10], Ops[11], rho_fr, rho_fi).item()-Q2f**2
Q4f = ExpVal(Ops[8], Ops[9], rho_fr, rho_fi).item()/2.0-Q2f*Q1f

#axs[0].plot(ts, Q1j, linewidth =3, color='blue', label = 'w control')
axs[0,0].plot(ts, Q1t, linewidth =3.5, color='g', linestyle='-')
axs[0,0].set_ylabel(r'$\left\langle\hat{X}\right\rangle$',fontsize=15)
axs[0,0].tick_params(labelsize=14)
axs[0,0].plot(0, Q1i, "o", color = 'b', markersize =12)
axs[0,0].plot(ts[-1], Q1f, "x", color = 'r', markersize =12)
#axs[1].plot(ts, Q2j, linewidth =3, color='blue')
axs[1,0].plot(ts, Q2t, linewidth =3.5, color='g', linestyle='-')
axs[1,0].set_ylabel(r'$\left\langle\hat{P}\right\rangle$',fontsize=15)
axs[1,0].tick_params(labelsize=14)
axs[1,0].plot(0, Q2i, "o", color = 'b', markersize =12)
axs[1,0].plot(ts[-1], Q2f, "x", color = 'r', markersize =12)

#axs[2].plot(ts, Q3j, linewidth =3, color='blue')
axs[2,0].plot(ts, Q3t, linewidth =3.5, color='g', linestyle='-')
axs[2,0].set_ylabel(r'$\textrm{var}\hat{X}$',fontsize=15)
axs[2,0].tick_params(labelsize=14)
axs[2,0].plot(0, Q3i, "o", color = 'b', markersize =12)
axs[2,0].plot(ts[-1], Q3f, "x", color = 'r', markersize =12)
#axs[3].plot(ts, Q4j, linewidth =3, color='blue')
axs[3,0].plot(ts, Q4t, linewidth =3.5, color='g', linestyle='-')
axs[3,0].set_ylabel(r'$\textrm{cov}\left(\hat{X},\hat{P}\right)$',fontsize=15)
axs[3,0].tick_params(labelsize=14)
axs[3,0].plot(0, Q4i, "o", color = 'b', markersize =12)
axs[3,0].plot(ts[-1], Q4f, "x", color = 'r', markersize =12)
axs[3,0].set_xlabel(r'$t$',fontsize=15)
axs[3,0].xaxis.set_label_coords(1.1, -0.15)


#axs[4].plot(ts, Q5j, linewidth =3, color='blue')
axs[0,1].plot(ts, Q5t, linewidth =3.5, color='g', linestyle='-')
axs[0,1].set_ylabel(r'$\textrm{var}\hat{P}$',fontsize=15)
axs[0,1].tick_params(labelsize=14)
axs[0,1].plot(0, Q5i, "o", color = 'b', markersize =12)
axs[0,1].plot(ts[-1], Q5f, "x", color = 'r', markersize =12)
#axs[5].plot(ts, rs,linewidth =3, color='blue')
axs[1,1].plot(ts, rop_stratj,linewidth =3.5, color='g', linestyle='-')
axs[1,1].set_ylabel(r'$r^\star$',fontsize=15)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(ts, theta_t,linewidth =3.5, color='g', linestyle='--')
axs[2,1].set_ylabel(r'$\theta^\star$',fontsize=15)
axs[2,1].axhline(y=np.pi/2.0, color='k', linestyle = 'dashed')
axs[2,1].axhline(y=-np.pi/2.0, color='k', linestyle = 'dashed')
axs[2,1].set_ylim(-np.pi/2.0-0.1, np.pi/2.0+0.1)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(ts, l1_t, linewidth =3.5, color='g', linestyle='--')
axs[3,1].set_ylabel(r'$\lambda_1^\star$',fontsize=15)
axs[3,1].tick_params(labelsize=14)

plt.subplots_adjust(wspace=0.22, hspace=0.08)
plt.savefig(Dirname+'/Plots/OC_plot.pdf',bbox_inches='tight')
