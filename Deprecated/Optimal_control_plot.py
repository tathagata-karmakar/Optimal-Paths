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


hf = h5py.File(script_dir+'/Data/Optimal_control_Extmp2.hdf5', 'r')

nlevels = int(np.array(hf['nlevels']))
a = destroy(nlevels)
tau = np.array(hf['tau']).item()
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

X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
X2 = X*X
CXP =X*P+P*X
P2 = P*P
#Ljump = X
#Mjump = P
#rho_i = rho_i*rho_i.dag()
#rho_f = rho_f*rho_f.dag()

jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f.full())
jnpX2= jnp.matmul(jnpX, jnpX)


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

Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, rho_f_simul2, rop_stratj = OP_wcontrol(Initvals, X.full(), P.full(), H.full(), X2.full(), CXP.full(), P2.full(),  rho_i.full(), l1_t, theta_t, ts, dt,  tau,  np_Idmat, np.identity(nlevels))

fig, axs = plt.subplots(4,2,figsize=(12,8),sharex='all')
q1i = expect(X,rho_i)
q1f = expect(X,rho_f)
q2i = expect(P,rho_i)
q2f = expect(P,rho_f)
q3i = expect(X2,rho_i)-q1i**2
q3f = expect(X2,rho_f)-q1f**2
P2 = P*P
cxp =X*P+P*X
q5i = expect(P2,rho_i)-q2i**2
q5f = expect(P2,rho_f)-q2f**2
q4i = expect(cxp,rho_i)/2-q1i*q2i
q4f = expect(cxp,rho_f)/2-q1f*q2f
#axs[0].plot(ts, Q1j, linewidth =3, color='blue', label = 'w control')
axs[0,0].plot(ts, Q1j1, linewidth =3.5, color='g', linestyle='-')
axs[0,0].set_ylabel(r'$\left\langle\hat{X}\right\rangle$',fontsize=15)
axs[0,0].tick_params(labelsize=14)
axs[0,0].plot(0, q1i, "o", color = 'b', markersize =12)
axs[0,0].plot(ts[-1], q1f, "x", color = 'r', markersize =12)
#axs[1].plot(ts, Q2j, linewidth =3, color='blue')
axs[1,0].plot(ts, Q2j1, linewidth =3.5, color='g', linestyle='-')
axs[1,0].set_ylabel(r'$\left\langle\hat{P}\right\rangle$',fontsize=15)
axs[1,0].tick_params(labelsize=14)
axs[1,0].plot(0, q2i, "o", color = 'b', markersize =12)
axs[1,0].plot(ts[-1], q2f, "x", color = 'r', markersize =12)

#axs[2].plot(ts, Q3j, linewidth =3, color='blue')
axs[2,0].plot(ts, Q3j1, linewidth =3.5, color='g', linestyle='-')
axs[2,0].set_ylabel(r'$\textrm{var}\hat{X}$',fontsize=15)
axs[2,0].tick_params(labelsize=14)
axs[2,0].plot(0, q3i, "o", color = 'b', markersize =12)
axs[2,0].plot(ts[-1], q3f, "x", color = 'r', markersize =12)
#axs[3].plot(ts, Q4j, linewidth =3, color='blue')
axs[3,0].plot(ts, Q4j1, linewidth =3.5, color='g', linestyle='-')
axs[3,0].set_ylabel(r'$\textrm{cov}\left(\hat{X},\hat{P}\right)$',fontsize=15)
axs[3,0].tick_params(labelsize=14)
axs[3,0].plot(0, q4i, "o", color = 'b', markersize =12)
axs[3,0].plot(ts[-1], q4f, "x", color = 'r', markersize =12)
axs[3,0].set_xlabel(r'$t$',fontsize=15)
axs[3,0].xaxis.set_label_coords(1.1, -0.15)


#axs[4].plot(ts, Q5j, linewidth =3, color='blue')
axs[0,1].plot(ts, Q5j1, linewidth =3.5, color='g', linestyle='-')
axs[0,1].set_ylabel(r'$\textrm{var}\hat{P}$',fontsize=15)
axs[0,1].tick_params(labelsize=14)
axs[0,1].plot(0, q5i, "o", color = 'b', markersize =12)
axs[0,1].plot(ts[-1], q5f, "x", color = 'r', markersize =12)
#axs[5].plot(ts, rs,linewidth =3, color='blue')
axs[1,1].plot(ts, rop_stratj,linewidth =3.5, color='g', linestyle='-')
axs[1,1].set_ylabel(r'$r^\star$',fontsize=15)
axs[1,1].tick_params(labelsize=14)

axs[2,1].plot(ts, theta_t,linewidth =3.5, color='g', linestyle='--')
axs[2,1].set_ylabel(r'$\theta^\star$',fontsize=15)
axs[2,1].tick_params(labelsize=14)

axs[3,1].plot(ts, l1_t, linewidth =3.5, color='g', linestyle='--')
axs[3,1].set_ylabel(r'$\lambda_1^\star$',fontsize=15)
axs[3,1].tick_params(labelsize=14)

plt.subplots_adjust(wspace=0.22, hspace=0.08)
#plt.savefig(script_dir+'/Plots/Cat_State_OC.pdf',bbox_inches='tight')

hf.close()

