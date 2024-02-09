#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:20:05 2024

@author: t_karmakar
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
from scipy.integrate import simps as intg
from google.colab import files
from google.colab import drive
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
from OP_Functions import *
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


import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
#torch.backends.cuda.cufft_plan_cache[0].max_size = 32
torch.autograd.set_detect_anomaly(True)

'''
Example Optimal Path Simulation for a Continuously Monitored
Quantum Harmonic Oscillator with Controlled Quadrature Measurement
'''

nlevels = 6
rho_i = basis(nlevels,0)
#rho_f = basis(nlevels,4)
rho_f = coherent(nlevels, 0.5+1j*0.17)
a = destroy(nlevels)
t_i = 0
t_f = 3
ts = np.linspace(t_i,t_f,300)
dt = ts[1]-ts[0]
tau = 10
nsteps = 1
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
Ljump = X
Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()

#xvec = np.linspace(-5,5,200)
#pvec = np.linspace(-5,5,200)
#W_i = wigner(rho_i,xvec,pvec)

q3, q4, q5, alr, ali, A, B, q1t, q2t = OP_PRXQ_Params(Ljump, Mjump, rho_i, rho_f, ts, tau)

rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul = OPsoln_SHO(Ljump, Mjump, H, rho_i, alr, ali, A, B, ts,  tau, 1)


fig, axs = plt.subplots(5,1,figsize=(6,12),sharex='all')
axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'PRXQ')
axs[0].plot(ts, X_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'General')
axs[0].plot(t_i, q1t[0], "o", color = 'b')
axs[0].plot(t_f, q1t[-1], "X" , color = 'r')
axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
axs[1].plot(ts, P_simul, linewidth =4, linestyle = 'dashed', color = 'blue')
axs[1].plot(t_i, q2t[0], "o", color = 'b')
axs[1].plot(t_f, q2t[-1], "X", color = 'r')
axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 15)
axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 15)
axs[0].tick_params(labelsize=15)
axs[1].tick_params(labelsize=15)
axs[0].legend(loc=4,fontsize=12)

axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[2].plot(ts, varX_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'General')
axs[2].plot(t_i, expect((Ljump-q1t[0])*(Ljump-q1t[0]), rho_i), "o", color = 'b')
axs[2].plot(t_f, expect((Ljump-q1t[-1])*(Ljump-q1t[-1]), rho_f), "X", color = 'r')

axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[3].plot(ts, covXP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'General')
axs[3].plot(t_i, expect((Ljump-q1t[0])*(Mjump-q2t[0])/2.0+(Mjump-q2t[0])*(Ljump-q1t[0])/2.0, rho_i), "o", color = 'b')
axs[3].plot(t_f, expect((Ljump-q1t[-1])*(Mjump-q2t[-1])/2.0+(Mjump-q2t[-1])*(Ljump-q1t[-1])/2.0, rho_f), "X", color = 'r')

axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[4].plot(ts, varP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'General')
axs[4].plot(t_i, expect((Mjump-q2t[0])*(Mjump-q2t[0]), rho_i), "o", color = 'b')
axs[4].plot(t_f, expect((Mjump-q2t[-1])*(Mjump-q2t[-1]), rho_f), "X", color = 'r')


axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/superoscillations/plots/tmp/fig2.pdf',bbox_inches='tight')
