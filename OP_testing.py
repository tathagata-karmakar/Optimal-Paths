#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 11:31:25 2024

@author: t_karmakar
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

nlevels = 25

#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 3
ts = np.linspace(t_i, t_f, 500)
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
in_alr = 0.0
in_ali = 0.0
fin_alr = 1.7
fin_ali = 0.8

rho_i = squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
#rho_f = basis(nlevels,4)
rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))

nsteps = 10
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
#Ljump = X
#Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()
rho_f_int = rho_f_int*rho_f_int.dag()
sigma_i = rand_herm(nlevels)
sigma_i = sigma_i-expect(sigma_i, rho_i)

XItf = X*np.cos(t_f)+P*np.sin(t_f)
PItf = P*np.cos(t_f)-X*np.sin(t_f)

Q1 = expect(XItf,rho_f_int)
Q2 = expect(PItf,rho_f_int)
V1 = expect(XItf*XItf,rho_f_int)-Q1**2
V3 = expect(PItf*PItf,rho_f_int)-Q2**2
V2 = (expect(XItf*PItf+PItf*XItf,rho_f_int)-2*Q1*Q2)/2.0

lrate = 1e-2
q3, q4, q5, alr, ali, A, B, q1t, q2t,rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)

#alr = np.random.rand()
#ali = np.random.rand()
#A = np.random.rand()
#B = np.random.rand()
Initials = jnp.array([alr, ali, A, B])
#Initials = jnp.array(np.random.rand(4))
theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)
tb = 0
jnpId = jnp.identity(nlevels)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnpX2 = jnp.matmul(jnpX, jnpX)
jnpP2 = jnp.matmul(jnpP, jnpP)
jnpXP = jnp.matmul(jnpX, jnpP)
jnpPX = jnp.matmul(jnpP, jnpX)
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f_int.full())

Q1i = jnp.trace(jnp.matmul(jnpX, jnp_rho_i)).real
Q2i = jnp.trace(jnp.matmul(jnpP, jnp_rho_i)).real
jnpVX = jnp.trace(jnp.matmul(jnpX2, jnp_rho_i)).real-Q1i**2
jnpVP = jnp.trace(jnp.matmul(jnpP2, jnp_rho_i)).real-Q1i**2
jnpCXP = jnp.trace(jnp.matmul(jnpXP+jnpPX, jnp_rho_i)).real/2.0-Q1i*Q2i

Q3i = 2*jnpVX
Q4i = 2*jnpCXP
Q5i = 2*jnpVP

Q1d = expect(X,rho_i)
Q2d = expect(P,rho_i)
Q3d = 2*(expect(X*X,rho_i)-Q1d**2)
Q5d = 2*(expect(P*P,rho_i)-Q2d**2)
Q4d = (expect(X*P+P*X,rho_i)-2*Q1d*Q2d)

I_tR = jnp.array([0.0])
I_tI = jnp.array([0.0])
#Initials1, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho, I_tR, I_tI, theta_t, ts, tau, dt, l1 = rho_update(0,(Initials, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho_i, I_tR, I_tI,  theta_t, ts, tau, dt, 0))
#
rho_f_simul1, X_simul1, P_simul1, varX_simul1, covXP_simul1, varP_simul1, rop_strat = OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)

fig, axs = plt.subplots(6,1,figsize=(6,12),sharex='all')
axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian Approx')
axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'General')
axs[0].plot(t_i, expect(X,rho_i), "o", color = 'b')
axs[0].plot(t_f, expect(X,rho_f), "X" , color = 'r')
axs[0].plot(t_f, Q1, "^" , color = 'k')

axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
axs[1].plot(ts, P_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
axs[1].plot(t_i, expect(P,rho_i), "o", color = 'b')
axs[1].plot(t_f, expect(P,rho_f), "X", color = 'r')
axs[1].plot(t_f, Q2, "^" , color = 'k')

axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
axs[2].plot(t_i, expect((X-q1t[0])*(X-q1t[0]), rho_i), "o", color = 'b')
axs[2].plot(t_f, expect((X-q1t[-1])*(X-q1t[-1]), rho_f), "X", color = 'r')
axs[2].plot(t_f, V1, "^" , color = 'k')



axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
axs[3].plot(t_i, expect((X-q1t[0])*(P-q2t[0])/2.0+(P-q2t[0])*(X-q1t[0])/2.0, rho_i), "o", color = 'b')
axs[3].plot(t_f, expect((X-q1t[-1])*(P-q2t[-1])/2.0+(P-q2t[-1])*(X-q1t[-1])/2.0, rho_f), "X", color = 'r')
axs[3].plot(t_f, V2, "^" , color = 'k')

axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
axs[4].plot(t_i, expect((P-q2t[0])*(P-q2t[0]), rho_i), "o", color = 'b')
axs[4].plot(t_f, expect((P-q2t[-1])*(P-q2t[-1]), rho_f), "X", color = 'r')
axs[4].plot(t_f, V3, "^" , color = 'k')

axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
axs[5].plot(ts, rop_strat,linewidth =3, color='red', linestyle='dashed')

#axs[5].plot(ts,4*(varX_simul1*varP_simul1-covXP_simul1**2),linewidth = 4)

axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

print ('Fidelity ', fidelity(rho_f_simul1, rho_f_int))