#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:44:46 2024

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

nlevels = 15

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
rparam = np.sqrt(q4f**2+(q3f-q5f)**2/4.0)+(q3f+q5f)/2.0
snh2r = np.sqrt(q4f**2+(q3f-q5f)**2/4.0)
csh2r = (q3f+q5f)/2.0
r_sq = np.log(rparam)/2
xiR = r_sq*(q5f-q3f)/(2*snh2r)
xiI = r_sq*(-q4f)/(snh2r)
in_alr = 0.0
in_ali = 0.0
fin_alr = -0.5
fin_ali = 0.8

rho_i = squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)

#rho_f = basis(nlevels,4)
rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
#rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))


nsteps = 30000
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
rho_i = rho_i*rho_i.dag()
#rho_i = thermal_dm(nlevels, 4)
rho_f = rho_f*rho_f.dag()


lrate = 5*1e-2
q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)
Initials = 0.1*jnp.array(np.random.rand(4)-0.5)

theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)

jnpId = jnp.identity(nlevels)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnpX2 = jnp.matmul(jnpX, jnpX)
jnpP2 = jnp.matmul(jnpP, jnpP)
jnpXP = jnp.matmul(jnpX, jnpP)
jnpPX = jnp.matmul(jnpP, jnpX)
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f.full())

I_tR = jnp.array([0.0])
I_tI = jnp.array([0.0])
#Initials1, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho, I_tR, I_tI, theta_t, ts, tau, dt, l1 = rho_update(0,(Initials, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho_i, I_tR, I_tI,  theta_t, ts, tau, dt, 0))
#jnp_rho_simul = OPsoln_JAX1(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp.array(theta_t), jnp.array(ts), dt, tau, jnpId)

for n in range(nsteps):
  stime = time.time()
  print (CostF_strat(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId))
  Initials = update_strat(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId, lrate)
  print (n, time.time()-stime)

Initvals = np.array(Initials)

with h5py.File("/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Data/Gaussian_OP_no_control_Ex1.hdf5", "w") as f:
    dset1 = f.create_dataset("nlevels", data = nlevels, dtype ='int')
    dset2 = f.create_dataset("rho_i", data = rho_i.full())
    dset3 = f.create_dataset("rho_f", data = rho_f.full())
    dset4 = f.create_dataset("ts", data = ts)
    dset5 = f.create_dataset("tau", data = tau)
    dset6 = f.create_dataset("theta_t", data = theta_t)
    dset7 = f.create_dataset("Initvals", data = Initvals)    
f.close()
    
PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfigs')

rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul, rop, nbar = OPsoln_strat_SHO(X, P, H, rho_i, Initvals[0], Initvals[1], Initvals[2], Initvals[3], ts, theta_t,  tau, 1)
#print ('Fidelity ', fidelity(rho_f_simul, rho_f_int))

