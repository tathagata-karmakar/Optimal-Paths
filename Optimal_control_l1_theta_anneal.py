#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:55:01 2024

@author: t_karmakar
"""
'''

This script finds optimal quadrature and parametric potential for a 
problem defined through Initialization.py

The results are saved as hdf5 file titled Optimal_control_solution

'''
import os,sys
os.environ['JAX_PLATFORMS'] = 'cpu'
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

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)
Dirname = script_dir+"/Data/testing2"
Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)

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

inits = (np.random.rand(10)-0.5)
#inits=np.array([ 3.167937  , -0.73915756, -2.5917065 , 19.205257  ,  0.76677966,
#       -5.1844788 , -5.5650477 ,  8.844423  , 10.119776  ,  4.2625775 ]
 #     )


#inits[4]=2*alr-k0r-4*(alr**2-ali**2+2*Dvm)*tau
Initials = jnp.array(inits)
'''
G100 = jnp.matmul(params[4][0], Initials)
G010 = jnp.matmul(params[4][1], Initials)
k100 = jnp.matmul(params[4][2], Initials)
k010 = jnp.matmul(params[4][3], Initials)
G200 = jnp.matmul(params[4][4], Initials)
G110 = jnp.matmul(params[4][5], Initials)
G020 = jnp.matmul(params[4][6], Initials)
k200 = jnp.matmul(params[4][7], Initials)
k110 = jnp.matmul(params[4][8], Initials)
k020 = jnp.matmul(params[4][9], Initials)
AGamma0 = (G100**2-G010**2-G200+G020)/2.0
BGamma0 = G100*G010-G110
theta0 = jnp.arctan2(BGamma0, AGamma0)/2.0
'''


theta_t = np.zeros(len(params[1]))
jnptheta_t = jnp.array(theta_t)
l1_t = np.zeros(len(params[1]))
jnpl1_t = jnp.array(l1_t)

#params = (l1max, ts, dt, tau, Idmat)
cost_b, J_b, rhotmpr, rhotmpi = CostF_control_l101(Initials, Ops, rho_ir, rho_ii, rho_fr, rho_fi,  params)
Initials_c, cost_c, J_c = Initials, cost_b, J_b
step_size = 0.1

#temp = temp0
metropolis = 1.0
tempf = 0.005
tempi = 1.0
temp = tempi
lrate = 1e-2

#Simulated annealing
nsteps = 5000
n=0
nbest = 0
stime = time.time()
while temp>tempf and (n<nsteps):
  #print (n)
  #stime = time.time()
  Initials_n = Initials_c+step_size*jnp.array(np.random.rand(10)-0.5)
  cost_n, J_n, rhotmpr, rhotmpi = CostF_control_l101(Initials_n, Ops, rho_ir, rho_ii, rho_fr, rho_fi, params)
  if (-cost_b<0.9):
      if (cost_n<cost_b):
          Initials, cost_b, J_b = Initials_n, cost_n, J_n
          nbest = n
      #print (nb, n,  -cost_b, temp) #Cost is the negative of fidelity
      diff = cost_n-cost_c
      diff2 = J_n-J_c
      #metropolis2 = jnp.exp(-50*diff2/temp)
      metropolis = jnp.exp(-100*diff/temp)
      if (diff<0) or (jnp.array(np.random.rand())<metropolis):
          Initials_c, cost_c, J_c = Initials_n, cost_n, J_n
          temp = temp/(1+0.02*temp)
      else:
          temp = temp/(1-0.002*temp)           
  else:
      if (J_n<J_b) and (cost_n<cost_b):
          Initials, cost_b, J_b = Initials_n, cost_n, J_n
          nbest = n
      #print (nb, n,  -cost_b, temp) #Cost is the negative of fidelity
      diff = (cost_n-cost_c)
      diff2 = J_n-J_c
      #print (diff, diff2)
      metropolis = jnp.exp(-100*(diff+diff2)/temp)
      #metropolis2 = jnp.exp(-50*diff2/temp)
      if ((diff<0) and(diff2<0)) or ((jnp.array(np.random.rand())<metropolis)):
          Initials_c, cost_c, J_c = Initials_n, cost_n, J_n
          temp = temp/(1+0.02*temp)
      else:
          temp = temp/(1-0.002*temp)    
      #Initials = update_control2_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
      #cost_b, J_b = CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
      
  n+=1
  step_size = step_size/(1+0.0001*step_size)
  print (nbest, n,  -cost_b, J_b, temp, metropolis)

print ('TT: ', time.time()-stime)  



Initvals = np.array(Initials)

stime  = time.time()
Q1j, Q2j, Q3j, Q4j, Q5j, theta_tj, l1_tj, rho_f_simul2r, rho_f_simul2i, rop_stratj, Jval = OPintegrate_strat(jnp.array(Initvals), Ops, rho_ir, rho_ii, params)
fid  = Fidelity_PS(rho_f_simul2r, rho_f_simul2i, rho_fr, rho_fi).item()
print (time.time()-stime, fid)


with h5py.File(Dirname+"/Optimal_control_solution.hdf5", "w") as f:
    dset1 = f.create_dataset("theta_t", data = theta_tj)
    dset2 = f.create_dataset("l1_t", data = l1_tj)
    dset3 = f.create_dataset("r_t", data = rop_stratj)
    dset4 = f.create_dataset("Initvals", data = Initvals)   
    dset5 = f.create_dataset("ML_readouts", data = rop_stratj) 
    dset6 = f.create_dataset("Jval", data = Jval) 
    dset7 = f.create_dataset("Final_fidelity", data = fid) 
    dset8 = f.create_dataset("Q1t", data = Q1j) 
    dset9 = f.create_dataset("Q2t", data = Q2j) 
    dset10 = f.create_dataset("Q3t", data = Q3j) 
    dset11 = f.create_dataset("Q4t", data = Q4j) 
    dset12 = f.create_dataset("Q5t", data = Q5j) 
    dset13 = f.create_dataset("rho_f_simulr", data = rho_f_simul2r)
    dset13 = f.create_dataset("rho_f_simuli", data = rho_f_simul2i)

#f.close()


t_i, t_f = params[1][0], params[1][-1]
fig, axs = plt.subplots(8,1,figsize=(6,14),sharex='all')
#axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian')
axs[0].plot(params[1], Q1j, linewidth =3, color='blue', label = 'w control')
#axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
#axs[0].plot(ts, X_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'General')
axs[0].plot(t_i, Q1i, "o", color = 'b')
axs[0].plot(t_f, Q1f, "X" , color = 'r')
#axs[0].plot(t_f, Q1, "^" , color = 'blue')
#axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
axs[1].plot(params[1], Q2j, linewidth =3, color='blue')
axs[1].plot(t_i, Q2i, "o", color = 'b')
axs[1].plot(t_f, Q2f, "X", color = 'r')
#axs[1].plot(t_f, Q2, "^" , color = 'k')
axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 15)
axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 15)
axs[0].tick_params(labelsize=15)
axs[1].tick_params(labelsize=15)
axs[0].legend(loc=1,fontsize=15)

#axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'Gaussian')
axs[2].plot(params[1], Q3j, linewidth =3, color='blue')
axs[2].plot(params[1][0], Q3i, "o", color = 'b')
axs[2].plot(t_f, Q3f, "X", color = 'r')

#axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'Gaussian')
axs[3].plot(params[1], Q4j, linewidth =3, color='blue')
axs[3].plot(params[1][0], Q4i, "o", color = 'b')
axs[3].plot(t_f, Q4f, "X", color = 'r')
#axs[3].plot(t_f, V2, "^" , color = 'k')

#axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'Gaussian')
axs[4].plot(params[1], Q5j, linewidth =3, color='blue')
axs[4].plot(params[1][0], Q5i, "o", color = 'b')
axs[4].plot(params[1][-1], Q5f, "X", color = 'r')
#axs[4].plot(t_f, V3, "^" , color = 'k')

#axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
axs[5].plot(params[1], rop_stratj,linewidth =3, color='blue', linestyle='dashed')

axs[6].plot(params[1], np.zeros(len(params[1])),linewidth =4, color='green')
axs[6].plot(params[1], theta_tj,linewidth =3, color='blue', linestyle='dashed')
axs[6].axhline(y=np.pi/2.0, color='k', linestyle = 'dashed')
axs[6].axhline(y=-np.pi/2.0, color='k', linestyle = 'dashed')

axs[7].plot(params[1], l1_tj, linewidth =3, color='blue', linestyle='dashed')

axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[5].set_ylabel(r'$r^\star$', fontsize = 15)
axs[6].set_ylabel(r'$\theta^\star$', fontsize = 15)
axs[7].set_ylabel(r'$\lambda_1$', fontsize = 15)
axs[7].set_xlabel(r'$t$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
axs[5].tick_params(labelsize=15)
axs[6].tick_params(labelsize=15)
axs[7].tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
#plt.savefig(script_dir+'/Plots/control_l10_method322.pdf',bbox_inches='tight')
#PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig_control')
