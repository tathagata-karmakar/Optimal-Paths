#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:23:37 2024

@author: t_karmakar
"""

'''

This script finds a sample quadrature and parametric potential for a 
problem defined through Initialization.py

The results are saved as hdf5 file titled Alternate_control

'''

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
from numba import njit, prange
import numba as nb

Dirname = script_dir+"/Data/testing2"
Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)

#q1i = expect(X,rho_i)
#q1f = expect(X,rho_f)
#q2i = expect(P,rho_i)
#q2f = expect(P,rho_f)

ts = params[1]
l1max = params[0]
nvars =4
Nc = 5
inits = (np.random.rand(nvars+4*Nc)-0.5)
Initials = jnp.array(inits)
theta_mat, l1_mat = Fourier_mat(nvars, Nc, ts[0], ts[-1], ts)
MMat = Multiply_Mat(nvars, Nc)
jnp_theta_mat = jnp.array(theta_mat)
jnp_l1_mat = jnp.array(l1_mat)
jnp_MMat = jnp.array(MMat)




'''
for n in range(nsteps):
  stime = time.time()
  #if (n%200==0):
  print (CostF_control_generate(Initials, jnpX, jnpP, jnpH, jnpX2, jnpCXP, jnpP2, jnp_rho_i, jnp_rho_f, jnp_theta_mat, jnp_l1_mat, l1max, ts, dt, tau, Nc, jnp_MMat, jnpId), time.time()-stime)
  Initials = update_control_generate(Initials, jnpX, jnpP, jnpH, jnpX2,  jnpCXP, jnpP2, jnp_rho_i, jnp_rho_f, jnp_theta_mat, jnp_l1_mat, l1max, ts, dt, tau, Nc, jnp_MMat, jnpId, nvars, lrate)
  #Initials = update_control2_l10_2input(Initials,  jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, Fmat, ts, dt, tau, Ncos, MMat1, jnpId, lrate)
  print (n, time.time()-stime)
'''  

#params = (l1max, ts, dt, tau, Idmat)
cost_b, J_b, rho_f_simulr, rho_f_simuli = CostF_control_generate(Initials, Ops, rho_ir, rho_ii, rho_fr, rho_fi, jnp_theta_mat, jnp_l1_mat, params, Nc, jnp_MMat)
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
diff = 0

while temp>tempf and (n<nsteps):
  stime = time.time()
  Initials_n = Initials_c+step_size*jnp.array(np.random.rand(nvars+4*Nc)-0.5)
  cost_n, J_n, rho_nr, rho_ni = CostF_control_generate(Initials_n, Ops, rho_ir, rho_ii, rho_fr, rho_fi,  jnp_theta_mat, jnp_l1_mat, params, Nc, jnp_MMat)
  if (cost_n<cost_b):
      #if cost_n<=2.0 and  (J_n<=J_b):
          #Initials, cost_b, J_b = Initials_n, cost_n, J_n
      #elif cost_n>2.0:
      Initials, cost_b, J_b, rho_f_simulr, rho_f_simuli = Initials_n, cost_n, J_n, rho_nr, rho_ni
      nbest = n
  print (nbest, n,  -cost_b, J_b, metropolis, temp) #Cost is the negative of fidelity
  diff = cost_n-cost_c
  metropolis = jnp.exp(-100*diff/temp)
  if (diff<0) or (jnp.array(np.random.rand())<metropolis):
      Initials_c, cost_c, J_c = Initials_n, cost_n, J_b
      temp = temp/(1+0.02*temp)
  else:
      temp = temp/(1-0.0002*temp)    
  #Initials = update_control2_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
  #cost_b, J_b = CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
  n+=1
  step_size = step_size/(1+0.0001*step_size)  
  
  
Initvals = np.array(Initials)
theta_t = (np.pi/2.0)*jnp.tanh(2*jnp.matmul(theta_mat,Initvals)/np.pi)
l1_t = (l1max)*jnp.tanh(jnp.matmul(l1_mat,Initials)/l1max)


Q1j1, Q2j1, Q3j1, Q4j1, Q5j1, rho_f_simul2r, rho_f_simul2i, rop_stratj, Jval = OP_wcontrol(jnp.array(Initvals)[:10], Ops, rho_ir, rho_ii,  l1_t, theta_t, params)
fid  = Fidelity_PS(rho_f_simul2r, rho_f_simul2i, rho_fr, rho_fi).item()
print (fid)
with h5py.File(Dirname+"/Alternate_control.hdf5", "w") as f:
    dset1 = f.create_dataset("theta_t_sample", data = theta_t)
    dset2 = f.create_dataset("l1_t_sample", data = l1_t)
    dset3 = f.create_dataset("Initials_sample", data = Initvals)
    dset4 = f.create_dataset("ML_readouts", data = rop_stratj)
    dset5 = f.create_dataset("Final_fidelity", data = fid) 
    dset6 = f.create_dataset("Jval", data = Jval)
    dset8 = f.create_dataset("Q1t", data = Q1j1) 
    dset9 = f.create_dataset("Q2t", data = Q2j1) 
    dset10 = f.create_dataset("Q3t", data = Q3j1) 
    dset11 = f.create_dataset("Q4t", data = Q4j1) 
    dset12 = f.create_dataset("Q5t", data = Q5j1) 

fig, axs = plt.subplots(3,1,figsize=(4,6),sharex='all')
axs[0].plot(ts, theta_t, linewidth =4, color = 'green')
axs[1].plot(ts, l1_t, linewidth =4, color = 'green')
axs[2].plot(ts, rop_stratj, linewidth =4, color = 'green')
axs[0].set_ylabel(r'$\theta(t)$', fontsize =12)
axs[1].set_ylabel(r'$\lambda_1(t)$',  fontsize =12)
axs[2].set_ylabel(r'$r(t)$',  fontsize =12)
axs[2].set_xlabel(r'$t$', fontsize =12)
plt.subplots_adjust(wspace=0.05, hspace=0.1)
plt.savefig(Dirname+'/sample_control_extmp.pdf',bbox_inches='tight')

'''
Initvals = np.array(Initials)
theta_t = (np.pi/2.0)*jnp.tanh(2*jnp.matmul(Fmat, Initials)/np.pi)


q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)
Q1j, Q2j, Q3j, Q4j, Q5j,  rho_f_simul2, rop, diff, Hs = OPintegrate_strat_2inputs(Initials,  X.full(), P.full(), H.full(), rho_i.full(), theta_t, ts, dt,  tau, Ncos, MMat1, np.identity(nlevels))


  
fig, axs = plt.subplots(8,1,figsize=(6,14),sharex='all')
axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'PRXQ')
axs[0].plot(ts, Q1j, color='k', label = 'w control')
#axs[0].plot(ts, X_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[0].plot(ts, Q1j, linewidth =3, color='k')
#axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'General')
#axs[0].plot(ts, X_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'General')
axs[0].plot(t_i, q1i, "o", color = 'b')
axs[0].plot(t_f, q1f, "X" , color = 'r')
#axs[0].plot(t_f, Q1, "^" , color = 'k')

axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
#axs[1].plot(ts, P_simul, linewidth =4, linestyle = 'dashed', color = 'blue')
axs[1].plot(ts, Q2j, linewidth =3, color='k')
#axs[1].plot(ts, P_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
#axs[1].plot(ts, P_simuld, linewidth =3, linestyle = 'dashed', color = 'blue')
axs[1].plot(t_i, q2i, "o", color = 'b')
axs[1].plot(t_f, q2f, "X", color = 'r')
#axs[1].plot(t_f, Q2, "^" , color = 'k')
axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 15)
axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 15)
axs[0].tick_params(labelsize=15)
axs[1].tick_params(labelsize=15)
axs[0].legend(loc=4,fontsize=12)

axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[2].plot(ts, Q3j, linewidth =3, color='k')
#axs[2].plot(ts, varX_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
#axs[2].plot(ts, varX_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Strato')
axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b')
axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "X", color = 'r')
#axs[2].plot(t_f, V1, "^" , color = 'k')

axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
#axs[3].plot(ts, covXP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
axs[3].plot(ts, Q4j, linewidth =3, color='k')
#axs[3].plot(ts, covXP_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Strato')
axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b')
axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "X", color = 'r')
#axs[3].plot(t_f, V2, "^" , color = 'k')

axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[4].plot(ts, Q5j, linewidth =3, color='k')
#axs[4].plot(ts, varP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
#axs[4].plot(ts, varP_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Starto')
axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b')
axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "X", color = 'r')
#axs[4].plot(t_f, V3, "^" , color = 'k')

axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
axs[5].plot(ts, rop,linewidth =3, color='k')
#axs[5].plot(ts, rop_stratd,linewidth =3, color='blue', linestyle='dashed')
#axs[5].plot(ts, nbar, color='red', linestyle ='dashed', linewidth = 3)
#axs[6].plot(ts, np.zeros(len(ts)),linewidth =4, color='green')
axs[6].plot(ts, theta_t,linewidth =3, color='k', linestyle='dashed')
#axs[6].plot(ts, theta_tj,linewidth =3, color='k', linestyle='dashed')

axs[7].plot(ts, diff, linewidth =3, color='k', linestyle='dashed')

axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[5].set_ylabel('$r^\star$', fontsize = 15)
axs[6].set_ylabel('$\\theta^\star$', fontsize = 15)
axs[7].set_ylabel('$\\theta^\star(t)-\\theta(t)$', fontsize = 15)
axs[7].set_xlabel('$t$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
axs[5].tick_params(labelsize=15)
axs[6].tick_params(labelsize=15)
axs[7].tick_params(labelsize=15)


plt.subplots_adjust(wspace=0.1, hspace=0.1)
#plt.savefig(script_dir+'/Plots/control_l10_method2_tmp.pdf',bbox_inches='tight')
#plt.plot(ts, np.matmul(Fmat, cinits))
#plt.plot(ts, jnp.matmul(Fmat, Coeffs))
'''

