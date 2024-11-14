#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 08:55:01 2024

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

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)

nlevels = 25

#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 5.0
ts = np.linspace(t_i, t_f, int(t_f/0.01))
dt = ts[1]-ts[0]
tau = 2.5
q4f = np.sqrt(1+4*tau*tau)-2*tau
q3f = np.sqrt(4*tau*q4f)
q5f = q3f*(1+q4f/(2*tau))

snh2r = -np.sqrt(q4f**2+(q3f-q5f)**2/4.0)
csh2r = (q3f+q5f)/2.0
rparam = snh2r+csh2r
r_sq = np.log(rparam)/2
xiR = r_sq*(q5f-q3f)/(2*snh2r)
xiI = r_sq*(-q4f)/snh2r
in_alr = .5
in_ali = -.7
fin_alr = 0#-0.1#in_alr*np.cos(t_f)+in_ali*np.sin(t_f)
fin_ali = 0#0.5#in_ali*np.cos(t_f)-in_alr*np.sin(t_f)


rho_i = (basis(nlevels, 0)+basis(nlevels,2))/np.sqrt(2)#squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
rho_f = (basis(nlevels, 0)-basis(nlevels,2))/np.sqrt(2)#coherent(nlevels, fin_alr+1j*fin_ali)
#rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
#rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))

X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
#Ljump = X
#Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()

Q1i = expect(X,rho_i)
Q2i = expect(P,rho_i)
Q3i = 2*(expect(X*X,rho_i)-Q1i**2)
Q5i = 2*(expect(P*P,rho_i)-Q2i**2)
Q4i = (expect(P*X+X*P,rho_i)-2*Q2i*Q1i)

inits = (np.random.rand(10)-0.5)
#inits=np.array([ 3.167937  , -0.73915756, -2.5917065 , 19.205257  ,  0.76677966,
#       -5.1844788 , -5.5650477 ,  8.844423  , 10.119776  ,  4.2625775 ]
 #     )
np_Idmat=np.identity(10)
Idmat = jnp.array(np_Idmat)


#inits[4]=2*alr-k0r-4*(alr**2-ali**2+2*Dvm)*tau
Initials = jnp.array(inits)
G100 = jnp.matmul(Idmat[0], Initials)
G010 = jnp.matmul(Idmat[1], Initials)
k100 = jnp.matmul(Idmat[2], Initials)
k010 = jnp.matmul(Idmat[3], Initials)
G200 = jnp.matmul(Idmat[4], Initials)
G110 = jnp.matmul(Idmat[5], Initials)
G020 = jnp.matmul(Idmat[6], Initials)
k200 = jnp.matmul(Idmat[7], Initials)
k110 = jnp.matmul(Idmat[8], Initials)
k020 = jnp.matmul(Idmat[9], Initials)
AGamma0 = (G100**2-G010**2-G200+G020)/2.0
BGamma0 = G100*G010-G110
theta0 = jnp.arctan2(BGamma0, AGamma0)/2.0



theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)
l1_t = np.zeros(len(ts))
jnpl1_t = jnp.array(l1_t)
l1max = 0.0
tb = 0
jnpId = jnp.identity(nlevels, dtype=complex)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f.full())


cost_b, J_b = CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, l1max, ts, dt, tau, Idmat, jnpId)
Initials_c, cost_c = Initials, cost_b
step_size = 0.1
#temp = temp0
metropolis = 1.0
tempf = 0.005
tempi = 2000.0
temp = tempi
lrate = 1e-2

#for n in range(nsteps):
nsteps = 1000
n=0
while temp>tempf and (n<nsteps):
  stime = time.time()
  Initials_n = Initials_c+step_size*jnp.array(np.random.rand(10)-0.5)
  cost_n, J_n = CostF_control_l101(Initials_n, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, l1max, ts, dt, tau, Idmat, jnpId)
  if (cost_n<cost_b):
      #if cost_n<=2.0 and  (J_n<=J_b):
          #Initials, cost_b, J_b = Initials_n, cost_n, J_n
      #elif cost_n>2.0:
      Initials, cost_b, J_b = Initials_n, cost_n, J_n
      print (n, -cost_b, J_b, metropolis, temp) #Cost is the negative of fidelity
  diff = cost_n-cost_c
  metropolis = jnp.exp(-diff/temp)
  if (diff<0) or (jnp.array(np.random.rand())<metropolis):
      Initials_c, cost_c = Initials_n, cost_n
      temp = temp/(1+0.02*temp)
  else:
      temp = temp/(1-0.002*temp)    
  #Initials = update_control2_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
  #cost_b, J_b = CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
  n+=1
  #print (CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId))
  #Initials = update_control_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
  #if (n>nsteps/4):
  #Initials = update_control2_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId, lrate)
  #print (Initials)
  #print (n, time.time()-stime)
  

  
Initvals = np.array(Initials)
#q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)

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

Q1j, Q2j, Q3j, Q4j, Q5j, theta_tj, l1_tj, rho_f_simul2, rop_stratj, diff = OPintegrate_strat(Initvals, X.full(), P.full(), H.full(), rho_i.full(), l1max, ts, dt,  tau,  np_Idmat, np.identity(nlevels))
a = (X+1j*P)/np.sqrt(2)
q1i = expect(X,rho_i)
q1f = expect(X,rho_f)
q2i = expect(P,rho_i)
q2f = expect(P,rho_f)


t_i, t_f = ts[0], ts[-1]
fig, axs = plt.subplots(8,1,figsize=(6,14),sharex='all')
#axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian')
axs[0].plot(ts, Q1j, linewidth =3, color='blue', label = 'w control')
#axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
#axs[0].plot(ts, X_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'General')
axs[0].plot(t_i, q1i, "o", color = 'b')
axs[0].plot(t_f, q1f, "X" , color = 'r')
#axs[0].plot(t_f, Q1, "^" , color = 'blue')
#axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
axs[1].plot(ts, Q2j, linewidth =3, color='blue')
axs[1].plot(t_i, q2i, "o", color = 'b')
axs[1].plot(t_f, q2f, "X", color = 'r')
#axs[1].plot(t_f, Q2, "^" , color = 'k')
axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 15)
axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 15)
axs[0].tick_params(labelsize=15)
axs[1].tick_params(labelsize=15)
axs[0].legend(loc=1,fontsize=15)

#axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'Gaussian')
axs[2].plot(ts, Q3j, linewidth =3, color='blue')
axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b')
axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "X", color = 'r')

#axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'Gaussian')
axs[3].plot(ts, Q4j, linewidth =3, color='blue')
axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b')
axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "X", color = 'r')
#axs[3].plot(t_f, V2, "^" , color = 'k')

#axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'Gaussian')
axs[4].plot(ts, Q5j, linewidth =3, color='blue')
axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b')
axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "X", color = 'r')
#axs[4].plot(t_f, V3, "^" , color = 'k')

#axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
axs[5].plot(ts, rop_stratj,linewidth =3, color='blue', linestyle='dashed')

axs[6].plot(ts, np.zeros(len(ts)),linewidth =4, color='green')
axs[6].plot(ts, theta_tj,linewidth =3, color='blue', linestyle='dashed')
axs[6].axhline(y=np.pi/2.0, color='k', linestyle = 'dashed')
axs[6].axhline(y=-np.pi/2.0, color='k', linestyle = 'dashed')

axs[7].plot(ts, l1_tj, linewidth =3, color='blue', linestyle='dashed')

axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[5].set_ylabel('$r^\star$', fontsize = 15)
axs[6].set_ylabel('$\\theta^\star$', fontsize = 15)
axs[7].set_ylabel('$\lambda_1$', fontsize = 15)
axs[7].set_xlabel('$t$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
axs[5].tick_params(labelsize=15)
axs[6].tick_params(labelsize=15)
axs[7].tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
#plt.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/Optimal_Path/Plots/control_l10_method322.pdf',bbox_inches='tight')
#PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig_control')
