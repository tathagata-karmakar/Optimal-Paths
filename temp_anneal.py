#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:46:04 2024

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
fin_alr = -0.1#in_alr*np.cos(t_f)+in_ali*np.sin(t_f)
fin_ali = 0.5#in_ali*np.cos(t_f)-in_alr*np.sin(t_f)

rho_i = squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
#rho_f = basis(nlevels,4)
rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))


X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
#Ljump = X
#Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()
rho_f_int = rho_f_int*rho_f_int.dag()
#sigma_i = rand_herm(nlevels)
#sigma_i = sigma_i-expect(sigma_i, rho_i)

XItf = X*np.cos(t_f)+P*np.sin(t_f)
PItf = P*np.cos(t_f)-X*np.sin(t_f)

Q1 = expect(XItf,rho_f_int)
Q2 = expect(PItf,rho_f_int)
V1 = expect(XItf*XItf,rho_f_int)-Q1**2
V3 = expect(PItf*PItf,rho_f_int)-Q2**2
V2 = (expect(XItf*PItf+PItf*XItf,rho_f_int)-2*Q1*Q2)/2.0

Q1i = expect(X,rho_i)
Q2i = expect(P,rho_i)
Q3i = 2*(expect(X*X,rho_i)-Q1i**2)
Q5i = 2*(expect(P*P,rho_i)-Q2i**2)
Q4i = (expect(P*X+X*P,rho_i)-2*Q2i*Q1i)

lrate0 = 1e-3#1/5
def normalize_sigma0(sigma0, rho_i, Id):
    expsigma = jnp.trace(jnp.matmul(sigma0,rho_i))
    sigma0 = sigma0+(1-expsigma)*Id
    return sigma0
def make_sigma0(sigma0r, sigma0i, rho_i, Id):
    sigma0r1 = (sigma0r+jnp.transpose(sigma0r))/2.0
    sigma0i1 = (sigma0i-jnp.transpose(sigma0i))/2.0
    sigma0 = (sigma0r1+1j*sigma0i1)
    #expsigma = jnp.trace(jnp.matmul(sigma0,rho_i))
    #sigma0 = sigma0+(1-expsigma)*Id
    return normalize_sigma0(sigma0, rho_i, Id)

scl=1e-9
sigma0r = scl*jnp.array(np.random.rand(nlevels, nlevels)-0.5)
#sigma0r = sigma0r+jnp.transpose(sigma0r)
sigma0i = scl*jnp.array(np.random.rand(nlevels, nlevels)-0.5)
#sigma0i = sigma0i-jnp.transpose(sigma0i)

#inits=np.array([-0.20454815,  3.12661525, -3.10051494,  2.90454994,  0.66154927,
      # -0.17223483,  3.55780393, -0.66664125, -4.43558925, -3.96721373])
np_Idmat=np.identity(10)
Idmat = jnp.array(np_Idmat)

theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)
tb = 0
jnpId = jnp.identity(nlevels, dtype=complex)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
X2 = X*X
P2 = P*P
CXP = (X*P+P*X)/2.0
jnpX2 = jnp.array(X2.full())
jnpP2 = jnp.array(P2.full())
jnpCXP = jnp.array(CXP.full())
jnpH = jnp.array(H.full())
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f_int = jnp.array(rho_f_int.full())
jnp_rho_f = jnp.array(rho_f.full())

sigma0 = make_sigma0(sigma0r, sigma0i, jnp_rho_i, jnpId)
#expsigma = jnp.trace(jnp.matmul(sigma0,jnp_rho_i))
#sigma0 = sigma0+(1-expsigma)*jnpId


#grads=grad(CostF_sigma_control_l10)(sigma0, jnpX, jnpP, jnpX2, jnpP2, jnpCXP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
cost_b, J_b = CostF_sigma_control_l101(sigma0, jnpX, jnpP, jnpX2, jnpP2, jnpCXP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
sigma0_c = sigma0
sigma0r_c = sigma0r
sigma0i_c = sigma0i
cost_c = cost_b

tempf = 0.005
tempi = 200.0
temp = tempi
step_size = 1e-4
nsteps = 10000
n=0
while temp>tempf and (n<nsteps):
  stime = time.time()
  #print (n)
  #temp = temp0/(1+n)
  sigma0r_n = sigma0r_c+step_size*jnp.array(np.random.rand(nlevels, nlevels)-0.5)
  sigma0i_n = sigma0i_c+step_size*jnp.array(np.random.rand(nlevels, nlevels)-0.5)
  sigma0_n = make_sigma0(sigma0r_n, sigma0i_n, jnp_rho_i, jnpId)
  cost_n, J_n = CostF_sigma_control_l101(sigma0_n, jnpX, jnpP, jnpX2, jnpP2, jnpCXP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, Idmat, jnpId)
  if (cost_n<cost_b):# and (J_n<=J_b):
      sigma0, cost_b, J_b = sigma0_n, cost_n, J_n
      print (n, cost_b, J_b, temp)
  diff = cost_n-cost_c
  metropolis = jnp.exp(-diff/temp)
  #print (n, metropolis)
  if (diff<0) or (jnp.array(np.random.rand())<metropolis):
      sigma0r_c, sigma0i_c, cost_c = sigma0r_n, sigma0i_n, cost_n
      temp = temp/(1+0.02*temp)
  else:
      temp = temp/(1-0.0002*temp)
  n+=1

  #print (n, time.time()-stime)
  
 
q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)

Q1j, Q2j, Q3j, Q4j, Q5j, theta_tj, rho_f_simul2, rop_stratj, diff = OPintegrate_sigma_strat(sigma0, X.full(), P.full(), X2.full(), P2.full(), CXP.full(), H.full(), rho_i.full(), ts, dt,  tau,  np_Idmat, np.identity(nlevels))
a = (X+1j*P)/np.sqrt(2)
q1i = expect(X,rho_i)
q1f = expect(X,rho_f)
q2i = expect(P,rho_i)
q2f = expect(P,rho_f)


t_i, t_f = ts[0], ts[-1]
fig, axs = plt.subplots(8,1,figsize=(6,14),sharex='all')
axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'PRXQ')
axs[0].plot(ts, Q1j, color='k')
#axs[0].plot(ts, X_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
axs[0].plot(ts, Q1j, linewidth =3, color='blue', label = 'w control')
#axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
#axs[0].plot(ts, X_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'General')
axs[0].plot(t_i, q1i, "o", color = 'b')
axs[0].plot(t_f, q1f, "X" , color = 'r')
axs[0].plot(t_f, Q1, "^" , color = 'blue')
axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
#axs[1].plot(ts, P_simul, linewidth =4, linestyle = 'dashed', color = 'blue')
axs[1].plot(ts, Q2j, linewidth =3, color='blue')
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
axs[2].plot(ts, Q3j, linewidth =3, color='blue')
#axs[2].plot(ts, varX_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
#axs[2].plot(ts, varX_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Strato')
axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b')
axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "X", color = 'r')
#axs[2].plot(t_f, V1, "^" , color = 'k')

axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
#axs[3].plot(ts, covXP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
axs[3].plot(ts, Q4j, linewidth =3, color='blue')
#axs[3].plot(ts, covXP_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Strato')
axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b')
axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "X", color = 'r')
#axs[3].plot(t_f, V2, "^" , color = 'k')

axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[4].plot(ts, Q5j, linewidth =3, color='blue')
#axs[4].plot(ts, varP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
#axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
#axs[4].plot(ts, varP_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Starto')
axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b')
axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "X", color = 'r')
#axs[4].plot(t_f, V3, "^" , color = 'k')

axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
axs[5].plot(ts, rop_stratj,linewidth =3, color='blue', linestyle='dashed')
#axs[5].plot(ts, rop_stratd,linewidth =3, color='blue', linestyle='dashed')
#axs[5].plot(ts, nbar, color='red', linestyle ='dashed', linewidth = 3)
axs[6].plot(ts, np.zeros(len(ts)),linewidth =4, color='green')
#axs[6].plot(ts, theta_t,linewidth =3, color='red', linestyle='dashed')
axs[6].plot(ts, theta_tj,linewidth =3, color='blue', linestyle='dashed')

axs[7].plot(ts, diff, linewidth =3, color='blue', linestyle='dashed')

axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[5].set_ylabel('$r^\star$', fontsize = 15)
axs[6].set_ylabel('$\\theta^\star$', fontsize = 15)
axs[7].set_ylabel('$\Gamma(1,0)$', fontsize = 15)
axs[7].set_xlabel('$t$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
axs[5].tick_params(labelsize=15)
axs[6].tick_params(labelsize=15)
axs[7].tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
#plt.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/Optimal_Path/Plots/control_l10_method35.pdf',bbox_inches='tight')
#PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig_control')







