#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:45:15 2024

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
t_f = 3
ts = np.linspace(t_i, t_f, 500)
dt = ts[1]-ts[0]
tau = 5.0
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
fin_alr = in_alr*np.cos(t_f)+in_ali*np.sin(t_f)
fin_ali = in_ali*np.cos(t_f)-in_alr*np.sin(t_f)

#xiR =0
#xiI = 0


rho_i = squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
#rho_f = basis(nlevels,4)
rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))

nsteps = 3000
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

Q1i = expect(X,rho_i)
Q2i = expect(P,rho_i)
Q3i = 2*(expect(X*X,rho_i)-Q1i**2)
Q5i = 2*(expect(P*P,rho_i)-Q2i**2)
Q4i = (expect(P*X+X*P,rho_i)-2*Q2i*Q1i)

lrate = 1e-1/5.
q3, q4, q5, alr, ali, A, B, q1t, q2t,rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)

#alr = np.random.rand()
#ali = np.random.rand()
#A = np.random.rand()
#B = np.random.rand()
#Initials = jnp.array([alr, ali, A, B])
inits = 10*(np.random.rand(9)-0.5)
#inits[0]=expect(X,rho_i).real
#inits = np.array([ 2.03119478, -0.64861863,  3.06237402,  3.65953078,  3.56056756,
       #-0.66798523, -3.0849589 , -2.22235786, -1.85641707])
alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm = inits[0], inits[1], inits[2], inits[3], inits[4], inits[5], inits[6], inits[7], inits[8]
#inits[4]=2*alr-k0r-4*(alr**2-ali**2+2*Dvm)*tau
Initials = jnp.array(inits)
#Initials = jnp.array([-25.605787  , -19.557974  , -44.8486    ,  -2.2532227 ,-0.3402054 ,   1.4204986 ,   0.21251394,   0.29398373,-1.9151409 ])


r0 = jnp.matmul(jnp.array([1.0,0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
gradcal= (-jnp.sin(r0)*Q1i+jnp.cos(r0)*Q2i+t_f*(2*jnp.cos(r0)*jnp.sin(r0)*(Q5i-Q3i)+2*(jnp.cos(r0)**2-jnp.sin(r0)**2)*Q4i))

theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)
tb = 0
jnpId = jnp.identity(nlevels, dtype=complex)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f_int = jnp.array(rho_f_int.full())
jnp_rho_f = jnp.array(rho_f.full())

def rfunct(Initials):
    #r = jnp.matmul(jnp.array([1.0,0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
    #w = jnp.matmul(jnp.array([0,0,1.0,0,0,0,0,0,0]),Initials)
    r = jnp.matmul(jnp.array([1.0,0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
    v = jnp.matmul(jnp.array([0,1.0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
    w = jnp.matmul(jnp.array([0,0,1.0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
    z = jnp.matmul(jnp.array([0,0,0,1.0,0,0,0,0,0]),Initials)#+jnp.array([0])
    GLL = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,-1.0]),Initials)#+jnp.array([0])
    kappaLL = jnp.matmul(jnp.array([0,0,0,0,1,-1.0,0,0,0]),Initials)#+jnp.array([0])
    kappaLM = jnp.matmul(jnp.array([0,0,0,0,0,0,1.0,0,0]),Initials)#+jnp.array([0])
    kappaMM = jnp.matmul(jnp.array([0,0,0,0,0,0,1.0,0,0]),Initials)#+jnp.array([0])
    GLL = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,-1.0]),Initials)#+jnp.array([0])
    GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)#+jnp.array([0])
    GLM = r*v
    kappaLL = kappaLL-2*r*w
    kappaLM = kappaLM-v*w-r*z
    kappaMM = kappaMM-2*v*z
    GLL+= -r**2-w**2/4.0
    #C1 = 2*(GLL-r**2-w**2/4.0)-(z**2-w**2)/4.0
    GLM+= -r*v-w*z/4.0
    GMM+= -v**2-z**2/4.0
    phi = r
    Lj = jnpX*jnp.cos(r)+jnpP*jnp.sin(r)
    expr = jnp.trace(jnp.matmul(Lj, jnp_rho_i)).real
    dL = Lj-expr*jnpId
    rho = jnp_rho_i
    j=0
    
    #while (j<len(ts)-1):
        #dphi = jnp.exp(1j*v)*dt
    #    rho+=(jnp.matmul(dL,jnp_rho_i)+jnp.matmul(jnp_rho_i,dL))*dt
     #   j+=1
    rho, dL, jnp_initial = jax.lax.fori_loop(0,len(ts)-1,test_rhou,(rho, dL, jnp_rho_i)) 
    expr = jnp.trace(jnp.matmul(Lj, rho)).real
    return expr

def test_rhou(i, in_values):
    rho, dL, jnp_rho_i = in_values
    rho1 = rho+(jnp.matmul(dL,jnp_rho_i)+jnp.matmul(jnp_rho_i,dL))*dt
    return rho1, dL, jnp_rho_i
    
for n in range(nsteps):
  stime = time.time()
  print (CostF_control_l101(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId))
  Initials = update_control_l10(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId, lrate)
  print (Initials)
  print (n, time.time()-stime)
  

  
Initvals = np.array(Initials)
q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)

alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm = Initvals[0], Initvals[1], Initvals[2], Initvals[3], Initvals[4], Initvals[5], Initvals[6], Initvals[7], Initvals[8]
rho_f_simul1, X_simul1, P_simul1, varX_simul1, covXP_simul1, varP_simul1, rop_strat,nbar, theta_t = OPsoln_control_l10(X, P, H, rho_i, alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm, ts,   tau, 1)
#rho_f_simuld, X_simuld, P_simuld, varX_simuld, covXP_simuld, varP_simuld, rop_stratd,nbard = OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)# OPsoln_control_l10(X, P, H, rho_i, alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm, ts,   tau, 1)

Q1j, Q2j, Q3j, Q4j, Q5j, theta_tj, rho_f_simul2, diff = OPintegrate_strat(Initials, X.full(), P.full(), H.full(), rho_i.full(), ts, dt,  tau, np.identity(nlevels))
a = (X+1j*P)/np.sqrt(2)
q1i = expect(X,rho_i)
q1f = expect(X,rho_f)
q2i = expect(P,rho_i)
q2f = expect(P,rho_f)


t_i, t_f = ts[0], ts[-1]
fig, axs = plt.subplots(8,1,figsize=(6,14),sharex='all')
axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian Approx')
axs[0].plot(ts, Q1j, color='k')
#axs[0].plot(ts, X_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
axs[0].plot(ts, Q1j, linewidth =3, color='k')
axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'General')
#axs[0].plot(ts, X_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'General')
axs[0].plot(t_i, q1i, "o", color = 'b')
axs[0].plot(t_f, q1f, "X" , color = 'r')
axs[0].plot(t_f, Q1, "^" , color = 'k')
axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
#axs[1].plot(ts, P_simul, linewidth =4, linestyle = 'dashed', color = 'blue')
axs[1].plot(ts, Q2j, linewidth =3, color='k')
axs[1].plot(ts, P_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
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
axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
#axs[2].plot(ts, varX_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Strato')
axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b')
axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "X", color = 'r')
#axs[2].plot(t_f, V1, "^" , color = 'k')

axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
#axs[3].plot(ts, covXP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
axs[3].plot(ts, Q4j, linewidth =3, color='k')
#axs[3].plot(ts, covXP_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Strato')
axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b')
axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "X", color = 'r')
#axs[3].plot(t_f, V2, "^" , color = 'k')

axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
axs[4].plot(ts, Q5j, linewidth =3, color='k')
#axs[4].plot(ts, varP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
#axs[4].plot(ts, varP_simuld, linewidth =3, linestyle = 'dashed', color = 'blue', label = 'Starto')
axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b')
axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "X", color = 'r')
#axs[4].plot(t_f, V3, "^" , color = 'k')

axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
axs[5].plot(ts, rop_strat,linewidth =3, color='red', linestyle='dashed')
#axs[5].plot(ts, rop_stratd,linewidth =3, color='blue', linestyle='dashed')
#axs[5].plot(ts, nbar, color='red', linestyle ='dashed', linewidth = 3)
axs[6].plot(ts, np.zeros(len(ts)),linewidth =4, color='green')
axs[6].plot(ts, theta_t,linewidth =3, color='red', linestyle='dashed')
axs[6].plot(ts, theta_tj,linewidth =3, color='k', linestyle='dashed')

axs[7].plot(ts, diff, linewidth =3, color='k', linestyle='dashed')

axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
axs[5].set_ylabel('$r^\star$', fontsize = 15)
axs[6].set_ylabel('$\\theta^\star$', fontsize = 15)
axs[6].set_xlabel('$t$', fontsize = 15)
axs[2].tick_params(labelsize=15)
axs[3].tick_params(labelsize=15)
axs[4].tick_params(labelsize=15)
axs[5].tick_params(labelsize=15)
axs[6].tick_params(labelsize=15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
#PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig_control')
