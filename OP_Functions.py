#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:16:27 2024

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

'''
Functions needed to Simulate Optimal Path for a Continuously Monitored
Quantum Harmonic Oscillator with Controlled Quadrature Measurement
'''

'''
def construct_sigma(cdiags, cndR, cndI, nlevels):
  i = 0
  while (i<nlevels):
    if (i == 0):
      temp_sigma = cdiags[0]*basis(nlevels,0)*basis(nlevels,0).dag()
    else:
      temp_sigma = temp_sigma+cdiags[i]*basis(nlevels,i)*basis(nlevels,i).dag()
    for j in range(i+1, nlevels):
      print (i,j,nlevels*i-i*(i+1)//2+j-(i+1))
      temp_sigma = temp_sigma+cndR[nlevels*i-i*(i+1)//2+j-(i+1)]*(basis(nlevels,i)*basis(nlevels,j).dag()+basis(nlevels,j)*basis(nlevels,i).dag())
      temp_sigma = temp_sigma+1j*cndI[nlevels*i-i*(i+1)//2+j-(i+1)]*(basis(nlevels,i)*basis(nlevels,j).dag()-basis(nlevels,j)*basis(nlevels,i).dag())
    i+=1
  return temp_sigma

def OPsoln(Ljump, H, rho_i, sigma_i, t_i, t_f,  tau, dt):
  rho = rho_i
  sigma = sigma_i-expect(sigma_i,rho)
  t=t_i
  while (t<t_f):
    Omega = commutator(Ljump, sigma, kind = 'anti')
    expL = expect(Ljump,rho)
    delL = Ljump - expL
    delL2 = delL*delL
    varL = expect(delL2,rho)
    addL = (delL2-varL)/(2*tau)
    xi = expect(Omega,rho)/2.0
    rho1 = rho+(-1j*commutator(H,rho)+(Ljump*rho*Ljump-Ljump*Ljump*rho/2.0-rho*Ljump*Ljump/2.0)/(4*tau)+xi*commutator(delL, rho,kind='anti')/(2*tau))*dt
    sigma1 = sigma+(-1j*commutator(H,sigma)-(Ljump*sigma*Ljump-Ljump*Ljump*sigma/2.0-sigma*Ljump*Ljump/2.0)/(4*tau)-xi*commutator(delL, sigma,kind='anti')/(2*tau)+addL)*dt
    rho = rho1#/rho1.tr()
    sigma = sigma1-expect(sigma1,rho1)
    #print (expect(sigma, rho))
    #xs.append(expect(sigmax(),rho))
    #ys.append(expect(sigmay(),rho))
    #zs.append(expect(sigmaz(),rho))
    t = t+dt
  return rho, sigma
'''

def OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, varReturn = 0):
  rho = rho_i
  #t=ts[0]
  t_f = ts[-1]
  dt = ts[1]-ts[0]
  I_t = 0
  expLs = np.zeros(len(ts))
  expMs = np.zeros(len(ts))
  varLs = np.zeros(len(ts))
  varMs = np.zeros(len(ts))
  covLMs = np.zeros(len(ts))
  rop_strat = np.zeros(len(ts))
  nbar = np.zeros(len(ts))
  i=0
  while (i<len(ts)):
    #print (ts[i])
    t = ts[i]
    theta = theta_t[i]
    phi = theta + t
    #print (t)
    #csth, snth = np.cos(theta), np.sin(theta)
    csph, snph = np.cos(phi), np.sin(phi)
    #cs2ph, sn2ph = np.cos(2*phi), np.sin(2*phi)
    #Ljump = csth*X+snth*P
    Ljump = csph*X+snph*P
    Ljump2 = Ljump*Ljump
    #Ljump2_a = (csph**2)*X*X+(snph**2)*P*P+csph*snph*(X*P+P*X)
    #Mjump = -snth*X+csth*P
    Mjump = -snph*X+csph*P
    expL = expect(Ljump,rho).real
    expM = expect(Mjump,rho).real
    expLs[i] = expL
    expMs[i] = expM
    delL = Ljump - expL
    delVjump = Ljump2-expect(Ljump2, rho).real
    #delL2 = delL*delL
    varL = expect(Ljump*Ljump,rho).real-expL**2
    #varL1 = expect(Ljump2_a,rho).real-expL**2
    if (varReturn ==1):
      delM = Mjump-expM
      varLs[i] = varL
      Mjump2 = Mjump*Mjump
      varMs[i] = expect(Mjump2,rho).real-expM**2
      covLMs[i] = (expect(Ljump*Mjump+Mjump*Ljump,rho).real-2*expL*expM)/2.0
      nbar[i] = expect((Ljump2+Mjump2-1)/2.0,rho)
    ht = np.exp(-1j*phi)*(alr+1j*ali+1j*t*(A+1j*B)/(8*tau))+np.exp(-1j*phi)*I_t
    r =  ht.real
    rop_strat[i]=r
    if(i<len(ts)-1):
        drhodt = ((-delVjump*rho-rho*delVjump)/(4*tau)+r*(delL*rho+rho*delL)/(2*tau))
        #print (drhodt.tr())
        rho1 = rho + drhodt*dt
        #rho1 = rho+((-delVjump*rho-rho*delVjump)/(4*tau)+r*(delL*rho+rho*delL)/(2*tau))*dt
        rho = rho1#/rho1.tr()
        #print ((A-1j*B)*(np.exp(1j*2*t)-1)/(16*tau)-I_t)
        #I_t = (A-1j*B)*(np.exp(1j*2*(t+dt))-1)/(16*tau)
        I_t+= 1j*(A-1j*B)*(np.exp(1j*2*phi))*dt/(8*tau)

    #t = t+dt
    i+=1
    #print (expM)
  if (varReturn == 1):
    return rho, expLs, expMs, varLs, covLMs, varMs,rop_strat,nbar
  else:
    return rho, expLs, expMs



def OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau):
  t_f = ts[-1]
  q1i = expect(X,rho_i)
  q2i = expect(P,rho_i)
  q1f = expect(X,rho_f)
  q2f = expect(P,rho_f)
  Q1f = q1i*np.cos(t_f)+q2i*np.sin(t_f)
  Q2f = -q1i*np.sin(t_f)+q2i*np.cos(t_f)
  q4 = np.sqrt(1+4*tau*tau)-2*tau
  q3 = np.sqrt(4*tau*q4)
  q5 = q3*(1+q4/(2*tau))
  b = np.array([[q3, 0],[q4,0]])/(2*np.sqrt(tau))
  Bmat = np.matmul(b,b.T)
 # Bmat1 = np.array([[q3**2, q3*q4],[q3*q4,q4**2]])/(4*tau)
  #print (Bmat, Bmat1)
  Amat00 = (Bmat[0,0]+Bmat[1,1])*t_f*np.cos(t_f)/2.0-(Bmat[1,1]-Bmat[0,0])*np.sin(t_f)/2.0
  Amat01 = ((Bmat[0,0]+Bmat[1,1])*t_f/2.0+Bmat[0,1])*np.sin(t_f)
  Amat11 = (Bmat[0,0]+Bmat[1,1])*t_f*np.cos(t_f)/2.0+(Bmat[1,1]-Bmat[0,0])*np.sin(t_f)/2.0
  Amat10 = (-(Bmat[0,0]+Bmat[1,1])*t_f/2.0+Bmat[0,1])*np.sin(t_f)
  Amat = np.array([[Amat00, Amat01], [Amat10, Amat11]])
  al12 = np.matmul(np.linalg.inv(Amat), np.array([q1f-Q1f,q2f-Q2f]))
  q1t = ((Bmat[0,0]+Bmat[1,1])*al12[0]*ts/2.0+q1i)*np.cos(ts)+(((Bmat[0,0]+Bmat[1,1])*al12[1]*ts/2.0+q2i)-((Bmat[1,1]-Bmat[0,0])*al12[0]/2.0-Bmat[1,0]*al12[1]))*np.sin(ts)
  q2t = ((Bmat[0,0]+Bmat[1,1])*al12[1]*ts/2.0+q2i)*np.cos(ts)-(((Bmat[0,0]+Bmat[1,1])*al12[0]*ts/2.0+q1i)-((Bmat[1,1]-Bmat[0,0])*al12[1]/2.0+Bmat[1,0]*al12[0]))*np.sin(ts)
  alr = (al12[0]*q3+al12[1]*q4)/2+q1i
  ali = (al12[0]*(Bmat[0,0]-q4/2)+al12[1]*(Bmat[0,1]+q3/2))+q2i
  A = 4*al12[1]*tau*(Bmat[0,0]+Bmat[1,1])
  B = -4*al12[0]*tau*(Bmat[0,0]+Bmat[1,1])
  p1t = al12[0]*np.cos(ts)+al12[1]*np.sin(ts)
  p2t = -al12[0]*np.sin(ts)+al12[1]*np.cos(ts)
  rt = q1t+(p1t*q3+p2t*q4)/2.0
  return q3, q4, q5, alr, ali, A, B, q1t, q2t,rt


def rho_update_strat(i, Input_Initials):
  Initials, X, P, H, rho, I_t, theta_t, ts, tau, dt, j, Id = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  theta = theta_t[j]
  phi = theta+t
  #csth, snth = jnp.cos(theta), jnp.sin(theta)
  csph, snph = jnp.cos(phi), jnp.sin(phi)
  #cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  Ljump = csph*X+snph*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  #Mjump = -snph*X+csph*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expV = jnp.trace(jnp.matmul(Ljump2, rho)).real
  expL = csph*expX + snph*expP
  #expM = -snph*expX + csph*expP
  delL = Ljump - expL*Id
  delV = Ljump2-expV*Id
  delh_t_Mat = jnp.exp(-1j*phi)*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau)])
  ht = jnp.matmul(delh_t_Mat, Initials) + jnp.exp(-1j*phi)*I_t
  r = ht.real
  #H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho1 = rho + (Lind_update+read_update)*dt
  delI_t_Mat = jnp.array([0.0  ,0. , 1j*jnp.exp(1j*2*phi)/(8.0*tau), jnp.exp(1j*2*phi)/(8.0*tau) ])
  I_t1 = I_t + jnp.matmul(delI_t_Mat, Initials)*dt
  return (Initials, X, P, H, rho1, I_t1, theta_t, ts, tau, dt, j+1, Id)


def OPsoln_strat_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Id):
  #I_tR = jnp.array([0.0])
  I_t = jnp.array([0.0 + 1j*0.0])
  rho = rho_i
  k1=0
  Initials, X, P, H,  rho, I_t, theta_t, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts), rho_update_strat,(Initials, X, P, H,  rho, I_t,  theta_t, ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho


def Tr_Distance(rho_f_simul, rho_f):
  delrho = rho_f_simul-rho_f
  eps = 1e-3
  delrho2 = jnp.matmul(delrho, delrho)
  return jnp.trace(delrho2).real#-jnp.exp(-jnp.trace(delrho2).real/eps)


def CostF_strat(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul = OPsoln_strat_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  return 1e2*Tr_Distance(rho_f_simul, rho_f)


@jit
def update_strat(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id,  step_size):
    grads=grad(CostF_strat)(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

def PlotOP(Initials, X, P, H, rho_i, rho_f, ts, theta_t, tau, figname):
    q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)
    alr, ali, A, B = Initials[0], Initials[1], Initials[2], Initials[3]
    #alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm = Initials[0], Initials[1], Initials[2], Initials[3], Initials[4], Initials[5], Initials[6], Initials[7], Initials[8]

    #rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul = OPsoln_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)
    rho_f_simul1, X_simul1, P_simul1, varX_simul1, covXP_simul1, varP_simul1, rop_strat,nbar = OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)
    #rho_f_simul1, X_simul1, P_simul1, varX_simul1, covXP_simul1, varP_simul1, rop_strat,nbar, theta_t = OPsoln_control_l10(X, P, H, rho_i, alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm, ts,   tau, 1)
    #jnpId = jnp.identity(25)
    #Q1j, Q2j, Q3j, Q4j, Q5j, thetaj = OPintegrate_strat_JAX(Initials, X, P, H, rho_i, ts, ts[1]-ts[0],  tau, jnpId)

    XItf = X*np.cos(ts[-1])+P*np.sin(ts[-1])
    PItf = P*np.cos(ts[-1])-X*np.sin(ts[-1])
    Q1 = expect(XItf,rho_f_simul1)
    Q2 = expect(PItf,rho_f_simul1)
    V1 = expect(XItf*XItf,rho_f_simul1)-Q1**2
    V3 = expect(PItf*PItf,rho_f_simul1)-Q2**2
    V2 = (expect(XItf*PItf+PItf*XItf,rho_f_simul1)-2*Q1*Q2)/2.0
    
    a = (X+1j*P)/np.sqrt(2)
    q1i = expect(X,rho_i)
    q1f = expect(X,rho_f)
    q2i = expect(P,rho_i)
    q2f = expect(P,rho_f)
    
    t_i, t_f = ts[0], ts[-1]
    fig, axs = plt.subplots(7,1,figsize=(6,14),sharex='all')
    axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian Approx')
    #axs[0].plot(ts, Q1j, color='k')
    #axs[0].plot(ts, X_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
    axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'General')
    axs[0].plot(t_i, q1i, "o", color = 'b')
    axs[0].plot(t_f, q1f, "X" , color = 'r')
    axs[0].plot(t_f, Q1, "^" , color = 'k')
    axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
    #axs[1].plot(ts, P_simul, linewidth =4, linestyle = 'dashed', color = 'blue')
    axs[1].plot(ts, P_simul1, linewidth =3, linestyle = 'dashed', color = 'red')

    axs[1].plot(t_i, q2i, "o", color = 'b')
    axs[1].plot(t_f, q2f, "X", color = 'r')
    #axs[1].plot(t_f, Q2, "^" , color = 'k')
    axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 15)
    axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 15)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[0].legend(loc=4,fontsize=12)

    axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    #axs[2].plot(ts, varX_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
    axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
    axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b')
    axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "X", color = 'r')
    #axs[2].plot(t_f, V1, "^" , color = 'k')
    
    axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    #axs[3].plot(ts, covXP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
    axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
    axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b')
    axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "X", color = 'r')
    #axs[3].plot(t_f, V2, "^" , color = 'k')

    axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    #axs[4].plot(ts, varP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'Ito')
    axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
    axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b')
    axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "X", color = 'r')
    #axs[4].plot(t_f, V3, "^" , color = 'k')

    axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
    axs[5].plot(ts, rop_strat,linewidth =3, color='red', linestyle='dashed')
    #axs[5].plot(ts, nbar, color='red', linestyle ='dashed', linewidth = 3)
    axs[6].plot(ts, np.zeros(len(ts)),linewidth =4, color='green')
    axs[6].plot(ts, theta_t,linewidth =3, color='red', linestyle='dashed')


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
    plt.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/Optimal_Path/Plots/'+figname+'.pdf',bbox_inches='tight')



def OPsoln_control_l10(X, P, H, rho_i, alr, ali, A, B, Cv, k0r, k0i, Dvp, Dvm,  ts,   tau, varReturn = 0):
  rho = rho_i
  #t=ts[0]
  t_f = ts[-1]
  dt = ts[1]-ts[0]
  I_t = 0
  I_k_t=0
  I_Gp_t=0
  I_G_t=0
  expLs = np.zeros(len(ts))
  expMs = np.zeros(len(ts))
  varLs = np.zeros(len(ts))
  varMs = np.zeros(len(ts))
  covLMs = np.zeros(len(ts))
  rop_strat = np.zeros(len(ts))
  theta_t = np.zeros(len(ts))
  nbar = np.zeros(len(ts))
  theta = 0
  i=0
  phi = 0
  #csph, snph = np.cos(phi), np.sin(phi)
  csth, snth = np.cos(theta), np.sin(theta)
  Ljump = csth*X+snth*P
  Mjump = -snth*X+csth*P
  Mjump2 = Mjump*Mjump
  expL = expect(Ljump,rho).real
  expM = expect(Mjump,rho).real
  varL = expect(Ljump*Ljump,rho).real-expL**2
  varM = expect(Mjump2,rho).real-expM**2
  covLM = (expect(Ljump*Mjump+Mjump*Ljump,rho).real-2*expL*expM)/2.0
  while (i<len(ts)):
    #print (ts[i])
    t = ts[i]
    #print (theta)
    theta_t[i]=theta
    #theta = theta_t[i]
    phi = theta + t
    #print (t)
    #csth, snth = np.cos(theta), np.sin(theta)
    csph, snph = np.cos(phi), np.sin(phi)
    #cs2ph, sn2ph = np.cos(2*phi), np.sin(2*phi)
    #Ljump = csth*X+snth*P
    
    #Ljump2 = Ljump*Ljump
    #Ljump2_a = (csph**2)*X*X+(snph**2)*P*P+csph*snph*(X*P+P*X)
    #Mjump = -snth*X+csth*P
    #Mjump = -snph*X+csph*P
    expLs[i] = expL
    expMs[i] = expM
    #delL = Ljump - expL
    #delVjump = Ljump2-expect(Ljump2, rho).real
    #delM = Mjump-expM
    #Mjump2 = Mjump*Mjump
    #delL2 = delL*delL
    #varL1 = expect(Ljump2_a,rho).real-expL**2
    if (varReturn ==1):
      varLs[i] = varL
      varMs[i] = varM
      covLMs[i] = covLM
      nbar[i] = 0
    
    ht = np.exp(-1j*phi)*(alr+1j*ali+1j*t*(A+1j*B)/(8*tau))+np.exp(-1j*phi)*I_t
    r =  ht.real
    v = ht.imag
    w = A*csph+B*snph
    z = -A*snph+B*csph
    kappa = np.exp(1j*2*phi)*(k0r+1j*k0i+I_k_t)
    kappaLM = kappa.imag
    kappaLL=Cv-kappa.real
    Gp = Dvp+I_Gp_t
    G = np.exp(1j*2*phi)*(Dvm+1j*alr*ali+I_G_t)
    Gr = G.real
    GLM=G.imag
    GLL=Gp-Gr
    rop_strat[i]=r
    #print (GLM-r*v)
    dtheta = ((2*r*w-kappaLL)/((4*r**2-4*v**2+8*Gr)*tau)-1)
    dphi = dtheta+1
    expL1 = expL+dt*(dphi*expM+(r-expL)*2*varL/(2*tau))
    expM1 = expM+dt*(-dphi*expL+(r-expL)*2*covLM/(2*tau))
    varL1 = varL+dt*(2*dphi*covLM-varL/tau)
    covLM1 = covLM+dt*(dphi*(varM-varL)-varL*covLM/tau)
    varM1 =  varM+dt*(-2*dphi*covLM+(1-4*covLM**2)/(4*tau))    
    I_t+= 1j*(A-1j*B)*(np.exp(1j*2*phi))*dt/(8*tau)
    I_k_t+=dt*1j*np.exp(-1j*2*phi)*(GLL-r**2)/tau
    I_Gp_t+=dt*(kappaLM-r*z)/(4*tau)
    I_G_t+=dt*(1j*(Cv-kappa)*np.exp(-1j*2*phi)-(B+1j*A)*r*np.exp(-1j*phi))/(4*tau)
    expL, expM, varL, covLM, varM = expL1, expM1, varL1, covLM1, varM1
    #print (dtheta)
    theta+=dt*dtheta
    #t = t+dt
    i+=1
    #print (expM)
  if (varReturn == 1):
    return rho, expLs, expMs, varLs, covLMs, varMs,rop_strat,nbar, theta_t
  else:
    return rho, expLs, expMs


def rho_update_control_l10(i, Input_Initials): #Optimal control integration with \lambda_1=0
  Initials, X, P, H, rho, I_t, I_kvp_t, I_k_t, I_Gvp_t, I_G_t,  kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0,  GMM0, Idth,   theta, ts, tau, dt, j, Id = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  #theta = theta_t[j]
  phi = theta+t
  
  #dphi  = jnp.array([1.0])#+0.1*jnp.tanh(10.0*(dphi -1))
  #dphi = jnp.array([1.0])
  
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  #csph, snph = jnp.cos(phi), jnp.sin(phi)
  #cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  #Mjump = -snth*X+csth*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expV = jnp.trace(jnp.matmul(Ljump2, rho)).real
  expL = csth*expX + snth*expP
  #exphi = 
  #expM = -snph*expX + csph*expP
  delL = Ljump - expL*Id
  delV = Ljump2-expV*Id
  e_jphi = jnp.exp(-1j*phi)
  delh_t_Mat = e_jphi*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
  ht = jnp.matmul(delh_t_Mat, Initials) + e_jphi*I_t
  r = ht.real 
  v = ht.imag
  wzmat = jnp.array([0,0,1,1j,0,0,0,0,0])
  wz = jnp.matmul(wzmat, Initials)*e_jphi
  w = wz.real
  z = wz.imag
  kappavp0 = (kappaLL0+kappaMM0)/2.0
  kappavp = I_kvp_t+kappavp0
  kappa0 = (kappaMM0-kappaLL0)/2.0+1j*kappaLM0
  kappa = (kappa0+I_k_t)/(e_jphi**2)
  kappar = kappa.real
  #kappaMM = kappar+kappavp
  kappaLL = kappavp-kappar
  kappaLM = kappa.imag
  Gvp0 = (GLL0+GMM0)/2.0
  Gvp = Gvp0+I_Gvp_t
  G0 = (GMM0-GLL0)/2.0+1j*GLM0
  G = (G0+I_G_t)/(e_jphi**2)
  Gr = G.real
  GLL = Gvp-Gr
  GLM = G.imag

  
  #GLL=(C1+(z**2-w**2)/4.0)/2.0
  dphi = -kappaLL/((z**2-w**2+8*Gr)*tau)
  dtheta = dphi-1.0
  #delk_t_Mat=jnp.exp(1j*2*phi)*jnp.array([0,0,0,0,0,1,1j,0,0])
  #kappa = jnp.matmul(delk_t_Mat,Initials)+jnp.exp(1j*2*phi)*I_k_t
  #kappaLM = kappa.imag
  #llcoeff = jnp.array([0,0,0,0,1+1j*0,0,0,0,0])
  #kappaLL=jnp.matmul(llcoeff, Initials).real-kappa.real
  #prodmat = np.zeros((9,9),dtype=np.complex_)
  #prodmat[0,1]=1j
  #prodmat = jnp.array(prodmat)
  #GLM = r*v
  #Gr = (GMM-GLL)/2.0
  #dtheta = (2*r*w-kappaLL)/((4*r**2-4*v**2+8*Gr)*tau)-1
  #dphi = jnp.array([1])+0.1*jnp.tanh(10*dtheta)
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho_update =H_update+Lind_update+read_update
  rho1 = rho + rho_update*dt
  delI_t_Mat = jnp.array([0.0 ,0.,1j/(8.0*tau), 1.0/(8.0*tau),0.0,0.0,0.0,0.0,0.0 ])/(e_jphi**2)
  I_t1 = I_t + jnp.matmul(delI_t_Mat, Initials)*dt
  I_kvp_t1 = I_kvp_t+dt*GLM/tau
  I_k_t1 = I_k_t+dt*(GLM+1j*GLL)*e_jphi**2/tau
  I_Gvp_t1 = I_Gvp_t+dt*kappaLM/(4*tau)
  I_G_t1 = I_G_t+dt*(kappaLM+1j*kappaLL)*e_jphi**2/(4*tau)
  
  #GLL1 = GLL+2*dphi*(GLM)*dt
  #GMM1 = GMM-2*dphi*(GLM)*dt+dt*kappaLM/(2*tau)
  #kappaLL1 = kappaLL+dt*(2*dphi*kappaLM)
  #kappaLM1 = kappaLM+dt*(dphi*(kappaMM-kappaLL)+GLL/tau)
  #kappaMM1 = kappaMM+dt*(-2*dphi*kappaLM + 2*(GLM)/tau)
  #GLM1 = GLM+dt*(dphi*(GMM-GLL)+kappaLL/(4*tau))
  #r1 = r+dt*dphi*v
  #v1 = v+dt*(-dphi*r+w/(4*tau))
  #w1 = w+dt*dphi*z
  #z1 = z-dt*w*dphi
  theta1 = theta + dt*dtheta
  Idth = GLM+w*z/4
  #Idth = -(GLL+w**2/4.0)/(2*tau)-(kappaLL+2*r*w)/2.0-(kappaMM+2*v*z)/2.0
  return (Initials, X, P, H, rho1, I_t1, I_kvp_t1, I_k_t1, I_Gvp_t1, I_G_t1, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0,  GMM0, Idth,  theta1, ts, tau, dt, j+1, Id)


def OPsoln_control_l10_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Id):
  #I_tR = jnp.array([0.0])
  I_t = jnp.array(0.0+1j*0.0)
  I_kvp_t = jnp.array(0.0)
  I_k_t = jnp.array(0.0+1j*0.0)
  I_Gvp_t = jnp.array(0.0)
  I_G_t = jnp.array(0.0+1j*0.0)
  #print (I_t)
  #I_k_t = jnp.array([0.0 + 1j*0.0])
  Idth =  0.0
  r0 = jnp.matmul(jnp.array([1.0,0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
  v0 = jnp.matmul(jnp.array([0,1.0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
  w0 = jnp.matmul(jnp.array([0,0,1.0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
  z0 = jnp.matmul(jnp.array([0,0,0,1.0,0,0,0,0,0]),Initials)#+jnp.array([0])
  #GLL = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,-1.0]),Initials)#+jnp.array([0])
  kappaLL = jnp.matmul(jnp.array([0,0,0,0,1,-1.0,0,0,0]),Initials)#+jnp.array([0])
  kappaLM = jnp.matmul(jnp.array([0,0,0,0,0,0,1.0,0,0]),Initials)#+jnp.array([0])
  kappaMM = jnp.matmul(jnp.array([0,0,0,0,1,1.0,0,0,0]),Initials)#+jnp.array([0])
  GLL = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,-1.0]),Initials)#+jnp.array([0])
  GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)#+jnp.array([0])
  GLM = r0*v0
  kappaLL = kappaLL-2*r0*w0
  kappaLM = kappaLM-v0*w0-r0*z0
  kappaMM = kappaMM-2*v0*z0
  GLL+= -r0**2-w0**2/4.0
  #C1 = 2*(GLL-r**2-w**2/4.0)-(z**2-w**2)/4.0
  GLM+= -r0*v0-w0*z0/4.0
  GMM+= -v0**2-z0**2/4.0
  GLL0 = GLL
  GLM0 = GLM
  GMM0 = GMM
  kappaLL0 = kappaLL
  kappaLM0 = kappaLM
  kappaMM0 = kappaMM
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  theta = jnp.array(0.0)
  k1=0
  Idth=0
  Initials, X, P, H,  rho, I_t, I_kvp_t, I_k_t, I_Gvp_t, I_G_t, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0, GMM0, Idth,  theta, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_control_l10,(Initials, X, P, H,  rho, I_t, I_kvp_t, I_k_t, I_Gvp_t, I_G_t, kappaLL0, kappaLM0, kappaMM0, GLL0, GLM0,  GMM0, Idth,  theta, ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho, Idth

def CostF_control_l10(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  #print (Idth)
  return 1e2*Tr_Distance(rho_f_simul, rho_f)

def CostF_control_l101(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  return 1e2*Tr_Distance(rho_f_simul, rho_f), Idth

@jit
def update_control_l10(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id,  step_size):
    grads=grad(CostF_control_l10)(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])


def OPintegrate_strat(Initials, X, P, H, rho_i, ts, dt,  tau, Id):
  #I_tR = jnp.array([0.0])
  r0 = np.matmul(np.array([1.0,0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
  v0 = np.matmul(np.array([0,1.0,0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
  w0 = np.matmul(np.array([0,0,1.0,0,0,0,0,0,0]),Initials)#+jnp.array([0])
  z0 = np.matmul(np.array([0,0,0,1.0,0,0,0,0,0]),Initials)#+jnp.array([0])
  GLL0 = np.matmul(np.array([0,0,0,0,0,0,0,1,-1.0]),Initials)#+jnp.array([0])
  kappaLL0 = np.matmul(np.array([0,0,0,0,1,-1.0,0,0,0]),Initials)#+jnp.array([0])
  kappaLM0 = np.matmul(np.array([0,0,0,0,0,0,1.0,0,0]),Initials)#+jnp.array([0])
  kappaMM0 = np.matmul(jnp.array([0,0,0,0,1,1.0,0,0,0]),Initials)#+jnp.array([0])
  GMM0 = np.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)#+jnp.array([0])
  GLM0 = r0*v0
  kappaLL0 = kappaLL0-2*r0*w0
  kappaLM0 = kappaLM0-v0*w0-r0*z0
  kappaMM0 = kappaMM0-2*v0*z0
  GLL0+= -r0**2-w0**2/4.0
  GLM0+= -r0*v0-w0*z0/4.0
  GMM0+= -v0**2-z0**2/4.0
  kappavp0 = (kappaLL0+kappaMM0)/2.0
  kappa0 = (kappaMM0-kappaLL0)/2.0+1j*kappaLM0
  Gvp0 = (GLL0+GMM0)/2.0
  
  #GLL0 = GLL
  #GLM0 = GLM
  #GMM0 = GMM
  #kappaLL0 = kappaLL
  #kappaLM0 = kappaLM
  #kappaMM0 = kappaMM
  rho = rho_i
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  theta_t = np.zeros(len(ts))
  diff = np.zeros(len(ts))
  phi = 0
  I_t=0
  I_kvp_t = 0
  I_k_t = 0
  I_Gvp_t = 0
  I_G_t = 0
  w = w0
  z = z0
  while (j<len(ts)):
      #print (j,r)
      #Initials, X, P, H, rho, I_t, I_k_t, I_Gp_t, I_G_t,   phi,  ts, tau, dt, j, Id, Q1, Q2, Q3, Q4, Q5 = Input_Initials
      #I_t = I_tR + 1j*I_tI
      #print (tau)
      t = ts[j]
      theta_t[j] = phi-t
      e_jphi = np.exp(-1j*phi)
      delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = ht.real
      v = ht.imag
      wzmat = np.array([0,0,1.0,1j,0,0,0,0,0])
      wz = np.matmul(wzmat, Initials)*e_jphi
      w = wz.real
      z = wz.imag
      kappavp = I_kvp_t+kappavp0
      kappa = (kappa0+I_k_t)/(e_jphi**2)
      kappar = kappa.real
      kappaMM = kappar+kappavp
      kappaLL = kappavp-kappar
      kappaLM = kappa.imag
      Gvp = Gvp0+I_Gvp_t
      G = (G0+I_G_t)/(e_jphi**2)
      Gr = G.real
      GLL = Gvp-Gr
      GLM = G.imag
      
      dphi = -kappaLL/((z**2-w**2+8*Gr)*tau)
      #dphi  = 1#+0.1*jnp.tanh(10.0*(dphi -1))
      #print (dphi)
      #dphi = jnp.array([1.0])
      #dtheta = dphi-1.0
      #csth, snth = jnp.cos(theta), jnp.sin(theta)
      #theta = theta_t[j]
      #theta_t[j]=phi.item()-t.item()
      theta = phi-t
      csth, snth = np.cos(theta), np.sin(theta)
      #csph, snph = np.cos(phi), np.sin(phi)
      
      #cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
      Ljump = csth*X+snth*P
      Ljump2 = np.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
      Mjump = -snth*X+csth*P
      Mjump2 = np.matmul(Mjump, Mjump)
      expX = np.trace(np.matmul(X, rho)).real
      expP = np.trace(np.matmul(P, rho)).real
      expV = np.trace(np.matmul(Ljump2, rho)).real
      expL = csth*expX + snth*expP
      #exphi = 
      expM = -snth*expX + csth*expP
      #print (expL)
      Q1[j]=expL
      Q2[j]=expM
      delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = np.trace(np.matmul(Ljump2,rho)).real-expL**2
      Q5[j] = np.trace(np.matmul(Mjump2,rho)).real-expM**2
      delV = Ljump2-expV*Id
      H_update = -1j*(np.matmul(H, rho)-np.matmul(rho, H))
      Lind_update = (-np.matmul(delV, rho)-np.matmul(rho, delV))/(4*tau)
      read_update = r*(np.matmul(delL, rho)+ np.matmul(rho, delL))*(1.0/(2*tau))
      drho = H_update+Lind_update+read_update
      #print (Lind_update)
      #print (np.trace(drho).real)
      rho+=drho*dt
      #rho=rho/np.trace(rho)
      delI_t_Mat =np.array([0.0 ,0.,1j/(8.0*tau), 1.0/(8.0*tau),0.0,0.0,0.0,0.0,0.0 ])/(e_jphi**2)
      I_t+= np.matmul(delI_t_Mat, Initials)*dt
      I_kvp_t+=dt*GLM/tau
      I_k_t+=dt*(GLM+1j*GLL)*e_jphi**2/tau
      I_Gvp_t+= dt*kappaLM/(4*tau)
      I_G_t+= dt*(kappaLM+1j*kappaLL)*e_jphi**2/(4*tau)
      #GLL1 = GLL+2*dphi*(GLM)*dt
      #GLM1 = GLM+dt*(dphi*(GMM-GLL)+kappaLL/(4.0*tau))
      #GMM1 = GMM-2*dphi*(GLM)*dt+dt*kappaLM/(2*tau)
      #kappaLL1 = kappaLL+dt*(2*dphi*kappaLM)
      #kappaLM1 = kappaLM+dt*(dphi*(kappaMM-kappaLL)+GLL/tau)
      #kappaMM1 = kappaMM+dt*(-2*dphi*kappaLM + 2*(GLM)/tau)
      phi+=dphi*dt
      diff[j]=GLM+w*z/4.0
      
      #GLL,GMM, GLM = GLL1, GMM1, GLM1
      #kappaLL, kappaMM, kappaLM = kappaLL1, kappaMM1, kappaLM1
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, theta_t, rho , diff