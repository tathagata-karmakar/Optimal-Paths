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
def OPsoln_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, varReturn = 0):
  rho = rho_i
  t=ts[0]
  t_f = ts[-1]
  dt = ts[1]-ts[0]
  I_t = 0
  expLs = np.zeros(len(ts))
  expMs = np.zeros(len(ts))
  varLs = np.zeros(len(ts))
  varMs = np.zeros(len(ts))
  covLMs = np.zeros(len(ts))
  i=0
  while (i<len(ts)):
    theta = theta_t[i]
    phi = theta + t
    csth, snth = np.cos(theta), np.sin(theta)
    csph, snph = np.cos(phi), np.sin(phi)
    cs2ph, sn2ph = np.cos(2*phi), np.sin(2*phi)
    Ljump = csth*X+snth*P
    Ljump2 = Ljump*Ljump
    Mjump = -snth*X+csth*P
    expL = expect(Ljump,rho).real
    expM = expect(Mjump,rho).real
    expLs[i] = expL
    expMs[i] = expM
    delL = Ljump - expL
    delL2 = delL*delL
    varL = expect(delL2,rho).real
    if (varReturn ==1):
      delM = Mjump-expM
      varLs[i] = varL
      varMs[i] = expect(delM*delM,rho).real
      covLMs[i] = (expect(Ljump*Mjump+Mjump*Ljump,rho).real-2*expL*expM)/2.0
    addL = (delL2-varL)/(2*tau)
    ft = expect(commutator(addL,Ljump,kind='anti')/2.0,rho).real
    #print (ft)
    gt = expect(commutator(addL,Mjump,kind='anti')/2.0,rho).real
    ht = np.exp(-1j*phi)*(alr+1j*ali+1j*t*(A+1j*B)/(8*tau))+np.exp(-1j*phi)*I_t
    u =  ht.real
    rho1 = rho+(-1j*commutator(H,rho)+(Ljump*rho*Ljump-Ljump*Ljump*rho/2.0-rho*Ljump*Ljump/2.0)/(4*tau)+u*commutator(delL, rho,kind='anti')/(2*tau))*dt
    rho = rho1#/rho1.tr()
    I_t+= ((ft+1j*gt)*np.exp(1j*phi)+1j*(A-1j*B)*np.exp(1j*2*phi)/(8*tau))*dt
    t = t+dt
    i+=1
    #print (expM)
  if (varReturn == 1):
    return rho, expLs, expMs, varLs, covLMs, varMs
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
  Amat00 = (Bmat[0,0]+Bmat[1,1])*t_f*np.cos(t_f)/2.0-(Bmat[1,1]-Bmat[0,0])*np.sin(t_f)/2.0
  Amat01 = ((Bmat[0,0]+Bmat[1,1])*t_f/2.0+Bmat[0,1])*np.sin(t_f)/2.0
  Amat11 = (Bmat[0,0]+Bmat[1,1])*t_f*np.cos(t_f)/2.0+(Bmat[1,1]-Bmat[0,0])*np.sin(t_f)/2.0
  Amat10 = (-(Bmat[0,0]+Bmat[1,1])*t_f/2.0+Bmat[0,1])*np.sin(t_f)/2.0
  Amat = np.array([[Amat00, Amat01], [Amat10, Amat11]])
  al12 = np.matmul(np.linalg.inv(Amat), np.array([q1f-Q1f,q2f-Q2f]))
  q1t = ((Bmat[0,0]+Bmat[1,1])*al12[0]*ts/2.0+q1i)*np.cos(ts)+(((Bmat[0,0]+Bmat[1,1])*al12[1]*ts/2.0+q2i)-((Bmat[1,1]-Bmat[0,0])*al12[0]/2.0-Bmat[1,0]*al12[1]))*np.sin(ts)
  q2t = ((Bmat[0,0]+Bmat[1,1])*al12[1]*ts/2.0+q2i)*np.cos(ts)-(((Bmat[0,0]+Bmat[1,1])*al12[0]*ts/2.0+q1i)-((Bmat[1,1]-Bmat[0,0])*al12[1]/2.0+Bmat[1,0]*al12[0]))*np.sin(ts)
  alr = (al12[0]*q3+al12[1]*q4)/2
  ali = (al12[0]*q4+al12[1]*q5)/2
  A = al12[1]
  B = -al12[0]
  return q3, q4, q5, alr, ali, A, B, q1t, q2t

def rho_update(i, Input_Initials):
  Initials, X, P, H, rho, I_t, theta_t, ts, tau, dt, j, Id = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  theta = theta_t[j]
  phi = theta+t
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  csph, snph = jnp.cos(phi), jnp.sin(phi)
  cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  Mjump = -snth*X+csth*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expL = csth*expX + snth*expP
  expM = -snth*expX + csth*expP
  delL = Ljump - expL*Id
  delL2 =jnp.matmul(delL, delL) #Ljump2+expL**2-2*expL*Ljump
  #delL2 = Ljump2+expL**2-2*expL*Ljump
  varL = jnp.trace(jnp.matmul(delL2,rho)).real
  delM = Mjump-expM*Id
  addL = (delL2-varL*Id)/(2*tau)
  ft = jnp.trace(jnp.matmul((jnp.matmul(addL,Ljump)+ jnp.matmul(Ljump,addL)) /2.0,rho).real)
  gt = jnp.trace(jnp.matmul((jnp.matmul(addL,Mjump)+ jnp.matmul(Mjump,addL)) /2.0,rho).real)
  #delh_tR_Mat = jnp.array([csph,snph,t*snph/(8.0*tau), -t*csph/(8.0*tau)])
  #delh_tI_Mat = jnp.array([-snph,csph,t*csph/(8.0*tau), t*snph/(8.0*tau)])
  delh_t_Mat = jnp.exp(-1j*phi)*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau)])
  #htR = jnp.matmul(delh_tR_Mat, Initials) + csph*I_tR + snph*I_tI
  #htI = jnp.matmul(delh_tI_Mat, Initials) - snph*I_tR + csph*I_tI
  ht = jnp.matmul(delh_t_Mat, Initials) + jnp.exp(-1j*phi)*I_t
  u = ht.real
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (jnp.matmul(jnp.matmul(Ljump,rho),Ljump)-jnp.matmul(Ljump2, rho)/2.0-jnp.matmul(rho, Ljump2)/2.0)/(4*tau)
  Back_update = u*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho1 = rho + (H_update+Lind_update+Back_update)*dt
  #rho = rho1#/rho1.tr()
  #delI_tR_Mat = jnp.array([0.0,0.0,-sn2ph/(8*tau), cs2ph/(8*tau)])
  #delI_tI_Mat = jnp.array([0.0,0.0,cs2ph/(8*tau), sn2ph/(8*tau)])
  delI_t_Mat = jnp.array([0.0  ,0. , 1j*jnp.exp(1j*2*phi)/(8.0*tau), jnp.exp(1j*2*phi)/(8.0*tau) ])
  #delI_tR_b = jnp.array([ft*csph-gt*snph])
  #delI_tI_b = jnp.array([ft*snph+gt*csph])
  delI_t_b = (ft+1j*gt)*jnp.exp(1j*phi) 
  #I_tR = I_tR + (jnp.matmul(delI_tR_Mat, Initials) + delI_tR_b)*dt
  #I_tI = I_tI + (jnp.matmul(delI_tI_Mat, Initials) + delI_tI_b)*dt
  I_t1 = I_t + (delI_t_b + jnp.matmul(delI_t_Mat, Initials))*dt
  return (Initials, X, P, H, rho1, I_t1, theta_t, ts, tau, dt, j+1, Id)

def OPsoln_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Id):
  #I_tR = jnp.array([0.0])
  I_t = jnp.array([0.0 + 1j*0.0])
  rho = rho_i
  Initials, X, P, H,  rho, I_t, theta_t, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts), rho_update,(Initials, X, P, H,  rho, I_t,  theta_t, ts, tau, dt, 0, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho

def Tr_Distance(rho_f_simul, rho_f):
  delrho = rho_f_simul-rho_f
  delrho2 = jnp.matmul(delrho, delrho)
  return jnp.trace(delrho2).real

def CostF(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul = OPsoln_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  return 1e2*Tr_Distance(rho_f_simul, rho_f)

@jit
def update(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id,  step_size):
    grads=grad(CostF)(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

def PlotOP(Initials, X, P, H, rho_i, rho_f, ts, theta_t, tau, figname):
    q3, q4, q5, alr, ali, A, B, q1t, q2t = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)
    alr, ali, A, B = Initials[0], Initials[1], Initials[2], Initials[3]
    rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul = OPsoln_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)
    t_i, t_f = ts[0], ts[-1]
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
    axs[2].plot(t_i, expect((X-q1t[0])*(X-q1t[0]), rho_i), "o", color = 'b')
    axs[2].plot(t_f, expect((X-q1t[-1])*(X-q1t[-1]), rho_f), "X", color = 'r')

    axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[3].plot(ts, covXP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'General')
    axs[3].plot(t_i, expect((X-q1t[0])*(P-q2t[0])/2.0+(P-q2t[0])*(X-q1t[0])/2.0, rho_i), "o", color = 'b')
    axs[3].plot(t_f, expect((X-q1t[-1])*(P-q2t[-1])/2.0+(P-q2t[-1])*(X-q1t[-1])/2.0, rho_f), "X", color = 'r')

    axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[4].plot(ts, varP_simul, linewidth =4, linestyle = 'dashed', color = 'blue', label = 'General')
    axs[4].plot(t_i, expect((P-q2t[0])*(P-q2t[0]), rho_i), "o", color = 'b')
    axs[4].plot(t_f, expect((P-q2t[-1])*(P-q2t[-1]), rho_f), "X", color = 'r')


    axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
    axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
    axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
    axs[2].tick_params(labelsize=15)
    axs[3].tick_params(labelsize=15)
    axs[4].tick_params(labelsize=15)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/Optimal_Path/Plots/'+figname+'.pdf',bbox_inches='tight')

#@jit
def rho_update1(Initials, H, Ljump, Mjump, rho, I_t, theta, t, tau, dt, Id):
  #Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, theta_t, ts, tau, dt, j = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #t = ts[j]
  #theta = theta_t[j]
  phi = theta+t
  #print (phi)
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  csph, snph = jnp.cos(phi), jnp.sin(phi)
  cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  #Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  #Mjump = -snth*X+csth*P
  #expX = jnp.trace(jnp.matmul(X, rho)).real
  #expP = jnp.trace(jnp.matmul(P, rho)).real
  expL = jnp.trace(jnp.matmul(Ljump, rho)).real
  expM =jnp.trace(jnp.matmul(Mjump, rho)).real
  delL = Ljump - expL*Id
  delL2 =jnp.matmul(delL, delL) #Ljump2+expL**2-2*expL*Ljump
  #delL2 = Ljump2+expL**2-2*expL*Ljump
  varL = jnp.trace(jnp.matmul(delL2,rho)).real
  delM = Mjump-expM*Id
  addL = (delL2-varL*Id)/(2*tau)
  ft = jnp.trace(jnp.matmul((jnp.matmul(addL,Ljump)+ jnp.matmul(Ljump,addL)) /2.0,rho).real)
  gt = jnp.trace(jnp.matmul((jnp.matmul(addL,Mjump)+ jnp.matmul(Mjump,addL)) /2.0,rho).real)
  #delh_tR_Mat = jnp.array([csph,snph,t*snph/(8.0*tau), -t*csph/(8.0*tau)])
  #delh_tI_Mat = jnp.array([-snph,csph,t*csph/(8.0*tau), t*snph/(8.0*tau)])
  delh_t_Mat = jnp.exp(-1j*phi)*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau)])
  #htR = jnp.matmul(delh_tR_Mat, Initials) + csph*I_tR + snph*I_tI
  #htI = jnp.matmul(delh_tI_Mat, Initials) - snph*I_tR + csph*I_tI
  ht = jnp.matmul(delh_t_Mat, Initials) + jnp.exp(-1j*phi)*I_t
  u = ht.real
  #print (u)
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (jnp.matmul(jnp.matmul(Ljump,rho),Ljump)-jnp.matmul(Ljump2, rho)/2.0-jnp.matmul(rho, Ljump2)/2.0)/(4*tau)
  Back_update = u*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho_update = (H_update+Lind_update+Back_update)*dt
  rho = rho + rho_update
  #rho = rho1#/rho1.tr()
  #delI_tR_Mat = jnp.array([0.0,0.0,-sn2ph/(8*tau), cs2ph/(8*tau)])
  #delI_tI_Mat = jnp.array([0.0,0.0,cs2ph/(8*tau), sn2ph/(8*tau)])
  delI_t_Mat = jnp.array([0.0  ,0. , 1j*jnp.exp(1j*2*phi)/(8.0*tau), jnp.exp(1j*2*phi)/(8.0*tau) ])
  #delI_tR_b = jnp.array([ft*csph-gt*snph])
  #delI_tI_b = jnp.array([ft*snph+gt*csph])
  delI_t_b = (ft+1j*gt)*jnp.exp(1j*phi) 
  #I_tR = I_tR + (jnp.matmul(delI_tR_Mat, Initials) + delI_tR_b)*dt
  #I_tI = I_tI + (jnp.matmul(delI_tI_Mat, Initials) + delI_tI_b)*dt
  I_t = I_t + (delI_t_b + jnp.matmul(delI_t_Mat, Initials))*dt
  #print (expM)
  return rho, I_t