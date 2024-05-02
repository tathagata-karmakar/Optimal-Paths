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
  delrho2 = jnp.matmul(delrho, delrho)
  return jnp.trace(delrho2).real


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
    #rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul = OPsoln_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)
    rho_f_simul1, X_simul1, P_simul1, varX_simul1, covXP_simul1, varP_simul1, rop_strat,nbar = OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, 1)
    
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
    fig, axs = plt.subplots(6,1,figsize=(6,12),sharex='all')
    axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian Approx')
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

    axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
    axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
    axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
    axs[5].set_ylabel('$r^\star$', fontsize = 15)
    axs[2].tick_params(labelsize=15)
    axs[3].tick_params(labelsize=15)
    axs[4].tick_params(labelsize=15)
    axs[5].tick_params(labelsize=15)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('/Users/t_karmakar/Library/CloudStorage/Box-Box/Research/Optimal_Path/Plots/'+figname+'.pdf',bbox_inches='tight')


def OPintegrate_strat_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt,  tau, Id):
  #I_tR = jnp.array([0.0])
  I_t = jnp.array([0.0 + 1j*0.0])
  rho = rho_i
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  while (j<len(ts)):
  #Initials, X, P, H,  rho, I_t, theta_t, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts), rho_update_strat,(Initials, X, P, H,  rho, I_t,  theta_t, ts, tau, dt, k1, Id))
  #Initials, X, P, H, rho, I_t, theta_t, ts, tau, dt, j, Id = Input_Initials
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
      Mjump = -snph*X+csph*P
      Mjump2 = jnp.matmul(Mjump, Mjump)
      expX = jnp.trace(jnp.matmul(X, rho)).real
      expP = jnp.trace(jnp.matmul(P, rho)).real
      expV = jnp.trace(jnp.matmul(Ljump2, rho)).real
      expL = csph*expX + snph*expP
      LMjump2 = jnp.matmul(Ljump, Mjump)+jnp.matmul(Mjump, Ljump)
      expLM = jnp.trace(jnp.matmul(LMjump2, rho)).real
      Q1[j]=expL
      Q3[j]=2*(expV-expL**2)
      expM = -snph*expX + csph*expP
      Q2[j]=expM
      delL = Ljump - expL*Id
      Q5[j]=2*(jnp.trace(jnp.matmul(Mjump2, rho)).real-expM**2)
      Q4[j]= expLM-2*expL*expM
      delV = Ljump2-expV*Id
      delh_t_Mat = jnp.exp(-1j*phi)*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau)])
      ht = jnp.matmul(delh_t_Mat, Initials) + jnp.exp(-1j*phi)*I_t
      r = ht.real
      #H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
      Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
      Back_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
      rho = rho + (Lind_update+Back_update)*dt
      delI_t_Mat = jnp.array([0.0  ,0. , 1j*jnp.exp(1j*2*phi)/(8.0*tau), jnp.exp(1j*2*phi)/(8.0*tau) ])
      I_t = I_t + jnp.matmul(delI_t_Mat, Initials)*dt
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return Q1,Q2,Q3,Q4,Q5