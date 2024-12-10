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

def OPsoln_strat_SHO(X, P, H, rho_i, alr, ali, A, B, ts, theta_t,  tau, varReturn = 0):
  rho = rho_i
  #t=ts[0]
  t_f = ts[-1]
  dt = ts[1]-ts[0]
  I_t = 0
  expXs = np.zeros(len(ts))
  expPs = np.zeros(len(ts))
  varXs = np.zeros(len(ts))
  varPs = np.zeros(len(ts))
  covXPs = np.zeros(len(ts))
  rop_strat = np.zeros(len(ts))
  nbar = np.zeros(len(ts))
  i=0
  nlevels = X.dims[0][0]
  Id = qeye(nlevels)
  while (i<len(ts)):
    #print (ts[i])
    t = ts[i]
    theta = theta_t[i]
    phi = theta + t
    #print (t)
    csth, snth = np.cos(theta), np.sin(theta)
    #csph, snph = np.cos(phi), np.sin(phi)
    #cs2ph, sn2ph = np.cos(2*phi), np.sin(2*phi)
    #Ljump = csth*X+snth*P
    Ljump = csth*X+snth*P
    Ljump2 = Ljump*Ljump
    #Ljump2_a = (csph**2)*X*X+(snph**2)*P*P+csph*snph*(X*P+P*X)
    #Mjump = -snth*X+csth*P
    #Mjump = -snth*X+csth*P
    expL = expect(Ljump,rho).real
    expX = expect(X,rho).real
    expP = expect(P,rho).real
    expXs[i] = expX
    expPs[i] = expP
    delL = Ljump - expL
    delVjump = Ljump2-expect(Ljump2, rho).real
    #delL2 = delL*delL
    X2 = X*X
    varX = expect(X2,rho).real-expX**2
    #varL1 = expect(Ljump2_a,rho).real-expL**2
    if (varReturn ==1):
      #delM = Mjump-expM
      varXs[i] = varX
      #Mjump2 = Mjump*Mjump
      P2 = P*P
      varPs[i] = expect(P2,rho).real-expP**2
      covXPs[i] = (expect(X*P+P*X,rho).real-2*expX*expP)/2.0
      nbar[i] = expect((X2+P2-1)/2.0,rho)
    ht = np.exp(-1j*phi)*(alr+1j*ali+1j*t*(A+1j*B)/(8*tau))+np.exp(-1j*phi)*I_t
    r =  ht.real
    rop_strat[i]=r
    if(i<len(ts)-1):
        #drhodt = ((-delVjump*rho-rho*delVjump)/(4*tau)+r*(delL*rho+rho*delL)/(2*tau)-1j*H*rho+1j*rho*H)
        F = -delVjump/(4*tau)+r*delL/(2*tau)
        rho1 = (Id+dt*(-1j*H+F))*rho*(Id+dt*(1j*H+F))
        #print (drhodt.tr())
        rho1 = rho1/rho1.tr()
        #rho1 = rho+((-delVjump*rho-rho*delVjump)/(4*tau)+r*(delL*rho+rho*delL)/(2*tau))*dt
        rho = rho1#/rho1.tr()
        #print ((A-1j*B)*(np.exp(1j*2*t)-1)/(16*tau)-I_t)
        #I_t = (A-1j*B)*(np.exp(1j*2*(t+dt))-1)/(16*tau)
        I_t+= 1j*(A-1j*B)*(np.exp(1j*2*phi))*dt/(8*tau)

    #t = t+dt
    i+=1
    #print (expM)
  if (varReturn == 1):
    return rho, expXs, expPs, varXs, covXPs, varPs,rop_strat,nbar
  else:
    return rho, expXs, expPs



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
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  #csph, snph = jnp.cos(phi), jnp.sin(phi)
  #cs2ph, sn2ph = jnp.cos(2*phi), jnp.sin(2*phi)
  Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  #Mjump = -snph*X+csph*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expV = jnp.trace(jnp.matmul(Ljump2, rho)).real
  expL = csth*expX + snth*expP
  #expM = -snph*expX + csph*expP
  delL = Ljump - expL*Id
  delV = Ljump2-expV*Id
  delh_t_Mat = jnp.exp(-1j*phi)*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau)])
  ht = jnp.matmul(delh_t_Mat, Initials) + jnp.exp(-1j*phi)*I_t
  r = ht.real
  #H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  #Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  #read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  F = -delV/(4*tau)+r*delL/(2*tau)
  rho1 = jnp.matmul(jnp.matmul(Id+dt*(-1j*H+F),rho), Id+dt*(1j*H+F))#+ (H_update+Lind_update+read_update)*dt
  rho1 = rho1/jnp.trace(rho1).real
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


#-jnp.exp(-jnp.trace(delrho2).real/eps)

def Fidelity_PS(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  fid = jnp.trace(jnp.matmul(rho_f_simul, rho_f).real)
  return fid

def np_Fidelity_PS(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  fid = np.trace(np.matmul(rho_f_simul, rho_f).real)
  return fid

def Tr_Distance(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  #dist = jnp.sqrt(jnp.trace(delrho2).real)
  return -Fidelity_PS(rho_f_simul, rho_f)#+1e2*dist

def CostF_strat(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul = OPsoln_strat_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  return Tr_Distance(rho_f_simul, rho_f)


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
    fig, axs = plt.subplots(3,1,figsize=(6,14),sharex='all')
    axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian assumption')
    axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'General')
    axs[0].plot(t_i, q1i, "o", color = 'b', markersize =12)
    axs[0].plot(t_f, q1f, "X" , color = 'r', markersize =12)
    
    axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
    axs[1].plot(ts, P_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
    axs[1].plot(t_i, q2i, "o", color = 'b', markersize =12)
    axs[1].plot(t_f, q2f, "X", color = 'r', markersize =12)
   
    axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 18)
    axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 18)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[0].legend(loc=1,fontsize=15)
    
    '''
    axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
    axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b', markersize =12)
    axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "X", color = 'r', markersize =12)
    
    axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
    axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b', markersize =12)
    axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "X", color = 'r', markersize =12)

    axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
    axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b', markersize =12)
    axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "X", color = 'r', markersize =12)
    '''
    axs[2].plot(ts, rop_prxq, linewidth =4, color='green')
    axs[2].plot(ts, rop_strat,linewidth =3, color='red', linestyle='dashed')
    

    #axs[2].set_ylabel('var('+r'$X)$', fontsize = 15)
    #axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 15)
    #axs[4].set_ylabel('var('+r'$P)$', fontsize = 15)
    axs[2].set_ylabel(r'$r^\star$', fontsize = 18)

    axs[2].set_xlabel(r'$t$', fontsize = 18)
    axs[2].tick_params(labelsize=18)
    #axs[3].tick_params(labelsize=15)
    #axs[4].tick_params(labelsize=15)
    #axs[5].tick_params(labelsize=15)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.savefig('/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Plots/'+figname+'.pdf',bbox_inches='tight')

def G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1):
    G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
    G011 = G01+dt*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*tau))
    k101 = k10+dt*k01
    k011 = k01-dt*(1+2*l1)*k10
    G201 = G20+dt*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*tau))
    G111 = G11+dt*(G02-(1+2*l1)*G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*tau))
    G021 = G02+dt*(-2*(1+2*l1)*G11+csth*(snth*k02+csth*k11-r*k01)/(2*tau))
    k201 = k20+dt*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/tau)
    k111 = k11+dt*(-(1+2*l1)*k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/tau)
    k021 = k02+dt*(-2*(1+2*l1)*k11+2*csth*(-snth*G02-csth*G11+r*G01)/tau)  
    return G101, G011, k101, k011, G201, G111, G021, k201, k111, k021

def G_k_updates_first_order(G10, G01, k10, k01, csth, snth, dt, tau, l1):
    G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
    G011 = G01+dt*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*tau))
    k101 = k10+dt*k01
    k011 = k01-dt*(1+2*l1)*k10
    return G101, G011, k101, k011


def rho_update_control_l10(i, Input_Initials): #Optimal control integration with \lambda_1=0
  X, P, H, X2, rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, j, Id = Input_Initials
  AGamma = (G10**2-G01**2-G20+G02)/2.0
  BGamma = G10*G01-G11
  theta = jnp.arctan2(BGamma, AGamma)/2.0
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  r = csth*G10+snth*G01
  l1 = -l1max*jnp.sign(k20)
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]
  #theta = theta_t[j]
  #phi = theta+t
  
  #dphi  = jnp.array([1.0])#+0.1*jnp.tanh(10.0*(dphi -1))
  #dphi = jnp.array([1.0])
  
  
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
  #e_jphi = jnp.exp(-1j*phi)
  #delh_t_Mat = e_jphi*jnp.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
  #ht = jnp.matmul(delh_t_Mat, Initials) + e_jphi*I_t
  #r = ht.real 
  #v = ht.imag
  #wzmat = jnp.array([0,0,1,1j,0,0,0,0,0])
  #wz = jnp.matmul(wzmat, Initials)*e_jphi
  #w = wz.real
  #z = wz.imag
  G101, G011, k101, k011, G201, G111, G021, k201, k111, k021 = G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1)
  '''
  G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
  G011 = G01+dt*(-G10+csth*(csth*k10+snth*k01)/(4*tau))
  k101 = k10+dt*k01
  k011 = k01-dt*k10
  G201 = G20+dt*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*tau))
  G111 = G11+dt*(G02-G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*tau))
  G021 = G02+dt*(-2*G11+csth*(snth*k02+csth*k11-r*k01)/(2*tau))
  k201 = k20+dt*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/tau)
  k111 = k11+dt*(-k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/tau)
  k021 = k02+dt*(-2*k11+2*csth*(-snth*G02-csth*G11+r*G01)/tau)  
  '''
  
  #H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  #Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  #read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  #rho_update =H_update+Lind_update+read_update
  H1 = H+l1*X2
  Fac1 = r*delL/(2*tau)-delV/(4*tau)
  rho1 = jnp.matmul(jnp.matmul(Id+dt*(Fac1-1j*H1),rho),Id+dt*(Fac1+1j*H1))
  tmptr = jnp.trace(rho1)
  rho1 = rho1/tmptr
  #rho1 = rho + rho_update*dt
  Idth1 = Idth+1e0*dt*(r**2-2*r*expL+expV)/(2*tau)
  return (X, P, H, X2, rho1, l1max, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021, Idth1, ts, tau, dt, j+1, Id)

def OPsoln_control_l10_JAX(Initials, X, P, H, X2, rho_i, l1max, ts, dt,  tau, Idmat,  Id):
  #I_tR = jnp.array([0.0])
  G10 = jnp.matmul(Idmat[0], Initials)
  G01 = jnp.matmul(Idmat[1], Initials)
  k10 = jnp.matmul(Idmat[2], Initials)
  k01 = jnp.matmul(Idmat[3], Initials)
  G20 = jnp.matmul(Idmat[4], Initials)
  G11 = jnp.matmul(Idmat[5], Initials)
  G02 = jnp.matmul(Idmat[6], Initials)
  k20 = jnp.matmul(Idmat[7], Initials)
  k11 = jnp.matmul(Idmat[8], Initials)
  k02 = jnp.matmul(Idmat[9], Initials)
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = 0.0
  X, P, H, X2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, rho_update_control_l10,(X, P, H, X2,  rho, l1max, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, Idth,  ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho, Idth

def rho_sigma_update_control_l10(i, Input_Initials): #Optimal control integration with \lambda_1=0
  X, P, X2, P2, CXP, H, rho, sigma,  Idth, ts, tau, dt, j, Id = Input_Initials
  rhosig = jnp.matmul(rho,sigma)
  sigrho = jnp.matmul(sigma, rho)
  Omega = (rhosig+sigrho)/2.0
  #Lambda = rhosig-sigrho
  G10 = jnp.trace(jnp.matmul(X,Omega)).real
  G01 = jnp.trace(jnp.matmul(P,Omega)).real
  G20 = jnp.trace(jnp.matmul(X2, Omega)).real
  G02 = jnp.trace(jnp.matmul(P2, Omega)).real
  G11 = jnp.trace(jnp.matmul(CXP, Omega)).real
  AGamma = (G10**2-G01**2-G20+G02)/2.0
  BGamma = G10*G01-G11
  theta = jnp.arctan2(BGamma, AGamma)/2.0
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  r = csth*G10+snth*G01
  #I_t = I_tR + 1j*I_tI
  #print (tau)
  t = ts[j]

  Ljump = csth*X+snth*P
  Ljump2 = jnp.matmul(Ljump, Ljump)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
  #Mjump = -snth*X+csth*P
  expX = jnp.trace(jnp.matmul(X, rho)).real
  expP = jnp.trace(jnp.matmul(P, rho)).real
  expV = jnp.trace(jnp.matmul(Ljump2, rho)).real
  expL = csth*expX + snth*expP
  delL = Ljump - expL*Id
  delV = Ljump2-expV*Id
  expsigma = jnp.trace(jnp.matmul(sigma, rho))
  
  H_update = -1j*(jnp.matmul(H, rho)-jnp.matmul(rho, H))
  Lind_update = (-jnp.matmul(delV, rho)-jnp.matmul(rho, delV))/(4*tau)
  read_update = r*(jnp.matmul(delL, rho)+ jnp.matmul(rho, delL))*(1.0/(2*tau))
  rho_update =H_update+Lind_update+read_update
  H_sigma_update = -1j*(jnp.matmul(H, sigma)-jnp.matmul(sigma, H))
  Lind_sigma_update = (jnp.matmul(delV, sigma)+jnp.matmul(sigma, delV)-2*(expsigma-1)*Ljump2)/(4*tau)
  read_sigma_update = -r*(jnp.matmul(delL, sigma)+ jnp.matmul(sigma, delL)-2*(expsigma-1)*Ljump)*(1.0/(2*tau))
  rho_update = H_update+Lind_update+read_update
  #print (jnp.trace(rho_update))
  #rho_update1 = rho_update-jnp.trace(rho_update)*rho
  sigma_update = H_sigma_update+Lind_sigma_update+read_sigma_update
  rho1 = rho + rho_update*dt
  sigma1 = sigma+sigma_update*dt
  expsigma = jnp.trace(jnp.matmul(sigma1, rho1))
  sigma1 = sigma1 + (1-expsigma)*Id
  
  Idth1 = Idth+1e1*dt*(r**2-2*r*expL+expV)/(2*tau)
  return (X, P,  X2, P2, CXP, H, rho1, sigma1, Idth1, ts, tau, dt, j+1, Id)

def OPsoln_sigma_control_l10_JAX(sigma0, X, P, X2, P2, CXP, H, rho_i, ts, dt,  tau, Idmat,  Id):
  #I_tR = jnp.array([0.0])
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rho = rho_i
  k1=0
  Idth = 0.0
  
  X, P, X2, P2, CXP, H,  rho, sigma, Idth, ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, rho_sigma_update_control_l10,(X, P, X2, P2, CXP, H,  rho, sigma0, Idth,  ts, tau, dt, k1, Id))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rho, Idth

@jit
def CostF_control_l10(Initials, X, P, H,  rho_i, rho_f, l1max, ts, dt, tau, Idmat, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX(Initials, X, P, H, rho_i, l1max, ts, dt, tau, Idmat, Id)
  #print (Idth)
  return Tr_Distance(rho_f_simul, rho_f)

@jit
def CostF_control2_l10(Initials, X, P, H,  rho_i, rho_f, l1max, ts, dt, tau, Idmat, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX(Initials, X, P, H, rho_i, l1max, ts, dt, tau, Idmat, Id)
  #print (Idth)
  return Idth

@jit
def CostF_control_l101(Initials, X, P, H, X2,  rho_i, rho_f, l1max, ts, dt, tau, Idmat, Id):
  rho_f_simul, Idth = OPsoln_control_l10_JAX(Initials, X, P, H, X2, rho_i, l1max, ts, dt, tau, Idmat, Id)
  return Tr_Distance(rho_f_simul, rho_f), Idth

def CostF_sigma_control_l10(sigma0, X, P, X2, P2, CXP, H,  rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id):
  rho_f_simul, Idth = OPsoln_sigma_control_l10_JAX(sigma0, X, P, X2, P2, CXP, H, rho_i, ts, dt, tau, Idmat, Id)
  #print (Idth)
  return Tr_Distance(rho_f_simul, rho_f)

def CostF_sigma_control2_l10(sigma0, X, P, X2, P2, CXP, H,  rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id):
  rho_f_simul, Idth = OPsoln_sigma_control_l10_JAX(sigma0, X, P, X2, P2, CXP, H, rho_i, ts, dt, tau, Idmat, Id)
  #print (Idth)
  return Idth

def CostF_sigma_control_l101(sigma0, X, P, X2, P2, CXP,  H,  rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id):
  rho_f_simul, Idth = OPsoln_sigma_control_l10_JAX(sigma0, X, P, X2, P2, CXP, H, rho_i,  ts, dt, tau, Idmat, Id)
  return Tr_Distance(rho_f_simul, rho_f), Idth

@jit
def update_control_l10(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id,  step_size):
    grads=grad(CostF_control_l10)(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id)
    #randlist = np.ones(10)
    #randlist[:5] = 0
    #np.random.shuffle(randlist)
    #grads = grads*randlist
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

@jit
def update_control2_l10(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id,  step_size):
    grads=grad(CostF_control2_l10)(Initials, X, P, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id)
    
    return jnp.array([w - step_size * dw
          for w, dw in zip(Initials, grads)])

@jit
def update_sigma_control_l10(sigma0, X, P, X2, P2, CXP, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id,  step_size):
    grads=grad(CostF_sigma_control_l10)(sigma0, X, P, X2, P2, CXP, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id)
    #print (jnp.shape(grads))
    return jnp.array([w - step_size * dw
          for w, dw in zip(sigma0, grads)])

@jit
def update_sigma_control2_l10(sigma0, X, P, X2, P2, CXP, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id,  step_size):
    grads=grad(CostF_sigma_control2_l10)(sigma0, X, P, X2, P2, CXP, H, rho_i, rho_f, theta_t, ts, dt, tau, Idmat, Id)
    return jnp.array([w - step_size * dw
          for w, dw in zip(sigma0, grads)])


def OPintegrate_strat(Initials, X, P, H, X2, rho_i, l1max, ts, dt,  tau, Idmat, Id):
  #I_tR = jnp.array([0.0])
  G10 = np.matmul(Idmat[0], Initials)
  G01 = np.matmul(Idmat[1], Initials)
  k10 = np.matmul(Idmat[2], Initials)
  k01 = np.matmul(Idmat[3], Initials)
  G20 = np.matmul(Idmat[4], Initials)
  G11 = np.matmul(Idmat[5], Initials)
  G02 = np.matmul(Idmat[6], Initials)
  k20 = np.matmul(Idmat[7], Initials)
  k11 = np.matmul(Idmat[8], Initials)
  k02 = np.matmul(Idmat[9], Initials)
  
  rho = rho_i
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  theta_t = np.zeros(len(ts))
  l1_t = np.zeros(len(ts))
  diff = np.zeros(len(ts))
  readout = np.zeros(len(ts))
  
  while (j<len(ts)):
      #print (j,r)
      #Initials, X, P, H, rho, I_t, I_k_t, I_Gp_t, I_G_t,   phi,  ts, tau, dt, j, Id, Q1, Q2, Q3, Q4, Q5 = Input_Initials
      #I_t = I_tR + 1j*I_tI
      #print (tau)
      
      diff[j]=G10
      t = ts[j]
      AGamma = (G10**2-G01**2-G20+G02)/2.0
      BGamma = G10*G01-G11
      theta = np.arctan2(BGamma, AGamma)/2.0
      theta_t[j] = theta
      csth, snth = np.cos(theta), np.sin(theta)
      l1 = -l1max*np.sign(k20)
      l1_t[j] = l1
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = csth*G10+snth*G01
      readout[j] = r

      Ljump = csth*X+snth*P
      Xjump2 = np.matmul(X, X)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
      Ljump2 = np.matmul(Ljump, Ljump)
      #Mjump = -snth*X+csth*P
      Pjump2 = np.matmul(P,P)
      expX = np.trace(np.matmul(X, rho)).real
      expP = np.trace(np.matmul(P, rho)).real
      expV = np.trace(np.matmul(Ljump2, rho)).real
      expL = csth*expX + snth*expP
      #exphi = 
      #expM = -snth*expX + csth*expP
      #print (expL)
      Q1[j]=expX
      Q2[j]=expP
      delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = np.trace(np.matmul(Xjump2,rho)).real-expX**2
      Q5[j] = np.trace(np.matmul(Pjump2,rho)).real-expP**2
      Q4[j] = np.trace(np.matmul(np.matmul(X, P)+np.matmul(P,X),rho)).real/2.0-expX*expP
      delV = Ljump2-expV*Id
      #H_update = -1j*(np.matmul(H, rho)-np.matmul(rho, H))
      #Lind_update = (-np.matmul(delV, rho)-np.matmul(rho, delV))/(4*tau)
      #read_update = r*(np.matmul(delL, rho)+ np.matmul(rho, delL))*(1.0/(2*tau))
      #drho = H_update+Lind_update+read_update
      Fac1 = r*delL/(2*tau)-delV/(4*tau)
      H1 = H+l1*X2
      rho = np.matmul(np.matmul(Id+dt*(Fac1-1j*H1),rho),Id+dt*(Fac1+1j*H1))
      temptr = np.trace(rho)
      rho = rho/temptr
      #print (Lind_update)
      #print (np.trace(drho).real)
      #rho+=(drho)*dt
      G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = G_k_updates(G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, csth, snth, r, dt, tau, l1)
      '''
      G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
      G011 = G01+dt*(-G10+csth*(csth*k10+snth*k01)/(4*tau))
      k101 = k10+dt*k01
      k011 = k01-dt*k10
      G201 = G20+dt*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*tau))
      G111 = G11+dt*(G02-G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*tau))
      G021 = G02+dt*(-2*G11+csth*(snth*k02+csth*k11-r*k01)/(2*tau))
      k201 = k20+dt*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/tau)
      k111 = k11+dt*(-k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/tau)
      k021 = k02+dt*(-2*k11+2*csth*(-snth*G02-csth*G11+r*G01)/tau)  
      G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = G101, G011, k101, k011, G201, G111, G021, k201, k111, k021
      '''
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, theta_t, l1_t, rho, readout, diff

def OP_stochastic_trajectory(X, P, H, X2, rho_i, l1_t, theta_t, ts, dt,  tau, Id):
  rho = rho_i
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  #theta_t = np.zeros(len(ts))
  #l1_t = np.zeros(len(ts))
  readout = np.zeros(len(ts))
  Xjump2 = np.matmul(X, X)
  Pjump2 = np.matmul(P, P)
  Cxp2 = np.matmul(X, P)+np.matmul(P,X)
  while (j<len(ts)):
      t = ts[j]
      theta = theta_t[j]
      l1 = l1_t[j]
      H1 = H+l1*Xjump2
      csth, snth = np.cos(theta), np.sin(theta)
      expX = np.trace(np.matmul(X, rho)).real
      expP = np.trace(np.matmul(P, rho)).real
      Ljump = csth*X+snth*P
      Ljump2 = np.matmul(Ljump,Ljump)
      Q1[j] = expX
      Q2[j] = expP
      Q3[j] = np.trace(np.matmul(Xjump2,rho)).real-expX**2
      Q5[j] = np.trace(np.matmul(Pjump2,rho)).real-expP**2
      Q4[j] = np.trace(np.matmul(Cxp2,rho)).real/2.0-expX*expP
      expL = csth*expX + snth*expP
      dW = np.random.normal(scale=np.sqrt(dt))
      #print (dW/np.sqrt(tau))
      delL = Ljump - expL*Id
      F = -np.matmul(delL, delL)*dt/(4*tau)+delL*dW/(2*np.sqrt(tau))
      F1 = Id+F+np.matmul(F,F)/2
      F2 = np.matmul(F1,(Id-1j*H1*dt))
      F3 = np.matmul((Id+1j*H1*dt),F1)
      #rho1 = np.matmul(np.matmul(F2,rho),F3)
      rho1 = rho+dt*(-1j*np.matmul(H1,rho)+1j*np.matmul(rho,H1)+(np.matmul(np.matmul(Ljump,rho),Ljump)-np.matmul(Ljump2, rho)/2.0-np.matmul(rho,Ljump2)/2.0)/(4*tau))+dW*(np.matmul(delL,rho)+np.matmul(rho,delL))/np.sqrt(4*tau)
      rho = rho1/np.trace(rho1)
      #print (np.trace(rho))
      readout[j] = expL+np.sqrt(tau)*dW/dt

      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, rho, readout

def OP_wcontrol(Initials, X, P, H, X2, rho_i, l1_t, theta_t, ts, dt,  tau, Idmat, Id):
  #I_tR = jnp.array([0.0])
  G10 = np.matmul(Idmat[0], Initials)
  G01 = np.matmul(Idmat[1], Initials)
  k10 = np.matmul(Idmat[2], Initials)
  k01 = np.matmul(Idmat[3], Initials)
  
  rho = rho_i
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  readout = np.zeros(len(ts))
  
  while (j<len(ts)):
      t = ts[j]
      theta = theta_t[j]
      csth, snth = np.cos(theta), np.sin(theta)
      l1 = l1_t[j]
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = csth*G10+snth*G01
      readout[j] = r

      Ljump = csth*X+snth*P
      Xjump2 = np.matmul(X, X)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
      Ljump2 = np.matmul(Ljump, Ljump)
      #Mjump = -snth*X+csth*P
      Pjump2 = np.matmul(P,P)
      expX = np.trace(np.matmul(X, rho)).real
      expP = np.trace(np.matmul(P, rho)).real
      expV = np.trace(np.matmul(Ljump2, rho)).real
      expL = csth*expX + snth*expP
      #exphi = 
      #expM = -snth*expX + csth*expP
      #print (expL)
      Q1[j]=expX
      Q2[j]=expP
      delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = np.trace(np.matmul(Xjump2,rho)).real-expX**2
      Q5[j] = np.trace(np.matmul(Pjump2,rho)).real-expP**2
      Q4[j] = np.trace(np.matmul(np.matmul(X, P)+np.matmul(P,X),rho)).real/2.0-expX*expP
      delV = Ljump2-expV*Id
      #H_update = -1j*(np.matmul(H, rho)-np.matmul(rho, H))
      #Lind_update = (-np.matmul(delV, rho)-np.matmul(rho, delV))/(4*tau)
      #read_update = r*(np.matmul(delL, rho)+ np.matmul(rho, delL))*(1.0/(2*tau))
      #drho = H_update+Lind_update+read_update
      Fac1 = r*delL/(2*tau)-delV/(4*tau)
      H1 = H+l1*X2
      rho = np.matmul(np.matmul(Id+dt*(Fac1-1j*H1),rho),Id+dt*(Fac1+1j*H1))
      temptr = np.trace(rho)
      rho = rho/temptr
      #print (Lind_update)
      #print (np.trace(drho).real)
      #rho+=(drho)*dt
      G10, G01, k10, k01 = G_k_updates_first_order(G10, G01, k10, k01,  csth, snth,  dt, tau, l1)
      '''
      G101 = G10+dt*(G01-snth*(csth*k10+snth*k01)/(4*tau))
      G011 = G01+dt*(-G10+csth*(csth*k10+snth*k01)/(4*tau))
      k101 = k10+dt*k01
      k011 = k01-dt*k10
      G201 = G20+dt*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*tau))
      G111 = G11+dt*(G02-G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*tau))
      G021 = G02+dt*(-2*G11+csth*(snth*k02+csth*k11-r*k01)/(2*tau))
      k201 = k20+dt*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/tau)
      k111 = k11+dt*(-k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/tau)
      k021 = k02+dt*(-2*k11+2*csth*(-snth*G02-csth*G11+r*G01)/tau)  
      G10, G01, k10, k01, G20, G11, G02, k20, k11, k02 = G101, G011, k101, k011, G201, G111, G021, k201, k111, k021
      '''
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, rho, readout

def OP_trajectory_JAX(i, Input_Initials):
    X, P, H, X2,  rho, l1_t, theta_t, dWt, Idth,  ts, tau, dt, j, Id = Input_Initials
    l1 = l1_t[j]
    theta = theta_t[j]
    H1 = H+l1*X2
    csth, snth = jnp.cos(theta), jnp.sin(theta)
    expX = jnp.trace(jnp.matmul(X, rho)).real
    expP = jnp.trace(jnp.matmul(P, rho)).real
    Ljump = csth*X+snth*P
    Ljump2 = jnp.matmul(Ljump,Ljump)
    expL = csth*expX + snth*expP
    dW = dWt[j]
    delL = Ljump - expL*Id
    F = -jnp.matmul(delL, delL)*dt/(4*tau)+delL*dW/(2*jnp.sqrt(tau))
    F1 = Id+F+jnp.matmul(F,F)/2
    F2 = jnp.matmul(F1,(Id-1j*H1*dt))
    F3 = jnp.matmul((Id+1j*H1*dt),F1)
    rho1 = jnp.matmul(jnp.matmul(F2,rho),F3)
    rho1 = rho1/jnp.trace(rho1)
    #rho1 = rho+dt*(-1j*jnp.matmul(H1,rho)+1j*jnp.matmul(rho,H1)+(jnp.matmul(jnp.matmul(Ljump,rho),Ljump)-jnp.matmul(Ljump2, rho)/2.0-jnp.matmul(rho,Ljump2)/2.0)/(4*tau))+dW*(jnp.matmul(delL,rho)+jnp.matmul(rho,delL))/jnp.sqrt(4*tau)
    return X, P, H, X2,  rho1, l1_t, theta_t, dWt, Idth,  ts, tau, dt, j+1, Id

@jit    
def OP_stochastic_trajectory_JAX(X, P, H, X2, rho_i, l1_t, theta_t, dWt, ts, dt,  tau, Id):
  #rho = rho_i
  rho = rho_i
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = 0.0
  X, P, H, X2,  rho, l1_t, theta_t, dWt, Idth,  ts, tau, dt, k1, Id = jax.lax.fori_loop(0, len(ts)-1, OP_trajectory_JAX,(X, P, H, X2,  rho, l1_t, theta_t, dWt, Idth,  ts, tau, dt, k1, Id))
  return rho


def OPintegrate_sigma_strat(sigma0, X, P, X2, P2, CXP, H, rho_i, ts, dt,  tau, Idmat, Id):
  rho = rho_i
  sigma = sigma0
  j=0
  Q1 = np.zeros(len(ts))
  Q2 = np.zeros(len(ts))
  Q3 = np.zeros(len(ts))
  Q4 = np.zeros(len(ts))
  Q5 = np.zeros(len(ts))
  theta_t = np.zeros(len(ts))
  diff = np.zeros(len(ts))
  readout = np.zeros(len(ts))
  
  while (j<len(ts)):
      #print (j,r)
      #Initials, X, P, H, rho, I_t, I_k_t, I_Gp_t, I_G_t,   phi,  ts, tau, dt, j, Id, Q1, Q2, Q3, Q4, Q5 = Input_Initials
      #I_t = I_tR + 1j*I_tI
      #print (tau)
      rhosig = np.matmul(rho,sigma)
      sigrho = np.matmul(sigma, rho)
      Omega = (rhosig+sigrho)/2.0
      #Lambda = rhosig-sigrho
      G10 = np.trace(np.matmul(X,Omega)).real
      G01 = np.trace(np.matmul(P,Omega)).real
      G20 = np.trace(np.matmul(X2, Omega)).real
      G02 = np.trace(np.matmul(P2, Omega)).real
      G11 = np.trace(np.matmul(CXP, Omega)).real
      AGamma = (G10**2-G01**2-G20+G02)/2.0
      BGamma = G10*G01-G11
      theta = jnp.arctan2(BGamma, AGamma)/2.0
      csth, snth = jnp.cos(theta), jnp.sin(theta)
      diff[j]=G10
      t = ts[j]
      theta_t[j] = theta
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = csth*G10+snth*G01
      readout[j] = r

      Ljump = csth*X+snth*P
      Xjump2 = np.matmul(X, X)#X2*csth**2 + P2*snth**2 + (XP + PX)*csth*snth
      Ljump2 = np.matmul(Ljump, Ljump)
      #Mjump = -snth*X+csth*P
      Pjump2 = np.matmul(P,P)
      expX = np.trace(np.matmul(X, rho)).real
      expP = np.trace(np.matmul(P, rho)).real
      expV = np.trace(np.matmul(Ljump2, rho)).real
      expL = csth*expX + snth*expP
      #exphi = 
      #expM = -snth*expX + csth*expP
      #print (expL)
      Q1[j]=expX
      Q2[j]=expP
      delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = np.trace(np.matmul(Xjump2,rho)).real-expX**2
      Q5[j] = np.trace(np.matmul(Pjump2,rho)).real-expP**2
      Q4[j] = np.trace(np.matmul(np.matmul(X, P)+np.matmul(P,X),rho)).real/2.0-expX*expP
      delV = Ljump2-expV*Id
      H_update = -1j*(np.matmul(H, rho)-np.matmul(rho, H))
      Lind_update = (-np.matmul(delV, rho)-np.matmul(rho, delV))/(4*tau)
      read_update = r*(np.matmul(delL, rho)+ np.matmul(rho, delL))*(1.0/(2*tau))
      H_sigma_update = -1j*(np.matmul(H, sigma)-np.matmul(sigma, H))
      Lind_sigma_update = (np.matmul(delV, sigma)+np.matmul(sigma, delV))/(4*tau)
      read_sigma_update = -r*(np.matmul(delL, sigma)+ np.matmul(sigma, delL))*(1.0/(2*tau))
      drho = H_update+Lind_update+read_update
      dsigma = H_sigma_update+Lind_sigma_update+read_sigma_update
      #print (Lind_update)
      #print (np.trace(drho).real)
      rho+=drho*dt
      sigma+=dsigma*dt
      expsigma = np.trace(np.matmul(sigma, rho))
      sigma = sigma + (1-expsigma)*Id
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1,Q2,Q3,Q4,Q5, theta_t, rho, readout, diff