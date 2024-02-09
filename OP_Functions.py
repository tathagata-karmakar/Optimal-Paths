#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:16:27 2024

@author: t_karmakar
"""

import os
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
from scipy.integrate import simps as intg
from google.colab import files
from google.colab import drive
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
'''
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

def OPsoln_SHO(Ljump, Mjump, H, rho_i, alr, ali, A, B, ts,  tau, varReturn = 0):
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
    ht = np.exp(-1j*t)*(alr+1j*ali+1j*t*(A+1j*B)/(8*tau))+np.exp(-1j*t)*I_t
    u =  ht.real
    rho1 = rho+(-1j*commutator(H,rho)+(Ljump*rho*Ljump-Ljump*Ljump*rho/2.0-rho*Ljump*Ljump/2.0)/(4*tau)+u*commutator(delL, rho,kind='anti')/(2*tau))*dt
    rho = rho1#/rho1.tr()
    I_t+= ((ft+1j*gt)*np.exp(1j*t)+1j*(A-1j*B)*np.exp(1j*2*t)/(8*tau))*dt
    t = t+dt
    i+=1
  if (varReturn == 1):
    return rho, expLs, expMs, varLs, covLMs, varMs
  else:
    return rho, expLs, expMs

def OP_PRXQ_Params(Ljump, Mjump, rho_i, rho_f, ts, tau):
  t_f = ts[-1]
  q1i = expect(Ljump,rho_i)
  q2i = expect(Mjump,rho_i)
  q1f = expect(Ljump,rho_f)
  q2f = expect(Mjump,rho_f)
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


