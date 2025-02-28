#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:14:39 2025

@author: tatha_k
"""

import os,sys
os.environ['JAX_PLATFORMS'] = 'cpu'
#os.environ['JAX_DISABLE_JIT'] = '1'

os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
os.environ['CHECKPOINT_PATH']='${path_to_checkpoints}'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
'''
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)
'''

import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
import h5py
from scipy.integrate import simpson as intg
#from google.colab import files
#from google.colab import drive
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
#from Initialization import *
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
'''
All the operators and states are split into their real and imaginary parts
'''
'''
Matrix multiplication functions
'''
def CompMultR(Ar, Ai, Br, Bi): 
    return Ar @ Br - Ai @ Bi#jnp.matmul(Ar, Br)-jnp.matmul(Ai, Bi)
def CompMultI(Ar, Ai, Br, Bi):
    return Ar @ Bi + Ai @ Br#jnp.matmul(Ar, Bi)+jnp.matmul(Ai, Br)
def CompMult(Ar, Ai, Br, Bi):
    return CompMultR(Ar, Ai, Br, Bi), CompMultI(Ar, Ai, Br, Bi)

def ExpVal(Xr, Xi, rhor, rhoi): #Expectation value of a hermitian operator wrt state rho
    return jnp.trace(CompMultR(Xr, Xi, rhor, rhoi))

def Moment_calc(Ops, rhor, rhoi):
    expX = ExpVal(Ops[0], Ops[1], rhor, rhoi)#.item()
    expP = ExpVal(Ops[2], Ops[3], rhor, rhoi)#.item()
    expX2 = ExpVal(Ops[6], Ops[7], rhor, rhoi)#.item()
    expP2 = ExpVal(Ops[10], Ops[11], rhor, rhoi)#.item()
    expCXP = ExpVal(Ops[8], Ops[9], rhor, rhoi)#.item()
    return expX, expP, expX2, expCXP, expP2

'''
Fidelity calculation
'''
def Fidelity_PS(rho_f_simulr, rho_f_simuli, rhotr, rhoti):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  tmpr  = CompMultR(rho_f_simulr, rho_f_simuli, rhotr, rhoti)
  fid = jnp.trace(tmpr)
  return fid

'''
Negative fidelity, to be used as cost function elsewhere
'''
def Tr_Distance(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi):
  
  return -Fidelity_PS(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi)#+1e2*dist

'''
params = (l1max, ts, dt, tau, Idmat)
Idmat is 10x10 identity matrix 
'''

'''
Update (differential) of first order Gamma terms (see article)
'''
def G_k_updates_first_order(G1s, cs, s2, c2, l1u, params):
    Atmp = params[2]*jnp.array([[0, 1.0, -cs/4.0, -s2/4.0], [l1u, 0, c2/4, cs/4], [0, 0, 0, 1.0], [0, 0, l1u, 0.0] ])
    
    return Atmp @ G1s

'''
Update (differential) of the first and second order Gamma terms
'''
def Del_G_k_updates(G1s, G2s, r,  csth, snth, l1,  params):
    c1 = csth/params[3]
    s1 = snth/params[3]
    cs = snth*csth/params[3]
    s2 = snth**2/params[3]
    c2 = csth**2/params[3]
    l1u = -(1+2*l1)
    G1s1 = G_k_updates_first_order(G1s, cs, s2, c2, l1u, params)
    #G101 = params[2]*(G01-snth*(csth*k10+snth*k01)/(4*params[3]))
    #G011 = params[2]*(-(1+2*l1)*G10+csth*(csth*k10+snth*k01)/(4*params[3]))
    #k101 = params[2]*k01
    #k011 = -params[2]*(1+2*l1)*k10
    #r = jnp.array([csth, snth, 0, 0]) @ G1s
    Btmp = jnp.array([[0, 2, 0, -cs/2, -s2/2, 0 ], [l1u, 0, 1, c2/4, 0, -s2/4], [0, 2*l1u, 0, 0, c2/2, cs/2], [2*cs, 2*s2, 0, 0, 2, 0], [-c2, 0, s2, l1u, 0, 1], [0, -2*c2, -2*cs, 0, 2*l1u, 0]])
    Ctmp = jnp.array([[0, 0, s1/2, 0], [0, 0, -c1/4, s1/4], [0, 0, 0, -c1/2.0], [-2*s1, 0, 0, 0], [c1, -s1, 0, 0], [0, 2*c1, 0, 0]])
    G2s1 = params[2]*(Btmp @ G2s + r*Ctmp @ G1s)
    #G201 = +params[2]*(2*G11+snth*(r*k10-csth*k20-snth*k11)/(2*params[3]))
    #G111 = +params[2]*(G02-(1+2*l1)*G20+(r*(snth*k01-csth*k10)+(csth**2*k20-snth**2*k02))/(4*params[3]))
    #G021 = +params[2]*(-2*(1+2*l1)*G11+csth*(snth*k02+csth*k11-r*k01)/(2*params[3]))
    #k201 = +params[2]*(2*k11+2*snth*(csth*G20+snth*G11-r*G10)/params[3])
    #k111 = +params[2]*(-(1+2*l1)*k20+k02+(r*(csth*G10-snth*G01)-csth**2*G20+snth**2*G02)/params[3])
    #k021 = +params[2]*(-2*(1+2*l1)*k11+2*csth*(-snth*G02-csth*G11+r*G01)/params[3])  
    return G1s1, G2s1

'''
Total of the first and second order Gamma terms
'''
def G_k_updates(G1s, G2s, csth, snth, l1, r, params):
    G1s1, G2s1 = Del_G_k_updates(G1s, G2s, r, csth, snth, l1, params)
    return G1s+G1s1, G2s+G2s1

'''
State update for given Kraus operator
'''
def rho_kraus_update(rhor, rhoi, Fr, Fi):
    tmp1r, tmp1i = CompMult(rhor, rhoi, Fr.T, -Fi.T)
    rho1r, rho1i = CompMult(Fr, Fi, tmp1r, tmp1i)
    Nr = jnp.trace(rho1r) 
    Ni = jnp.trace(rho1i)
    rho2r = (Nr*rho1r+Ni*rho1i)/(Nr**2+Ni**2) #Normalization
    rho2i = (Nr*rho1i-Ni*rho1r)/(Nr**2+Ni**2) #Normalization
    return rho2r, rho2i

'''
Calculation of optimal theta and lambda1, given the first and second order Gamma parameters
'''
def Optimal_theta_l1(G1s, G2s, params):
    G10 = jnp.array([1.0, 0, 0, 0,]) @ G1s
    G01 = jnp.array([0.0, 1.0, 0, 0,]) @ G1s
    G11 = jnp.array([0.0, 1, 0, 0, 0, 0]) @ G2s
    G20 = jnp.array([1.0, 0, 0, 0, 0, 0]) @ G2s
    G02 = jnp.array([0.0, 0, 1.0, 0, 0, 0]) @ G2s
    k20 = jnp.array([0.0, 0, 0.0, 1.0, 0, 0]) @ G2s
    AGamma = (G10**2-G01**2-G20+G02)/2.0
    BGamma = G10*G01-G11
    theta = jnp.arctan2(BGamma, AGamma)/2.0
    #csth, snth = jnp.cos(theta), jnp.sin(theta)
    #r = csth*G10+snth*G01
    l1 = -params[0]*jnp.sign(k20)
    return theta, l1

'''
Straonovich measurement Kraus operator
'''
def M_step(Ops, r, csth, snth,  l1, params): #Optimal control integration with \lambda_1=0
  #csth, snth = jnp.cos(theta), jnp.sin(theta)
  #r = jnp.array([csth, snth, 0, 0]) @ G1s
  #G1s1, G2s1 = Del_G_k_updates(G1s, G2s, csth, snth, l1, params)
  H1r = Ops[4]+l1*Ops[6]
  H1i = Ops[5]+l1*Ops[7]
  #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
  Fac2r = -params[2]*(Ops[6]*csth**2+csth*snth*Ops[8]+Ops[10]*snth**2)/(4*params[3])+params[2]*r*(csth*Ops[0]+snth*Ops[2])/(2*params[3])+params[2]*H1i
  Fac2i = -params[2]*(Ops[7]*csth**2+csth*snth*Ops[9]+Ops[11]*snth**2)/(4*params[3])+params[2]*r*(csth*Ops[1]+snth*Ops[3])/(2*params[3])-params[2]*H1r
  return Fac2r, Fac2i#, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021

'''
Ito measurement Kraus operator
'''
def M_stochastic_step(Ops, r, csth, snth,  l1, params): #Optimal control integration with \lambda_1=0
  #csth, snth = jnp.cos(theta), jnp.sin(theta)
  #r = jnp.array([csth, snth, 0, 0]) @ G1s
  #G1s1, G2s1 = Del_G_k_updates(G1s, G2s, csth, snth, l1, params)
  H1r = Ops[4]+l1*Ops[6]
  H1i = Ops[5]+l1*Ops[7]
  #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
  Fac2r = -params[2]*(Ops[6]*csth**2+csth*snth*Ops[8]+Ops[10]*snth**2)/(8*params[3])+params[2]*r*(csth*Ops[0]+snth*Ops[2])/(2*params[3])+params[2]*H1i
  Fac2i = -params[2]*(Ops[7]*csth**2+csth*snth*Ops[9]+Ops[11]*snth**2)/(8*params[3])+params[2]*r*(csth*Ops[1]+snth*Ops[3])/(2*params[3])-params[2]*H1r
  return Fac2r, Fac2i#, G101, G011, k101, k011, G201, G111, G021, k201, k111, k021

def RK4_delyn(k1, k2, k3, k4):
    return k1/6.0+k2/3.0+k3/3.0+k4/6.0

def integrator_step(G1s, G2s, Ops, params):
    #expX, expP, expX2, expCXP, expP2 = Moment_calc(Ops, rhor, rhoi)
    theta, l1 = Optimal_theta_l1(G1s, G2s, params)
    csth, snth = jnp.cos(theta), jnp.sin(theta)
    r = jnp.array([csth, snth, 0, 0]) @ G1s
    Fr, Fi  = M_step(Ops, r, csth, snth, l1, params)
    dG1s, dG2s = Del_G_k_updates(G1s, G2s, r,  csth, snth, l1, params)
    #dq0 = params[2]*(r**2 - 2*r*(csth*expX+snth*expP)+(csth**2*expX2+csth*snth*expCXP+snth**2*expP2))/(2*params[3])
    return Fr, Fi, dG1s, dG2s
    

@jit
def RK4_step(Ops, rhor, rhoi, G1s, G2s, q0, params): 
  #= Input_Initials
  theta, l1 = Optimal_theta_l1(G1s, G2s, params)
  csth, snth = jnp.cos(theta), jnp.sin(theta)
  r = jnp.array([csth, snth, 0, 0]) @ G1s
  expX, expP, expX2, expCXP, expP2 = Moment_calc(Ops, rhor, rhoi)
  Fk1r, Fk1i, G1k1, G2k1 = integrator_step(G1s, G2s, Ops, params)
  Fk2r, Fk2i, G1k2, G2k2 = integrator_step(G1s+G1k1/2.0, G2s+G2k1/2.0, Ops,   params)
  Fk3r, Fk3i, G1k3, G2k3 = integrator_step(G1s+G1k2/2.0, G2s+G2k2/2.0, Ops,  params)
  Fk4r, Fk4i, G1k4, G2k4 = integrator_step(G1s+G1k3, G2s+G2k3, Ops,  params)
  G1s1 = G1s+RK4_delyn(G1k1, G1k2, G1k3, G1k4)
  G2s1 = G2s+RK4_delyn(G2k1, G2k2, G2k3, G2k4)
  Fr = Ops[12]+RK4_delyn(Fk1r, Fk2r, Fk3r, Fk4r)
  Fi = RK4_delyn(Fk1i, Fk2i, Fk3i, Fk4i)
  dq0 = params[2]*(r**2 - 2*r*(csth*expX+snth*expP)+(csth**2*expX2+csth*snth*expCXP+snth**2*expP2))/(2*params[3])
  '''
  #theta, l1 = Optimal_theta_l1(G1s, G2s, params)
  
  #csth, snth = jnp.cos(theta), jnp.sin(theta)
  #r = jnp.array([csth, snth, 0, 0]) @ G1s
  #Fk1r, Fk1i, G10k1, G01k1, k10k1, k01k1, G20k1, G11k1, G02k1, k20k1, k11k1, k02k1 = integrator_step(Ops, G10, G01, k10, k01, G20, G11, G02, k20, k11, k02, theta, l1, params)
  #Fk1r, Fk1i  = M_step(Ops, r, csth, snth, l1, params)
  #G1k1, G2k1 = Del_G_k_updates(G1s, G2s, csth, snth, l1, params)
  #rhok1r = rhor+(rho1r-rhor)/2.0
  #rhok1i = rhoi+(rho1i-rhoi)/2.0
  theta_tmp, l1tmp = Optimal_theta_l1(G10+G10k1/2.0, G01+G01k1/2.0, G20+G20k1/2.0, G11+G11k1/2.0, G02+G02k1/2.0, k20+k20k1/2.0, params)
  Fk2r, Fk2i, G10k2, G01k2, k10k2, k01k2, G20k2, G11k2, G02k2, k20k2, k11k2, k02k2 = integrator_step(Ops, G10+G10k1/2.0, G01+G01k1/2.0, k10+k10k1/2.0, k01+k01k1/2.0, G20+G20k1/2.0, G11+G11k1/2.0, G02+G02k1/2.0, k20+k20k1/2.0, k11+k11k1/2.0, k02+k02k1/2.0, theta_tmp, l1tmp, params)
  theta_tmp, l1tmp = Optimal_theta_l1(G10+G10k2/2.0, G01+G01k2/2.0, G20+G20k2/2.0, G11+G11k2/2.0, G02+G02k2/2.0, k20+k20k2/2.0, params)
  Fk3r, Fk3i, G10k3, G01k3, k10k3, k01k3, G20k3, G11k3, G02k3, k20k3, k11k3, k02k3 = integrator_step(Ops, G10+G10k2/2.0, G01+G01k2/2.0, k10+k10k2/2.0, k01+k01k2/2.0, G20+G20k2/2.0, G11+G11k2/2.0, G02+G02k2/2.0, k20+k20k2/2.0, k11+k11k2/2.0, k02+k02k2/2.0, theta_tmp, l1tmp, params)
  theta_tmp, l1tmp = Optimal_theta_l1(G10+G10k3, G01+G01k3, G20+G20k3, G11+G11k3, G02+G02k3, k20+k20k3, params)
  Fk4r, Fk4i, G10k4, G01k4, k10k4, k01k4, G20k4, G11k4, G02k4, k20k4, k11k4, k02k4 = integrator_step(Ops, G10+G10k3, G01+G01k3, k10+k10k3, k01+k01k3, G20+G20k3, G11+G11k3, G02+G02k3, k20+k20k3, k11+k11k3, k02+k02k3, theta_tmp, l1tmp, params)
  #rho1r = rhor+rhok1r/6.0+rhok2r/3.0+rhok3r/3.0+rhok4r/6.0
  #rho1i = rhoi+rhok1i/6.0+rhok2i/3.0+rhok3i/3.0+rhok4i/6.0
  G101 = G10+RK4_delyn(G10k1, G10k2, G10k3, G10k4)
  G011 = G01+RK4_delyn(G01k1, G01k2, G01k3, G01k4)
  k101 = k10+RK4_delyn(k10k1, k10k2, k10k3, k10k4)
  k011 = k01+RK4_delyn(k01k1, k01k2, k01k3, k01k4)
  G201 = G20+RK4_delyn(G20k1, G20k2, G20k3, G20k4)
  G111 = G11+RK4_delyn(G11k1, G11k2, G11k3, G11k4)
  G021 = G02+RK4_delyn(G02k1, G02k2, G02k3, G02k4)
  k201 = k20+RK4_delyn(k20k1, k20k2, k20k3, k20k4)
  k111 = k11+RK4_delyn(k11k1, k11k2, k11k3, k11k4)
  k021 = k02+RK4_delyn(k02k1, k02k2, k02k3, k02k4)
  Fr = Ops[12]+RK4_delyn(Fk1r, Fk2r, Fk3r, Fk4r)
  Fi = RK4_delyn(Fk1i, Fk2i, Fk3i, Fk4i)
  '''
  rho1r, rho1i = rho_kraus_update(rhor, rhoi, Fr, Fi)
  
  #Idth1 = Idth
  return  rho1r, rho1i, G1s1, G2s1, q0+dq0

def OC_plot_step(G1s, Ops, csth, snth, l1, params):
    #theta, l1 = Optimal_theta_l1(G1s, G2s, params)
    #c1 = csth/params[3]
    #s1 = snth/params[3]
    cs = snth*csth/params[3]
    s2 = snth**2/params[3]
    c2 = csth**2/params[3]
    l1u = -(1+2*l1)
    
    r = jnp.array([csth, snth, 0, 0]) @ G1s
    Fr, Fi  = M_step(Ops, r, csth, snth, l1, params)
    dG1s = G_k_updates_first_order(G1s, cs, s2, c2, l1u, params)
    return Fr, Fi, dG1s

@jit
def RK4_wcontrol(Ops, rhor, rhoi, G1s, csth, snth, l1, csthi, snthi, l1i, csth1, snth1, l11, params):
    Fk1r, Fk1i, G1k1 = OC_plot_step(G1s, Ops, csth, snth, l1, params)
    Fk2r, Fk2i, G1k2 = OC_plot_step(G1s+G1k1/2.0, Ops, csthi, snthi, l1i, params)
    Fk3r, Fk3i, G1k3 = OC_plot_step(G1s+G1k2/2.0, Ops,  csthi, snthi, l1i, params)
    Fk4r, Fk4i, G1k4 = OC_plot_step(G1s+G1k3, Ops, csth1, snth1, l11, params)
    G1s1 = G1s+RK4_delyn(G1k1, G1k2, G1k3, G1k4)
    #G2s1 = G2s+RK4_delyn(G2k1, G2k2, G2k3, G2k4)
    Fr = Ops[12]+RK4_delyn(Fk1r, Fk2r, Fk3r, Fk4r)
    Fi = RK4_delyn(Fk1i, Fk2i, Fk3i, Fk4i)
    rho1r, rho1i = rho_kraus_update(rhor, rhoi, Fr, Fi)
    
    #Idth1 = Idth
    return  rho1r, rho1i, G1s1


@jit
def Euler_wcontrol(Ops, rhor, rhoi, G1s, csth, snth, l1, params):
    Fk1r, Fk1i, G1k1 = OC_plot_step(G1s,csth, snth, l1, params)
    #Fk2r, Fk2i, G1k2 = OC_plot_step(G1s+G1k1/2.0, Ops, csthi, snthi, l1i, params)
    #Fk3r, Fk3i, G1k3 = OC_plot_step(G1s+G1k2/2.0, Ops, csthi, snthi, l1i, params)
    #Fk4r, Fk4i, G1k4 = OC_plot_step(G1s+G1k3, Ops, csth1, snth1, l11, params)
    G1s1 = G1s+G1k1
    #G2s1 = G2s+RK4_delyn(G2k1, G2k2, G2k3, G2k4)
    Fr = Ops[12]+Fk1r
    Fi = Fk1i
    rho1r, rho1i = rho_kraus_update(rhor, rhoi, Fr, Fi)
    
    #Idth1 = Idth
    return  rho1r, rho1i, G1s1

def RK4_stepJAX(i, Input_Initials): #Optimal control integration with \lambda_1=0
  Ops, rhor, rhoi, G1s, G2s, params, j, Idth = Input_Initials
  #Idth1 = Idth
  
  rho1r, rho1i, G1s1, G2s1, Idth1 = RK4_step(Ops, rhor, rhoi, G1s, G2s, Idth, params)
  return (Ops, rho1r, rho1i, G1s1, G2s1, params, j+1, Idth1)


def OPsoln_control_l10_JAX(Initials, Ops, rho_ir, rho_ii,  params):
  #I_tR = jnp.array([0.0])
  #G10 = jnp.matmul(params[4][0], Initials)
  #G01 = jnp.matmul(params[4][1], Initials)
  #k10 = jnp.matmul(params[4][2], Initials)
  #k01 = jnp.matmul(params[4][3], Initials)
  #G20 = jnp.matmul(params[4][4], Initials)
  #G11 = jnp.matmul(params[4][5], Initials)
  #G02 = jnp.matmul(params[4][6], Initials)
  #k20 = jnp.matmul(params[4][7], Initials)
  #k11 = jnp.matmul(params[4][8], Initials)
  #k02 = jnp.matmul(params[4][9], Initials)
  G1s = params[4][0:4] @ Initials #First order $\Gamma$
  G2s = params[4][4:] @ Initials #Second order $\Gamma$
  
  #print (GLL)
  #GMM = jnp.matmul(jnp.array([0,0,0,0,0,0,0,1,1.0]),Initials)+jnp.array([0])
  rhor = rho_ir
  rhoi = rho_ii
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = jnp.array(0)
  Ops, rhor, rhoi, G1s, G2s, params, k1, Idth = jax.lax.fori_loop(0, len(params[1])-1, RK4_stepJAX,(Ops, rho_ir, rho_ii, G1s, G2s, params, k1, Idth))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rhor, rhoi, Idth

@jit
def CostF_control_l101(Initials, Ops, rho_ir, rho_ii, rho_fr, rho_fi,  params):
  rho_f_simulr, rho_f_simuli, Idth = OPsoln_control_l10_JAX(Initials, Ops, rho_ir, rho_ii, params)
  return Tr_Distance(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi), Idth, rho_f_simulr, rho_f_simuli


def OPintegrate_strat(Initials, Ops, rho_ir, rho_ii, params):
  #I_tR = jnp.array([0.0])
  '''
  G10 = params[4][0] @ Initials
  G01 = params[4][1] @ Initials
  k10 = params[4][2] @ Initials
  k01 = params[4][3] @ Initials
  G20 = params[4][4] @ Initials
  G11 = params[4][5] @ Initials
  G02 = params[4][6] @ Initials
  k20 = params[4][7] @ Initials
  k11 = params[4][8] @ Initials
  k02 = params[4][9] @ Initials
  '''
  G1s = params[4][0:4] @ Initials
  G2s = params[4][4:] @ Initials
  rhor = rho_ir
  rhoi = rho_ii
  j=0
  npoints = len(params[1])
  Q1 = np.zeros(npoints)
  Q2 = np.zeros(npoints)
  Q3 = np.zeros(npoints)
  Q4 = np.zeros(npoints)
  Q5 = np.zeros(npoints)
  theta_t = np.zeros(npoints)
  l1_t = np.zeros(npoints)
  diff = np.zeros(npoints)
  readout = np.zeros(npoints)
  #k1 = 0
  Idth = 0
  while (j<npoints):
      
      #print (j,r)
      #Initials, X, P, H, rho, I_t, I_k_t, I_Gp_t, I_G_t,   phi,  ts, tau, dt, j, Id, Q1, Q2, Q3, Q4, Q5 = Input_Initials
      #I_t = I_tR + 1j*I_tI
      #print (tau)
      
      diff[j]=G1s[0]
      t = params[1][j]
      #print (jnp.shape(params[0]), jnp.shape(params[1]))
      expX = ExpVal(Ops[0], Ops[1], rhor, rhoi).item()
      expP = ExpVal(Ops[2], Ops[3], rhor, rhoi).item()
      Q1[j]=expX
      Q2[j]=expP
      #delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = ExpVal(Ops[6], Ops[7], rhor, rhoi).item()-expX**2
      Q5[j] = ExpVal(Ops[10], Ops[11], rhor, rhoi).item()-expP**2
      Q4[j] = ExpVal(Ops[8], Ops[9], rhor, rhoi).item()/2.0-expX*expP
      
      theta, l1 = Optimal_theta_l1(G1s, G2s, params)
      #AGamma = (G10**2-G01**2-G20+G02)/2.0
      #BGamma = G10*G01-G11
      #theta = np.arctan2(BGamma, AGamma)/2.0
      theta_t[j] = theta.item()
      csth, snth = jnp.cos(theta), jnp.sin(theta)
      #l1 = -l1max*np.sign(k20)
      l1_t[j] = l1.item()
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = jnp.array([csth, snth, 0, 0]) @ G1s
      readout[j] = r.item()
      #Idth += params[2]*(r**2-2*r*(csth*Q1[j]+snth*Q2[j])+csth**2*expX+snth**2*expP+2*snth*csth*(Q4[j]+expX*expP))/params[3]
      rhor, rhoi, G1s, G2s, Idth = RK4_step(Ops, rhor, rhoi, G1s, G2s, Idth, params)
      j+=1
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1, Q2, Q3, Q4, Q5, theta_t, l1_t, rhor, rhoi, readout, Idth


def OP_wcontrol(Initials, Ops, rho_ir, rho_ii,  l1_t, theta_t, params):
  #I_tR = jnp.array([0.0])
  G1s = params[4][0:4] @ Initials
  
  rhor, rhoi = rho_ir, rho_ii
  j=0
  npoints = len(params[1])
  Q1 = np.zeros(npoints)
  Q2 = np.zeros(npoints)
  Q3 = np.zeros(npoints)
  Q4 = np.zeros(npoints)
  Q5 = np.zeros(npoints)
  readout = np.zeros(npoints)
  rhor = rho_ir
  rhoi = rho_ii
  
  tsi = np.linspace(params[1][0], params[1][-1], 2*len(params[1])-1)
  l1_ti = np.interp(tsi, params[1], l1_t)
  theta_ti = np.interp(tsi, params[1], theta_t)
  istep = 0
  Idth = 0
  while (j<len(params[1])):
      t = params[1][j]
      theta = theta_t[j]
      
      csth, snth = np.cos(theta), np.sin(theta)
      
      l1 = l1_t[j]
      
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = jnp.array([csth, snth, 0, 0]) @ G1s
      readout[j] = r.item()

      expX = ExpVal(Ops[0], Ops[1], rhor, rhoi).item()
      expP = ExpVal(Ops[2], Ops[3], rhor, rhoi).item()
      Q1[j]=expX
      Q2[j]=expP
      #delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = ExpVal(Ops[6], Ops[7], rhor, rhoi).item()-expX**2
      Q5[j] = ExpVal(Ops[10], Ops[11], rhor, rhoi).item()-expP**2
      Q4[j] = ExpVal(Ops[8], Ops[9], rhor, rhoi).item()/2.0-expX*expP

      Idth += params[2]*(r**2-2*r*(csth*Q1[j]+snth*Q2[j])+csth**2*expX+snth**2*expP+2*snth*csth*(Q4[j]+expX*expP))/params[3]
      if (j<len(params[1])-1):
          theta1 = theta_t[j+1]
          #thetai = theta_ti[istep+1]
          csth1, snth1 = np.cos(theta1), np.sin(theta1)
          #csthi, snthi = np.cos(thetai), np.sin(thetai)
          l11 = l1_t[j+1]
          #l1i = l1_ti[istep]
          thetai = theta_ti[istep+1]
          csthi, snthi = np.cos(thetai), np.sin(thetai)
          l1i = l1_ti[istep]
          rhor, rhoi, G1s = RK4_wcontrol(Ops, rhor, rhoi, G1s, csth, snth, l1, csthi, snthi,  l1i, csth1, snth1, l11, params)
      
      j+=1
      istep+=2
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1, Q2, Q3, Q4, Q5, rhor, rhoi, readout, Idth

def OP_wcontrol_Euler(Initials, Ops, rho_ir, rho_ii, l1_t, theta_t, params):
  #I_tR = jnp.array([0.0])
  G1s = params[4][0:4] @ Initials
  
  rhor, rhoi = rho_ir, rho_ii
  j=0
  npoints = len(params[1])
  Q1 = np.zeros(npoints)
  Q2 = np.zeros(npoints)
  Q3 = np.zeros(npoints)
  Q4 = np.zeros(npoints)
  Q5 = np.zeros(npoints)
  readout = np.zeros(npoints)
  rhor = rho_ir
  rhoi = rho_ii
  
  #tsi = np.linspace(params[1][0], params[1][-1], 2*len(params[1])-1)
  #l1_ti = np.interp(tsi, params[1], l1_t)
  #theta_ti = np.interp(tsi, params[1], theta_t)
  #istep = 0
  while (j<npoints):
      t = params[1][j]
      theta = theta_t[j]
      
      csth, snth = np.cos(theta), np.sin(theta)
      
      l1 = l1_t[j]
      
      #e_jphi = np.exp(-1j*phi)
      #delh_t_Mat = e_jphi*np.array([1,1j,1j*t/(8.0*tau), -t/(8.0*tau),0,0,0,0,0])
      #ht = np.matmul(delh_t_Mat, Initials) + e_jphi*I_t
      r = jnp.array([csth, snth, 0, 0]) @ G1s
      readout[j] = r.item()

      expX = ExpVal(Ops[0], Ops[1], rhor, rhoi).item()
      expP = ExpVal(Ops[2], Ops[3], rhor, rhoi).item()
      Q1[j]=expX
      Q2[j]=expP
      #delL = Ljump - expL*Id
      #print (delL)
      Q3[j] = ExpVal(Ops[6], Ops[7], rhor, rhoi).item()-expX**2
      Q5[j] = ExpVal(Ops[10], Ops[11], rhor, rhoi).item()-expP**2
      Q4[j] = ExpVal(Ops[8], Ops[9], rhor, rhoi).item()/2.0-expX*expP

      
      if (j<npoints-1):
          #theta1 = theta_t[j+1]
          #thetai = theta_ti[istep+1]
          #csth1, snth1 = np.cos(theta1), np.sin(theta1)
          #csthi, snthi = np.cos(thetai), np.sin(thetai)
          #l11 = l1_t[j+1]
          #l1i = l1_ti[istep]
          #thetai = theta_ti[istep+1]
          #csthi, snthi = np.cos(thetai), np.sin(thetai)
          #l1i = l1_ti[istep]
          rhor, rhoi, G1s = Euler_wcontrol(rhor, rhoi, G1s, csth, snth, l1, params)
      
      j+=1
      #istep+=2
  #Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5 = jax.lax.fori_loop(0, len(ts), rho_integrate_JAX,(Initials, X, P, H,  rho, I_t, I_k_t, I_Gp_t, I_G_t,  phi, ts, tau, dt, k1, Id, Q1, Q2, Q3, Q4, Q5))
  return Q1, Q2, Q3, Q4, Q5, rhor, rhoi, readout





def Multiply_Mat(nvars, Ncos):
    tmp1 = np.zeros((nvars,nvars+4*Ncos))
    #tmp2 = np.zeros((4,10+2*Ncos), dtype=complex)
    zerovec = np.zeros(4*Ncos) 
    idmat = np.identity(nvars)
    for i in range(nvars):
        tmp1[i] = np.concatenate((idmat[i],zerovec))
    return tmp1

def Fourier_fns(nvars, Ncos, t_i, t_f, t):
    tmp = np.zeros(2*Ncos)
    for i in range(2*Ncos):
        #if i==0:
            #tmp[i]=1.0
        if (i<=Ncos-1):
            tmp[i]=np.cos(2*np.pi*(i)*(t-t_i)/(t_f-t_i))#-1
        else:
            tmp[i]=np.sin(2*np.pi*(i+1-Ncos)*(t-t_i)/(t_f-t_i))
    theta_mat = np.zeros(nvars+4*Ncos)
    l1_mat = np.zeros(nvars+4*Ncos)
    theta_mat[nvars:nvars+2*Ncos] = tmp
    l1_mat[nvars+2*Ncos:] = tmp
    
    return theta_mat, l1_mat

def Fourier_mat(nvars, Ncos, t_i, t_f, ts):
    theta_mat = np.zeros((len(ts),nvars+4*Ncos))
    l1_mat = np.zeros((len(ts),nvars+4*Ncos))
    for i in range(len(ts)):
        tmp1, tmp2=Fourier_fns(nvars, Ncos,t_i, t_f, ts[i])
        theta_mat[i,:] = tmp1
        l1_mat[i,:] = tmp2
    return theta_mat, l1_mat

def rho_update_control_generate(i, Input_Initials): #Optimal control integration with \lambda_1=0
    Initials, Ops, rhor, rhoi, G1s, Idth, theta_t, l1_t, params, j, Ncos = Input_Initials
  #I_t = I_tR + 1j*I_tI
  #print (tau)
    t = params[1][j]
    theta = theta_t[j]
    l1 = l1_t[j]
    #theta = (np.pi/2.0)*jnp.tanh(2*jnp.matmul(theta_mat[j],Initials)/np.pi)
    #l1 = (params[0])*jnp.tanh(jnp.matmul(l1_mat[j],Initials)/params[0])
  
    theta1 = theta_t[j+1]#(np.pi/2.0)*jnp.tanh(2*jnp.matmul(theta_mat[j+1],Initials)/np.pi)
    l11 = l1_t[j+1]#(params[0])*jnp.tanh(jnp.matmul(l1_mat[j+1],Initials)/params[0])
  
    thetai = (theta+theta1)/2.0
    l1i = (l1+l11)/2.0
    csth, snth = jnp.cos(theta), jnp.sin(theta)
    csthi, snthi = jnp.cos(thetai), jnp.sin(thetai)
    csth1, snth1 = jnp.cos(theta1), jnp.sin(theta1)
    rhor1, rhoi1, G1s1 = RK4_wcontrol(Ops, rhor, rhoi, G1s, csth, snth, l1, csthi, snthi, l1i, csth1, snth1, l11, params)
  
    Idth1 = Idth#+dt*(theta_star-theta)**2
  #Idth1 = -(-(GLL-w**2/4.0)/(2*tau)-(kappaLL+2*r*w)/2.0-(kappaMM+2*v*z)/2.0)
    return (Initials, Ops, rhor1, rhoi1, G1s1, Idth1, theta_t, l1_t, params, j+1, Ncos)


def MLP_control_generate(Initials, Ops, rho_ir, rho_ii, theta_t, l1_t, params, Ncos, MMat):
  Idth =  0.0
  #G10 = jnp.matmul(MMat[0],Initials)#+jnp.array([0])
  #G01 = jnp.matmul(MMat[1],Initials)#+jnp.array([0])
  #k10 = jnp.matmul(MMat[2],Initials)#+jnp.array([0])
  #k01 = jnp.matmul(MMat[3],Initials)#+jnp.array([0])
  G1s = MMat[:4] @ Initials
  rhor = rho_ir
  rhoi = rho_ii
  k1=0
  Idth=0
  Initials, Ops, rhor, rhoi, G1s, Idth,  theta_t, l1_t, params, k1, Ncos = jax.lax.fori_loop(0, len(params[1])-1, rho_update_control_generate,(Initials, Ops, rhor, rhoi, G1s, Idth,  theta_t, l1_t, params, k1, Ncos))
  #rho_update(Initials, X, P, H, X2, P2, XP, PX, rho, I_tR, I_tI, i, theta_t, ts, tau, dt)
  return rhor, rhoi

@jit
def CostF_control_generate(Initials, Ops, rho_ir, rho_ii, rho_fr, rho_fi, theta_mat, l1_mat, params, Ncos, MMat):
    theta_t = theta_mat @ Initials
    l1_t = l1_mat @ Initials
    theta_t = (np.pi/2.0)*jnp.tanh(2*theta_t/np.pi)
    l1_t = (params[0])*jnp.tanh(l1_t/params[0])
    rho_f_simulr, rho_f_simuli =  MLP_control_generate(Initials, Ops, rho_ir, rho_ii, theta_t, l1_t, params, Ncos, MMat)
    #print (Idth)
    return Tr_Distance(rho_f_simulr, rho_f_simuli, rho_fr, rho_fi), rho_f_simulr, rho_f_simuli


def OP_trajectory_JAX(i, Input_Initials):
    dWt, Ops, rhor, rhoi, l1_t, theta_t, r_t, Idth, params, j = Input_Initials
    l1 = l1_t[j]
    #print(j)
    theta = theta_t[j]
    rOP = r_t[j]
    #H1 = H+l1*X2
    csth, snth = jnp.cos(theta), jnp.sin(theta)
    expX = ExpVal(Ops[0], Ops[1], rhor, rhoi)
    expP = ExpVal(Ops[2], Ops[3], rhor, rhoi)
    #Ljump = csth*X+snth*P
    #Ljump2 = jnp.matmul(Ljump,Ljump)
    expL = csth*expX + snth*expP
    dW = dWt[j]
    r = expL+jnp.sqrt(params[3])*dW/params[2]
    Idth1 = Idth+params[2]*(r-rOP)**2
    #delL = Ljump - expL*Id
    #F = -jnp.matmul(delL, delL)*dt/(8*tau)+delL*dW/(2*jnp.sqrt(tau))
    #F1 = Id+F#+jnp.matmul(F,F)/2
    #F2 = jnp.matmul(F1,(Id-1j*H1*dt))
    #F3 = jnp.matmul((Id+1j*H1*dt),F1)
    #rho1 = jnp.matmul(jnp.matmul(F2,rho),F3)
    #H1 = H+l1*X2
    #Fac1 = r*Ljump/(2*tau)#-Ljump2/(4*tau)
    Fr, Fi = M_stochastic_step(Ops, r, csth, snth,  l1, params)
    rhor, rhoi = rho_kraus_update(rhor, rhoi, Ops[12]+Fr, Fi)
    #Fac2 = Id-dt*(X2*csth**2+csth*snth*CXP+P2*snth**2)/(8*tau)+dr*(csth*X+snth*P)/(2*tau)
    #rho1 = jnp.matmul(jnp.matmul(Fac2-dt*1j*H1,rho),Fac2+dt*1j*H1)
    #tmptr = jnp.trace(rho1)
    #rho1 = rho1/tmptr
    #rho1 = rho1/jnp.trace(rho1)
    #rho1 = rho+dt*(-1j*jnp.matmul(H1,rho)+1j*jnp.matmul(rho,H1)+(jnp.matmul(jnp.matmul(Ljump,rho),Ljump)-jnp.matmul(Ljump2, rho)/2.0-jnp.matmul(rho,Ljump2)/2.0)/(4*tau))+dW*(jnp.matmul(delL,rho)+jnp.matmul(rho,delL))/jnp.sqrt(4*tau)
    return (dWt, Ops, rhor, rhoi, l1_t, theta_t, r_t, Idth1, params, j+1)


@jit    
def OP_stochastic_trajectory_JAX(dWt, Ops, rho_ir, rho_ii, rho_fr, rho_fi,  l1_t, theta_t, r_t, params):
  #rho = rho_i
  rhor = rho_ir
  rhoi = rho_ii
  #phi = jnp.array([theta_t[0]+ts[0]])
  #theta = jnp.array(0.0)
  k1=0
  Idth = 0.0
  dWt, Ops, rhor, rhoi, l1_t, theta_t, r_t, Idth, params, k1 = jax.lax.fori_loop(0, len(params[1])-1, OP_trajectory_JAX,(dWt, Ops, rhor, rhoi, l1_t, theta_t, r_t, Idth, params, k1))
  return Fidelity_PS(rhor, rhoi, rho_fr, rho_fi), jnp.sqrt(Idth/params[1][-1])

def RdParams(Dirname):
    with h5py.File(Dirname+'/Parameters.hdf5', 'r') as f:
        Ops =  jnp.array(f['Ops'])
        ts =  np.array(f['ts'])
        rhoi =  np.array(f['rho_i'])
        rhof =  np.array(f['rho_f_target'])
        rho_ir = jnp.array(rhoi.real)
        rho_ii = jnp.array(rhoi.imag)
        rho_fr = jnp.array(rhof.real)
        rho_fi = jnp.array(rhof.imag)
        tau = np.array(f['tau']).item()
        #dt = np.array(f['dt']).item()
        Idmat = jnp.array(f['Idmat'])
        l1max = np.array(f['l1max']).item()
    return Ops, rho_ir, rho_ii,  rho_fr, rho_fi, (l1max, ts, ts[1]-ts[0], tau, Idmat)