#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:44:46 2024

@author: t_karmakar
"""

'''
This script finds the optimal path of an oscillator
under position measurements and compares the results
with those found through the Gaussian state assumption 
See Karmakar et al., PRX Quantum 3, 010327 (2022).
The comparison makes sense only when the initial and 
final state are both Gaussian.
'''

import os,sys
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
from scipy.integrate import simps as intg
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
#from Eff_OP_Functions import *
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
#from jaxopt import OptaxSolver
#import optax
script_dir = os.path.dirname(__file__)

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



def Fidelity_PS1(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  fid = jnp.trace(jnp.matmul(rho_f_simul, rho_f)).real
  return fid

def np_Fidelity_PS1(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  fid = np.trace(np.matmul(rho_f_simul, rho_f)).real
  return fid

def Tr_Distance1(rho_f_simul, rho_f):
  #delrho = rho_f_simul-rho_f
  #eps = 1e-2
  #delrho2 = jnp.matmul(delrho, delrho)
  #dist = jnp.sqrt(jnp.trace(delrho2).real)
  return -Fidelity_PS1(rho_f_simul, rho_f)#+1e2*dist


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

def CostF_strat(Initials, X, P, H,  rho_i, rho_f, theta_t, ts, dt, tau, Id):
  rho_f_simul = OPsoln_strat_JAX(Initials, X, P, H, rho_i, theta_t, ts, dt, tau, Id)
  return Tr_Distance1(rho_f_simul, rho_f)

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
    fig, axs = plt.subplots(6,1,figsize=(6,14),sharex='all')
    axs[0].plot(ts, q1t, linewidth =4, color = 'green', label = 'Gaussian assumption')
    axs[0].plot(ts, X_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'General')
    axs[0].plot(t_i, q1i, "o", color = 'b', markersize =15)
    axs[0].plot(t_f, q1f, "x" , color = 'r', markersize =18)
    
    axs[1].plot(ts, q2t, linewidth = 4, color = 'green')
    axs[1].plot(ts, P_simul1, linewidth =3, linestyle = 'dashed', color = 'red')
    axs[1].plot(t_i, q2i, "o", color = 'b', markersize =15)
    axs[1].plot(t_f, q2f, "x", color = 'r', markersize =18)
   
    axs[0].set_ylabel(r'$\left\langle \hat{X} \right\rangle$', fontsize = 20)
    axs[1].set_ylabel(r'$\left\langle \hat{P} \right\rangle$', fontsize = 20)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[2].tick_params(labelsize=15)
    axs[0].legend(loc=1,fontsize=15)
    
    
    axs[2].axhline(q3/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[2].plot(ts, varX_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
    axs[2].plot(t_i, expect((X-q1i)*(X-q1i), rho_i), "o", color = 'b', markersize =15)
    axs[2].plot(t_f, expect((X-q1f)*(X-q1f), rho_f), "x", color = 'r', markersize =18)
    
    axs[3].axhline(q4/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[3].plot(ts, covXP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Strato')
    axs[3].plot(t_i, expect((X-q1i)*(P-q2i)/2.0+(P-q2i)*(X-q1i)/2.0, rho_i), "o", color = 'b', markersize =15)
    axs[3].plot(t_f, expect((X-q1f)*(P-q2f)/2.0+(P-q2f)*(X-q1f)/2.0, rho_f), "x", color = 'r', markersize =18)

    axs[4].axhline(q5/2.0, linewidth =4, color = 'green', label = 'PRXQ')
    axs[4].plot(ts, varP_simul1, linewidth =3, linestyle = 'dashed', color = 'red', label = 'Starto')
    axs[4].plot(t_i, expect((P-q2i)*(P-q2i), rho_i), "o", color = 'b', markersize =15)
    axs[4].plot(t_f, expect((P-q2f)*(P-q2f), rho_f), "x", color = 'r', markersize =18)
    
    axs[5].plot(ts, rop_prxq, linewidth =4, color='green')
    axs[5].plot(ts, rop_strat,linewidth =3, color='red', linestyle='dashed')
    

    axs[2].set_ylabel('var('+r'$X)$', fontsize = 20)
    axs[3].set_ylabel('cov('+r'$X,P)$', fontsize = 20)
    axs[4].set_ylabel('var('+r'$P)$', fontsize = 20)
    axs[5].set_ylabel(r'$r^\star$', fontsize = 20)

    axs[5].set_xlabel(r'$t$', fontsize = 20)
    axs[5].tick_params(labelsize=18)
    axs[3].tick_params(labelsize=15)
    axs[4].tick_params(labelsize=15)
    axs[5].tick_params(labelsize=15)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    axs[0].set_xlim(0,3)
    #plt.savefig(script_dir+'/Plots/'+figname+'.pdf',bbox_inches='tight')



nlevels = 15

#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 3
ts = np.linspace(t_i, t_f, 3000)
dt = ts[1]-ts[0]
tau = 15.0
q4f = np.sqrt(1+4*tau*tau)-2*tau
q3f = np.sqrt(4*tau*q4f)
q5f = q3f*(1+q4f/(2*tau))
rparam = np.sqrt(q4f**2+(q3f-q5f)**2/4.0)+(q3f+q5f)/2.0
snh2r = np.sqrt(q4f**2+(q3f-q5f)**2/4.0)
csh2r = (q3f+q5f)/2.0
r_sq = np.log(rparam)/2
xiR = r_sq*(q5f-q3f)/(2*snh2r)
xiI = r_sq*(-q4f)/(snh2r)
in_alr = 0.0
in_ali = 0.0
fin_alr = -0.5
fin_ali = 0.8

rho_i = squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)

#rho_i = (basis(nlevels, 0)-basis(nlevels,4))/np.sqrt(2)#squeeze(nlevels, xiR+1j*xiI)*coherent(nlevels, in_alr+1j*in_ali)
#rho_f = (basis(nlevels, 0)+basis(nlevels,4))/np.sqrt(2)
rho_f = squeeze(nlevels,  xiR+1j*xiI)*coherent(nlevels, fin_alr+1j*fin_ali)
#rho_f_int = squeeze(nlevels,  xiR*np.cos(2*t_f)-xiI*np.sin(2*t_f)+1j*(xiI*np.cos(2*t_f)+xiR*np.sin(2*t_f)))*coherent(nlevels, fin_alr*np.cos(t_f)-fin_ali*np.sin(t_f)+1j*(fin_ali*np.cos(t_f)+fin_alr*np.sin(t_f)))


nsteps = 0
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
rho_i = rho_i*rho_i.dag()
#rho_i = thermal_dm(nlevels, 4)
rho_f = rho_f*rho_f.dag()


lrate = 1.0
q3, q4, q5, alr, ali, A, B, q1t, q2t, rop_prxq = OP_PRXQ_Params(X, P, rho_i, rho_f, ts, tau)
Initials = jnp.array(np.random.rand(4)-0.5)

theta_t = np.zeros(len(ts))
jnptheta_t = jnp.array(theta_t)

jnpId = jnp.identity(nlevels)
jnpX = jnp.array(X.full())
jnpP = jnp.array(P.full())
jnpH = jnp.array(H.full())
jnpX2 = jnp.matmul(jnpX, jnpX)
jnpP2 = jnp.matmul(jnpP, jnpP)
jnpXP = jnp.matmul(jnpX, jnpP)
jnpPX = jnp.matmul(jnpP, jnpX)
jnp_rho_i = jnp.array(rho_i.full())
jnp_rho_f = jnp.array(rho_f.full())

I_tR = jnp.array([0.0])
I_tI = jnp.array([0.0])
#Initials1, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho, I_tR, I_tI, theta_t, ts, tau, dt, l1 = rho_update(0,(Initials, jnpX, jnpP, jnpH, jnpX2, jnpP2, jnpXP, jnpPX, jnp_rho_i, I_tR, I_tI,  theta_t, ts, tau, dt, 0))
#jnp_rho_simul = OPsoln_JAX1(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp.array(theta_t), jnp.array(ts), dt, tau, jnpId)


#Gradient descent
for n in range(nsteps):
  stime = time.time()
  print (CostF_strat(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId))
  Initials = update_strat(Initials, jnpX, jnpP, jnpH, jnp_rho_i, jnp_rho_f, theta_t, ts, dt, tau, jnpId, lrate)
  print (n, time.time()-stime)

Initvals = np.array(Initials)

with h5py.File("/Users/tatha_k/Library/CloudStorage/Box-Box/Research/Optimal_Path/Codes/Optimal-Paths/Data/Gaussian_OP_no_control_Ex_bc.hdf5", "w") as f:
    dset1 = f.create_dataset("nlevels", data = nlevels, dtype ='int')
    dset2 = f.create_dataset("rho_i", data = rho_i.full())
    dset3 = f.create_dataset("rho_f", data = rho_f.full())
    dset4 = f.create_dataset("ts", data = ts)
    dset5 = f.create_dataset("tau", data = tau)
    dset6 = f.create_dataset("theta_t", data = theta_t)
    dset7 = f.create_dataset("Initvals", data = Initvals)    
f.close()
    
PlotOP(Initvals, X, P, H, rho_i, rho_f, ts, theta_t, tau, 'fgtmps')

rho_f_simul, X_simul, P_simul, varX_simul, covXP_simul, varP_simul, rop, nbar = OPsoln_strat_SHO(X, P, H, rho_i, Initvals[0], Initvals[1], Initvals[2], Initvals[3], ts, theta_t,  tau, 1)
#print ('Fidelity ', fidelity(rho_f_simul, rho_f_int))

