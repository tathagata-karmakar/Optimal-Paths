#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:52:49 2024

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

nlevels = 8
rho_i = basis(nlevels,0)
#rho_f = basis(nlevels,4)
rho_f = coherent(nlevels, 0.5+1j*1.0)
#rho_f = coherent(nlevels, 0.5)
a = destroy(nlevels)
t_i = 0
t_f = 3
ts = np.linspace(t_i, t_f, 100)
dt = ts[1]-ts[0]
tau = 15
nsteps = 10
theta_t = np.zeros(len(ts))
X = (a+a.dag())/np.sqrt(2)
P = (a-a.dag())/(np.sqrt(2)*1j)
H = (X*X+P*P)/2.0
Ljump = X
Mjump = P
rho_i = rho_i*rho_i.dag()
rho_f = rho_f*rho_f.dag()
sigma_i = rand_herm(nlevels)
sigma_i = sigma_i-expect(sigma_i, rho_i)
xvec = np.linspace(-5,5,200)
pvec = np.linspace(-5,5,200)
W_i = wigner(rho_i,xvec,pvec)

dc = 0.01
lrate = 1
q3, q4, q5, alr, ali, A, B, q1t, q2t = OP_PRXQ_Params(Ljump, Mjump, rho_i, rho_f, ts, tau)
alr = np.random.rand()
ali = np.random.rand()
A = np.random.rand()
B = np.random.rand()
tb = 0

def fidelity1(rho_1, rho_2):
    delrho = rho_1 - rho_2
    #del2rho = delrho*delrho
    return -expect(delrho, delrho).real
for n in range(nsteps):

  rho_f_simul, X0, P0 = OPsoln_SHO(Ljump, Mjump, H, rho_i, alr, ali, A, B, ts,   theta_t, tau, 0)
  f0 = -fidelity(rho_f_simul,rho_f)
  print (n, f0, time.time()-tb)
  tb = time.time()

  rho_f_simul1, Xtmp, Ptmp = OPsoln_SHO(Ljump, Mjump, H, rho_i, alr+dc, ali, A, B, ts, theta_t, tau, 0)
  f1 = -fidelity(rho_f_simul1,rho_f)
  delC1 = lrate*(f1-f0)/dc
  alr_update = -delC1

  rho_f_simul2, Xtmp, Ptmp = OPsoln_SHO(Ljump, Mjump, H, rho_i, alr, ali+dc, A, B, ts, theta_t, tau, 0)
  f2 = -fidelity(rho_f_simul2,rho_f)
  delC2 = lrate*(f2-f0)/dc
  ali_update = -delC2

  rho_f_simul3, Xtmp, Ptmp = OPsoln_SHO(Ljump, Mjump, H, rho_i, alr, ali, A+dc, B, ts, theta_t, tau, 0)
  f3 = -fidelity(rho_f_simul3,rho_f)
  delC3 = lrate*(f3-f0)/dc
  A_update = -delC3

  rho_f_simul4, Xtmp, Ptmp = OPsoln_SHO(Ljump, Mjump, H, rho_i, alr, ali, A, B+dc, ts,  theta_t, tau, 0)
  f4 = -fidelity1(rho_f_simul4,rho_f)
  delC4 = lrate*(f4-f0)/dc
  B_update = -delC4

  alr = alr + alr_update
  ali = ali + alr_update
  A = A + A_update
  B = B + B_update

PlotOP(np.array([alr, ali, A, B]), X, P, H, rho_i, rho_f, ts, theta_t, tau, 'tmpfig')
'''
W_f = wigner(rho_f,xvec,pvec)
fig,axes = plt.subplots(1,3, figsize=(12,3))
cont0 = axes[0].contourf(xvec, pvec, W_i, 100)
lbl0 = axes[0].set_title('$\\rho_i$')
cont1 = axes[1].contourf(xvec, pvec, W_f, 100)
lbl1 = axes[1].set_title('$\\rho_f$')
W_f_simul = wigner(rho_f_simul,xvec,pvec)
cont2 = axes[2].contourf(xvec, pvec, W_f_simul, 100)
lbl2 = axes[2].set_title('Simulated '+'$\\rho_f$')
plt.show()
'''