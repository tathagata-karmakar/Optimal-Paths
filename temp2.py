#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:44:04 2024

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


ts = np.linspace(0,1,100)
dt = ts[1]-ts[0]
nlevels = 5
x0 = jnp.array(np.random.rand(nlevels))
r0 = jnp.array(np.ones(nlevels))

lrate = 1e-2
nsteps = 200

def updatef(i, inputs):
    r0, x, ts, dt = inputs
    return r0, x+dt*r0, ts, dt

def solf(r0, x0, ts, dt):
    #x = x0
    r, x, ts, dt = jax.lax.fori_loop(0, len(ts)-1, updatef,(r0, x0, ts, dt))
    return x

def costf(x0, r0, ts, dt):
    x = solf(r0, x0, ts, dt)
    return (jnp.sum(x**2)-1)**2

def update_r0_f(r0, x, ts, dt, step_size):
    grads=grad(costf)(x, r0, ts, dt)
    return jnp.array([w - step_size * dw
          for w, dw in zip(x, grads)])

for n in range(nsteps):
  stime = time.time()
  #lrate = lrate0#/(1+n)
  print (costf(r0, x0, ts, dt))
  x0 = update_r0_f(r0, x0, ts, dt, lrate)
  print (n, time.time()-stime)


    
