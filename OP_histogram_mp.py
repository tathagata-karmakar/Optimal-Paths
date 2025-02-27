#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 21:16:29 2025

@author: tatha_k
"""
import os,sys
os.environ['JAX_PLATFORMS'] = 'cpu'
#os.environ['JAX_DISABLE_JIT'] = '1'

#os.environ['JAX_DISABLE_JIT'] = '1'

os.environ['ENABLE_PJRT_COMPATIBILITY'] = '1'
os.environ['CHECKPOINT_PATH']='${path_to_checkpoints}'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import scipy
from scipy.integrate import simpson as intg
#from google.colab import files
#from google.colab import drive
from matplotlib import rc
from pylab import rcParams
from matplotlib import colors
from qutip import *
from Eff_OP_Functions import *
#from Initialization import *
import h5py
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text',usetex=True)
script_dir = os.path.dirname(__file__)

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

from multiprocessing import Pool, Value, Process
import multiprocessing as mp
import multiprocessing.pool

def OP_ST_JAX(dWs, Ops, rho_ir, rho_ii, rho_fr, rho_fi, l10i, theta0i, newparams):
        # Reconstruct the JAX function within the process
        local_f = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9: OP_stochastic_trajectory_JAX(x1, x2, x3, x4, x5, x6, x7, x8, x9)
        return local_f(dWs, Ops, rho_ir, rho_ii, rho_fr, rho_fi, l10i, theta0i, newparams)

if __name__=="__main__":
    Dirname = script_dir+"/Data/Cat_to_ground"
    Ops, rho_ir, rho_ii,  rho_fr, rho_fi, params = RdParams(Dirname)
    '''
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
    '''
    #dt = ts[1]-ts[0]
        
    with h5py.File(Dirname+'/Optimal_control_solution.hdf5', 'r') as f:
        Initvals = np.array(f['Initvals'])
        l1_t = np.array(f['l1_t'])
        theta_t = np.array(f['theta_t'])
        

    with h5py.File(Dirname+'/Alternate_control.hdf5', 'r') as f:
        Initvals_s = np.array(f['Initials_sample'])
        l10 = np.array(f['l1_t_sample'])
        theta0 = np.array(f['theta_t_sample'])
        

    tsi = np.linspace(params[1][0], params[1][-1], 5*len(params[1]))
    l1_ti = np.interp(tsi, params[1], l1_t)
    theta_ti = np.interp(tsi, params[1], theta_t)
    l10i = np.interp(tsi, params[1], l10)
    theta0i = np.interp(tsi, params[1], theta0)
    newparams = (params[0], tsi, tsi[1]-tsi[0], params[3], params[4])
    batchsize = 1000
    ns = 1
    samplesize = ns*batchsize
    #fidelitiesC = np.zeros(samplesize)
    #fidelities_OC = np.zeros(samplesize)
    
    resultsC = jnp.zeros((ns, batchsize))
    resultsOC = jnp.zeros((ns, batchsize))
    
    stime = time.time()
    for n in range(ns):
        btime  = time.time()
        print ("Histogram generation for batch: ", n+1)
        with mp.pool.Pool() as pool:
            dWt = jnp.array(np.random.normal(scale=np.sqrt(newparams[2]), size = (batchsize,len(theta0i))))
            input_tuples = [(dWs, Ops, rho_ir, rho_ii, rho_fr, rho_fi, l10i, theta0i, newparams) for dWs in dWt]
            results =pool.starmap(OP_ST_JAX, input_tuples)
        resultsC = resultsC.at[n,:].set(jnp.array(results))
            
        with mp.pool.Pool() as pool:
            dWt = jnp.array(np.random.normal(scale=np.sqrt(newparams[2]), size = (batchsize,len(theta_ti))))
            input_tuples = [(dWs, Ops, rho_ir, rho_ii, rho_fr, rho_fi, l1_ti, theta_ti, newparams) for dWs in dWt]
            results1 =pool.starmap(OP_ST_JAX, input_tuples)
        resultsOC = resultsOC.at[n,:].set(jnp.array(results1))
        print ("Total batch time :", time.time()-btime)
        
    
    fidelitiesC = jnp.array(resultsC.flatten())
    fidelitiesOC = jnp.array(resultsOC.flatten())
    print (time.time()-stime)
    
    fig, ax = plt.subplots()

    ax.hist(results)
    ax.hist(results1)
    ax.set_xlabel(r'$\mathcal{F}\left(\hat{\rho}_f,\hat{\rho}(t_f)\right)$')

    with h5py.File(Dirname+"/Histogram.hdf5", "w") as f:
        dset1 = f.create_dataset("Fidelities_wo_control", data = fidelitiesC)
        dset2 = f.create_dataset("Fidelities_w_control", data = fidelitiesOC)
        #dset10 = f.create_dataset("Initvals", data = Initvals)   
        dset3 = f.create_dataset("Sample_size", data = samplesize)   

    '''
    #print (nsample)
    #Q1j, Q2j, Q3j, Q4j, Q5j, rho_f_simul, rs= OP_stochastic_trajectory(X.full(), P.full(), H.full(), X2.full(), rho_i.full(), l10, theta0, ts, dt,  tau,  np.identity(nlevels))
    dWt = jnp.array(np.random.normal(scale=np.sqrt(dt), size = (samplesize,len(theta0i))))
    batch_stochastic_trajectory = jax.vmap(OP_stochastic_trajectory_JAX, in_axes=[0,  None, None, None], out_axes = 0)
    fidelitiesC = batch_stochastic_trajectory(dWt, l10i, theta0i, newparams)
    #fidelitiesC[nsample] = Fidelity_PS(rho_f_simul0, jnp_rho_f).item()

    dWt = jnp.array(np.random.normal(scale=np.sqrt(dt), size = (samplesize,len(theta_ti))))
    fidelities_OC = batch_stochastic_trajectory(dWt,   l1_ti, theta_ti, newparams)

    fidelitiesC = np.array(fidelitiesC)
    fidelities_OC = np.array(fidelities_OC)
    '''