#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 11:22:59 2024

@author: tatha_k
"""

import os,sys
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
from Initialization  import *
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

#import torch
#from torch import nn
#from torch.utils.data import DataLoader,Dataset
#from torchvision import datasets
#from torchvision.io import read_image
#from torchvision.transforms import ToTensor, Lambda
#import torchvision.models as models
##torch.backends.cuda.cufft_plan_cache[0].max_size = 32
#torch.autograd.set_detect_anomaly(True)


with h5py.File(Dirname+'/Histogram.hdf5', 'r') as f:
    fidelities0 = np.array(f['Fidelities_wo_control'])
    fidelities_OP = np.array(f['Fidelities_w_control'])
    samplesize =np.array(f['Sample_size']).item()

#np_Idmat=np.identity(10)
#Idmat = jnp.array(np_Idmat)


fig, ax = plt.subplots(figsize=(6,4))

bins = np.linspace(0,1,50)
ax.hist(fidelities_OP, bins=bins, label = 'Optimal control', hatch ='|')
ax.hist(fidelities0, bins = bins, label="Sample control", alpha = 0.6, hatch ='\\')
ax.set_xlabel(r'$\mathcal{F}\left(\hat{\rho}_f,\hat{\rho}(t_f)\right)$', fontsize=18)
ax.set_ylabel('Number of Trajectories', fontsize=18)
ax.tick_params(labelsize=15)
ax.legend(loc=2,fontsize=15)
#ax.set_xlim(0,1)
plt.savefig(Dirname+'/Plots/histogram.pdf',bbox_inches='tight')
