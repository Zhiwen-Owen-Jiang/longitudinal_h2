import os
import logging
import h5py
import numpy as np
import pandas as pd
import concurrent.futures
import scipy.sparse as sp
from tqdm import tqdm
from functools import partial
from script.utils import inv
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, hstack, eye
from sklearn.decomposition import IncrementalPCA
from scipy.interpolate import make_interp_spline
from image import ImageManager

"""
input:
- raw longitudinal images
- spatial bases
- covariates, which are cateogrical, which are time-variant

methods:
- for each time t, get the mean image across subjects
- for each subject i at time t, remove the mean image
- get kth ldr at time t
- regress out time-variant covariates for each ldr at time t
- local linear estimate for pooling across time

output:
- two-step ldr zeta_{ikl}
- time bases psi_{kl}

data structure:
- image
    - id
    - time
    - image
    - coordinates
- 
"""





