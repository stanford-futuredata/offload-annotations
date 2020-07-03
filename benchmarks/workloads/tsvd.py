#!/usr/bin/python
# https://github.com/rapidsai/notebooks/blob/branch-0.12/cuml/tsvd_demo.ipynb
import sys
from time import time

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import TruncatedSVD as skTSVD

import cupy as cp
from cuml.decomposition import TruncatedSVD as cumlTSVD

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 14
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26

n_features = 512
n_components = 2
random_state = 42


def gen_data(size):
    """Parameters:
    - size: number of samples
    """
    # make a synthetic dataset
    X, y = make_blobs(
        n_samples=size, n_features=n_features, centers=1, random_state=7)
    return X, y


def run_bach(X, y):
    import sa.annotated.estimator.sklearn as sklearn
    tsvd = sklearn.TruncatedSVD(n_components=n_components,
                 algorithm="arpack",
                 n_iter=5000,
                 tol=0.00001,
                 random_state=random_state)

    result = sklearn.fit_transform(tsvd, X)
    result.materialize = Backend.CPU
    sklearn.evaluate(
        workers=1,
        batch_size={
            Backend.CPU: DEFAULT_CPU,
            Backend.GPU: DEFAULT_GPU,
        },
        force_cpu=False,
    )

    return result.value


def run_cpu(X, y):
    tsvd_sk = skTSVD(n_components=n_components,
                 algorithm="arpack",
                 n_iter=5000,
                 tol=0.00001,
                 random_state=random_state)

    result_sk = tsvd_sk.fit_transform(X)
    return result_sk


def run_gpu(X, y):
    tsvd_cuml = cumlTSVD(n_components=n_components,
                     algorithm="full",
                     n_iter=50000,
                     tol=0.00001,
                     random_state=random_state)

    # Transfer inputs to GPU
    X = cp.array(X)
    result_cuml = tsvd_cuml.fit_transform(X)

    # Transfer outputs to CPU
    result = np.asarray(result_cuml.as_gpu_matrix())
    # result = cp.asnumpy(result_cuml)
    return result


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE

    # Initialize CUDA driver...
    import cuml
    cuml.linear_model.LinearRegression()

    # Get inputs
    start = time()
    inputs = gen_data(size)
    init_time = time() - start
    sys.stdout.write('Init: {}\n'.format(init_time))

    # Run program
    start = time()
    if mode == Mode.BACH:
        result = run_bach(*inputs)
    elif mode == Mode.CPU:
        result = run_cpu(*inputs)
    elif mode == Mode.GPU:
        result = run_gpu(*inputs)
    else:
        raise ValueError
    runtime = time() - start

    cpu_result = run_cpu(*inputs)
    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()

    passed = np.allclose(cpu_result, result, atol=0.2)
    # larger error margin due to different algorithms: arpack vs full
    print('compare tsvd: cuml vs sklearn transformed results %s'%('equal'if passed else 'NOT equal'))
    assert passed
    return init_time, runtime
