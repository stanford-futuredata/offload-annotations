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


def gen_data(_mode, size):
    # make a synthetic dataset
    X, y = make_blobs(
        n_samples=size, n_features=n_features, centers=1, random_state=7)
    return X, y


def run_composer(mode, X, y, batch_size, threads):
    import sa.annotated.sklearn as sklearn
    force_cpu = mode == Mode.MOZART
    tsvd = sklearn.TruncatedSVD(n_components=n_components,
                 algorithm="arpack",
                 n_iter=5000,
                 tol=0.00001,
                 random_state=random_state)

    result = sklearn.fit_transform(tsvd, X)
    result.materialize = Backend.CPU
    sklearn.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)

    return result


def run_naive(X, y):
    tsvd_sk = skTSVD(n_components=n_components,
                 algorithm="arpack",
                 n_iter=5000,
                 tol=0.00001,
                 random_state=random_state)

    result_sk = tsvd_sk.fit_transform(X)
    return result_sk


def run_cuda(X, y):
    tsvd_cuml = cumlTSVD(n_components=n_components,
                     algorithm="full",
                     n_iter=50000,
                     tol=0.00001,
                     random_state=random_state)

    X = cp.array(X)
    result_cuml = tsvd_cuml.fit_transform(X)
    result_cuml = np.asarray(result_cuml.as_gpu_matrix())
    return result_cuml


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        threads = 1

    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    # Initialize CUDA driver...
    import cuml
    cuml.linear_model.LinearRegression()

    # Get inputs
    start = time()
    inputs = gen_data(mode, size)
    init_time = time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time()
    if mode.is_composer():
        result = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        result = run_naive(*inputs)
    elif mode == Mode.CUDA:
        result = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time() - start

    naive_result = run_naive(*inputs)
    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()

    # passed = np.allclose(naive_result[0], result[0], atol=0.01)
    # print('compare tsvd: cuml vs sklearn singular_values_ {}'.format('equal' if passed else 'NOT equal'))
    # passed = np.allclose(naive_result[1], result[1], atol=1e-2)
    # print('compare tsvd: cuml vs sklearn components_ {}'.format('equal' if passed else 'NOT equal'))
    # compare the reduced matrix
    passed = np.allclose(naive_result, result, atol=0.2)
    # larger error margin due to different algorithms: arpack vs full
    print('compare tsvd: cuml vs sklearn transformed results %s'%('equal'if passed else 'NOT equal'))
    assert passed
    return init_time, runtime
