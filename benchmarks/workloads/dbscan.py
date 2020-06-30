#!/usr/bin/python
import sys
import time
import math
import numpy as np
import cupy as cp
import pandas as pd
import cuml
import cudf
import sklearn
import sklearn.cluster

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 13
DEFAULT_CPU = 1 << 22
MAX_BATCH_SIZE = 1 << 22

DEFAULT_FEATURES = 256
DEFAULT_CENTERS = 32
DEFAULT_CLUSTER_STD = 1.0


def gen_data(size,
             n_features=DEFAULT_FEATURES,
             centers=DEFAULT_CENTERS,
             cluster_std=DEFAULT_CLUSTER_STD):
    """Generate input data (included in total runtime).

    Parameters include number of features, number of centers, and the
    standard deviation of the clusters.
    """
    X, _labels_true = sklearn.datasets.make_blobs(n_samples=size,
                                                  n_features=n_features,
                                                  centers=centers,
                                                  cluster_std=cluster_std,
                                                  center_box=(-2.0,2.0),
                                                  random_state=42)
    eps = (n_features * cluster_std**2)**0.5
    min_samples = max(2, size / centers * 0.05)
    return X, eps, min_samples


def clusters(labels):
    """Number of clusters in labels, ignoring noise if present.

    Used to validate classification results.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return n_clusters, n_noise


def run_bach(X, eps, min_samples):
    import sa.annotated.numpy_cupy as np
    import sa.annotated.sklearn as sklearn

    if X.shape[0] > MAX_BATCH_SIZE:
        print('WARNING: will run out of memory')

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    np.subtract(X, mean, out=X)
    np.divide(X, std, out=X)

    db = sklearn.DBSCAN(eps=eps, min_samples=min_samples)
    db = sklearn.fit_x(db, X)
    labels = sklearn.labels(db)
    labels.materialize = Backend.CPU

    sklearn.evaluate(
        workers=1,
        batch_size={
            Backend.CPU: DEFAULT_CPU,
            Backend.GPU: MAX_BATCH_SIZE,
        },
        force_cpu=False,
        paging=False,
    )
    return labels.value


def run_cpu(X, eps, min_samples):
    # Begin computation
    t0 = time.time()
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    np.subtract(X, mean, out=X)
    np.divide(X, std, out=X)
    print('Preprocessing:', time.time() - t0)

    # Run DBSCAN
    db = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    db = db.fit(X)
    labels = db.labels_
    return labels


def run_gpu(X, eps, min_samples):
    # Transfer inputs to GPU
    X = cp.array(X)

    # Begin computation
    t0 = time.time()
    mean = cp.mean(X, axis=0)
    std = cp.std(X, axis=0)
    cp.subtract(X, mean, out=X)
    cp.divide(X, std, out=X)
    print('Preprocessing:', time.time() - t0)

    # Run DBSCAN
    db = cuml.DBSCAN(eps=eps, min_samples=min_samples)
    db = db.fit(X)
    labels = db.labels_

    # Transfer outputs to CPU
    labels = labels.to_pandas().to_numpy()
    return labels


def run(mode, size=None, _cpu=None, _gpu=None, threads=None, data_mode='file'):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if threads is None:
        threads = 1

    # Get inputs
    start = time.time()
    inputs = gen_data(size)
    init_time = time.time() - start
    sys.stdout.write('Init: {}\n'.format(init_time))

    # This workload can't be split, so always default to the max size
    batch_size = {
        Backend.CPU: inputs[0].shape[0],
        Backend.GPU: inputs[0].shape[0],
    }

    # Run program
    start = time.time()
    if mode == Mode.BACH:
        labels = run_bach(*inputs)
    elif mode == Mode.CPU:
        labels = run_cpu(*inputs)
    elif mode == Mode.GPU:
        labels = run_gpu(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print('Clusters, noise:', clusters(labels))
    return init_time, runtime
