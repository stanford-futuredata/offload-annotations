#!/usr/bin/python
import sys
import time
import math
import numpy as np
import pandas as pd
import cuml
import cudf

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 13
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26

DEFAULT_FEATURES = 256
DEFAULT_CENTERS = 32
DEFAULT_CLUSTER_STD = 1.0


def gen_data(mode,
             size,
             n_features=DEFAULT_FEATURES,
             centers=DEFAULT_CENTERS,
             cluster_std=DEFAULT_CLUSTER_STD):
    X, _labels_true = make_blobs(n_samples=size,
                                n_features=n_features,
                                centers=centers,
                                cluster_std=cluster_std,
                                center_box=(-2.0,2.0),
                                random_state=42)
    X = StandardScaler().fit_transform(X)
    if mode == Mode.CUDA:
        X = cudf.from_pandas(pd.DataFrame(X))

    eps = (n_features * cluster_std**2)**0.5
    min_samples = max(2, size / centers * 0.05)
    return X, eps, min_samples


def clusters(labels):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(X, labels))
    return n_clusters, n_noise


def run_composer(mode, X, batch_size, threads):
    raise Exception


def run_naive(X, eps, min_samples):
    size = len(X)
    print('eps={} min_samples={}'.format(eps, min_samples))

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    return labels


def run_cuda(X, eps, min_samples):
    size = len(X)
    print('eps={} min_samples={}'.format(eps, min_samples))

    # Run DBSCAN
    db = cuml.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    labels = labels.to_pandas().to_numpy()
    return labels


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        if mode == Mode.MOZART:
            threads = 16
        else:
            threads = 1

    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    # Get inputs
    start = time.time()
    inputs = gen_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        labels = run_naive(*inputs)
    elif mode == Mode.CUDA:
        labels = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print('Clusters, noise:', clusters(labels))
    return init_time, runtime

