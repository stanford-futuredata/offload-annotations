#!/usr/bin/python
import sys
import time
import numpy as np
import cuml

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 16
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26

DEFAULT_FEATURES = 256
DEFAULT_CENTERS = 4
DEFAULT_CLUSTER_STD = 0.3


def _gen_data_cuda(size):
    X, labels_true = cuml.datasets.make_blobs(n_samples=size,
                                              n_features=DEFAULT_FEATURES,
                                              centers=DEFAULT_CENTERS,
                                              cluster_std=DEFAULT_CLUSTER_STD,
                                              random_state=42)
    X = StandardScaler().fit_transform(X)
    return X, labels_true


def gen_data(mode, size):
    if mode == Mode.CUDA:
        return _gen_data_cuda(size)

    X, labels_true = make_blobs(n_samples=size,
                                n_features=DEFAULT_FEATURES,
                                centers=DEFAULT_CENTERS,
                                cluster_std=DEFAULT_CLUSTER_STD,
                                random_state=42)
    X = StandardScaler().fit_transform(X)
    return X, labels_true


def run_composer(mode, X, batch_size, threads):
    raise Exception


def run_naive(X, labels_true):
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

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

    return n_clusters_, n_noise_


def run_cuda(X, labels_true):
    db = cuml.DBSCAN(eps=0.3, min_samples=10)
    db = db.fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return n_clusters_, n_noise_


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
        results = run_naive(*inputs)
    elif mode == Mode.CUDA:
        results = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(results)
    return init_time, runtime

