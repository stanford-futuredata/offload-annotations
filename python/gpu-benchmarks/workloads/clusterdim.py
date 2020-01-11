#!/usr/bin/python
import sys
import time
import math
import numpy as np
import pandas as pd
import cuml
import cudf
import sklearn
import matplotlib.pyplot as plt

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 10
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def gen_data(mode, size):
    df = pd.read_csv('datasets/clusterdim/Wholesale customers data.csv')
    for _ in range(int(math.log2(size))):
        df = df.append(df)

    if mode == Mode.CUDA:
        df = cudf.from_pandas(df)
    return df


def run_composer(mode, X, eps, min_samples, _, threads):
    import sa.annotated.sklearn as sklearn

    if mode == Mode.MOZART:
        force_cpu = True
    elif mode == Mode.BACH:
        force_cpu = False
    else:
        raise Exception

    pass

    # Note: batch sizes must be max size
    batch_size = { Backend.CPU: X.shape[0], Backend.GPU: X.shape[0], }
    sklearn.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    return labels.value


def run_naive(df):
    # https://mclguide.readthedocs.io/en/latest/sklearn/clusterdim.html

    df = df.drop(labels=['Channel', 'Region'], axis=1)
    # print(df.head())

    # preprocessing
    T = sklearn.preprocessing.Normalizer().fit_transform(df)

    # change n_clusters to 2, 3 and 4 etc. to see the output patterns
    n_clusters = 3 # number of cluster

    # Clustering using KMeans
    kmean_model = sklearn.cluster.KMeans(n_clusters=n_clusters)
    kmean_model.fit(T)
    centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_
    # print(centroids)
    # print(labels)

    # Dimesionality reduction to 2
    pca_model = sklearn.decomposition.PCA(n_components=2)
    pca_model.fit(T) # fit the model
    T = pca_model.transform(T) # transform the 'normalized model'
    # transform the 'centroids of KMean'
    centroid_pca = pca_model.transform(centroids)
    print(centroid_pca)

    # # colors for plotting
    # colors = ['blue', 'red', 'green', 'orange', 'black', 'brown']
    # # assign a color to each features (note that we are using features as target)
    # features_colors = [ colors[labels[i]] for i in range(len(T)) ]

    # # plot the PCA components
    # plt.scatter(T[:, 0], T[:, 1],
    #             c=features_colors, marker='o',
    #             alpha=0.4
    #         )

    # # plot the centroids
    # plt.scatter(centroid_pca[:, 0], centroid_pca[:, 1],
    #             marker='x', s=100,
    #             linewidths=3, c=colors
    #         )

    # # store the values of PCA component in variable: for easy writing
    # xvector = pca_model.components_[0] * max(T[:,0])
    # yvector = pca_model.components_[1] * max(T[:,1])
    # columns = df.columns

    # # plot the 'name of individual features' along with vector length
    # for i in range(len(columns)):
    #     # plot arrows
    #     plt.arrow(0, 0, xvector[i], yvector[i],
    #                 color='b', width=0.0005,
    #                 head_width=0.02, alpha=0.75
    #             )
    #     # plot name of features
    #     plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)

    # plt.show()


def run_cuda(df):
    df = df.drop(labels=['Channel', 'Region'], axis=1)

    # preprocessing
    df = df.to_pandas()
    T = sklearn.preprocessing.Normalizer().fit_transform(df)

    # change n_clusters to 2, 3 and 4 etc. to see the output patterns
    n_clusters = 3 # number of cluster

    # Clustering using KMeans
    T = cudf.from_pandas(pd.DataFrame(T))
    kmean_model = cuml.KMeans(n_clusters=n_clusters)
    kmean_model.fit(T)
    centroids, labels = kmean_model.cluster_centers_, kmean_model.labels_
    # print(centroids)
    # print(labels)

    # Dimesionality reduction to 2
    pca_model = cuml.PCA(n_components=2)
    pca_model.fit(T) # fit the model
    T = pca_model.transform(T) # transform the 'normalized model'
    # transform the 'centroids of KMean'
    centroid_pca = pca_model.transform(centroids)
    centroid_pca = centroid_pca.to_pandas()
    print(centroid_pca)

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

    # Get inputs
    start = time.time()
    df = gen_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, df, batch_size, threads)
    elif mode == Mode.NAIVE:
        results = run_naive(df)
    elif mode == Mode.CUDA:
        results = run_cuda(df)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print('Results:', results)
    return init_time, runtime

