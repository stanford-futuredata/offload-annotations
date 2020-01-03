#!/usr/bin/python
# Based on sklearn example "Clustering text documents using k-means"
# Link: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
import sys
import time
import logging
import numpy as np

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 14
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


# Preprocess documents with latent semantic analysis.
DEFAULT_LSA = True
# Desired dimensionality of Truncated SVD output.
DEFAULT_N_COMPONENTS = 100
# Use minibatch k-means algorithm as opposed to ordinary k-means.
DEFAULT_MINIBATCH = False
# Use Inverse Document Frequency feature weighting.
DEFAULT_IDF = False
# Use a hashing feature vectorizer as opposed to Tfidf.
DEFAULT_HASHING = True
# Maximum number of features (dimensions) to extract from text.
DEFAULT_N_FEATURES = 100000


def gen_data(
    mode,
    # categories=['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'],
    categories=None,
):
    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)

    print("%d documents" % len(dataset.data))
    print("%d categories" % len(dataset.target_names))
    print()
    return dataset


def run_composer(mode, dataset, batch_size, threads):
    raise Exception


def run_naive(
    dataset,
    use_hashing=DEFAULT_HASHING,
    use_idf=DEFAULT_IDF,
    use_lsa=DEFAULT_LSA,
    use_minibatch=DEFAULT_MINIBATCH,
    n_components=DEFAULT_N_COMPONENTS,
    n_features=DEFAULT_N_FEATURES,
):
    print("Extracting features from the training dataset "
          "using a sparse vectorizer")
    t0 = time.time()
    if use_hashing:
        if use_idf:
            # # Perform an IDF normalization on the output of HashingVectorizer
            # hasher = HashingVectorizer(n_features=n_features,
            #                            stop_words='english', alternate_sign=False,
            #                            norm=None)
            # vectorizer = make_pipeline(hasher, TfidfTransformer())
            raise Exception
        else:
            vectorizer = HashingVectorizer(n_features=n_features,
                                           stop_words='english',
                                           alternate_sign=False, norm='l2')
    else:
        # vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features,
        #                              min_df=2, stop_words='english',
        #                              use_idf=use_idf)
        raise Exception

    X = vectorizer.fit_transform(dataset.data)
    print("done in %fs" % (time.time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if use_lsa:
        print("Performing dimensionality reduction using LSA")
        t0 = time.time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time.time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    # #############################################################################
    # Do the actual clustering

    labels = dataset.target
    true_k = np.unique(labels).shape[0]
    if use_minibatch:
        # km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
        #                      init_size=1000, batch_size=1000, verbose=True)
        raise Exception
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                    verbose=True)

    print("Clustering sparse data with %s" % km)
    t0 = time.time()
    km.fit(X)
    print("done in %0.3fs" % (time.time() - t0))
    print()

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    print()


    if not use_hashing:
        # print("Top terms per cluster:")

        # if use_lsa:
        #     original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        #     order_centroids = original_space_centroids.argsort()[:, ::-1]
        # else:
        #     order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        # terms = vectorizer.get_feature_names()
        # for i in range(true_k):
        #     print("Cluster %d:" % i, end='')
        #     for ind in order_centroids[i, :10]:
        #         print(' %s' % terms[ind], end='')
        #     print()
        raise Exception


def run_cuda(dataset):
    results = None
    return results


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

    # Display progress logs on stdout
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(levelname)s %(message)s')

    # Get inputs
    start = time.time()
    dataset = gen_data(mode)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, dataset, batch_size, threads)
    elif mode == Mode.NAIVE:
        results = run_naive(dataset)
    elif mode == Mode.CUDA:
        results = run_cuda(dataset)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(results)
    return init_time, runtime

