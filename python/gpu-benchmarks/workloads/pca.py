#!/usr/bin/python
import sys
import time
import math
import cudf
import cuml
import numpy as np
import pandas as pd

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, datasets
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 10
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def gen_data(mode, size):
    features, target = datasets.load_wine(return_X_y=True)
    # Make a train/test split using 30% test size
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        train_size=0.70,
                                                        random_state=42)
    # Increase the data size
    for _ in range(int(math.log2(size))):
        X_train = np.append(X_train, X_train, axis=0)
        X_test = np.append(X_test, X_test, axis=0)
        y_train = np.append(y_train, y_train)
        y_test = np.append(y_test, y_test)
    return X_train, X_test, y_train, y_test


def accuracy(y_test, pred_test):
    if isinstance(pred_test, cudf.Series) or isinstance(pred_test, cudf.DataFrame):
        pred_test = pred_test.to_pandas()
    return metrics.accuracy_score(y_test, pred_test)


def run_composer(mode, inputs, batch_size, threads):
    raise Exception


def run_naive_unscaled(X_train, X_test, y_train, y_test):
    pca = PCA(n_components=2)
    knc = KNeighborsClassifier()
    X_train_ = pca.fit_transform(X_train)
    knc.fit(X_train_, y_train)

    X_test_ = pca.transform(X_test)
    pred_test = knc.predict(X_test_)
    return pred_test


def run_naive_scaled(X_train, X_test, y_train, y_test):
    # Fit to data and predict using pipelined scaling, GNB and PCA.
    t0 = time.time()
    ss = StandardScaler()
    pca = PCA(n_components=2)
    knc = KNeighborsClassifier()
    X_train_ = ss.fit_transform(X_train)
    X_train_ = pca.fit_transform(X_train_)
    knc.fit(X_train_, y_train)
    print('Fit(naive):', time.time() - t0)

    t0 = time.time()
    X_test_ = ss.transform(X_test)
    X_test_ = pca.transform(X_test_)
    pred_test_std = knc.predict(X_test_)
    print('Predict(naive):', time.time() - t0)
    return pred_test_std


def run_cuda_unscaled(X_train, X_test, y_train, y_test):
    pca = cuml.PCA(n_components=2)
    knc = cuml.neighbors.KNeighborsClassifier()
    X_train = cudf.from_pandas(pd.DataFrame(X_train))
    X_train_ = pca.fit_transform(X_train)
    y_train = cudf.from_pandas(pd.Series(y_train))
    knc.fit(X_train_, y_train)

    X_test = cudf.from_pandas(pd.DataFrame(X_test))
    X_test_ = pca.transform(X_test)
    pred_test = knc.predict(X_test_)
    return pred_test


def run_cuda_scaled(X_train, X_test, y_train, y_test):
    # Fit to data and predict using pipelined scaling, GNB and PCA.
    t0 = time.time()
    ss = StandardScaler()
    pca = cuml.PCA(n_components=2)
    knc = cuml.neighbors.KNeighborsClassifier()
    X_train_ = ss.fit_transform(X_train)
    X_train_ = cudf.from_pandas(pd.DataFrame(X_train_))
    X_train_ = pca.fit_transform(X_train_)
    y_train = cudf.from_pandas(pd.Series(y_train))
    knc.fit(X_train_, y_train)
    print('Fit(cuda):', time.time() - t0)

    t0 = time.time()
    X_test_ = ss.transform(X_test)
    X_test_ = cudf.from_pandas(pd.DataFrame(X_test_))
    X_test_ = pca.transform(X_test_)
    pred_test_std = knc.predict(X_test_)
    print('Predict(cuda):', time.time() - t0)
    return pred_test_std


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
    inputs = gen_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    # if mode.is_composer():
    #     results = run_composer(mode, *inputs, batch_size, threads)
    # elif mode == Mode.NAIVE:
    #     pred_test = run_naive_unscaled(*inputs)
    # elif mode == Mode.CUDA:
    #     pred_test = run_cuda_unscaled(*inputs)
    # else:
    #     raise ValueError
    runtime_unscaled = time.time() - start

    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        pred_test_std = run_naive_scaled(*inputs)
    elif mode == Mode.CUDA:
        pred_test_std = run_cuda_scaled(*inputs)
    else:
        raise ValueError
    runtime_scaled = time.time() - start

    total_runtime = runtime_unscaled + runtime_scaled
    # sys.stdout.write('Runtime unscaled: {}\n'.format(runtime_unscaled))
    sys.stdout.write('Runtime scaled: {}\n'.format(runtime_scaled))
    sys.stdout.write('Total: {}\n'.format(init_time + total_runtime))
    sys.stdout.flush()
    # print('Accuracy unscaled: {:.2%}'.format(accuracy(inputs[3], pred_test)))
    print('Accuracy scaled: {:.2%}'.format(accuracy(inputs[3], pred_test_std)))
    print()
    return init_time, total_runtime

