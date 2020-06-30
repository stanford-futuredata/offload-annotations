#!/usr/bin/python
import sys
import time
import math
import cudf
import cuml
import numpy as np
import pandas as pd
import sklearn

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 10
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def gen_data(mode, size):
    """Parameters:
    - size: number of rows in the dataset before train/test split, and before
      scaling by original dataset size (178)
    """
    features, target = sklearn.datasets.load_wine(return_X_y=True)
    # Make a train/test split using 30% test size
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        features, target, train_size=0.70, random_state=42)
    # Increase the data size
    for _ in range(int(math.log2(size))):
        X_test = np.append(X_test, X_test, axis=0)
        y_test = np.append(y_test, y_test)
    return X_train, X_test, y_train, y_test


def accuracy(y_test, pred_test):
    """Accuracy of classifier. Used to validate results.

    Parameters:
    - y_test: actual results
    - pred_test: predicted results
    """
    return sklearn.metrics.accuracy_score(y_test, pred_test)


def run_bach(X_train, X_test, y_train, y_test, scaled=True):
    import sa.annotated.sklearn as sklearn
    batch_size = {
        Backend.CPU: X_train.shape[0],
        Backend.GPU: X_train.shape[0],
    }

    # Initialize models
    t0 = time.time()
    if scaled:
        ss = sklearn.StandardScaler()
        ss.dontsend = False
    pca = sklearn.PCA(n_components=2)
    knc = sklearn.KNeighborsClassifier()
    pca.dontsend = False
    knc.dontsend = False

    # Train models
    if scaled:
        X_train_ = sklearn.fit_transform_cpu(ss, X_train)
    else:
        X_train_ = X_train
    X_train_ = sklearn.fit_transform(pca, X_train_)
    sklearn.fit_xy(knc, X_train_, y_train)

    # Note: batch sizes must be max size
    sklearn.evaluate(
        workers=1,
        batch_size=batch_size,
        force_cpu=False,
    )
    print('Fit(composer):', time.time() - t0)

    # Test models
    t0 = time.time()
    if scaled:
        X_test_ = sklearn.transform_cpu(ss.value, X_test)
    else:
        X_test_ = X_test
    X_test_ = sklearn.transform(pca.value, X_test_)
    pred_test = sklearn.predict(knc.value, X_test_)
    pred_test.materialize = Backend.CPU

    # Note: here, the batch sizes can be anything because this section can be split
    sklearn.evaluate(
        workers=1,
        batch_size={
            Backend.CPU: DEFAULT_CPU,
            Backend.GPU: DEFAULT_GPU,
        },
        force_cpu=False,
    )
    t0 = print('Predict(composer):', time.time() - t0)
    return pred_test.value


def run_cpu(X_train, X_test, y_train, y_test, scaled=True):
    # Initialize models
    t0 = time.time()
    if scaled:
        ss = sklearn.preprocessing.StandardScaler()
    pca = PCA(n_components=2)
    knc = KNeighborsClassifier()

    # Train models
    if scaled:
        X_train_ = ss.fit_transform(X_train)
    else:
        X_train_ = X_train
    X_train_ = pca.fit_transform(X_train_)
    knc.fit(X_train_, y_train)
    print('Fit(cpu):', time.time() - t0)

    # Test models
    t0 = time.time()
    if scaled:
        X_test_ = ss.transform(X_test)
    else:
        X_test_ = X_test
    X_test_ = pca.transform(X_test_)
    pred_test = knc.predict(X_test_)
    print('Predict(cpu):', time.time() - t0)
    return pred_test


def run_gpu(X_train, X_test, y_train, y_test, scaled=True):
    import cupy as cp

    # Initialize models
    t0 = time.time()
    if scaled:
        ss = sklearn.preprocessing.StandardScaler()
    pca = cuml.PCA(n_components=2)
    knc = cuml.neighbors.KNeighborsClassifier()

    # Train models
    if scaled:
        X_train_ = ss.fit_transform(X_train)
    else:
        X_train_ = X_train
    X_train_ = cp.array(X_train_)
    X_train_ = pca.fit_transform(X_train_)
    y_train_ = cp.array(y_train)
    knc.fit(X_train_, y_train_)
    print('Fit(gpu):', time.time() - t0)

    # Test models
    t0 = time.time()
    if scaled:
        X_test_ = ss.transform(X_test)
    else:
        X_test_ = X_test
    X_test_ = cp.array(X_test_)
    X_test_ = pca.transform(X_test_)
    pred_test = knc.predict(X_test_)
    pred_test = pred_test.to_pandas().to_numpy()
    print('Predict(gpu):', time.time() - t0)
    return pred_test


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE

    # Get inputs
    start = time.time()
    inputs = gen_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Init: {}\n'.format(init_time))

    # Initialize CUDA
    if mode == Mode.BACH or mode == Mode.GPU:
        cuml.LogisticRegression()

    # Run program
    start = time.time()
    if mode == Mode.BACH:
        pred_test_std = run_bach(*inputs, scaled=True)
    elif mode == Mode.CPU:
        pred_test_std = run_cpu(*inputs, scaled=True)
    elif mode == Mode.GPU:
        pred_test_std = run_gpu(*inputs, scaled=True)
    else:
        raise ValueError
    runtime_scaled = time.time() - start

    sys.stdout.write('Runtime scaled: {}\n'.format(runtime_scaled))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime_scaled))
    sys.stdout.flush()
    print('Accuracy scaled: {:.2%}'.format(accuracy(inputs[3], pred_test_std)))
    print()
    return init_time, runtime_scaled
