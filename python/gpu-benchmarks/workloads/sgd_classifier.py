#!/usr/bin/python
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from cuml.solvers import SGD as cumlSGD
import cudf

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 14
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26

DEFAULT_INFORMATIVE = 32
DEFAULT_FEATURES = 256
DEFAULT_PREDICTIONS = 1 << 21

DEFAULT_REPEATED = 0
DEFAULT_REDUNDANT = 0


def _gen_data_cuda(X0, y0, columns, predictions):
    X = pd.DataFrame(X0[:-predictions], columns=columns)
    pred_data = pd.DataFrame(X0[-predictions:], columns=columns)
    X = cudf.from_pandas(X)
    pred_data = cudf.from_pandas(pred_data)

    y = cudf.Series(y0[:-predictions], dtype='float64')
    pred_results = pd.Series(y0[-predictions:], dtype='float64')
    return X, y, pred_data, pred_results


def _gen_data(X0, y0, columns, predictions):
    X = pd.DataFrame(X0[:-predictions], columns=columns)
    y = pd.Series(y0[:-predictions])
    pred_data = pd.DataFrame(X0[-predictions:], columns=columns)
    pred_results = pd.Series(y0[-predictions:])
    return X, y, pred_data, pred_results


def gen_data(
    mode,
    size,
    informative=DEFAULT_INFORMATIVE,
    features=DEFAULT_FEATURES,
    predictions=DEFAULT_PREDICTIONS,
):
    X0, y0 = datasets.make_classification(n_samples=size + predictions, n_features=features,
                                          n_informative=informative, n_classes=2, random_state=0,
                                          n_redundant=DEFAULT_REDUNDANT, n_repeated=DEFAULT_REPEATED)
    columns = [str(i) for i in range(features)]
    if mode == Mode.CUDA:
        return _gen_data_cuda(X0, y0, columns, predictions)
    else:
        return _gen_data(X0, y0, columns, predictions)


def accuracy(actual, expected):
    assert len(actual) == len(expected)
    total = len(actual)
    correct = sum([1 if x == y else 0 for (x, y) in zip(actual, expected)])
    return correct / total


def run_composer(mode, inputs, batch_size, threads):
    raise Exception


def run_naive(X, y, pred_data):
    sgd = SGDClassifier(eta0=0.005, max_iter=2000, fit_intercept=True, tol=1e-3)
    start = time.time()
    sgd.fit(X, y)
    print('Fit:', time.time() - start)
    start = time.time()
    pred = sgd.predict(pred_data)
    print('Prediction:', time.time() - start)
    return sgd.intercept_, pred


def run_cuda(X, y, pred_data):
    cu_sgd = cumlSGD(eta0=0.005, epochs=2000, fit_intercept=True, tol=1e-3)
    start = time.time()
    cu_sgd.fit(X, y)
    print('Fit:', time.time() - start)
    start = time.time()
    cu_pred = cu_sgd.predictClass(pred_data).to_array()
    print('Prediction:', time.time() - start)
    return cu_sgd.intercept_, cu_pred


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
    X, y, pred_data, pred_results = gen_data(mode, size)
    inputs = (X, y, pred_data)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        intercept, pred = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        intercept, pred = run_naive(*inputs)
    elif mode == Mode.CUDA:
        intercept, pred = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(intercept)
    print(pred)
    print('Accuracy:', accuracy(pred, pred_results))
    return init_time, runtime

