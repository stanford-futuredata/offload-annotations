#!/usr/bin/python
import sys
import time
import math
import numpy as np

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

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


def run_composer(mode, inputs, batch_size, threads):
    raise Exception


def run_naive(X_train, X_test, y_train, y_test):
    pca = PCA(n_components=2)
    knc = KNeighborsClassifier()
    X_train_ = pca.fit_transform(X_train)
    knc.fit(X_train_, y_train)

    X_test_ = pca.transform(X_test)
    pred_test = knc.predict(X_test_)
    return pred_test


def run_cuda(X_train, X_test, y_train, y_test):
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

