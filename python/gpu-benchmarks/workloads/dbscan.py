#!/usr/bin/python
import sys
import time

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 14
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def gen_data(mode, size):
    X = None
    y = None
    pred_data = None
    pred_results = None
    return X, y, pred_data, pred_results


def accuracy(actual, expected):
    return 0.0


def run_composer(mode, inputs, batch_size, threads):
    raise Exception


def run_naive(X, y, pred_data):
    results = None
    return results


def run_cuda(X, y, pred_data):
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

    # Get inputs
    start = time.time()
    X, y, pred_data, pred_results = gen_data(mode, size)
    inputs = (X, y, pred_data)
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
    print('Accuracy:', accuracy(results, pred_results))
    return init_time, runtime

