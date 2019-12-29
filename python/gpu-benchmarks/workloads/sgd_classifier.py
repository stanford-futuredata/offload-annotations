#!/usr/bin/python
import sys
import time

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def get_data(mode, size):
    raise Exception


def run_composer(mode, inputs, batch_size, threads):
    raise Exception


def run_naive(inputs):
    raise Exception


def run_cuda(inputs):
    raise Exception


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
    inputs = get_data(mode, size)
    init_time = time.time() - start
    print('Initialization:', init_time)

    # Run program
    start = time.time()
    if mode.is_composer():
        result = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        result = run_naive(*inputs)
    elif mode == Mode.CUDA:
        result = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(result)
    return init_time, runtime
