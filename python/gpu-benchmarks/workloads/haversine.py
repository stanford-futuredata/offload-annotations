#!/usr/bin/python
import os
import math
import sys
import time
import numpy as np
import cupy as cp
import torch

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 13
DEFAULT_GPU = 1 << 26


def get_data(size):
    lats = np.ones(size, dtype='float64') * 0.0698132
    lons = np.ones(size, dtype='float64') * 0.0698132
    return lats, lons


def _get_tmp_arrays_cuda(size, use_torch):
    if use_torch:
        a = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
        dlat = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
        dlon = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    else:
        a = cp.empty(size, dtype='float64')
        dlat = cp.empty(size, dtype='float64')
        dlon = cp.empty(size, dtype='float64')
    return a, dlat, dlon

def get_tmp_arrays(mode, size, use_torch=True):
    if mode == Mode.CUDA:
        return _get_tmp_arrays_cuda(size, use_torch=use_torch)

    if mode.is_composer() and use_torch:
        import sa.annotated.numpy as np
    elif mode.is_composer() and not use_torch:
        import sa.annotated.cupy as np
    else:
        import numpy as np

    a = np.empty(size, dtype='float64')
    dlat = np.empty(size, dtype='float64')
    dlon = np.empty(size, dtype='float64')
    return a, dlat, dlon


def haversine(lat2, lon2, a, dlat, dlon, composer, use_torch=None):
    if composer and use_torch:
        import sa.annotated.numpy as np
    elif composer and not use_torch:
        import sa.annotated.cupy as np
    else:
        import numpy as np

    lat1 = 0.70984286
    lon1 = 1.23892197
    MILES_CONST = 3959.0

    np.subtract(lat2, lat1, out=dlat)
    np.subtract(lon2, lon1, out=dlon)

    # dlat = sin(dlat / 2.0) ** 2.0
    np.divide(dlat, 2.0, out=dlat)
    np.sin(dlat, out=dlat)
    np.multiply(dlat, dlat, out=dlat)

    # a = cos(lat1) * cos(lat2)
    lat1_cos = math.cos(lat1)
    np.cos(lat2, out=a)
    np.multiply(a, lat1_cos, out=a)

    # a = a + sin(dlon / 2.0) ** 2.0
    np.divide(dlon, 2.0, out=dlon)
    np.sin(dlon, out=dlon)
    np.multiply(dlon, dlon, out=dlon)
    np.multiply(a, dlon, out=a)
    np.add(dlat, a, out=a)

    c = a
    np.sqrt(a, out=a)
    np.arcsin(a, out=a)
    np.multiply(a, 2.0, out=c)

    mi = c
    np.multiply(c, MILES_CONST, out=mi)


def run_composer(mode, lat2, lon2, a, dlat, dlon, batch_size, threads, use_torch):
    if use_torch:
        import sa.annotated.numpy as np
    else:
        import sa.annotated.cupy as np

    a.materialize = Backend.CPU

    if mode == Mode.MOZART:
        force_cpu = True
    elif mode == Mode.BACH:
        force_cpu = False
        # TODO: don't allow different batch sizes for now
        batch_size[Backend.CPU] = batch_size[Backend.GPU]
    else:
        raise Exception

    haversine(lat2, lon2, a, dlat, dlon, composer=True, use_torch=use_torch)
    np.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    return a.value


def run_naive(lat2, lon2, a, dlat, dlon):
    haversine(lat2, lon2, a, dlat, dlon, composer=False)
    return a


def run_cuda_torch(lat2, lon2, a, dlat, dlon):
    lat1 = 0.70984286
    lon1 = 1.23892197
    MILES_CONST = 3959.0

    t = time.time()
    lat2 = torch.from_numpy(lat2).cuda()
    lon2 = torch.from_numpy(lon2).cuda()
    transfer_inputs = time.time() - t
    print('Transfer(inputs):', transfer_inputs)

    t = time.time()
    torch.sub(lat2, lat1, out=dlat)
    torch.sub(lon2, lon1, out=dlon)

    # dlat = sin(dlat / 2.0) ** 2.0
    torch.div(dlat, 2.0, out=dlat)
    torch.sin(dlat, out=dlat)
    torch.mul(dlat, dlat, out=dlat)

    # a = cos(lat1) * cos(lat2)
    lat1_cos = math.cos(lat1)
    torch.cos(lat2, out=a)
    torch.mul(a, lat1_cos, out=a)

    # a = a + sin(dlon / 2.0) ** 2.0
    torch.div(dlon, 2.0, out=dlon)
    torch.sin(dlon, out=dlon)
    torch.mul(dlon, dlon, out=dlon)
    torch.mul(a, dlon, out=a)
    torch.add(dlat, a, out=a)

    c = a
    torch.sqrt(a, out=a)
    torch.asin(a, out=a)
    torch.mul(a, 2.0, out=c)

    mi = c
    torch.mul(c, MILES_CONST, out=mi)
    print('Compute:', time.time() - t)

    t = time.time()
    mi = mi.cpu().numpy()
    transfer_outputs = time.time() - t
    print('Transfer(outputs):', transfer_outputs)
    print('Transfer(total):', transfer_inputs + transfer_outputs)
    return mi


def run_cuda_cupy(lat2, lon2, a, dlat, dlon):
    lat1 = 0.70984286
    lon1 = 1.23892197
    MILES_CONST = 3959.0

    t = time.time()
    lat2 = cp.array(lat2)
    lon2 = cp.array(lon2)
    transfer_inputs = time.time() - t
    print('Transfer(inputs):', transfer_inputs)

    t = time.time()
    cp.subtract(lat2, lat1, out=dlat)
    cp.subtract(lon2, lon1, out=dlon)

    # dlat = sin(dlat / 2.0) ** 2.0
    cp.divide(dlat, 2.0, out=dlat)
    cp.sin(dlat, out=dlat)
    cp.multiply(dlat, dlat, out=dlat)

    # a = cos(lat1) * cos(lat2)
    lat1_cos = math.cos(lat1)
    cp.cos(lat2, out=a)
    cp.multiply(a, lat1_cos, out=a)

    # a = a + sin(dlon / 2.0) ** 2.0
    cp.divide(dlon, 2.0, out=dlon)
    cp.sin(dlon, out=dlon)
    cp.multiply(dlon, dlon, out=dlon)
    cp.multiply(a, dlon, out=a)
    cp.add(dlat, a, out=a)

    c = a
    cp.sqrt(a, out=a)
    cp.arcsin(a, out=a)
    cp.multiply(a, 2.0, out=c)

    mi = c
    cp.multiply(c, MILES_CONST, out=mi)
    print('Compute:', time.time() - t)

    t = time.time()
    mi = cp.asnumpy(mi)
    transfer_outputs = time.time() - t
    print('Transfer(outputs):', transfer_outputs)
    print('Transfer(total):', transfer_inputs + transfer_outputs)
    return mi


def run(mode, size=None, cpu=None, gpu=None, threads=None):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        threads = 1

    use_torch = threads % 2 == 1
    threads = 1
    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    start = time.time()
    inputs = get_data(size)
    print('Inputs (doesn\'t count):', time.time() - start)

    # Get inputs
    start = time.time()
    tmp_arrays = get_tmp_arrays(mode, size, use_torch=use_torch)
    init_time = time.time() - start
    print('Initialization:', init_time)

    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, *inputs, *tmp_arrays, batch_size, threads, use_torch=use_torch)
    elif mode == Mode.NAIVE:
        results = run_naive(*inputs, *tmp_arrays)
    elif mode == Mode.CUDA and use_torch:
        results = run_cuda_torch(*inputs, *tmp_arrays)
    elif mode == Mode.CUDA and not use_torch:
        results = run_cuda_cupy(*inputs, *tmp_arrays)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(results)
    return init_time, runtime
