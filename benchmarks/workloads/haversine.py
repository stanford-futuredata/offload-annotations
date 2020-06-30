#!/usr/bin/python
import os
import math
import sys
import time
import numpy as np
import torch

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 13
MAX_BATCH_SIZE = 1 << 28


def get_data(size):
    """Input data starts on the CPU, as if it came from another
    step in the pipeline.

    Parameters:
    - size: number of elements
    """

    lats = np.ones(size, dtype='float64') * 0.0698132
    lons = np.ones(size, dtype='float64') * 0.0698132
    return lats, lons


def haversine(lat2, lon2, oas=True, use_torch=True):
    """The "original" workload.

    Parameters:
    - oas: true if using OAs, false if using the CPU library
    - use_torch: true if torch, false if cupy (only relevant if using OAs)
    """
    if oas:
        if use_torch:
            import sa.annotated.numpy_torch as np
            import sa.annotated.numpy_torch as ss
        else:
            import sa.annotated.numpy_cupy as np
            import sa.annotated.numpy_cupy as ss
    else:
        import numpy as np
        import scipy.special as ss

    # Allocate output array and temporary arrays
    size = len(lat2)
    a = np.empty(size, dtype='float64')
    dlat = np.empty(size, dtype='float64')
    dlon = np.empty(size, dtype='float64')

    if oas:
        a.materialize = Backend.CPU

    # Begin computation
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

    # Materialize outputs
    if oas:
        np.evaluate(
            workers=1,
            batch_size={
                Backend.CPU: DEFAULT_CPU,
                Backend.GPU: MAX_BATCH_SIZE,
            },
            force_cpu=False,
            paging=size > MAX_BATCH_SIZE,
        )
        return a.value
    else:
        return a


def run_numpy(lats, lons):
    return haversine(lats, lons, oas=False)


def run_bach_torch(lats, lons):
    return haversine(lats, lons, oas=True, use_torch=True)


def run_bach_cupy(lats, lons):
    return haversine(lats, lons, oas=True, use_torch=False)


def run_torch(lat2, lon2):
    import torch

    # Allocate temporary arrays
    size = len(lat2)
    a = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    dlat = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    dlon = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))

    # Transfer inputs to the GPU
    lat2 = torch.from_numpy(lat2).cuda()
    lon2 = torch.from_numpy(lon2).cuda()

    # Begin computation
    lat1 = 0.70984286
    lon1 = 1.23892197
    MILES_CONST = 3959.0

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

    # Transfer outputs back to CPU
    torch.cuda.synchronize()
    a = a.cpu().numpy()

    return a


def run_cupy(lat2, lon2):
    import cupy as cp

    # Allocate temporary arrays
    size = len(lat2)
    a = cp.empty(size, dtype='float64')
    dlat = cp.empty(size, dtype='float64')
    dlon = cp.empty(size, dtype='float64')

    # Transfer inputs to the GPU
    lat2 = cp.array(lat2)
    lon2 = cp.array(lon2)

    # Begin computation
    lat1 = 0.70984286
    lon1 = 1.23892197
    MILES_CONST = 3959.0

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

    # Transfer outputs back to CPU
    a = cp.asnumpy(a)

    return a


def run(mode, use_torch, size, cpu, gpu, threads):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if mode == Mode.GPU or mode == Mode.BACH:
        torch.cuda.init()
        torch.cuda.synchronize()

    start = time.time()
    inputs = get_data(size)
    print('Inputs:', time.time() - start)

    start = time.time()
    if mode == Mode.CPU:
        result = run_numpy(*inputs)
    elif mode == Mode.GPU:
        if use_torch:
            result = run_torch(*inputs)
        else:
            result = run_cupy(*inputs)
    elif mode == Mode.BACH:
        if use_torch:
            result = run_bach_torch(*inputs)
        else:
            result = run_bach_cupy(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start
    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.flush()

    print(result)
    return 0, runtime


def run_torch_main(mode, size=None, cpu=None, gpu=None, threads=1):
    return run(mode, True, size=size, cpu=cpu, gpu=gpu, threads=threads)


def run_cupy_main(mode, size=None, cpu=None, gpu=None, threads=1):
    return run(mode, False, size=size, cpu=cpu, gpu=gpu, threads=threads)
