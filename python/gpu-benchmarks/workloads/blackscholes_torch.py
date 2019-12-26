#!/usr/bin/python
import os
import math
import sys
import time
import torch

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 18
# Same for both cuda streams and bach gpu piece size
DEFAULT_GPU = 1 << 26


def get_data(mode, size):
    """
    Inputs.

    Allocated on the cpu as if they were tensors passed in by another application.
    """
    price = torch.ones(size, dtype=torch.float64) * 4.0
    strike = torch.ones(size, dtype=torch.float64) * 4.0
    t = torch.ones(size, dtype=torch.float64) * 4.0
    rate = torch.ones(size, dtype=torch.float64) * 4.0
    vol = torch.ones(size, dtype=torch.float64) * 4.0
    return price, strike, t, rate, vol


def get_tmp_arrays(mode, size):
    if mode.is_composer():
        import sa.annotated.torch as torch
    else:
        import torch

    # Tmp arrays
    tmp = torch.empty(size, dtype=torch.float64)
    vol_sqrt = torch.empty(size, dtype=torch.float64)
    rsig = torch.empty(size, dtype=torch.float64)
    d1 = torch.empty(size, dtype=torch.float64)
    d2 = torch.empty(size, dtype=torch.float64)

    # Outputs
    call = torch.empty(size, dtype=torch.float64)
    put = torch.empty(size, dtype=torch.float64)
    return tmp, vol_sqrt, rsig, d1, d2, call, put


def torch_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put, composer):
    if composer:
        import sa.annotated.torch as torch
    else:
        import torch

    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)

    torch.mul(vol, vol, out=rsig)
    torch.mul(rsig, c05, out=rsig)
    torch.add(rsig, rate, out=rsig)

    torch.sqrt(t, out=vol_sqrt)
    torch.mul(vol_sqrt, vol, out=vol_sqrt)

    torch.mul(rsig, t, out=tmp)
    torch.div(price, strike, out=d1)
    torch.log2(d1, out=d1)
    torch.add(d1, tmp, out=d1)

    torch.div(d1, vol_sqrt, out=d1)
    torch.sub(d1, vol_sqrt, out=d2)

    # d1 = c05 + c05 * erf(d1 * invsqrt2)
    torch.mul(d1, invsqrt2, out=d1)
    torch.erf(d1, out=d1)
    torch.mul(d1, c05, out=d1)
    torch.add(d1, c05, out=d1)

    # d2 = c05 + c05 * erf(d2 * invsqrt2)
    torch.mul(d2, invsqrt2, out=d2)
    torch.erf(d2, out=d2)
    torch.mul(d2, c05, out=d2)
    torch.add(d2, c05, out=d2)

    # Reuse existing buffers
    e_rt = vol_sqrt
    tmp2 = rsig

    # e_rt = exp(-rate * t)
    torch.mul(rate, -1.0, out=e_rt)
    torch.mul(e_rt, t, out=e_rt)
    torch.exp(e_rt, out=e_rt)

    # call = price * d1 - e_rt * strike * d2
    #
    # tmp = price * d1
    # tmp2 = e_rt * strike * d2
    # call = tmp - tmp2
    torch.mul(price, d1, out=tmp)
    torch.mul(e_rt, strike, out=tmp2)
    torch.mul(tmp2, d2, out=tmp2)
    torch.sub(tmp, tmp2, out=call)

    # put = e_rt * strike * (c10 - d2) - price * (c10 - d1)
    # tmp = e_rt * strike
    # tmp2 = (c10 - d2)
    # put = tmp - tmp2
    # tmp = c10 - d1
    # tmp = price * tmp
    # put = put - tmp
    torch.mul(e_rt, strike, out=tmp)
    torch.sub(c10, d2, out=tmp2)
    torch.mul(tmp, tmp2, out=put)
    torch.sub(c10, d1, out=tmp)
    torch.mul(price, tmp, out=tmp)
    torch.sub(put, tmp, out=put)


def run_composer(
    mode,
    price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put,
    batch_size,
    threads,
):
    start = time.time()
    import sa.annotated.torch as torch
    call.materialize = Backend.CPU
    put.materialize = Backend.CPU

    if mode == Mode.MOZART:
        force_cpu = True
    elif mode == Mode.BACH:
        force_cpu = False
    else:
        raise Exception

    torch_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put, composer=True)
    print('Build time:', time.time() - start)
    torch.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    if not force_cpu:
        torch.cuda.synchronize()
    if hasattr(call, 'value'):
        return call.value, put.value
    else:
        return call, put


def run_naive(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put):
    torch_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put, composer=False)
    return call, put


def run(mode, size=None, cpu=None, gpu=None, threads=None):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        threads = 16

    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    # Single-threaded allocation
    torch.set_num_threads(1)

    start = time.time()
    inputs = get_data(mode, size)
    print('Inputs (doesn\'t count):', time.time() - start)

    # Get inputs
    start = time.time()
    tmp_arrays = get_tmp_arrays(mode, size)
    init_time = time.time() - start
    print('Temporary arrays:', init_time)

    start = time.time()
    if mode == Mode.MOZART:
        # Composer runs on a single thread for manual parallelization
        torch.set_num_threads(1)
        call, put = run_composer(mode, *inputs, *tmp_arrays, batch_size, threads)
    elif mode == Mode.NAIVE:
        # Allow naive PyTorch the number of requested threads for parallel execution
        torch.set_num_threads(threads)
        call, put = run_naive(*inputs, *tmp_arrays)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    print('Total:', init_time + runtime)
    print(call)
    print(put)
    return init_time, runtime
