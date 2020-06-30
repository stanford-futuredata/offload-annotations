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
DEFAULT_CPU = 1 << 27
MAX_BATCH_SIZE = 1 << 27


def get_data(size):
    """Input data starts on the CPU, as if it came from another
    step in the pipeline.

    Parameters:
    - size: number of elements in the array
    """

    price = np.ones(size, dtype='float64') * 4.0
    strike = np.ones(size, dtype='float64') * 4.0
    t = np.ones(size, dtype='float64') * 4.0
    rate = np.ones(size, dtype='float64') * 4.0
    vol = np.ones(size, dtype='float64') * 4.0
    return price, strike, t, rate, vol


def blackscholes(price, strike, t, rate, vol, oas=True, use_torch=True):
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

    # Allocate temporary arrays
    size = len(price)
    tmp = np.empty(size, dtype='float64')
    vol_sqrt = np.empty(size, dtype='float64')
    rsig = np.empty(size, dtype='float64')
    d1 = np.empty(size, dtype='float64')
    d2 = np.empty(size, dtype='float64')

    # Outputs
    call = np.empty(size, dtype='float64')
    put = np.empty(size, dtype='float64')

    if oas:
        call.materialize = Backend.CPU
        put.materialize = Backend.CPU

    # Begin computation
    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)

    np.multiply(vol, vol, out=rsig)
    np.multiply(rsig, c05, out=rsig)
    np.add(rsig, rate, out=rsig)

    np.sqrt(t, out=vol_sqrt)
    np.multiply(vol_sqrt, vol, out=vol_sqrt)

    np.multiply(rsig, t, out=tmp)
    np.divide(price, strike, out=d1)
    np.log2(d1, out=d1)
    np.add(d1, tmp, out=d1)

    np.divide(d1, vol_sqrt, out=d1)
    np.subtract(d1, vol_sqrt, out=d2)

    # d1 = c05 + c05 * erf(d1 * invsqrt2)
    np.multiply(d1, invsqrt2, out=d1)
    ss.erf(d1, out=d1)
    np.multiply(d1, c05, out=d1)
    np.add(d1, c05, out=d1)

    # d2 = c05 + c05 * erf(d2 * invsqrt2)
    np.multiply(d2, invsqrt2, out=d2)
    ss.erf(d2, out=d2)
    np.multiply(d2, c05, out=d2)
    np.add(d2, c05, out=d2)

    # Reuse existing buffers
    e_rt = vol_sqrt
    tmp2 = rsig

    # e_rt = exp(-rate * t)
    np.multiply(rate, -1.0, out=e_rt)
    np.multiply(e_rt, t, out=e_rt)
    np.exp(e_rt, out=e_rt)

    # call = price * d1 - e_rt * strike * d2
    #
    # tmp = price * d1
    # tmp2 = e_rt * strike * d2
    # call = tmp - tmp2
    np.multiply(price, d1, out=tmp)
    np.multiply(e_rt, strike, out=tmp2)
    np.multiply(tmp2, d2, out=tmp2)
    np.subtract(tmp, tmp2, out=call)

    # put = e_rt * strike * (c10 - d2) - price * (c10 - d1)
    # tmp = e_rt * strike
    # tmp2 = (c10 - d2)
    # put = tmp - tmp2
    # tmp = c10 - d1
    # tmp = price * tmp
    # put = put - tmp
    np.multiply(e_rt, strike, out=tmp)
    np.subtract(c10, d2, out=tmp2)
    np.multiply(tmp, tmp2, out=put)
    np.subtract(c10, d1, out=tmp)
    np.multiply(price, tmp, out=tmp)
    np.subtract(put, tmp, out=put)

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
        return call.value, put.value
    else:
        return call, put


def run_numpy(price, strike, t, rate, vol):
    return blackscholes(price, strike, t, rate, vol, oas=False)


def run_bach_torch(price, strike, t, rate, vol):
    return blackscholes(price, strike, t, rate, vol, oas=True, use_torch=True)


def run_bach_cupy(price, strike, t, rate, vol):
    return blackscholes(price, strike, t, rate, vol, oas=True, use_torch=False)


def run_torch(price, strike, t, rate, vol):
    import torch

    # Allocate temporary arrays
    size = len(price)
    tmp = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    vol_sqrt = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    rsig = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    d1 = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    d2 = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))

    # Outputs
    call = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    put = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))

    # Transfer inputs to the GPU
    price = torch.from_numpy(price).to(torch.device('cuda'))
    strike = torch.from_numpy(strike).to(torch.device('cuda'))
    t = torch.from_numpy(t).to(torch.device('cuda'))
    rate = torch.from_numpy(rate).to(torch.device('cuda'))
    vol = torch.from_numpy(vol).to(torch.device('cuda'))

    # Begin computation
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

    # Transfer outputs back to CPU
    torch.cuda.synchronize()
    call = call.cpu().numpy()
    put = put.cpu().numpy()

    return call, put


def run_cupy(price, strike, t, rate, vol):
    import cupy as cp

    # Allocate temporary arrays
    size = len(price)
    tmp = cp.empty(size, dtype='float64')
    vol_sqrt = cp.empty(size, dtype='float64')
    rsig = cp.empty(size, dtype='float64')
    d1 = cp.empty(size, dtype='float64')
    d2 = cp.empty(size, dtype='float64')

    # Outputs
    call = cp.empty(size, dtype='float64')
    put = cp.empty(size, dtype='float64')

    # Transfer inputs to the GPU
    price = cp.array(price)
    strike = cp.array(strike)
    t = cp.array(t)
    rate = cp.array(rate)
    vol = cp.array(vol)

    # Create an erf function that doesn't exist
    cp_erf = cp.core.create_ufunc(
        'cupyx_scipy_erf', ('f->f', 'd->d'),
        'out0 = erf(in0)',
        doc='''Error function.
        .. seealso:: :meth:`scipy.special.erf`
        ''')

    # Begin computation
    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)

    cp.multiply(vol, vol, out=rsig)
    cp.multiply(rsig, c05, out=rsig)
    cp.add(rsig, rate, out=rsig)

    cp.sqrt(t, out=vol_sqrt)
    cp.multiply(vol_sqrt, vol, out=vol_sqrt)

    cp.multiply(rsig, t, out=tmp)
    cp.divide(price, strike, out=d1)
    cp.log2(d1, out=d1)
    cp.add(d1, tmp, out=d1)

    cp.divide(d1, vol_sqrt, out=d1)
    cp.subtract(d1, vol_sqrt, out=d2)

    # d1 = c05 + c05 * erf(d1 * invsqrt2)
    cp.multiply(d1, invsqrt2, out=d1)
    cp_erf(d1, out=d1)
    cp.multiply(d1, c05, out=d1)
    cp.add(d1, c05, out=d1)

    # d2 = c05 + c05 * erf(d2 * invsqrt2)
    cp.multiply(d2, invsqrt2, out=d2)
    cp_erf(d2, out=d2)
    cp.multiply(d2, c05, out=d2)
    cp.add(d2, c05, out=d2)

    # Reuse existing buffers
    e_rt = vol_sqrt
    tmp2 = rsig

    # e_rt = exp(-rate * t)
    cp.multiply(rate, -1.0, out=e_rt)
    cp.multiply(e_rt, t, out=e_rt)
    cp.exp(e_rt, out=e_rt)

    # call = price * d1 - e_rt * strike * d2
    #
    # tmp = price * d1
    # tmp2 = e_rt * strike * d2
    # call = tmp - tmp2
    cp.multiply(price, d1, out=tmp)
    cp.multiply(e_rt, strike, out=tmp2)
    cp.multiply(tmp2, d2, out=tmp2)
    cp.subtract(tmp, tmp2, out=call)

    # put = e_rt * strike * (c10 - d2) - price * (c10 - d1)
    # tmp = e_rt * strike
    # tmp2 = (c10 - d2)
    # put = tmp - tmp2
    # tmp = c10 - d1
    # tmp = price * tmp
    # put = put - tmp
    cp.multiply(e_rt, strike, out=tmp)
    cp.subtract(c10, d2, out=tmp2)
    cp.multiply(tmp, tmp2, out=put)
    cp.subtract(c10, d1, out=tmp)
    cp.multiply(price, tmp, out=tmp)
    cp.subtract(put, tmp, out=put)

    # Transfer outputs back to CPU
    call = cp.asnumpy(call)
    put = cp.asnumpy(put)

    return call, put


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
        call, put = run_numpy(*inputs)
    elif mode == Mode.GPU:
        if use_torch:
            call, put = run_torch(*inputs)
        else:
            call, put = run_cupy(*inputs)
    elif mode == Mode.BACH:
        if use_torch:
            call, put = run_bach_torch(*inputs)
        else:
            call, put = run_bach_cupy(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start
    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.flush()

    print(call)
    print(put)
    return 0, runtime


def run_torch_main(mode, size=None, cpu=None, gpu=None, threads=1):
    return run(mode, True, size=size, cpu=cpu, gpu=gpu, threads=threads)


def run_cupy_main(mode, size=None, cpu=None, gpu=None, threads=1):
    return run(mode, False, size=size, cpu=cpu, gpu=gpu, threads=threads)
