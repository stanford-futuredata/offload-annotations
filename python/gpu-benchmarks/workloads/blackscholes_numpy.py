#!/usr/bin/python
import os
import math
import sys
import time
import scipy.special as ss
import numpy as np
import torch

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 13
DEFAULT_GPU = 1 << 26
DEFAULT_STREAM_SIZE = 1 << 22
DEFAULT_STREAMS = 16


def get_data(mode, size):
    # Inputs
    import numpy as np
    price = np.ones(size, dtype='float64') * 4.0
    strike = np.ones(size, dtype='float64') * 4.0
    t = np.ones(size, dtype='float64') * 4.0
    rate = np.ones(size, dtype='float64') * 4.0
    vol = np.ones(size, dtype='float64') * 4.0
    return price, strike, t, rate, vol


def _get_tmp_arrays_cuda(size):
    # Tmp arrays
    tmp = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    vol_sqrt = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    rsig = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    d1 = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    d2 = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))

    # Outputs
    call = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    put = torch.empty(size, dtype=torch.float64, device=torch.device('cuda'))
    return tmp, vol_sqrt, rsig, d1, d2, call, put


def get_tmp_arrays(mode, size):
    if mode == Mode.CUDA:
        return _get_tmp_arrays_cuda(size)

    if mode.is_composer():
        import sa.annotated.numpy as np
    else:
        import numpy as np

    # Tmp arrays
    tmp = np.empty(size, dtype='float64')
    vol_sqrt = np.empty(size, dtype='float64')
    rsig = np.empty(size, dtype='float64')
    d1 = np.empty(size, dtype='float64')
    d2 = np.empty(size, dtype='float64')

    # Outputs
    call = np.empty(size, dtype='float64')
    put = np.empty(size, dtype='float64')
    return tmp, vol_sqrt, rsig, d1, d2, call, put


def numpy_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put, composer):
    if composer:
        import sa.annotated.numpy as np
    else:
        import numpy as np

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

    if composer:
        np.erf(d1, out=d1)
    else:
        ss.erf(d1, out=d1)

    np.multiply(d1, c05, out=d1)
    np.add(d1, c05, out=d1)

    # d2 = c05 + c05 * erf(d2 * invsqrt2)
    np.multiply(d2, invsqrt2, out=d2)

    if composer:
        np.erf(d2, out=d2)
    else:
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


def run_composer(
    mode,
    price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put,
    batch_size,
    threads,
):
    import sa.annotated.numpy as np
    call.materialize = Backend.CPU
    put.materialize = Backend.CPU

    if mode == Mode.MOZART:
        force_cpu = True
    elif mode == Mode.BACH:
        force_cpu = False
    else:
        raise Exception

    numpy_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put, composer=True)
    np.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    if not force_cpu:
        torch.cuda.synchronize()
    return call.value, put.value


def run_naive(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put):
    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)
    numpy_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put, composer=False)
    return call, put


def torch_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put):
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


def run_cuda_nostream(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put):
    # Transfer to GPU
    price = torch.from_numpy(price).cuda()
    strike = torch.from_numpy(strike).cuda()
    t = torch.from_numpy(t).cuda()
    rate = torch.from_numpy(rate).cuda()
    vol = torch.from_numpy(vol).cuda()

    # Compute
    torch_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put)

    # Transfer to CPU
    call = call.cpu().numpy()
    put = put.cpu().numpy()
    return call, put


def run_cuda(stream_size, a, b, c, d, e, f, g, h, i, j, k, l):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    c = torch.from_numpy(c)
    d = torch.from_numpy(d)
    e = torch.from_numpy(e)

    size = len(a)
    n = stream_size
    num_pieces = int((size+n-1)/n)
    nstreams = min(num_pieces, DEFAULT_STREAMS)
    x = torch.empty(size, dtype=torch.float64)
    y = torch.empty(size, dtype=torch.float64)
    ais,bis,cis,dis,eis,fis,gis,his,iis,jis,kis,lis = [],[],[],[],[],[],[],[],[],[],[],[]
    streams = [torch.cuda.Stream() for _ in range(nstreams)]

    # Transfer to GPU
    for index in range(num_pieces):
        m = index * n
        s = streams[index % nstreams]
        with torch.cuda.stream(s):
            ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
            fi, gi, hi, ii, ji = f[m:m+n], g[m:m+n], h[m:m+n], i[m:m+n], j[m:m+n]
            ki, li = k[m:m+n], l[m:m+n]

            ai = ai.to(torch.device('cuda'), non_blocking=True)
            bi = bi.to(torch.device('cuda'), non_blocking=True)
            ci = ci.to(torch.device('cuda'), non_blocking=True)
            di = di.to(torch.device('cuda'), non_blocking=True)
            ei = ei.to(torch.device('cuda'), non_blocking=True)
            ais.append(ai)
            bis.append(bi)
            cis.append(ci)
            dis.append(di)
            eis.append(ei)
            fis.append(fi)
            gis.append(gi)
            his.append(hi)
            iis.append(ii)
            jis.append(ji)
            kis.append(ki)
            lis.append(li)
    # Compute
    for m in range(num_pieces):
        s = streams[m % nstreams]
        with torch.cuda.stream(s):
            price, strike, t, rate, vol = ais[m], bis[m], cis[m], dis[m], eis[m]
            tmp, vol_sqrt, rsig, d1, d2 = fis[m], gis[m], his[m], iis[m], jis[m]
            call, put = kis[m], lis[m]
            torch_bs(price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put)

    # Transfer to CPU
    for index in range(num_pieces):
        m = index * n
        s = streams[index % nstreams]
        with torch.cuda.stream(s):
            kis[index] = kis[index].to(torch.device('cpu'), non_blocking=True)
            lis[index] = lis[index].to(torch.device('cpu'), non_blocking=True)
            x[m:m+n]=kis[index][:]
            y[m:m+n]=lis[index][:]
    torch.cuda.synchronize()
    return x.numpy(), y.numpy()


def run(mode, size=None, cpu=None, gpu=None, threads=None):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        if mode == Mode.CUDA:
            gpu = DEFAULT_STREAM_SIZE
        else:
            gpu = DEFAULT_GPU
    if threads is None:
        threads = 1

    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    if mode == Mode.CUDA or mode == Mode.BACH:
        torch.cuda.init()
        torch.cuda.synchronize()

    start = time.time()
    inputs = get_data(mode, size)
    print('Inputs (doesn\'t count):', time.time() - start)

    # Get inputs
    start = time.time()
    tmp_arrays = get_tmp_arrays(mode, size)
    init_time = time.time() - start
    print('Initialization:', init_time)

    start = time.time()
    if mode.is_composer():
        call, put = run_composer(mode, *inputs, *tmp_arrays, batch_size, threads)
    elif mode == Mode.NAIVE:
        call, put = run_naive(*inputs, *tmp_arrays)
    elif mode == Mode.CUDA:
        call, put = run_cuda_nostream(*inputs, *tmp_arrays)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(call)
    print(put)
    return init_time, runtime
