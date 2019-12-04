
import sys
sys.path.append("../../lib/")
sys.path.append("../../pycomposer/")

import argparse
import math
import time
from enum import Enum

from sa.annotation import Backend

class Mode(Enum):
    NAIVE = 0
    COMPOSER = 1

def get_data(size, mode, allocation, compute):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    # Allocate input arrays on the given backend for allocation
    device = torch.device(allocation)
    dtype = torch.float64
    price = torch.ones(size, device=device, dtype=dtype) * 4.0
    strike = torch.ones(size, device=device, dtype=dtype) * 4.0
    t = torch.ones(size, device=device, dtype=dtype) * 4.0
    rate = torch.ones(size, device=device, dtype=dtype) * 4.0
    vol = torch.ones(size, device=device, dtype=dtype) * 4.0

    # Allocate intermediate and output arrays on the given backend for compute
    device = torch.device(compute)
    tmp = torch.ones(size, device=device, dtype=dtype)
    vol_sqrt = torch.ones(size, device=device, dtype=dtype)
    rsig = torch.ones(size, device=device, dtype=dtype)
    d1 = torch.ones(size, device=device, dtype=dtype)
    d2 = torch.ones(size, device=device, dtype=dtype)

    # Outputs
    call = torch.ones(size, device=device, dtype=dtype)
    put = torch.ones(size, device=device, dtype=dtype)

    start = time.time()
    price = price.pin_memory()
    strike = strike.pin_memory()
    t = t.pin_memory()
    rate = rate.pin_memory()
    vol = vol.pin_memory()
    print('Pin memory time:', time.time() - start)

    return price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put

def transfer_to(price, strike, t, rate, vol, mode, compute):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    # Transfer input arrays if necessary
    start = time.time()
    def f(x): return x.to(torch.device('cuda'), non_blocking=True)
    if price.device.type == 'cpu' and compute == 'cuda':
        price = f(price)
        strike = f(strike)
        t = f(t)
        rate = f(rate)
        vol = f(vol)
        print('Transfer H2D:', time.time() - start)
    elif price.device.type == 'cuda' and compute == 'cpu':
        raise ValueError
    return (price, strike, t, rate, vol)

    # price = f(price)
    # strike = f(strike)
    # t = f(t)
    # rate = f(rate)
    # vol = f(vol)
    # print('Transfer H2D:', time.time() - start)

def transfer_from(call, put, mode):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    # Transfer output arrays if necessary
    start = time.time()
    def f(x): return x.to(torch.device('cpu'), non_blocking=True)
    if call.device.type == 'cuda':
        call = f(call)
        put = f(put)
        print('Transfer D2H:', time.time() - start)
    return (call, put)

    # call = f(call)
    # put = f(put)
    # print('Transfer D2H:', time.time() - start)

def bs(
    price, strike, t, rate, vol,    # original data
    tmp, vol_sqrt, rsig, d1, d2,    # temporary arrays
    call, put,                      # outputs
    mode, threads, compute,         # experiment figuration
    gpu_piece_size, cpu_piece_size  # piece sizes
):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
        torch.set_num_threads(1)
    else:
        import torch
        torch.set_num_threads(threads)

    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)

    start = time.time()

    call.materialize = True
    put.materialize = True

    # Computation
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

    if mode == Mode.COMPOSER:
        print("Build time:", time.time() - start)
        batch_size = {
            Backend.CPU: cpu_piece_size,
            Backend.GPU: gpu_piece_size,
        }
        torch.evaluate(workers=threads, batch_size=batch_size)
    # if compute == 'cuda':
    #     torch.cuda.synchronize()
    # print('Evaluation:', time.time() - start)

    return call, put

def run():
    import torch
    parser = argparse.ArgumentParser(
        description="Chained Adds pipelining test on a single thread."
    )
    parser.add_argument('-s', "--size", type=int, default=27, help="Size of each array")
    parser.add_argument('-cpu', "--cpu_piece_size", type=int, default=14, help="Log size of each CPU piece.")
    parser.add_argument('-gpu', "--gpu_piece_size", type=int, default=19, help="Log size of each GPU piece.")
    parser.add_argument('-t', "--threads", type=int, default=1, help="Number of threads.")
    parser.add_argument('-v', "--verbosity", type=str, default="none", help="Log level (debug|info|warning|error|critical|none)")
    parser.add_argument('-m', "--mode", type=str, required=True, help="Mode (naive|composer)")
    parser.add_argument('-a', "--allocation", type=str, default="cpu", help="Allocation backend (cpu|cuda)")
    parser.add_argument('-c', "--compute", type=str, default="cuda", help="Compute backend (cpu|cuda)")
    args = parser.parse_args()

    size = (1 << args.size)
    gpu_piece_size = 1<<args.gpu_piece_size
    cpu_piece_size = 1<<args.cpu_piece_size
    threads = args.threads
    loglevel = args.verbosity
    mode = args.mode.strip().lower()
    allocation = args.allocation.strip().lower()
    compute = args.compute.strip().lower()

    assert threads >= 1

    print("Size:", size)
    print("GPU Piece Size:", gpu_piece_size)
    print("CPU Piece Size:", cpu_piece_size)
    print("Threads:", threads)
    print("Log Level", loglevel)
    print("Mode:", mode)
    print("Allocation:", allocation)
    print("Compute:", compute)

    # Parse the mode
    if mode == 'composer':
        mode = Mode.COMPOSER
    elif mode == 'naive':
        mode = Mode.NAIVE
    else:
        raise ValueError("invalid mode", mode)

    # Parse the allocation and compute backend
    if allocation not in ['cpu', 'cuda']:
        raise ValueError("invalid allocation backend", allocation)
    if compute not in ['cpu', 'cuda']:
        raise ValueError("invalid compute backend", compute)

    start = time.time()
    a, b, c, d, e, f, g, h, i, j, k, l = get_data(size, mode, allocation, compute)
    print("Initialization: {}s".format(time.time() - start))

    start = time.time()
    n = gpu_piece_size

    # ABCABCABC
    for m in range(0, size, n):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
            fi, gi, hi, ii, ji = f[m:m+n], g[m:m+n], h[m:m+n], i[m:m+n], j[m:m+n]
            ki, li = k[m:m+n], l[m:m+n]
            ai,bi,ci,di,ei = transfer_to(ai, bi, ci, di, ei, mode, compute)
            bs(
                ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li,
                mode, threads, compute, gpu_piece_size, cpu_piece_size
            )
            ki, li = transfer_from(ki, li, mode)
            k[m:m+n]=ki[:]
            l[m:m+n]=li[:]
    torch.cuda.synchronize()

    # AAABBBCCC
    ais,bis,cis,dis,eis,fis,gis,his,iis,jis,kis,lis = [],[],[],[],[],[],[],[],[],[],[],[]
    streams = []
    for m in range(0, size, n):
        s = torch.cuda.Stream()
        streams.append(s)
        with torch.cuda.stream(s):
            ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
            fi, gi, hi, ii, ji = f[m:m+n], g[m:m+n], h[m:m+n], i[m:m+n], j[m:m+n]
            ki, li = k[m:m+n], l[m:m+n]
            ai,bi,ci,di,ei = transfer_to(ai, bi, ci, di, ei, mode, compute)
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
    for m in range(0, int(size/n)):
        s = streams[m]
        with torch.cuda.stream(s):
            bs(
                ais[m], bis[m], cis[m], dis[m], eis[m],
                fis[m], gis[m], his[m], iis[m], jis[m], kis[m], lis[m],
                mode, threads, compute, gpu_piece_size, cpu_piece_size
            )
    for m in range(0, int(size/n)):
        s = streams[m]
        with torch.cuda.stream(s):
            kis[m], lis[m] = transfer_from(kis[m], lis[m], mode)
            k[m:m+n]=kis[m][:]
            l[m:m+n]=lis[m][:]
    torch.cuda.synchronize()

    call = k
    put = l

    print("Total Runtime: {}s".format(time.time() - start))
    print("Call (len {}): {}".format(len(call), call))
    print("Put (len {}): {}".format(len(put), put))

if __name__ == "__main__":
    run()
