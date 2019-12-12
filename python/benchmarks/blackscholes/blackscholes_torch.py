
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
    AAABBBCCC = 2
    ABCABCABC = 3
    ABC = 4

def get_inputs(size, mode, device):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    # Allocate input arrays on the given backend for allocation
    device = torch.device(device)
    dtype = torch.float64
    price = torch.ones(size, device=device, dtype=dtype) * 4.0
    strike = torch.ones(size, device=device, dtype=dtype) * 4.0
    t = torch.ones(size, device=device, dtype=dtype) * 4.0
    rate = torch.ones(size, device=device, dtype=dtype) * 4.0
    vol = torch.ones(size, device=device, dtype=dtype) * 4.0

    start = time.time()
    price = price.pin_memory()
    strike = strike.pin_memory()
    t = t.pin_memory()
    rate = rate.pin_memory()
    vol = vol.pin_memory()
    print('Pin memory time:', time.time() - start)
    return price, strike, t, rate, vol

def get_tmp_arrays(size, mode, device, reuse_memory, gpu_piece_size):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    # Allocate intermediate and output arrays on the given backend for compute
    kwargs = {'dtype': torch.float64}
    if mode != Mode.COMPOSER:
        kwargs['device'] = device
    if reuse_memory:
        assert device.type == 'cuda'
        size = gpu_piece_size

    tmp = torch.empty(size, **kwargs)
    vol_sqrt = torch.empty(size, **kwargs)
    rsig = torch.empty(size, **kwargs)
    d1 = torch.empty(size, **kwargs)
    d2 = torch.empty(size, **kwargs)
    call = torch.empty(size, **kwargs)
    put = torch.empty(size, **kwargs)

    return tmp, vol_sqrt, rsig, d1, d2, call, put

def transfer_to(cpu_arrays, gpu_arrays=None):
    # Transfer input arrays if necessary
    if cpu_arrays[0].device.type == 'cuda':
        return

    import torch
    start = time.time()
    if gpu_arrays is None:
        assert len(cpu_arrays) == 5
        out = (
            cpu_arrays[0].to(torch.device('cuda'), non_blocking=True),
            cpu_arrays[1].to(torch.device('cuda'), non_blocking=True),
            cpu_arrays[2].to(torch.device('cuda'), non_blocking=True),
            cpu_arrays[3].to(torch.device('cuda'), non_blocking=True),
            cpu_arrays[4].to(torch.device('cuda'), non_blocking=True),
        )
        # print('Transfer H2D:', time.time() - start)
        return out
    else:
        assert len(cpu_arrays) == 5
        assert len(gpu_arrays) == 5
        gpu_arrays[0][:] = cpu_arrays[0][:]
        gpu_arrays[1][:] = cpu_arrays[1][:]
        gpu_arrays[2][:] = cpu_arrays[2][:]
        gpu_arrays[3][:] = cpu_arrays[3][:]
        gpu_arrays[4][:] = cpu_arrays[4][:]
        # print('Transfer H2D:', time.time() - start)
        return (gpu_arrays[0], gpu_arrays[1], gpu_arrays[2], gpu_arrays[3], gpu_arrays[4])

def transfer_from(gpu_arrays, cpu_arrays=None):
    # Transfer output arrays if necessary
    if gpu_arrays[0].device.type == 'cpu':
        return

    import torch
    start = time.time()
    if cpu_arrays is None:
        assert len(gpu_arrays) == 2
        out = (
            gpu_arrays[0].to(torch.device('cpu'), non_blocking=True),
            gpu_arrays[1].to(torch.device('cpu'), non_blocking=True),
        )
        # print('Transfer D2H:', time.time() - start)
        return out
    else:
        assert len(cpu_arrays) == 2
        assert len(gpu_arrays) == 2
        cpu_arrays[0][:] = gpu_arrays[0][:]
        cpu_arrays[1][:] = gpu_arrays[1][:]
        # print('Transfer D2H:', time.time() - start)
        return (cpu_arrays[0], cpu_arrays[1])

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

    return call.value, put.value

def run_abcabcabc(
    a, b, c, d, e, f, g, h, i, j, k, l,
    size, mode, threads, compute, gpu_piece_size, cpu_piece_size, nstreams,
):
    import torch
    n = gpu_piece_size
    start = time.time()
    streams = [torch.cuda.Stream() for _ in range(nstreams)]
    print('Create streams:', time.time() - start)
    start = time.time()
    call = torch.empty(len(a), dtype=torch.float64)
    put = torch.empty(len(a), dtype=torch.float64)
    print('Allocate cpu arrays:', time.time() - start)
    for index in range(0, int(size/n)):
        m = index * n
        s = streams[index % nstreams]
        with torch.cuda.stream(s):
            ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
            fi, gi, hi, ii, ji = f[m:m+n], g[m:m+n], h[m:m+n], i[m:m+n], j[m:m+n]
            ki, li = k[m:m+n], l[m:m+n]
            ai,bi,ci,di,ei = transfer_to([ai, bi, ci, di, ei])
            bs(
                ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li,
                mode, threads, compute, gpu_piece_size, cpu_piece_size
            )
            ki, li = transfer_from([ki, li])
            call[m:m+n]=ki[:]
            put[m:m+n]=li[:]
    torch.cuda.synchronize()
    return (call, put)

def run_abc(
    a, b, c, d, e, f, g, h, i, j, k, l,
    size, mode, threads, compute, gpu_piece_size, cpu_piece_size
):
    import torch
    n = gpu_piece_size
    call_times = {'to_cpu':0,'to_gpu':0,'call':0,'split':0,'merge':0}

    call = torch.empty(len(a), dtype=torch.float64)
    put = torch.empty(len(a), dtype=torch.float64)
    for m in range(0, size, n):
        start = time.time()
        ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
        fi, gi, hi, ii, ji = f[m:m+n], g[m:m+n], h[m:m+n], i[m:m+n], j[m:m+n]
        ki, li = k[m:m+n], l[m:m+n]
        call_times['split'] += time.time() - start

        start = time.time()
        ai,bi,ci,di,ei = transfer_to([ai, bi, ci, di, ei])
        call_times['to_gpu'] += time.time() - start

        start = time.time()
        bs(
            ai, bi, ci, di, ei, fi, gi, hi, ii, ji, ki, li,
            mode, threads, compute, gpu_piece_size, cpu_piece_size
        )
        call_times['call'] += time.time() - start

        start = time.time()
        ki, li = transfer_from([ki, li], cpu_arrays=[call[m:m+n], put[m:m+n]])
        call_times['to_cpu'] += time.time() - start

    start = time.time()
    call_times['merge'] += time.time() - start
    torch.cuda.synchronize()

    for key, val in sorted(call_times.items()):
        print('{}: {}'.format(key, val))
    return (call, put)


def run_abc_reuse_memory(
    a, b, c, d, e, f, g, h, i, j, k, l,
    size, mode, threads, compute, gpu_piece_size, cpu_piece_size
):
    import torch
    n = gpu_piece_size
    gpu_arrays = [torch.empty(n, device=torch.device('cuda'), dtype=torch.float64) for _ in range(5)]  # 5 inputs
    call = torch.empty(len(a), dtype=torch.float64)
    put = torch.empty(len(a), dtype=torch.float64)

    for m in range(0, size, n):
        ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
        ai,bi,ci,di,ei = transfer_to([ai, bi, ci, di, ei], gpu_arrays)
        bs(
            ai, bi, ci, di, ei, f, g, h, i, j, k, l,
            mode, threads, compute, gpu_piece_size, cpu_piece_size
        )
        ki, li = transfer_from([k, l], cpu_arrays=[call[m:m+n], put[m:m+n]])
    torch.cuda.synchronize()
    return (call, put)


def run_aaabbbccc(
    a, b, c, d, e, f, g, h, i, j, k, l,
    size, mode, threads, compute, gpu_piece_size, cpu_piece_size, nstreams,
):
    import torch
    n = gpu_piece_size
    ais,bis,cis,dis,eis,fis,gis,his,iis,jis,kis,lis = [],[],[],[],[],[],[],[],[],[],[],[]
    start = time.time()
    streams = [torch.cuda.Stream() for _ in range(nstreams)]
    print('Create streams:', time.time() - start)
    start = time.time()
    call = torch.empty(len(a), dtype=torch.float64)
    put = torch.empty(len(a), dtype=torch.float64)
    print('Allocate cpu arrays:', time.time() - start)
    for index in range(0, int(size/n)):
        m = index * n
        s = streams[index % nstreams]
        with torch.cuda.stream(s):
            ai, bi, ci, di, ei = a[m:m+n], b[m:m+n], c[m:m+n], d[m:m+n], e[m:m+n]
            fi, gi, hi, ii, ji = f[m:m+n], g[m:m+n], h[m:m+n], i[m:m+n], j[m:m+n]
            ki, li = k[m:m+n], l[m:m+n]
            ai,bi,ci,di,ei = transfer_to([ai, bi, ci, di, ei])
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
        s = streams[m % nstreams]
        with torch.cuda.stream(s):
            bs(
                ais[m], bis[m], cis[m], dis[m], eis[m],
                fis[m], gis[m], his[m], iis[m], jis[m], kis[m], lis[m],
                mode, threads, compute, gpu_piece_size, cpu_piece_size
            )
    for m in range(0, int(size/n)):
        s = streams[m % nstreams]
        with torch.cuda.stream(s):
            kis[m], lis[m] = transfer_from([kis[m], lis[m]])
            call[m:m+n]=kis[m][:]
            put[m:m+n]=lis[m][:]
    torch.cuda.synchronize()
    return (call, put)

def run(args):
    import torch
    size = (1 << args.size)
    gpu_piece_size = 1<<args.gpu_piece_size
    cpu_piece_size = 1<<args.cpu_piece_size
    threads = args.threads
    loglevel = args.verbosity
    nstreams = args.streams
    mode = args.mode.strip().lower()
    allocation = args.allocation.strip().lower()
    compute = args.compute.strip().lower()
    reuse_memory = args.reuse_memory

    assert threads >= 1

    print("Size:", size)
    print("GPU Piece Size:", gpu_piece_size)
    print("CPU Piece Size:", cpu_piece_size)
    print("Threads:", threads)
    print("Streams:", nstreams)
    print("Log Level", loglevel)
    print("Mode:", mode)
    print("Allocation:", allocation)
    print("Compute:", compute)
    print('------------------------------------------------------')

    # Parse the mode
    if mode == 'composer':
        mode = Mode.COMPOSER
    elif mode == 'naive':
        mode = Mode.NAIVE
    elif mode == 'aaabbbccc':
        mode = Mode.AAABBBCCC
    elif mode == 'abcabcabc':
        mode = Mode.ABCABCABC
    elif mode == 'abc':
        mode = Mode.ABC
    else:
        raise ValueError("invalid mode", mode)

    # Parse the allocation and compute backend
    if allocation not in ['cpu', 'cuda']:
        raise ValueError("invalid allocation backend", allocation)
    if compute not in ['cpu', 'cuda']:
        raise ValueError("invalid compute backend", compute)

    start = time.time()
    a, b, c, d, e = get_inputs(size, mode, allocation)
    f, g, h, i, j, k, l = get_tmp_arrays(size, mode, compute, reuse_memory, gpu_piece_size)
    print("Initialization: {}s".format(time.time() - start))

    torch.cuda.synchronize()
    start = time.time()
    n = gpu_piece_size

    if mode == Mode.ABCABCABC:
        call, put = run_abcabcabc(
            a, b, c, d, e, f, g, h, i, j, k, l,
            size, mode, threads, compute, gpu_piece_size, cpu_piece_size, nstreams
        )
    elif mode == Mode.AAABBBCCC:
        call, put = run_aaabbbccc(
            a, b, c, d, e, f, g, h, i, j, k, l,
            size, mode, threads, compute, gpu_piece_size, cpu_piece_size, nstreams
        )
    elif mode == Mode.ABC:
        if reuse_memory:
            func = run_abc_reuse_memory
        else:
            func = run_abc
        call, put = func(
            a, b, c, d, e, f, g, h, i, j, k, l,
            size, mode, threads, compute, gpu_piece_size, cpu_piece_size
        )
    elif mode == Mode.COMPOSER:
        call, put = bs(a, b, c, d, e, f, g, h, i, j, k, l, mode, threads, compute, gpu_piece_size, cpu_piece_size)
    else:
        a,b,c,d,e = transfer_to(a, b, c, d, e, mode, compute)
        bs(a, b, c, d, e, f, g, h, i, j, k, l, mode, threads, compute, gpu_piece_size, cpu_piece_size)
        call, put = transfer_from(k, l, mode)

    runtime = time.time() - start
    print('------------------------------------------------------')
    print("Total Runtime: {}s".format(runtime))
    print("Call (len {}): {}".format(len(call), call))
    print("Put (len {}): {}".format(len(put), put))
    return runtime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chained Adds pipelining test on a single thread."
    )
    parser.add_argument('-s', "--size", type=int, default=27, help="Size of each array")
    parser.add_argument('-cpu', "--cpu_piece_size", type=int, default=14, help="Log size of each CPU piece.")
    parser.add_argument('-gpu', "--gpu_piece_size", type=int, default=19, help="Log size of each GPU piece.")
    parser.add_argument('-t', "--threads", type=int, default=1, help="Number of threads.")
    parser.add_argument('-streams', type=int, default=16, help="Number of streams.")
    parser.add_argument('-v', "--verbosity", type=str, default="none", help="Log level (debug|info|warning|error|critical|none)")
    parser.add_argument('-m', "--mode", type=str, required=True, help="Mode (naive|composer)")
    parser.add_argument('-a', "--allocation", type=str, default="cpu", help="Allocation backend (cpu|cuda)")
    parser.add_argument('-c', "--compute", type=str, default="cuda", help="Compute backend (cpu|cuda)")
    parser.add_argument('--reuse_memory', action='store_true', help='Whether to reuse arrays for each piece per stream.')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials.')
    args = parser.parse_args()

    res = [run(args) for _ in range(args.trials)]
    if args.trials > 1:
        m = int(len(res) / 2)
        if args.trials % 2 == 1:
            print('Median:', res[m])
        else:
            print('Median:', (res[m] + res[m-1]) / 2)
