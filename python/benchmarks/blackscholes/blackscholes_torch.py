
import sys
sys.path.append("../../lib/")
sys.path.append("../../pycomposer/")

import argparse
import math
import time
from enum import Enum

class Mode(Enum):
    TORCH_CPU = 0
    TORCH_COMPOSER = 1
    TORCH_CUDA = 2
    TORCH_MANUALCUDA = 3

def get_data(size, mode, threads):
    if mode == Mode.TORCH_COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    torch.set_num_threads(threads);
    if mode == Mode.TORCH_CUDA:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float64

    price = torch.ones(size, device=device, dtype=dtype) * 4.0
    strike = torch.ones(size, device=device, dtype=dtype) * 4.0
    t = torch.ones(size, device=device, dtype=dtype) * 4.0
    rate = torch.ones(size, device=device, dtype=dtype) * 4.0
    vol = torch.ones(size, device=device, dtype=dtype) * 4.0

    if mode == Mode.TORCH_MANUALCUDA:
        device = torch.device('cuda')
    tmp = torch.ones(size, device=device, dtype=dtype)
    vol_sqrt = torch.ones(size, device=device, dtype=dtype)
    rsig = torch.ones(size, device=device, dtype=dtype)
    d1 = torch.ones(size, device=device, dtype=dtype)
    d2 = torch.ones(size, device=device, dtype=dtype)

    # Outputs
    call = torch.ones(size, device=device, dtype=dtype)
    put = torch.ones(size, device=device, dtype=dtype)

    return price, strike, t, rate, vol, tmp, vol_sqrt, rsig, d1, d2, call, put

def bs(
    price, strike, t, rate, vol,  # original data
    tmp, vol_sqrt, rsig, d1, d2,  # temporary arrays
    call, put,                    # outputs
    mode, threads, piece_size     # experiment configuration
):
    if mode == Mode.TORCH_COMPOSER:
        import sa.annotated.torch as torch
        torch.set_num_threads(1)
    else:
        import torch
        torch.set_num_threads(threads)

    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)

    start = time.time()

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

    if mode == Mode.TORCH_COMPOSER:
        end = time.time()
        print("Build time:", end - start)
        torch.evaluate(workers=threads, batch_size=piece_size)

    return call, put

def manual_bs(
    price, strike, t, rate, vol,  # original data
    tmp, vol_sqrt, rsig, d1, d2,  # temporary arrays
    call, put,                    # outputs
    mode, threads, piece_size     # experiment configuration
):
    import torch
    torch.set_num_threads(threads)

    c05 = 3.0
    c10 = 1.5
    invsqrt2 = 1.0 / math.sqrt(2.0)

    start = time.time()
    cuda = torch.device('cuda')
    price = price.to(cuda)
    strike = strike.to(cuda)
    t = t.to(cuda)
    rate = rate.to(cuda)
    vol = vol.to(cuda)
    print('To device:', time.time() - start)

    start = time.time()
    call, put = bs(price, strike, t, rate, vol,
                   tmp, vol_sqrt, rsig, d1, d2,
                   call, put,
                   mode, threads, piece_size)
    torch.cuda.synchronize()
    print('Compute:', time.time() - start)

    start = time.time()
    cpu = torch.device('cpu')
    call = call.to(cpu)
    put = put.to(cpu)
    print('To host:', time.time() - start)
    return call, put


def run():
    parser = argparse.ArgumentParser(
        description="Chained Adds pipelining test on a single thread."
    )
    parser.add_argument('-s', "--size", type=int, default=27, help="Size of each array")
    parser.add_argument('-p', "--piece_size", type=int, default=16384, help="Size of each piece.")
    parser.add_argument('-t', "--threads", type=int, default=1, help="Number of threads.")
    parser.add_argument('-v', "--verbosity", type=str, default="none", help="Log level (debug|info|warning|error|critical|none)")
    parser.add_argument('-m', "--mode", type=str, required=True, help="Mode (naive|composer|cuda|manualcuda)")
    parser.add_argument('-a', "--allocation", type=str, default="single", help="Mode (single|multi)")
    args = parser.parse_args()

    size = (1 << args.size)
    piece_size = args.piece_size
    threads = args.threads
    loglevel = args.verbosity
    mode = args.mode.strip().lower()
    alloc = args.allocation.strip().lower()

    assert threads >= 1

    print("Size:", size)
    print("Piece Size:", piece_size)
    print("Threads:", threads)
    print("Log Level", loglevel)
    print("Mode:", mode)
    print("Allocation:", alloc)

    # Parse the mode
    if mode == "naive":
        mode = Mode.TORCH_CPU
    elif mode == "composer":
        mode = Mode.TORCH_COMPOSER
    elif mode == "cuda":
        mode = Mode.TORCH_CUDA
    elif mode == "manualcuda":
        mode = Mode.TORCH_MANUALCUDA
    else:
        raise ValueError("invalid mode", mode)

    # Parse the allocation type
    if alloc == "single":
        alloc_threads = 1
    elif alloc == "multi":
        alloc_threads = 16
    else:
        raise ValueError("invalid allocation type", alloc)

    start = time.time()
    sys.stdout.write("Initializing...")
    sys.stdout.flush()
    a, b, c, d, e, f, g, h, i, j, k, l = get_data(size, mode, alloc_threads)
    print("done: {}s".format(time.time() - start))

    start = time.time()
    if mode == Mode.TORCH_MANUALCUDA:
        call, put = manual_bs(a, b, c, d, e, f, g, h, i, j, k, l, mode, threads, piece_size)
    else:
        call, put = bs(a, b, c, d, e, f, g, h, i, j, k, l, mode, threads, piece_size)

    print("Runtime: {}s".format(time.time() - start))
    print("Call (len {}): {}".format(len(call), call))
    print("Put (len {}): {}".format(len(put), put))

if __name__ == "__main__":
    run()
