import sys
sys.path.append('../../lib/')
sys.path.append('../../pycomposer/')

import argparse
import time
from enum import Enum

from sa.annotation import Backend


class Mode(Enum):
    NAIVE = 0
    COMPOSER = 1


def arrays(size, mode):
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    dtype = torch.float64
    a = torch.ones(size, dtype=dtype, device=torch.device('cpu')) * 2.0
    b = torch.ones(size, dtype=dtype, device=torch.device('cpu')) * 3.0
    c = torch.empty(size, dtype=dtype)
    d = torch.empty(size, dtype=dtype)
    d.materialize = Backend.CPU
    return (a, b, c, d)

def verify(d, size):
    assert len(d) == size
    assert d[0] == 7.0
    assert d[-1] == 7.0
    assert d.device.type == 'cpu'

def run(args):
    """
    a + b = c
    a + c = d
    print(d)
    """
    size = (1 << args.size)
    batch_size = {}
    batch_size[Backend.GPU] = 1<<args.gpu_piece_size
    batch_size[Backend.CPU] = 1<<args.cpu_piece_size
    threads = args.threads
    mode = args.mode.strip().lower()

    assert threads >= 1

    print('Size: 1 << {}'.format(args.size))
    print('GPU Piece Size: 1 << {}'.format(args.gpu_piece_size))
    print('CPU Piece Size: 1 << {}'.format(args.cpu_piece_size))
    print('Mode:', mode)
    print('------------------------------------------------------')

    # Parse the mode
    if mode == 'composer':
        mode = Mode.COMPOSER
    elif mode == 'naive':
        mode = Mode.NAIVE
    else:
        raise ValueError("invalid mode", mode)

    # Import torch
    if mode == Mode.COMPOSER:
        import sa.annotated.torch as torch
    else:
        import torch

    start = time.time()
    a, b, c, d = arrays(size, mode)
    if mode == Mode.COMPOSER:
        torch.cuda.synchronize()
    init_time = time.time() - start
    print('Initialization:', init_time)

    start = time.time()
    torch.add(a, b, out=c)
    torch.add(a, c, out=d)
    if mode == Mode.COMPOSER:
        torch.evaluate(workers=threads, batch_size=batch_size)
        torch.cuda.synchronize()
    runtime = time.time() - start
    print('Runtime:', runtime)

    print('------------------------------------------------------')
    print(d.value)
    verify(d.value, size)
    print('Verified!')
    return runtime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=25, help='Size of each array')
    parser.add_argument('-cpu', '--cpu_piece_size', type=int, default=14, help='Log size of each CPU piece.')
    parser.add_argument('-gpu', '--gpu_piece_size', type=int, default=19, help='Log size of each GPU piece.')
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads')
    parser.add_argument('-m', '--mode', type=str, required=True, help='Mode (naive|composer)')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials.')
    args = parser.parse_args()

    res = [run(args) for _ in range(args.trials)]
    if args.trials > 1:
        m = int(len(res) / 2)
        if args.trials % 2 == 1:
            print('Median:', res[m])
        else:
            print('Median:', (res[m] + res[m-1]) / 2)
