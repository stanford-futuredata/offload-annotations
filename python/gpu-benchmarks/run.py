import argparse
import enum
import sys

sys.path.append("../lib/")
sys.path.append("../pycomposer/")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode
from workloads import *

class Benchmark(enum.Enum):
    BIRTH_ANALYSIS = 0
    BLACKSCHOLES_NUMPY = 1
    BLACKSCHOLES_TORCH = 2
    CRIME_INDEX = 3
    # IMAGE_PROCESSING = 4
    # LINEAR_REGRESSION = 5
    # PREPROCESSING = 6
    # XGBOOST = 7


def to_function(bm):
    if bm == Benchmark.BIRTH_ANALYSIS:
        return birth_analysis.run
    elif bm == Benchmark.BLACKSCHOLES_NUMPY:
        return blackscholes_numpy.run
    elif bm == Benchmark.BLACKSCHOLES_TORCH:
        return blackscholes_torch.run
    elif bm == Benchmark.CRIME_INDEX:
        return crime_index.run
    else:
        raise Exception


def median(arr):
    arr.sort()
    m = int(len(arr) / 2)
    if len(arr) % 2 == 1:
        return arr[m]
    else:
        return (arr[m] + arr[m-1]) / 2


def run(ntrials, bm, mode, size, cpu, gpu, threads):
    print('Trials:', ntrials)
    print('Benchmark:', bm.name.lower())
    print('Mode:', mode.name.lower())
    print('Size:', size)
    print('CPU piece size:', cpu)
    print('GPU piece size:', gpu)
    print('Threads:', threads)

    bm_func = to_function(bm)

    init_times = []
    runtimes = []
    for _ in range(ntrials):
        init_time, runtime = bm_func(mode, size, cpu, gpu, threads)
        init_times.append(init_time)
        runtimes.append(runtime)
    print('Median Init:', median(init_times))
    print('Median Runtime:', median(runtimes))
    print('Median Total:', median(init_times) + median(runtimes))


if __name__ == '__main__':
    bm_names = [bm.name.lower() for bm in Benchmark]
    mode_names = [mode.name.lower() for mode in Mode]

    parser = argparse.ArgumentParser('Benchmark for accelerator-aware split annotations.')
    parser.add_argument('-b', '--benchmark', type=str, required=True,
        help='Benchmark name ({}) or (0-{})'.format('|'.join(bm_names), len(Benchmark) - 1))
    parser.add_argument('-m', '--mode', type=str, required=True, help='Mode (naive|mozart|bach|cuda)')
    parser.add_argument('-s', '--size', type=int, help='Log data size')
    parser.add_argument('--cpu', type=int, help='Log CPU piece size')
    parser.add_argument('--gpu', type=int, help='Log GPU piece size in bach or stream size in cuda')
    parser.add_argument('--threads', type=int, help='Number of threads (naive only)')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials')
    args = parser.parse_args()

    # Parse benchmark
    bm = args.benchmark.lower()
    if bm.isdigit():
        bm = Benchmark(int(bm))
    elif bm in bm_names:
        bm = Benchmark(bm_names.index(bm))
    else:
        raise ValueError('Invalid benchmark:', bm)

    # Parse mode
    mode = args.mode.lower()
    if mode in mode_names:
        mode = Mode(mode_names.index(mode))
    else:
        raise ValueError('Invalid mode:', mode)

    # Parse other arguments
    size = None if args.size is None else 1 << args.size
    cpu = None if args.cpu is None else 1 << args.cpu
    gpu = None if args.gpu is None else 1 << args.gpu
    threads = args.threads
    ntrials = args.trials
    assert ntrials > 0

    run(ntrials, bm, mode, size, cpu, gpu, threads)
