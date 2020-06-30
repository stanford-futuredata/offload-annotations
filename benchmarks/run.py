import argparse
import enum
import sys

sys.path.append("../pycomposer/")
sys.path.append("./pycomposer/")

from sa.annotation import Backend
from mode import Mode
from workloads import *

class Benchmark(enum.Enum):
    BLACKSCHOLES_TORCH = 0
    BLACKSCHOLES_CUPY = 1
    CRIME_INDEX = 2
    TSVD = 3
    PCA = 4
    DBSCAN = 5
    HAVERSINE_TORCH = 6
    HAVERSINE_CUPY = 7


def to_function(bm):
    if bm == Benchmark.BLACKSCHOLES_TORCH:
        return blackscholes.run_torch_main
    elif bm == Benchmark.BLACKSCHOLES_CUPY:
        return blackscholes.run_cupy_main
    elif bm == Benchmark.CRIME_INDEX:
        return crime_index.run
    elif bm == Benchmark.TSVD:
        return tsvd.run
    elif bm == Benchmark.PCA:
        return pca.run
    elif bm == Benchmark.DBSCAN:
        return dbscan.run
    elif bm == Benchmark.HAVERSINE_TORCH:
        return haversine.run_torch_main
    elif bm == Benchmark.HAVERSINE_CUPY:
        return haversine.run_cupy_main
    else:
        raise Exception


def median(arr):
    arr.sort()
    m = int(len(arr) / 2)
    if len(arr) % 2 == 1:
        return arr[m]
    else:
        return (arr[m] + arr[m-1]) / 2


def run(ntrials, bm, mode, size):
    print('Trials:', ntrials)
    print('Benchmark:', bm.name.lower())
    print('Mode:', mode.name.lower())
    print('Size:', size)

    bm_func = to_function(bm)

    init_times = []
    runtimes = []
    for _ in range(ntrials):
        init_time, runtime = bm_func(mode, size)
        init_times.append(init_time)
        runtimes.append(runtime)
    print('Median Init:', median(init_times))
    print('Median Runtime:', median(runtimes))
    print('Median Total:', median(init_times) + median(runtimes))


if __name__ == '__main__':
    bm_names = [bm.name.lower() for bm in Benchmark]
    mode_names = [mode.name.lower() for mode in Mode]

    parser = argparse.ArgumentParser('Benchmark for offload annotations.')
    parser.add_argument('-b', '--benchmark', type=str, required=True,
        help='Benchmark name ({}) or (0-{})'.format('|'.join(bm_names), len(Benchmark) - 1))
    parser.add_argument('-m', '--mode', type=str, required=True, help='Mode (cpu|gpu|bach)')
    parser.add_argument('-s', '--size', type=int, help='Log2 data size')
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
    ntrials = args.trials
    assert ntrials > 0

    run(ntrials, bm, mode, size)
