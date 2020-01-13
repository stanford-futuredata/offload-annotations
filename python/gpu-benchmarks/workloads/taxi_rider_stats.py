#!/usr/bin/python
import subprocess
import sys
import time

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask_cudf
import pandas as pd
from sa.annotation import Backend
from mode import Mode


DEFAULT_NUM_LINES = 1 << 15
DEFAULT_CPU = 0
DEFAULT_GPU = 0
MAX_BATCH_SIZE = 0

filename = '/lfs/1/deepak/nyc_taxi/split-annotations/python/gpu-benchmarks/workloads/nyc_taxi.csv'


def _read_data_cuda(filename):
    df = dask_cudf.read_csv(filename, parse_dates=['tpep_pickup_datetime'])
    df.persist()
    return df

def read_data(mode, tmp_filename):
    if mode == Mode.CUDA:
        df = _read_data_cuda(tmp_filename)
    else:
        df = pd.read_csv(tmp_filename, parse_dates=['tpep_pickup_datetime'])

    return df

def run_naive(df):
    # Average trip distance, grouped by passenger count.
    df.groupby('passenger_count').trip_distance.mean()
    # Average tip fraction, grouped by hour of trip.
    df2 = df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount']]
    df2 = df2.query('tip_amount > 0 and fare_amount > 0')
    df2['hour'] = df2.tpep_pickup_datetime.dt.hour
    df2['tip_fraction'] = df2.tip_amount / df2.fare_amount
    mean = df2.groupby('hour').tip_fraction.mean()
    return mean

def run_cuda(df):
    # Initialize dask cluster
    # start = time.time()
    # cluster = LocalCUDACluster(n_workers=4)
    # dask_init_time = time.time() - start
    # sys.stdout.write('Initialization(dask): {}\n'.format(dask_init_time))

    # Average trip distance, grouped by passenger count.
    df.groupby('passenger_count').trip_distance.mean().compute()
    # Average tip fraction, grouped by hour of trip.
    df2 = df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount']]
    df2 = df2.query('tip_amount > 0 and fare_amount > 0')
    df2['hour'] = df2.tpep_pickup_datetime.dt.hour
    df2['tip_fraction'] = df2.tip_amount / df2.fare_amount
    mean = df2.groupby('hour').tip_fraction.mean().compute().to_pandas()
    return mean

def run_composer(mode, df, batch_size, threads):
    pass

def run(mode, size=None, cpu=None, gpu=None, threads=None):
    # Optimal defaults
    if size == None:
        size = DEFAULT_NUM_LINES
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        threads = 1

    batch_size = {
        Backend.CPU: min(cpu, max(1, int(size / threads))),
        Backend.GPU: min(gpu, MAX_BATCH_SIZE),
    }

    # Get inputs
    tmp_filename = 'nyc_taxi_tmp.csv'
    if size is None:
        subprocess.call('cat %s > %s' % (filename, tmp_filename), shell=True)
    else:
        subprocess.call('head -n %d %s > %s' % (size, filename, tmp_filename),
            shell=True)

    start = time.time()
    df = read_data(mode, tmp_filename)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    start = time.time()
    if mode.is_composer():
        result = run_composer(mode, df, batch_size, threads)
    elif mode == Mode.NAIVE:
        result = run_naive(df)
    elif mode == Mode.CUDA:
        result = run_cuda(df)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(result)
    return init_time, runtime


if __name__ == '__main__':
    for num_lines in [10, 100, 1000, 10000, 100000, 1000000, None]:
        print("Number of lines:", num_lines)
        print('=' * 40)
        print('CUDA:')
        print('=' * 40)
        run_cuda(num_lines)
        print('=' * 40)
        print('Naive:')
        print('=' * 40)
        run_naive(num_lines)
        print('=' * 80)
