#!/usr/bin/python
import subprocess
import sys
import time

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask_cudf
import pandas as pd
from sa.annotation import Backend
from mode import Mode
import dask.dataframe as dd
import dask

DEFAULT_NUM_LINES = 1 << 15
MAX_BATCH_SIZE = 1 << 30

filename = '/lfs/1/deepak/nyc_taxi/split-annotations/python/gpu-benchmarks/workloads/nyc_taxi.csv'
file_size = 59626358


def _write_tmp_file(tmp_filename, size):
    subprocess.call('rm %s' % tmp_filename, shell=True)
    if size is None:
        subprocess.call('cat %s >> %s' % (filename, tmp_filename), shell=True)
    elif size <= file_size:
        subprocess.call('head -n %d %s >> %s' % (size, filename, tmp_filename), shell=True)
    else:
        d = int(size / file_size)
        mod = size - file_size * d
        for _ in range(d):
            subprocess.call('cat %s >> %s' % (filename, tmp_filename), shell=True)
        subprocess.call('head -n %d %s >> %s' % (mod, filename, tmp_filename), shell=True)

def _read_data_cuda(filename):
    df = dask_cudf.read_csv(filename, parse_dates=['tpep_pickup_datetime'])
    df.persist()
    return df

def read_data(mode, tmp_filename):
    if mode == Mode.CUDA:
        df = _read_data_cuda(tmp_filename)
    elif mode.is_composer():
        import sa.annotated.dask as dask
        df = dask.read_csv(tmp_filename, parse_dates=['tpep_pickup_datetime'])
    else:
        df = dd.read_csv(tmp_filename, parse_dates=['tpep_pickup_datetime'])

    return df

def run_naive(df):
    # Average trip distance, grouped by passenger count.
    distance = df.groupby('passenger_count').trip_distance.mean().compute()
    # Average tip fraction, grouped by hour of trip.
    df2 = df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount']]
    df2 = df2.query('tip_amount > 0 and fare_amount > 0')
    df2['hour'] = df2.tpep_pickup_datetime.dt.hour
    df2['tip_fraction'] = df2.tip_amount / df2.fare_amount
    tip_mean = df2.groupby('hour').tip_fraction.mean().compute()
    return distance, tip_mean

def run_cuda(df):
    # Initialize dask cluster
    # start = time.time()
    # cluster = LocalCUDACluster(n_workers=4)
    # dask_init_time = time.time() - start
    # sys.stdout.write('Initialization(dask): {}\n'.format(dask_init_time))

    # Average trip distance, grouped by passenger count.
    distance = df.groupby('passenger_count').trip_distance.mean().compute()
    # Average tip fraction, grouped by hour of trip.
    df2 = df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount']]
    df2 = df2.query('tip_amount > 0 and fare_amount > 0')
    df2['hour'] = df2.tpep_pickup_datetime.dt.hour
    df2['tip_fraction'] = df2.tip_amount / df2.fare_amount
    tip_mean = df2.groupby('hour').tip_fraction.mean().compute().to_pandas()
    return distance, tip_mean

def run_composer(mode, df, batch_size, threads):
    import sa.annotated.dask as dask
    force_cpu = mode == Mode.MOZART

    # Average trip distance, grouped by passenger count.
    tmp = dask.groupby(df, 'passenger_count')
    tmp = dask.index(tmp, 'trip_distance')
    tmp = dask.mean(tmp)
    distance = dask.compute(tmp)
    # Average tip fraction, grouped by hour of trip.
    tmp = dask.index(df, ['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount'])
    tmp = dask.query(tmp, 'tip_amount > 0 and fare_amount > 0')

    tmp1 = dask.index(tmp, 'tpep_pickup_datetime')
    tmp1 = dask.dt(tmp1)
    tmp1 = dask.hour(tmp1)
    dask.set(tmp, 'hour', tmp1)

    tmp1 = dask.index(tmp, 'tip_amount')
    tmp2 = dask.index(tmp, 'fare_amount')
    tmp1 = dask.divide(tmp1, tmp2)
    dask.set(tmp, 'tip_fraction', tmp1)

    tmp = dask.groupby(tmp, 'hour')
    tmp = dask.index(tmp, 'tip_fraction')
    tmp = dask.mean(tmp)
    tip_mean = dask.compute(tmp)

    distance.materialize = Backend.CPU
    tip_mean = Backend.CPU
    dask.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    return distance.value, tip_mean.value

def run(mode, size=None, cpu=None, gpu=None, threads=None):
    # Optimal defaults
    if size == None:
        size = DEFAULT_NUM_LINES
    if threads is None:
        threads = 1

    batch_size = {
        Backend.CPU: size,
        Backend.GPU: size,
    }

    # Get inputs
    tmp_filename = 'nyc_taxi_tmp.csv'
    _write_tmp_file(tmp_filename, size)

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
