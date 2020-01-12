import sys
import time

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask_cudf
import pandas as pd

def run_naive():
    start = time.time()
    df = pd.read_csv('nyc_taxi.csv', parse_dates=['tpep_pickup_datetime'])
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    start = time.time()
    # Average trip distance, grouped by passenger count.
    df.groupby('passenger_count').trip_distance.mean()
    # Average tip fraction, grouped by hour of trip.
    df2 = df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount']]
    df2 = df2.query('tip_amount > 0 and fare_amount > 0')
    df2['hour'] = df2.tpep_pickup_datetime.dt.hour
    df2['tip_fraction'] = df2.tip_amount / df2.fare_amount
    df2.groupby('hour').tip_fraction.mean()
    runtime = time.time() - start
    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))

def run_cuda():
    cluster = LocalCUDACluster(ip='0.0.0.0',n_workers=4, device_memory_limit='10000 MiB')
    client = Client(cluster)

    start = time.time()
    df = dask_cudf.read_csv('nyc_taxi.csv', parse_dates=['tpep_pickup_datetime'])
    df.persist()
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    start = time.time()
    # Average trip distance, grouped by passenger count.
    df.groupby('passenger_count').trip_distance.mean().compute()
    # Average tip fraction, grouped by hour of trip.
    df2 = df[['tpep_pickup_datetime', 'trip_distance', 'tip_amount', 'fare_amount']]
    df2 = df2.query('tip_amount > 0 and fare_amount > 0')
    df2['hour'] = df2.tpep_pickup_datetime.dt.hour
    df2['tip_fraction'] = df2.tip_amount / df2.fare_amount
    df2.groupby('hour').tip_fraction.mean().compute().to_pandas()
    runtime = time.time() - start
    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))


if __name__ == '__main__':
    run_cuda()
    run_naive()
