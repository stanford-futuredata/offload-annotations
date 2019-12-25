#!/usr/bin/python
import os
import sys
import time
import cudf
import numpy as np

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

prefix = '../benchmarks/crime_index/'
filenames = ['total_population.csv', 'adult_population.csv', 'num_robberies.csv']
filenames = [prefix + f for f in filenames]
values = [500000, 250000, 1000]

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def _write_data(size, filenames=filenames):
    for i, filename in enumerate(filenames):
        if os.path.exists(filename):
            # os.remove(filename)
            print('File {} already exists.'.format(filename))
            continue
        sys.stdout.write('Writing {}...'.format(filename))
        sys.stdout.flush()
        with open(filename, 'w+') as f:
            f.write('\n'.join([str(values[i]) for _ in range(size)]))
            f.write('\n')
        print('done.')


def _read_data_cuda(filenames):
    total_population = cudf.read_csv(filenames[0], header=None).iloc[:,0]
    adult_population = cudf.read_csv(filenames[1], header=None).iloc[:,0]
    num_robberies = cudf.read_csv(filenames[2], header=None).iloc[:,0]
    return total_population, adult_population, num_robberies


def read_data(mode, filenames=filenames):
    if mode == Mode.CUDA:
        return _read_data_cuda(filenames)

    if mode.is_composer():
        import sa.annotated.pandas as pd
    else:
        import pandas as pd

    total_population = pd.read_csv(filenames[0], squeeze=True, header=None)
    adult_population = pd.read_csv(filenames[1], squeeze=True, header=None)
    num_robberies = pd.read_csv(filenames[2], squeeze=True, header=None)
    return total_population, adult_population, num_robberies


def _gen_data_cuda(size):
    total_population = np.ones(size, dtype="float64") * 500000
    adult_population = np.ones(size, dtype="float64") * 250000
    num_robberies = np.ones(size, dtype="float64") * 1000
    return cudf.Series(total_population), cudf.Series(adult_population), cudf.Series(num_robberies)


def gen_data(mode, size):
    if mode == Mode.CUDA:
        return _gen_data_cuda(size)

    if mode.is_composer():
        import sa.annotated.pandas as pd
    else:
        import pandas as pd

    total_population = np.ones(size, dtype="float64") * 500000
    adult_population = np.ones(size, dtype="float64") * 250000
    num_robberies = np.ones(size, dtype="float64") * 1000
    return pd.Series(total_population), pd.Series(adult_population), pd.Series(num_robberies)


def run_composer(
    mode,
    total_population,
    adult_population,
    num_robberies,
    batch_size,
    threads,
):
    import sa.annotated.pandas as pd
    if mode == Mode.MOZART:
        force_cpu = True
    elif mode == Mode.BACH:
        force_cpu = False
    else:
        raise Exception

    # Get all city information with total population greater than 500,000
    big_cities = pd.greater_than(total_population, 500000.0)
    big_cities = pd.mask(total_population, big_cities, 0.0)

    double_pop = pd.multiply(adult_population, 2.0)
    double_pop = pd.add(big_cities, double_pop)
    multiplied = pd.multiply(num_robberies, 2000.0)
    double_pop = pd.subtract(double_pop, multiplied)
    crime_index = pd.divide(double_pop, 100000.0)

    gt = pd.greater_than(crime_index, 0.02)
    crime_index = pd.mask(crime_index, gt, 0.032)
    lt = pd.less_than(crime_index, 0.01)
    crime_index = pd.mask(crime_index, lt, 0.005)

    result = pd.pandasum(crime_index)
    result.dontsend = False
    pd.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    return result.value


def run_naive(total_population, adult_population, num_robberies):
    big_cities = total_population > 500000
    big_cities = total_population.mask(big_cities, 0.0)
    double_pop = adult_population * 2 + big_cities - (num_robberies * 2000.0)
    crime_index = double_pop / 100000
    crime_index = crime_index.mask(crime_index > 0.02, 0.032)
    crime_index = crime_index.mask(crime_index < 0.01, 0.005)
    return crime_index.sum()


def run_cuda(total_population, adult_population, num_robberies):
    def mask(series, cond, val):
        clone = series.copy()
        clone.loc[cond] = val
        return clone

    big_cities = total_population > 500000
    big_cities = mask(total_population, big_cities, 0.0)
    double_pop = adult_population * 2 + big_cities - (num_robberies * 2000.0)
    crime_index = double_pop / 100000
    crime_index = mask(crime_index, crime_index > 0.02, 0.032)
    crime_index = mask(crime_index, crime_index < 0.01, 0.005)
    return crime_index.sum()


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        if mode == Mode.MOZART:
            threads = 16
        else:
            threads = 1

    # Initialize data
    if data_mode == 'file':
        _write_data(size)
    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    # Get inputs
    start = time.time()
    if data_mode == 'generated':
        inputs = gen_data(mode, size)
    elif data_mode == 'file':
        inputs = read_data(mode)
    else:
        raise ValueError
    init_time = time.time() - start
    print("Get {} data: {}".format(data_mode, init_time))

    start = time.time()
    if mode.is_composer():
        result = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        result = run_naive(*inputs)
    elif mode == Mode.CUDA:
        result = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    print('Total:', init_time + runtime)
    print(result)
    return init_time, runtime
