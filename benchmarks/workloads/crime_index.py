#!/usr/bin/python
import os
import sys
import time
import math
import cudf
import numpy as np

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

prefix = 'datasets/crime_index'
filenames = ['total_population', 'adult_population', 'num_robberies']
values = [500000, 250000, 1000]

DEFAULT_SIZE = 1 << 26
DEFAULT_CPU = 1 << 25
MAX_BATCH_SIZE = 1 << 26


def _write_data(size, filenames=filenames):
    """Writes input files to disk.

    Titled datasets/crime_index/total_population_${log(size)}.csv, etc.
    Writes the file only if a file with the name doesn't already exist.

    Parameters:
    - size: number of rows
    """
    for i, filename in enumerate(filenames):
        filename = '{}/{}_{}.csv'.format(prefix, filename, int(math.log2(size)))
        if os.path.exists(filename):
            continue
        sys.stdout.write('Writing {}...'.format(filename))
        sys.stdout.flush()
        with open(filename, 'w+') as f:
            f.write('\n'.join([str(values[i]) for _ in range(size)]))
            f.write('\n')
        print('done.')


def read_data(mode, size=None, filenames=filenames):
    """Reads data from disk.

    Parameters:
    - size: number of rows
    """
    if size is None:
        fs = filenames
    else:
        fs = ['{}/{}_{}.csv'.format(prefix, filename, int(math.log2(size))) for filename in filenames]
    if mode == Mode.CUDA:
        total_population = cudf.read_csv(fs[0], header=None).iloc[:,0]
        adult_population = cudf.read_csv(fs[1], header=None).iloc[:,0]
        num_robberies = cudf.read_csv(fs[2], header=None).iloc[:,0]
    else:
        if mode == Mode.BACH:
            import sa.annotated.pandas as pd
        elif mode == Mode.NAIVE:
            import pandas as pd
        total_population = pd.read_csv(fs[0], squeeze=True, header=None)
        adult_population = pd.read_csv(fs[1], squeeze=True, header=None)
        num_robberies = pd.read_csv(fs[2], squeeze=True, header=None)

    # Validate data
    if size is not None and mode != Mode.BACH:
        assert len(total_population) == size
        assert len(adult_population) == size
        assert len(num_robberies) == size
    return total_population, adult_population, num_robberies


def run_bach_cudf(
    size,
    total_population,
    adult_population,
    num_robberies,
):
    import sa.annotated.pandas as pd

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
    result.materialize = Backend.CPU
    pd.evaluate(
        workers=1,
        batch_size={
            Backend.CPU: DEFAULT_CPU,
            Backend.GPU: MAX_BATCH_SIZE,
        },
        force_cpu=False,
        paging=size > MAX_BATCH_SIZE,
    )
    return result.value


def run_pandas(total_population, adult_population, num_robberies):
    big_cities = total_population > 500000
    big_cities = total_population.mask(big_cities, 0.0)
    double_pop = adult_population * 2 + big_cities - (num_robberies * 2000.0)
    crime_index = double_pop / 100000
    crime_index = crime_index.mask(crime_index > 0.02, 0.032)
    crime_index = crime_index.mask(crime_index < 0.01, 0.005)
    return crime_index.sum()


def run_cudf(total_population, adult_population, num_robberies):
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


def run(mode, size=None, cpu=None, gpu=None, threads=1):
    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE

    # Initialize data
    _write_data(size)

    # Get inputs
    start = time.time()
    inputs = read_data(mode, size)
    init_time = time.time() - start
    print("Init: {}".format(init_time))

    start = time.time()
    if mode == Mode.NAIVE:
        result = run_pandas(*inputs)
    elif mode == Mode.CUDA:
        result = run_cudf(*inputs)
    elif mode == Mode.BACH:
        result = run_bach_cudf(size, *inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    print('Runtime:', runtime)
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(result)
    return init_time, runtime
