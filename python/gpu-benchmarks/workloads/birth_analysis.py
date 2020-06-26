#!/usr/bin/python
import sys
import math
import time
import pandas as pd
import cudf
import matplotlib.pyplot as plt

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1
MAX_SIZE = 2011-1880+1
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26

PREFIX = 'datasets/birth_analysis'
FIRST_YEAR = 1880


def _read_data_cuda(num_years):
    pieces = []
    columns = ['name', 'sex', 'births']
    for i in range(num_years):
        year = FIRST_YEAR + i
        path = '{}/yob{}.txt'.format(PREFIX, year)
        frame = cudf.read_csv(path, names=columns)
        frame['year'] = year
        pieces.append(frame)
    names = cudf.concat(pieces, ignore_index=True)
    return names


def read_data(mode, num_years):
    if mode == Mode.CUDA:
        return _read_data_cuda(num_years)

    pieces = []
    columns = ['name', 'sex', 'births']
    for i in range(num_years):
        year = FIRST_YEAR + i
        path = '{}/yob{}.txt'.format(PREFIX, year)
        frame = pd.read_csv(path, names=columns)
        frame['year'] = year
        pieces.append(frame)
    names = pd.concat(pieces, ignore_index=True)
    return names


def run_composer(mode, inputs, batch_size, threads):
    raise Exception


def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[0:1000]


def run_naive(names):
    grouped = names.groupby(['year', 'sex'])
    top1000 = grouped.apply(get_top1000)
    top1000.reset_index(inplace=True, drop=True)

    all_names = pd.Series(top1000.name.unique())
    lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
    filtered = top1000[top1000.name.isin(lesley_like)]

    table = filtered.pivot_table('births', index='year',
                                 columns='sex', aggfunc='sum')
    table = table.div(table.sum(1), axis=0)
    return table


def run_cuda(names):
    grouped = names.groupby(['year', 'sex'], as_index=False, method='cudf')
    top1000 = grouped.apply(get_top1000)
    top1000.reset_index(inplace=True, drop=True)

    all_names = cudf.Series(top1000.name.unique())
    lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
    filtered = top1000[top1000.name.isin(lesley_like)]

    filtered = filtered.to_pandas()
    table = filtered.pivot_table('births', index='year',
                                 columns='sex', aggfunc='sum')
    table = table.div(table.sum(1), axis=0)
    return table


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
    if size is not None:
        size = int(math.log2(size))
        assert size <= MAX_SIZE

    # Optimal defaults
    if size == None:
        size = DEFAULT_SIZE
    if cpu is None:
        cpu = DEFAULT_CPU
    if gpu is None:
        gpu = DEFAULT_GPU
    if threads is None:
        threads = 1

    batch_size = {
        Backend.CPU: cpu,
        Backend.GPU: gpu,
    }

    # Get inputs
    start = time.time()
    inputs = read_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        results = run_naive(inputs)
    elif mode == Mode.CUDA:
        results = run_cuda(inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print(results.tail())
    return init_time, runtime

