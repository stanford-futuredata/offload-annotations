#!/usr/bin/python
import sys
import time
import math
import numpy as np
import pandas as pd
import cuml
import cudf
import sklearn
import matplotlib.pyplot as plt

sys.path.append("../../lib")
sys.path.append("../../pycomposer")
sys.path.append(".")

from sa.annotation import Backend
from mode import Mode

DEFAULT_SIZE = 1 << 10
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def gen_data(mode, size):
    pd.set_option('display.max_columns', None)
    data_train = pd.read_excel('datasets/feature_engineering/Data_Train.xlsx')
    data_test = pd.read_excel('datasets/feature_engineering/Test_set.xlsx')

    if mode == Mode.CUDA:
        return cudf.from_pandas(data_train), cudf.from_pandas(data_test)
    return data_train, data_test


def run_composer(mode, X, eps, min_samples, _, threads):
    import sa.annotated.sklearn as sklearn

    if mode == Mode.MOZART:
        force_cpu = True
    elif mode == Mode.BACH:
        force_cpu = False
    else:
        raise Exception

    pass

    # Note: batch sizes must be max size
    batch_size = { Backend.CPU: X.shape[0], Backend.GPU: X.shape[0], }
    sklearn.evaluate(workers=threads, batch_size=batch_size, force_cpu=force_cpu)
    return labels.value


def run_naive(data_train, data_test):
    # https://towardsdatascience.com/feature-engineering-in-python-part-i-the-most-powerful-way-of-dealing-with-data-8e2447e7c69e
    # https://www.machinehack.com/course/predict-the-flight-ticket-price-hackathon/
    price_train = data_train.Price
    data = pd.concat([data_train.drop(['Price'], axis=1), data_test])
    data = data.drop_duplicates()
    # data = data.drop(data.loc[data['Route'].isnull()].index)

    data['Airline'].loc[data['Airline']=='Vistara Premium economy'] = 'Vistara'
    data['Airline'].loc[data['Airline']=='Jet Airways Business'] = 'Jet Airways'
    data['Airline'].loc[data['Airline']=='Multiple carriers Premium economy'] = 'Multiple carriers'

    data['Destination'].loc[data['Destination']=='Delhi'] = 'New Delhi'

    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])
    data['day_of_week'] = data['Date_of_Journey'].dt.day_name()
    data['Journey_Month'] = pd.to_datetime(data.Date_of_Journey, format='%d/%m/%Y').dt.month_name()

    data['Departure_t'] = pd.to_datetime(data.Dep_Time, format='%H:%M')
    a = data.assign(dept_session=pd.cut(data.Departure_t.dt.hour,[0,6,12,18,24],labels=['Night','Morning','Afternoon','Evening']))
    data['Departure_S'] = a['dept_session']
    data['Departure_S'].fillna("Night", inplace = True)


def run_cuda(_data_train, _data_test):
    _price_train = _data_train.Price
    _data = cudf.concat([_data_train.drop(['Price'], axis=1), _data_test])
    _data = _data.drop_duplicates()
    # _data = _data.drop(_data.loc[_data['Route'].isnull()].index)

    _data['Airline'].loc[_data['Airline']=='Vistara Premium economy'] = 'Vistara'
    _data['Airline'].loc[_data['Airline']=='Jet Airways Business'] = 'Jet Airways'
    _data['Airline'].loc[_data['Airline']=='Multiple carriers Premium economy'] = 'Multiple carriers'


def run(mode, size=None, cpu=None, gpu=None, threads=None, data_mode='file'):
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
    inputs = gen_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, *inputs, batch_size, threads)
    elif mode == Mode.NAIVE:
        results = run_naive(*inputs)
    elif mode == Mode.CUDA:
        results = run_cuda(*inputs)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print('Results:', results)
    return init_time, runtime

