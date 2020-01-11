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

DEFAULT_SIZE = 1 << 6
DEFAULT_CPU = 1 << 16
DEFAULT_GPU = 1 << 26


def gen_data(mode, size):
    # create header for dataset
    header = ['age','bp','sg','al','su','rbc','pc','pcc',
        'ba','bgr','bu','sc','sod','pot','hemo','pcv',
        'wbcc','rbcc','htn','dm','cad','appet','pe','ane',
        'classification']
    # read the dataset
    df = pd.read_csv("datasets/kidneys/chronic_kidney_disease_full.arff",
            header=None,
            names=header
           )
    for _ in range(int(math.log2(size))):
        df = df.append(df)
    return df


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


def run_naive(df):
    # https://mclguide.readthedocs.io/en/latest/sklearn/preprocessing.html

    ###########################################################
    # Clean the data.
    #
    # dataset has '?' in it, convert these into NaN
    df = df.replace('?', np.nan)
    # drop the NaN
    df = df.dropna(axis=0, how="any")

    # print total samples
    # print("Total samples:", len(df))
    # print 4-rows and 6-columns
    # print("Partial data\n", df.iloc[0:4, 0:6])

    ###########################################################
    # Saving targets with different color names.
    #
    # covert 'ckd' and 'notckd' labels as '0' and '1'
    targets = df['classification'].astype('category')
    # save target-values as color for plotting
    # red: disease,  green: no disease
    label_color = ['red' if i==0 else 'green' for i in targets]
    # print(label_color[0:3], label_color[-3:-1])

    ###########################################################
    # Preparing data for PCA analysis using pandas and sklearn
    #
    # list of categorical features
    categorical_ = ['rbc', 'pc', 'pcc', 'ba', 'htn',
            'dm', 'cad', 'appet', 'pe', 'ane']

    # drop the classification column
    df = df.drop(labels=['classification'], axis=1)

    # drop using 'inplace' which is equivalent to df = df.drop()
    # df.drop(labels=categorical_, axis=1, inplace=True)
    # print("Partial data\n", df.iloc[0:4, 0:6]) # print partial data
    # convert categorical features into dummy variable
    df = pd.get_dummies(df, columns=categorical_)
    # StandardScaler: mean=0, variance=1
    ss = sklearn.preprocessing.StandardScaler()
    df = ss.fit_transform(df)

    ###########################################################
    # Dimensionality reduction.
    #
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(df)
    T = pca.transform(df) # transformed data
    # change 'T' to Pandas-DataFrame to plot using Pandas-plots
    T = pd.DataFrame(T)

    ###########################################################
    # Plot the data.
    T.columns = ['PCA component 1', 'PCA component 2']
    T.plot.scatter(x='PCA component 1', y='PCA component 2', marker='o',
            alpha=0.7, # opacity
            color=label_color,
            title="red: ckd, green: not-ckd" )
    plt.show()


def run_cuda(df):
    # transfer
    df = df.replace('?', np.nan)
    df = df.dropna(axis=0, how="any")
    df = cudf.from_pandas(df)

    targets = df['classification'].astype('category')
    label_color = ['red' if i==0 else 'green' for i in targets]
    categorical_ = ['rbc', 'pc', 'pcc', 'ba', 'htn',
            'dm', 'cad', 'appet', 'pe', 'ane']

    df = df.drop(labels=['classification'], axis=1)
    df = cudf.get_dummies(df, columns=categorical_)
    # StandardScaler: mean=0, variance=1
    ss = sklearn.preprocessing.StandardScaler()
    df = df.to_pandas() # transfer
    df = ss.fit_transform(df)

    ###########################################################
    # Dimensionality reduction.
    #
    df = cudf.from_pandas(pd.DataFrame(df)) # transfer
    pca = cuml.PCA(n_components=2)
    pca.fit(df)
    T = pca.transform(df)
    T = T.to_pandas().to_numpy()
    T = pd.DataFrame(T)

    T.columns = ['PCA component 1', 'PCA component 2']
    T.plot.scatter(x='PCA component 1', y='PCA component 2', marker='o',
            alpha=0.7, # opacity
            color=label_color,
            title="red: ckd, green: not-ckd" )
    plt.show()


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
    df = gen_data(mode, size)
    init_time = time.time() - start
    sys.stdout.write('Initialization: {}\n'.format(init_time))

    # Run program
    start = time.time()
    if mode.is_composer():
        results = run_composer(mode, df, batch_size, threads)
    elif mode == Mode.NAIVE:
        results = run_naive(df)
    elif mode == Mode.CUDA:
        results = run_cuda(df)
    else:
        raise ValueError
    runtime = time.time() - start

    sys.stdout.write('Runtime: {}\n'.format(runtime))
    sys.stdout.write('Total: {}\n'.format(init_time + runtime))
    sys.stdout.flush()
    print('Results:', results)
    return init_time, runtime

