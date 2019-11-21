#!/usr/bin/python

import argparse
import sys

sys.path.append("../../lib")
sys.path.append("../../pycomposer")

import numpy as np
import time

import sa.annotated.pandas as pd
from sa.annotation import Backend
import cudf

def gen_data(size):
    total_population = np.ones(size, dtype="float64") * 500000
    adult_population = np.ones(size, dtype="float64") * 250000
    num_robberies = np.ones(size, dtype="float64") * 1000
    return cudf.Series(total_population), cudf.Series(adult_population), cudf.Series(num_robberies)

def crime_index_composer(
    total_population,
    adult_population,
    num_robberies,
    threads,
    gpu_piece_size,
    cpu_piece_size,
):
    # Get all city information with total population greater than 500,000
    big_cities = pd.greater_than(total_population, 500000.0)
    big_cities.dontsend = True
    big_cities = pd.mask(total_population, big_cities, 0.0)
    big_cities.dontsend = True

    double_pop = pd.multiply(adult_population, 2.0)
    double_pop.dontsend = True
    double_pop = pd.add(big_cities, double_pop)
    double_pop.dontsend = True
    multiplied = pd.multiply(num_robberies, 2000.0)
    multiplied.dontsend = True
    double_pop = pd.subtract(double_pop, multiplied)
    double_pop.dontsend = True
    crime_index = pd.divide(double_pop, 100000.0)
    crime_index.dontsend = True

    gt = pd.greater_than(crime_index, 0.02)
    gt.dontsend = True
    crime_index = pd.mask(crime_index, gt, 0.032)
    crime_index.dontsend = True
    lt = pd.less_than(crime_index, 0.01)
    crime_index = pd.mask(crime_index, lt, 0.005)
    crime_index.dontsend = True

    result = pd.pandasum(crime_index)
    result.dontsend = False
    batch_size = {
        Backend.CPU: cpu_piece_size,
        Backend.GPU: gpu_piece_size,
    }
    pd.evaluate(workers=threads, batch_size=batch_size)
    return result.value

def crime_index_pandas(total_population, adult_population, num_robberies):
    print(len(total_population))
    big_cities = total_population > 500000
    big_cities = total_population.mask(big_cities, 0.0)
    double_pop = adult_population * 2 + big_cities - (num_robberies * 2000.0)
    crime_index = double_pop / 100000
    crime_index = crime_index.mask(crime_index > 0.02, 0.032)
    crime_index = crime_index.mask(crime_index < 0.01, 0.005)
    return crime_index.sum()

def run():
    parser = argparse.ArgumentParser(description="Crime Index")
    parser.add_argument('-s', "--size", type=int, default=26, help="Size of each array")
    parser.add_argument('-gpu', "--gpu_piece_size", type=int, default=524288, help="Size of each GPU piece.")
    parser.add_argument('-cpu', "--cpu_piece_size", type=int, default=16384*2, help="Size of each CPU piece.")
    parser.add_argument('-t', "--threads", type=int, default=1, help="Number of threads.")
    parser.add_argument('-m', "--mode", type=str, required=True, help="Mode (composer|naive)")
    args = parser.parse_args()

    size = (1 << args.size)
    gpu_piece_size = args.gpu_piece_size
    cpu_piece_size = args.cpu_piece_size
    threads = args.threads
    mode = args.mode.strip().lower()

    assert mode == "composer" or mode == "naive"
    assert threads >= 1

    print("Size:", size)
    print("GPU Piece Size:", gpu_piece_size)
    print("CPU Piece Size:", cpu_piece_size)
    print("Threads:", threads)
    print("Mode:", mode)

    sys.stdout.write("Generating data...")
    sys.stdout.flush()
    inputs = gen_data(size)
    print("done.")

    start = time.time()
    if mode == "composer":
        result = crime_index_composer(inputs[0], inputs[1], inputs[2], threads, gpu_piece_size, cpu_piece_size)
    elif mode == "naive":
        result = crime_index_pandas(*inputs)
    end = time.time()

    print(end - start, "seconds")
    print(result)

if __name__ == "__main__":
    run()

