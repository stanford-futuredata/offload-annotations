import random
import sys
import unittest
import numpy as np

sys.path.append("../lib")
sys.path.append("../pycomposer")
sys.path.append('.')

from mode import Mode
from sa.annotation import Backend
from workloads import *


class TestBlackscholesNumpy(unittest.TestCase):

    def setUp(self):
        self.data_size = 1 << 20
        self.batch_size = {
            Backend.CPU: blackscholes_numpy.DEFAULT_CPU,
            Backend.GPU: blackscholes_numpy.DEFAULT_GPU,
        }
        # The expected result for the given data size
        self.expected_call = 24.0
        self.expected_put = 18.0

    def validateArray(self, arr, val):
        self.assertAlmostEqual(arr[0], val, places=5)
        self.assertAlmostEqual(arr[-1], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)

    def test_naive(self):
        inputs = blackscholes_numpy.get_data(Mode.NAIVE, self.data_size)
        tmp_arrays = blackscholes_numpy.get_tmp_arrays(Mode.NAIVE, self.data_size)
        call, put = blackscholes_numpy.run_naive(*inputs, *tmp_arrays)
        self.assertTrue(isinstance(call, np.ndarray))
        self.assertTrue(isinstance(put, np.ndarray))
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    def test_cuda(self):
        inputs = blackscholes_numpy.get_data(Mode.CUDA, self.data_size)
        tmp_arrays = blackscholes_numpy.get_tmp_arrays(Mode.CUDA, self.data_size)
        call, put = blackscholes_numpy.run_cuda(*inputs, *tmp_arrays)
        self.assertTrue(isinstance(call, np.ndarray))
        self.assertTrue(isinstance(put, np.ndarray))
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    def test_mozart(self):
        inputs = blackscholes_numpy.get_data(Mode.MOZART, self.data_size)
        tmp_arrays = blackscholes_numpy.get_tmp_arrays(Mode.MOZART, self.data_size)
        call, put = blackscholes_numpy.run_composer(
            Mode.MOZART, *inputs, *tmp_arrays, self.batch_size, threads=16)
        self.assertTrue(isinstance(call, np.ndarray))
        self.assertTrue(isinstance(put, np.ndarray))
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    def test_bach(self):
        inputs = blackscholes_numpy.get_data(Mode.BACH, self.data_size)
        tmp_arrays = blackscholes_numpy.get_tmp_arrays(Mode.BACH, self.data_size)
        call, put = blackscholes_numpy.run_composer(
            Mode.BACH, *inputs, *tmp_arrays, self.batch_size, threads=1)
        self.assertTrue(isinstance(call, np.ndarray))
        self.assertTrue(isinstance(put, np.ndarray))
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)


class TestCrimeIndex(unittest.TestCase):

    def setUp(self):
        prefix = 'datasets/crime_index/test/'
        filenames = ['total_population.csv', 'adult_population.csv', 'num_robberies.csv']
        self.filenames = [prefix + f for f in filenames]
        self.data_size = 1 << 20
        self.batch_size = {
            Backend.CPU: crime_index.DEFAULT_CPU,
            Backend.GPU: crime_index.DEFAULT_GPU,
        }
        # The expected result for the given data size
        self.expected = 5242.88

    def test_read_naive(self):
        inputs = crime_index.read_data(Mode.NAIVE, filenames=self.filenames)
        result = crime_index.run_naive(*inputs)
        self.assertAlmostEqual(result, self.expected)

    def test_read_cuda(self):
        inputs = crime_index.read_data(Mode.CUDA, filenames=self.filenames)
        result = crime_index.run_cuda(*inputs)
        self.assertAlmostEqual(result, self.expected)

    def test_read_mozart(self):
        inputs = crime_index.read_data(Mode.MOZART, filenames=self.filenames)
        result = crime_index.run_composer(Mode.MOZART, *inputs, self.batch_size, threads=16)
        self.assertAlmostEqual(result, self.expected)

    def test_read_bach(self):
        inputs = crime_index.read_data(Mode.BACH, filenames=self.filenames)
        result = crime_index.run_composer(Mode.BACH, *inputs, self.batch_size, threads=1)
        self.assertAlmostEqual(result, self.expected)

    def test_gen_naive(self):
        inputs = crime_index.gen_data(Mode.NAIVE, self.data_size)
        result = crime_index.run_naive(*inputs)
        self.assertAlmostEqual(result, self.expected)

    def test_gen_cuda(self):
        inputs = crime_index.gen_data(Mode.CUDA, self.data_size)
        result = crime_index.run_cuda(*inputs)
        self.assertAlmostEqual(result, self.expected)

    def test_gen_mozart(self):
        inputs = crime_index.gen_data(Mode.MOZART, self.data_size)
        result = crime_index.run_composer(Mode.MOZART, *inputs, self.batch_size, threads=16)
        self.assertAlmostEqual(result, self.expected)

    def test_gen_bach(self):
        inputs = crime_index.gen_data(Mode.BACH, self.data_size)
        result = crime_index.run_composer(Mode.BACH, *inputs, self.batch_size, threads=1)
        self.assertAlmostEqual(result, self.expected)


if __name__ == '__main__':
    unittest.main()
