import sys
import unittest

sys.path.append("../lib")
sys.path.append("../pycomposer")
sys.path.append('.')

from mode import Mode
from sa.annotation import Backend
from workloads import *


class TestCrimeIndex(unittest.TestCase):

    def setUp(self):
        prefix = 'datasets/crime_index/test/'
        filenames = ['total_population.csv', 'adult_population.csv', 'num_robberies.csv']
        self.filenames = [prefix + f for f in filenames]
        self.data_size = 1 << 20
        self.batch_size = {
            Backend.CPU: 1 << 14,
            Backend.GPU: 1 << 19,
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
