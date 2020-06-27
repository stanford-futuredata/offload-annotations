import random
import sys
import unittest
import numpy as np
import cupy as cp
import torch
import pandas as pd
import cudf
import pytest

sys.path.append("../pycomposer")
sys.path.append('./pycomposer')

from mode import Mode
from sa.annotation import Backend
from sa.annotation import dag
from workloads import *


class TestHaversine(unittest.TestCase):

    def setUp(self):
        self.data_size = 1 << 16
        # The expected result for the given data size
        self.expected = 4839.95983063

    def validateResult(self, arr, val):
        self.assertIsInstance(arr, np.ndarray)
        self.assertAlmostEqual(arr[0], val, places=5)
        self.assertAlmostEqual(arr[-1], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)

    def test_get_data(self):
        size = 16
        arrays = haversine.get_data(size)
        self.assertEqual(len(arrays), 2)
        for array in arrays:
            self.assertEqual(len(array), size)
            self.assertIsInstance(array, np.ndarray)

    def test_cpu(self):
        inputs = haversine.get_data(self.data_size)
        result = haversine.run_numpy(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)

    def test_gpu_torch(self):
        inputs = haversine.get_data(self.data_size)
        result = haversine.run_torch(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)

    @pytest.mark.bach
    def test_bach_torch(self):
        inputs = haversine.get_data(self.data_size)
        result = haversine.run_bach_torch(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)

    @pytest.mark.paging
    @pytest.mark.bach
    def test_bach_torch_paging(self):
        data_size = haversine.MAX_BATCH_SIZE << 2
        inputs = haversine.get_data(data_size)
        result = haversine.run_bach_torch(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)

    @pytest.mark.cupy
    def test_gpu_cupy(self):
        inputs = haversine.get_data(self.data_size)
        result = haversine.run_cupy(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)

    @pytest.mark.bach
    @pytest.mark.cupy
    def test_bach_cupy(self):
        inputs = haversine.get_data(self.data_size)
        result = haversine.run_bach_cupy(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)

    @pytest.mark.paging
    @pytest.mark.bach
    @pytest.mark.cupy
    def test_bach_cupy_paging(self):
        data_size = haversine.MAX_BATCH_SIZE << 2
        inputs = haversine.get_data(data_size)
        result = haversine.run_bach_cupy(*inputs)
        self.assertIsInstance(result, np.ndarray)
        self.validateResult(result, self.expected)


class TestPCA(unittest.TestCase):

    def setUp(self):
        self.data_size = 1 << 8
        self.batch_size = {
            Backend.CPU: (1 << 6) * pca.NUM_TEST,
            Backend.GPU: (1 << 6) * pca.NUM_TEST,
        }
        self.base_data_size = 178

    def test_gen_data(self):
        size = 1 << 4
        for mode in Mode:
            X_train, X_test, y_train, y_test = pca.gen_data(mode, size=size)
            self.assertIsInstance(X_train, np.ndarray)
            self.assertIsInstance(X_test, np.ndarray)
            self.assertIsInstance(y_train, np.ndarray)
            self.assertIsInstance(y_test, np.ndarray)
            self.assertEqual(len(X_train) + len(X_test), self.base_data_size * size)
            self.assertEqual(len(y_train) + len(y_test), self.base_data_size * size)

    def test_naive(self):
        X_train, X_test, y_train, y_test = pca.gen_data(Mode.NAIVE, size=self.data_size)
        pred_test = pca.run_naive_unscaled(X_train, X_test, y_train, y_test)
        self.assertIsInstance(pred_test, np.ndarray)
        pred_test_std = pca.run_naive_scaled(X_train, X_test, y_train, y_test)
        self.assertIsInstance(pred_test_std, np.ndarray)

        accuracy = pca.accuracy(y_test, pred_test)
        accuracy_std = pca.accuracy(y_test, pred_test_std)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy_std, 0.9)
        self.assertGreater(accuracy_std, accuracy)

    def test_cuda(self):
        X_train, X_test, y_train, y_test = pca.gen_data(Mode.CUDA, size=self.data_size)
        pred_test = pca.run_cuda_unscaled(X_train, X_test, y_train, y_test)
        self.assertIsInstance(pred_test, np.ndarray)
        pred_test_std = pca.run_cuda_scaled(X_train, X_test, y_train, y_test)
        self.assertIsInstance(pred_test_std, np.ndarray)

        accuracy = pca.accuracy(y_test, pred_test)
        accuracy_std = pca.accuracy(y_test, pred_test_std)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy_std, 0.9)
        self.assertGreater(accuracy_std, accuracy)

    def test_mozart(self):
        inputs = pca.gen_data(Mode.MOZART, size=self.data_size)
        pred_test = pca.run_composer_unscaled(Mode.MOZART, *inputs, self.batch_size, threads=1)
        self.assertIsInstance(pred_test, np.ndarray)
        pred_test_std = pca.run_composer_scaled(Mode.MOZART, *inputs, self.batch_size, threads=1)
        self.assertIsInstance(pred_test_std, np.ndarray)

        accuracy = pca.accuracy(inputs[3], pred_test)
        accuracy_std = pca.accuracy(inputs[3], pred_test_std)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy_std, 0.9)
        self.assertGreater(accuracy_std, accuracy)

    @pytest.mark.bach
    def test_bach(self):
        inputs = pca.gen_data(Mode.BACH, size=self.data_size)
        pred_test = pca.run_composer_unscaled(Mode.BACH, *inputs, self.batch_size, threads=1)
        self.assertIsInstance(pred_test, np.ndarray)
        pred_test_std = pca.run_composer_scaled(Mode.BACH, *inputs, self.batch_size, threads=1)
        self.assertIsInstance(pred_test_std, np.ndarray)

        accuracy = pca.accuracy(inputs[3], pred_test)
        accuracy_std = pca.accuracy(inputs[3], pred_test_std)
        self.assertGreater(accuracy, 0.7)
        self.assertGreater(accuracy_std, 0.9)
        self.assertGreater(accuracy_std, accuracy)


class TestDBSCAN(unittest.TestCase):

    def setUp(self):
        self.data_size = 1 << 10
        self.centers = dbscan.DEFAULT_CENTERS

    def test_cpu(self):
        X, eps, min_samples = dbscan.gen_data(Mode.NAIVE, self.data_size, centers=self.centers)
        self.assertIsInstance(X, np.ndarray)
        labels = dbscan.run_cpu(X, eps, min_samples)
        self.assertIsInstance(labels, np.ndarray)

        clusters, noise = dbscan.clusters(labels)
        self.assertEqual(clusters, self.centers)
        self.assertLess(noise, self.data_size * 0.3)

    @pytest.mark.cupy
    def test_gpu(self):
        X, eps, min_samples = dbscan.gen_data(Mode.CUDA, self.data_size, centers=self.centers)
        self.assertIsInstance(X, cudf.DataFrame)
        labels = dbscan.run_gpu(X, eps, min_samples)
        self.assertIsInstance(labels, np.ndarray)

        clusters, noise = dbscan.clusters(labels)
        self.assertEqual(clusters, self.centers)
        self.assertLess(noise, self.data_size * 0.3)

    def validateLabels(self, labels, size, centers):
        clusters, noise = dbscan.clusters(labels)
        self.assertGreater(clusters, centers * 0.5)
        self.assertLess(clusters, centers * 2.0)
        self.assertLess(noise, size * 0.3)

    def validateResults(self, size, centers, cluster_std):
        inputs = dbscan.gen_data(Mode.NAIVE, size, centers=centers, cluster_std=cluster_std)
        labels = dbscan.run_cpu(*inputs)
        self.validateLabels(labels, size, centers)

        inputs = dbscan.gen_data(Mode.CUDA, size, centers=centers, cluster_std=cluster_std)
        labels = dbscan.run_gpu(*inputs)
        self.validateLabels(labels, size, centers)

        inputs = dbscan.gen_data(Mode.BACH, size, centers=centers, cluster_std=cluster_std)
        labels = dbscan.run_gpu(*inputs)
        self.validateLabels(labels, size, centers)

    @pytest.mark.bach
    @pytest.mark.cupy
    def test_parameter_sensitivity(self):
        self.validateResults(self.data_size, centers=32, cluster_std=1.0)
        self.validateResults(self.data_size, centers=32, cluster_std=1.1)
        self.validateResults(self.data_size, centers=32, cluster_std=0.9)
        self.validateResults(self.data_size, centers=4, cluster_std=1.0)
        self.validateResults(self.data_size, centers=128, cluster_std=1.0)

    @pytest.mark.bach
    @pytest.mark.cupy
    def test_bach(self):
        X, eps, min_samples = dbscan.gen_data(Mode.BACH, self.data_size, centers=self.centers)
        self.assertIsInstance(X, np.ndarray)
        labels = dbscan.run_bach(X, eps, min_samples)
        self.assertIsInstance(labels, np.ndarray)

        clusters, noise = dbscan.clusters(labels)
        self.assertEqual(clusters, self.centers)
        self.assertLess(noise, self.data_size * 0.3)


class TestBlackscholes(unittest.TestCase):

    def setUp(self):
        self.data_size = 1 << 20
        # The expected result for the given data size
        self.expected_call = 24.0
        self.expected_put = 18.0

    def validateArray(self, arr, val):
        self.assertAlmostEqual(arr[0], val, places=5)
        self.assertAlmostEqual(arr[-1], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)

    def test_get_data(self):
        size = 16
        arrays = blackscholes.get_data(size)
        self.assertEqual(len(arrays), 5)
        for array in arrays:
            self.assertEqual(len(array), size)
            self.assertIsInstance(array, np.ndarray)

    def test_cpu(self):
        inputs = blackscholes.get_data(self.data_size)
        call, put = blackscholes.run_numpy(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    def test_gpu_torch(self):
        inputs = blackscholes.get_data(self.data_size)
        call, put = blackscholes.run_torch(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    @pytest.mark.bach
    def test_bach_torch(self):
        inputs = blackscholes.get_data(self.data_size)
        call, put = blackscholes.run_bach_torch(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    @pytest.mark.paging
    @pytest.mark.bach
    def test_bach_torch_paging(self):
        data_size = blackscholes.MAX_BATCH_SIZE << 2
        inputs = blackscholes.get_data(data_size)
        call, put = blackscholes.run_bach_torch(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    @pytest.mark.cupy
    def test_gpu_cupy(self):
        inputs = blackscholes.get_data(self.data_size)
        call, put = blackscholes.run_cupy(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    @pytest.mark.bach
    @pytest.mark.cupy
    def test_bach_cupy(self):
        inputs = blackscholes.get_data(self.data_size)
        call, put = blackscholes.run_bach_cupy(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)

    @pytest.mark.paging
    @pytest.mark.bach
    @pytest.mark.cupy
    def test_bach_cupy_paging(self):
        data_size = blackscholes.MAX_BATCH_SIZE << 2
        inputs = blackscholes.get_data(data_size)
        call, put = blackscholes.run_bach_cupy(*inputs)
        self.assertIsInstance(call, np.ndarray)
        self.assertIsInstance(put, np.ndarray)
        self.validateArray(call, self.expected_call)
        self.validateArray(put, self.expected_put)


class TestCrimeIndex(unittest.TestCase):

    def setUp(self):
        prefix = 'datasets/crime_index/test/'
        filenames = ['total_population.csv', 'adult_population.csv', 'num_robberies.csv']
        self.filenames = [prefix + f for f in filenames]
        self.data_size = 1 << 20
        # The expected result for the given data size
        self.expected = 5242.88

    def test_read_data(self):
        for array in crime_index.read_data(Mode.NAIVE, filenames=self.filenames):
            self.assertIsInstance(array, pd.Series)
        for array in crime_index.read_data(Mode.CUDA, filenames=self.filenames):
            self.assertIsInstance(array, cudf.Series)
        for array in crime_index.read_data(Mode.BACH, filenames=self.filenames):
            self.assertIsInstance(array, dag.Operation)

    def test_write_data(self):
        crime_index._write_data(self.data_size)
        for array in crime_index.read_data(Mode.NAIVE, size=self.data_size):
            self.assertIsInstance(array, pd.Series)
        for array in crime_index.read_data(Mode.CUDA, size=self.data_size):
            self.assertIsInstance(array, cudf.Series)
        for array in crime_index.read_data(Mode.BACH, size=self.data_size):
            self.assertIsInstance(array, dag.Operation)

    def test_cpu(self):
        inputs = crime_index.read_data(Mode.NAIVE, size=self.data_size)
        result = crime_index.run_pandas(*inputs)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, self.expected)

    def test_gpu(self):
        inputs = crime_index.read_data(Mode.CUDA, size=self.data_size)
        result = crime_index.run_cudf(*inputs)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, self.expected)

    @pytest.mark.bach
    def test_bach(self):
        inputs = crime_index.read_data(Mode.BACH, size=self.data_size)
        result = crime_index.run_bach_cudf(self.data_size, *inputs)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, self.expected)

    @pytest.mark.paging
    @pytest.mark.bach
    def test_bach_paging(self):
        size = 1 << 28
        self.assertGreater(size, crime_index.MAX_BATCH_SIZE)
        crime_index._write_data(1 << 28)
        inputs = crime_index.read_data(Mode.BACH, size=size)
        result = crime_index.run_bach_cudf(size, *inputs)
        self.assertAlmostEqual(result, 1342177.28, places=3)


if __name__ == '__main__':
    unittest.main()
