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
        self.expected = 4839.95983063
        self.batch_size = {
            Backend.CPU: 1 << 14,
            Backend.GPU: 1 << 14,
        }

    def validateResult(self, arr, val):
        self.assertIsInstance(arr, np.ndarray)
        self.assertAlmostEqual(arr[0], val, places=5)
        self.assertAlmostEqual(arr[-1], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)
        self.assertAlmostEqual(arr[random.randrange(len(arr))], val, places=5)

    def test_get_data(self):
        size = 2
        for array in haversine.get_data(size):
            self.assertIsInstance(array, np.ndarray)
        for array in haversine.get_tmp_arrays(Mode.NAIVE, size):
            self.assertIsInstance(array, np.ndarray)
        for array in haversine.get_tmp_arrays(Mode.CUDA, size, use_torch=True):
            self.assertIsInstance(array, torch.Tensor)
            self.assertEqual(array.device.type, 'cuda')
        for array in haversine.get_tmp_arrays(Mode.CUDA, size, use_torch=False):
            self.assertIsInstance(array, cp.ndarray)

    def test_naive(self):
        inputs = haversine.get_data(self.data_size)
        tmp_arrays = haversine.get_tmp_arrays(Mode.NAIVE, self.data_size)
        im = haversine.run_naive(*inputs, *tmp_arrays)
        self.validateResult(im, self.expected)

    def test_cuda_torch(self):
        inputs = haversine.get_data(self.data_size)
        tmp_arrays = haversine.get_tmp_arrays(Mode.CUDA, self.data_size, use_torch=True)
        im = haversine.run_cuda_torch(*inputs, *tmp_arrays)
        self.validateResult(im, self.expected)

    def test_cuda_cupy(self):
        inputs = haversine.get_data(self.data_size)
        tmp_arrays = haversine.get_tmp_arrays(Mode.CUDA, self.data_size, use_torch=False)
        im = haversine.run_cuda_cupy(*inputs, *tmp_arrays)
        self.validateResult(im, self.expected)

    @pytest.mark.skip(reason='fatal Python error')
    def test_mozart(self):
        inputs = haversine.get_data(self.data_size)
        tmp_arrays = haversine.get_tmp_arrays(Mode.MOZART, self.data_size)
        im = haversine.run_composer(
            Mode.MOZART, *inputs, *tmp_arrays, self.batch_size, threads=16, use_torch=None)
        self.validateResult(im, self.expected)

    @pytest.mark.bach
    @pytest.mark.skip(reason='fatal Python error')
    def test_bach_torch(self):
        inputs = haversine.get_data(self.data_size)
        tmp_arrays = haversine.get_tmp_arrays(Mode.BACH, self.data_size, use_torch=True)
        im = haversine.run_composer(
            Mode.BACH, *inputs, *tmp_arrays, self.batch_size, threads=1, use_torch=True)
        self.validateResult(im, self.expected)

    @pytest.mark.bach
    @pytest.mark.skip(reason='fatal Python error')
    def test_bach_cupy(self):
        inputs = haversine.get_data(self.data_size)
        tmp_arrays = haversine.get_tmp_arrays(Mode.BACH, self.data_size, use_torch=False)
        im = haversine.run_composer(
            Mode.BACH, *inputs, *tmp_arrays, self.batch_size, threads=1, use_torch=False)
        self.validateResult(im, self.expected)


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

    def test_naive(self):
        X, eps, min_samples = dbscan.gen_data(Mode.NAIVE, self.data_size, centers=self.centers)
        self.assertIsInstance(X, np.ndarray)
        labels = dbscan.run_naive(X, eps, min_samples)
        self.assertIsInstance(labels, np.ndarray)

        clusters, noise = dbscan.clusters(labels)
        self.assertEqual(clusters, self.centers)
        self.assertLess(noise, self.data_size * 0.3)

    def test_cuda(self):
        X, eps, min_samples = dbscan.gen_data(Mode.CUDA, self.data_size, centers=self.centers)
        self.assertIsInstance(X, cudf.DataFrame)
        labels = dbscan.run_cuda(X, eps, min_samples)
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
        labels = dbscan.run_naive(*inputs)
        self.validateLabels(labels, size, centers)

        inputs = dbscan.gen_data(Mode.CUDA, size, centers=centers, cluster_std=cluster_std)
        labels = dbscan.run_cuda(*inputs)
        self.validateLabels(labels, size, centers)

        batch_size = { Backend.CPU: size, Backend.GPU: size }
        inputs = dbscan.gen_data(Mode.MOZART, size, centers=centers)
        labels = dbscan.run_composer(Mode.MOZART, *inputs, batch_size, threads=1)
        self.validateLabels(labels, size, centers)

        inputs = dbscan.gen_data(Mode.BACH, size, centers=centers)
        labels = dbscan.run_composer(Mode.BACH, *inputs, batch_size, threads=1)
        self.validateLabels(labels, size, centers)

    def test_parameter_sensitivity(self):
        self.validateResults(self.data_size, centers=32, cluster_std=1.0)
        self.validateResults(self.data_size, centers=32, cluster_std=1.1)
        self.validateResults(self.data_size, centers=32, cluster_std=0.9)
        self.validateResults(self.data_size, centers=4, cluster_std=1.0)
        self.validateResults(self.data_size, centers=128, cluster_std=1.0)

    def test_mozart(self):
        X, eps, min_samples = dbscan.gen_data(Mode.MOZART, self.data_size, centers=self.centers)
        self.assertIsInstance(X, np.ndarray)
        labels = dbscan.run_composer(Mode.MOZART, X, eps, min_samples, None, threads=1)
        self.assertIsInstance(labels, np.ndarray)

        clusters, noise = dbscan.clusters(labels)
        self.assertEqual(clusters, self.centers)
        self.assertLess(noise, self.data_size * 0.3)

    @pytest.mark.bach
    def test_bach(self):
        X, eps, min_samples = dbscan.gen_data(Mode.BACH, self.data_size, centers=self.centers)
        self.assertIsInstance(X, np.ndarray)
        labels = dbscan.run_composer(Mode.BACH, X, eps, min_samples, None, threads=1)
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
