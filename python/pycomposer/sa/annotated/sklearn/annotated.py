
import sklearn
import cuml

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *
from sa.annotated.numpy_cudf import NdArraySplit


class CPUModelSplit(SplitType):

    def combine(self, values, original=None):
        return values[0]

    def split(self, _start, _end, value):
        return value

    def elements(self, value):
        return None

    def __str__(self):
        return "CPUModelSplit"


class ModelSplit(SplitType):

    def __init__(self):
        self.supported_backends = [Backend.CPU, Backend.GPU]
        self.cpu_models = set([
            sklearn.cluster.DBSCAN,
            sklearn.neighbors.KNeighborsClassifier,
            sklearn.decomposition.PCA,
        ])
        self.gpu_models = set([
            cuml.DBSCAN,
            cuml.neighbors.KNeighborsClassifier,
            cuml.PCA,
        ])

    def combine(self, values, original=None):
        return values[0]

    def split(self, _start, _end, value):
        return value

    def elements(self, value):
        return None

    def backend(self, value):
        if type(value) in self.cpu_models:
            return Backend.CPU
        elif type(value) in self.gpu_models:
            return Backend.GPU
        else:
            raise Exception('unknown device: {}'.format(type(value)))

    def to(self, value, backend):
        old_backend = self.backend(value)
        if old_backend == backend:
            return value
        else:
            raise Exception('cannot transfer from {} to {}'.format(old_backend, backend))

    def __str__(self):
        return "ModelSplit"


# *************************************************************************************************
# Models
DBSCAN = alloc(ModelSplit(), gpu=True, gpu_func=cuml.DBSCAN)(sklearn.cluster.DBSCAN)
KNeighborsClassifier = alloc(ModelSplit(), gpu=True, gpu_func=cuml.neighbors.KNeighborsClassifier)(
    sklearn.neighbors.KNeighborsClassifier)
PCA = alloc(ModelSplit(), gpu=True, gpu_func=cuml.PCA)(sklearn.decomposition.PCA)
StandardScaler = alloc(CPUModelSplit())(sklearn.preprocessing.StandardScaler)

# *************************************************************************************************
# Method wrappers that CANNOT be split
@sa((ModelSplit(), NdArraySplit()), {}, ModelSplit(), gpu=True)
def fit_x(model, X):
    return model.fit(X)

@sa((ModelSplit(), NdArraySplit(), NdArraySplit()), {}, ModelSplit(), gpu=True)
def fit_xy(model, X, y):
    return model.fit(X, y)

@sa((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), gpu=True)
def fit_transform(model, X):
    return model.fit_transform(X)

@sa((CPUModelSplit(), NdArraySplit()), {}, NdArraySplit())
def fit_transform_cpu(model, X):
    return model.fit_transform(X)

@sa((ModelSplit(),), {}, NdArraySplit(), gpu=True)
def labels(model):
    return model.labels_

# *************************************************************************************************
# Method wrappers that CAN be split
@sa((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), gpu=True)
def transform(model, X):
    return model.transform(X)

@sa((CPUModelSplit(), NdArraySplit()), {}, NdArraySplit())
def transform_cpu(model, X):
    return model.transform(X)

@sa((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), gpu=True)
def predict(model, X):
    return model.predict(X)
