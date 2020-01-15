
import sklearn
import cuml

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.scheduling import *
from sa.annotation.split_types import *
from sa.annotated.cupy import NdArraySplit


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

    estimator = transfer_estimator

    def __init__(self):
        self.supported_backends = [Backend.CPU, Backend.GPU]
        self.cpu_models = [
            sklearn.cluster.DBSCAN,
            sklearn.neighbors.KNeighborsClassifier,
            sklearn.decomposition.PCA,
        ]
        self.gpu_models = [
            cuml.DBSCAN,
            cuml.neighbors.KNeighborsClassifier,
            cuml.PCA,
        ]

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
        if True:
        # try:
        #     sklearn.utils.validation.check_is_fitted(value)
        #     raise Exception('cannot transfer model, already fitted')
        # except sklearn.exceptions.NotFittedError:
            if old_backend == Backend.CPU:
                model = self.gpu_models[self.cpu_models.index(type(value))]
            else:
                model = self.cpu_models[self.gpu_models.index(type(value))]
            return model()


    def __str__(self):
        return "ModelSplit"


# *************************************************************************************************
# Models
DBSCAN = alloc_gpu(ModelSplit(), func=cuml.DBSCAN)(sklearn.cluster.DBSCAN)
KNeighborsClassifier = alloc_gpu(ModelSplit(), func=cuml.neighbors.KNeighborsClassifier)(
    sklearn.neighbors.KNeighborsClassifier)
PCA = alloc_gpu(ModelSplit(), func=cuml.PCA)(sklearn.decomposition.PCA)
StandardScaler = alloc(CPUModelSplit())(sklearn.preprocessing.StandardScaler)

# *************************************************************************************************
# Method wrappers that CANNOT be split
@sa_gpu((ModelSplit(), NdArraySplit()), {}, ModelSplit(), estimator=compute_estimator)
def fit_x(model, X):
    return model.fit(X)

@sa_gpu((ModelSplit(), NdArraySplit(), NdArraySplit()), {}, ModelSplit(), estimator=compute_estimator)
def fit_xy(model, X, y):
    return model.fit(X, y)

@sa_gpu((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), estimator=compute_estimator)
def fit_transform(model, X):
    return model.fit_transform(X)

@sa((CPUModelSplit(), NdArraySplit()), {}, NdArraySplit())
def fit_transform_cpu(model, X):
    return model.fit_transform(X)

@sa_gpu((ModelSplit(),), {}, NdArraySplit())
def labels(model):
    return model.labels_

# *************************************************************************************************
# Method wrappers that CAN be split
@sa_gpu((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), estimator=compute_estimator)
def transform(model, X):
    return model.transform(X)

@sa((CPUModelSplit(), NdArraySplit()), {}, NdArraySplit())
def transform_cpu(model, X):
    return model.transform(X)

@sa_gpu((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), estimator=compute_estimator)
def predict(model, X):
    return model.predict(X)
