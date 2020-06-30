
import sklearn
import cuml

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.scheduling import *
from sa.annotation.split_types import *
from sa.annotated.cupy_estimator import NdArraySplit


class CPUModelSplit(SplitType):

    def combine(self, values, original=None):
        return values[0]

    def split(self, _start, _end, value):
        return value

    def elements(self, value):
        return None

    def __str__(self):
        return "CPUModelSplit"


class ModelSplit(OffloadSplitType):

    estimator = gen_linear_transfer_estimator(1, 0)

    def __init__(self):
        self.supported_backends = [Backend.CPU, Backend.GPU]
        self.cpu_models = [
            sklearn.cluster.DBSCAN,
            sklearn.neighbors.KNeighborsClassifier,
            sklearn.decomposition.PCA,
            sklearn.decomposition.TruncatedSVD,
        ]
        self.gpu_models = [
            cuml.DBSCAN,
            cuml.neighbors.KNeighborsClassifier,
            cuml.PCA,
            cuml.decomposition.TruncatedSVD,
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
def custom_TruncatedSVD(*args, **kwargs):
    if 'algorithm' in kwargs and kwargs['algorithm'] == 'arpack':
        kwargs['algorithm'] = 'full'
    return cuml.TruncatedSVD(*args, **kwargs)

DBSCAN = oa_alloc(ModelSplit(), func=cuml.DBSCAN)(sklearn.cluster.DBSCAN)
KNeighborsClassifier = oa_alloc(ModelSplit(), func=cuml.neighbors.KNeighborsClassifier)(
    sklearn.neighbors.KNeighborsClassifier)
PCA = oa_alloc(ModelSplit(), func=cuml.PCA)(sklearn.decomposition.PCA)
TruncatedSVD = oa_alloc(ModelSplit(), func=custom_TruncatedSVD)(sklearn.decomposition.TruncatedSVD)
StandardScaler = alloc(CPUModelSplit())(sklearn.preprocessing.StandardScaler)

# *************************************************************************************************
# Method wrappers that CANNOT be split
@oa((ModelSplit(), NdArraySplit()), {}, ModelSplit())
def fit_x(model, X):
    return model.fit(X)

@oa((ModelSplit(), NdArraySplit(), NdArraySplit()), {}, ModelSplit())
def fit_xy(model, X, y):
    return model.fit(X, y)

compute_estimator = gen_linear_compute_estimator(2, 0, 0, 14)
@oa((ModelSplit(), NdArraySplit()), {}, NdArraySplit(), estimator=compute_estimator)
def fit_transform(model, X):
    return model.fit_transform(X)

@sa((CPUModelSplit(), NdArraySplit()), {}, NdArraySplit())
def fit_transform_cpu(model, X):
    return model.fit_transform(X)

@oa((ModelSplit(),), {}, NdArraySplit())
def labels(model):
    return model.labels_

# *************************************************************************************************
# Method wrappers that CAN be split
@oa((ModelSplit(), NdArraySplit()), {}, NdArraySplit())
def transform(model, X):
    return model.transform(X)

@sa((CPUModelSplit(), NdArraySplit()), {}, NdArraySplit())
def transform_cpu(model, X):
    return model.transform(X)

@oa((ModelSplit(), NdArraySplit()), {}, NdArraySplit())
def predict(model, X):
    return model.predict(X)
