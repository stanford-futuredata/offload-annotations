
import sklearn
import cuml

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *
from sa.annotated.numpy_cudf import NdArraySplit


class ModelSplit(SplitType):

    def __init__(self):
        self.supported_backends = [Backend.CPU, Backend.GPU]
        self.cpu_models = set([
            sklearn.cluster.DBSCAN,
        ])
        self.gpu_models = set([
            cuml.DBSCAN,
        ])

    def combine(self, values, original=None):
        pass

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
        pass

    def __str__(self):
        return "ModelSplit"


DBSCAN = alloc(ModelSplit(), gpu=True, gpu_func=cuml.DBSCAN)(sklearn.cluster.DBSCAN)

# CANNOT BE SPLIT
@sa((ModelSplit(), NdArraySplit()), {}, ModelSplit())
def fit_x(model, X):
    return model.fit(X)

# CANNOT BE SPLIT
@sa((ModelSplit(),), {}, NdArraySplit())
def labels(model):
    return model.labels_
