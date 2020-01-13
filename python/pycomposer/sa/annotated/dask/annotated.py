"""
Annotations for Dask functions.

Note: For convinience, we just write a wrapper function that calls the Pandas function, and then
use those functions instead. We could equivalently just replace methods on the DataFrame class too and
split `self` instead of the DataFrame passed in here.
"""

import dask
import time
import dask_cudf

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *
from sa.annotation.backend import Backend
from sa.annotated.pandas import DataFrameSplit

class DaskDfSplit(SplitType):
    supported_backends = [Backend.CPU, Backend.GPU]
    cpu_types = set([])
    gpu_types = set([])

    def combine(self, values):
        raise Exception('need to call compute() to compute future')

    def split(self, start, end, value):
        return value

    def backend(self, value):
        if type(value) in self.cpu_types:
            return Backend.CPU
        elif type(value) in self.gpu_types:
            return Backend.GPU
        # elif isinstance(value, str):
        else:
            return Backend.SCALAR
        # else:
        #   raise Exception('unsupported type:', type(value))

    def to(self, value, backend):
        raise Exception('cannot transfer DaskDfSplit, use compute() to compute future')

    def elements(self, value):
        return len(value)

    def __str__(self):
        return 'DaskDfSplit'

class DateTimeSplit(SplitType):
    supported_backends = [Backend.CPU, Backend.GPU]

    def combine(self, values):
        raise Exception

    def split(self, start, end, value):
        return value

    def backend(self, value):
        if isinstance(value, dask.accessor.DateTimeAccessor):
            return Backend.CPU
        elif isinstance(value, dask_cudf.accessor.DateTimeAccessor):
            return Backend.GPU
        else:
            raise Exception

    def __str__(self):
        return 'DateTimeSplit'


# Tertiary broadcast op
@sa((DaskDfSplit(), DaskDfSplit(), DaskDfSplit()), {}, DaskDfSplit(), gpu=True)
def set(df, index, val):
    df[index] = val

# Binary broadcast ops
_args = (DaskDfSplit(), DaskDfSplit())
_ret = DaskDfSplit()

@sa(dc(_args), {}, dc(_ret), gpu=True)
def groupby(df, keys):
    return df.groupby(keys)

@sa(dc(_args), {}, dc(_ret), gpu=True)
def index(df, index):
    return df[index]

@sa(dc(_args), {}, dc(_ret), gpu=True)
def query(df, query):
    return df.query(query)

@sa(dc(_args), {}, dc(_ret), gpu=True)
def divide(df, val):
    return df / val

# Unary broadcast ops
@sa((DaskDfSplit(),), {}, DaskDfSplit(), gpu=True)
def mean(df):
    return df.mean()

# Datetime
@sa((DaskDfSplit(),), {}, DateTimeSplit(), gpu=True)
def dt(df):
    return df.dt

@sa((DateTimeSplit(),), {}, DaskDfSplit(), gpu=True)
def hour(dt):
    return dt.hour

# Compute
@sa((DaskDfSplit(),), {}, DataFrameSplit(), gpu=True)
def compute(df):
    return df.compute()

# Allocation
read_csv = alloc(DaskDfSplit(), gpu=True, gpu_func=dask_cudf.read_csv)(dask.dataframe.read_csv)

