"""
Annotations for Pandas functions.

Note: For convinience, we just write a wrapper function that calls the Pandas function, and then
use those functions instead. We could equivalently just replace methods on the DataFrame class too and
split `self` instead of the DataFrame passed in here.
"""

import numpy as np
import pandas as pd
import time
import cudf

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *
from sa.annotation.backend import Backend

class UniqueSplit(SplitType):
    supported_backends = [Backend.CPU, Backend.GPU]

    """ For the result of Unique """
    def combine(self, values, original=None):
        if len(values) > 0:
            result = np.unique(np.concatenate(values))
        else:
            result = np.array([])
        if original is not None:
            assert isinstance(original, np.ndarray)
            original.data = result
        return result

    def split(self, values):
        raise ValueError

    def backend(self, value):
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            return Backend.CPU
        elif isinstance(value, cudf.DataFrame) or isinstance(value, cudf.Series):
            return Backend.GPU
        elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return Backend.SCALAR
        else:
            raise Exception('unknown backend: {}'.format(type(value)))

    def to(self, value, backend):
        current_backend = self.backend(value)
        if current_backend == Backend.SCALAR or current_backend == backend:
            return value
        elif current_backend == Backend.CPU and backend == Backend.GPU:
            return cudf.from_pandas(value)
        elif current_backend == Backend.GPU and backend == Backend.CPU:
            return value.to_pandas()
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return 'UniqueSplit'

class DataFrameSplit(SplitType):
    supported_backends = [Backend.CPU, Backend.GPU]

    def combine(self, values, original=None):
        do_combine = False
        for val in values:
            if val is not None:
                do_combine = True

        if do_combine and len(values) > 0:
            backends = set([self.backend(val) for val in values])
            assert len(backends) == 1
            backend = backends.pop()
            if backend == Backend.CPU:
                result = pd.concat(values)
            elif backend == Backend.GPU:
                result = cudf.concat(values)
            else:
                raise ValueError
            if original is not None:
                assert isinstance(original, np.ndarray)
                original.data = result
            return result

    def split(self, start, end, value):
        if self.backend(value) not in self.supported_backends:
            # Assume this is a constant (str, int, etc.).
            return value
        return value[start:end]

    def elements(self, value):
        if self.backend(value) not in self.supported_backends:
            return None
        return len(value)

    def backend(self, value):
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            return Backend.CPU
        elif isinstance(value, cudf.DataFrame) or isinstance(value, cudf.Series):
            return Backend.GPU
        elif isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return Backend.SCALAR
        else:
            return None

    def to(self, value, backend):
        current_backend = self.backend(value)
        if current_backend == Backend.SCALAR or current_backend == backend:
            return value
        elif current_backend == Backend.CPU and backend == Backend.GPU:
            return cudf.from_pandas(value)
        elif current_backend == Backend.GPU and backend == Backend.CPU:
            return value.to_pandas()
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return 'DataFrameSplit'

class SumSplit(SplitType):
    supported_backends = [Backend.CPU, Backend.GPU]

    def combine(self, values):
        return sum(values)

    def split(self, start, end, value):
        raise ValueError("can't split sum values")

    def backend(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return Backend.SCALAR
        else:
            raise Exception('unknown backend: {}'.format(type(value)))

    def to(self, value, backend):
        if self.backend(value) == Backend.SCALAR:
            return value
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return 'SumSplit'

class GroupBySplit(SplitType):
    def combine(self, values):
        return None

    def split(self, start, end, value):
        raise ValueError("can't split groupby values")

class SizeSplit(SplitType):
    def combine(self, values):
        return pd.concat(values)

    def split(self, start, end, value):
        raise ValueError("can't split size values")

def dfgroupby(df, keys):
    return df.groupby(keys)

def merge(left, right):
    return pd.merge(left, right)

def gbapply(grouped, func):
    return grouped.apply(func)

def gbsize(grouped):
    return grouped.size()

def filter(df, column, target):
    return df[df[column] > target]

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def divide(series, value):
    result = (series / value)
    return result

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def multiply(series, value):
    result = (series * value)
    return result

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def subtract(series, value):
    result = (series - value)
    return result

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def add(series, value):
    result = (series + value)
    return result

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def equal(series, value):
    result = (series == value)
    return result

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def greater_than(series, value):
    result = (series >= value)
    return result

@sa((DataFrameSplit(), DataFrameSplit()), {}, DataFrameSplit(), gpu=False)
def less_than(series, value):
    result = (series < value)
    return result

@sa((DataFrameSplit(),), {}, SumSplit(), gpu=False)
def pandasum(series):
    result = series.sum()
    return result

@sa((DataFrameSplit(),), {}, UniqueSplit(), gpu=False)
def unique(series):
    result = series.unique()
    return result

@sa((DataFrameSplit(),), {}, DataFrameSplit())
def series_str(series):
    result = series.str
    return result

def gpu_mask(series, cond, val):
    clone = series.copy()
    clone.loc[cond] = val
    return clone

@sa((DataFrameSplit(), DataFrameSplit(), Broadcast()), {}, DataFrameSplit(), gpu=False, gpu_func=gpu_mask)
def mask(series, cond, val):
    result = series.mask(cond, val)
    return result

@sa((DataFrameSplit(), Broadcast(), Broadcast()), {}, DataFrameSplit(), gpu=False)
def series_str_slice(series, start, end):
    result = series.str.slice(start, end)
    return result

@sa((DataFrameSplit(),), {}, DataFrameSplit())
def pandanot(series):
    return ~series

@sa((DataFrameSplit(), Broadcast()), {}, DataFrameSplit())
def series_str_contains(series, target):
    result = series.str.contains(target)
    return result

@alloc(DataFrameSplit(), gpu=True, gpu_func=cudf.read_csv)
def read_csv(filename, names=None):
    return pd.read_csv(filename, names=names)

dfgroupby = sa((DataFrameSplit(), Broadcast()), {}, GroupBySplit())(dfgroupby)
merge = sa((DataFrameSplit(), Broadcast()), {}, DataFrameSplit())(merge)
filter = sa((DataFrameSplit(), Broadcast(), Broadcast()), {}, DataFrameSplit())(filter)

# Return split type should be ApplySplit(subclass of DataFrameSplit), and it
# should take the first argument as a parameter. The parameter is guaranteed to
# be a dag.Operation.  The combiner can then use the `by` arguments to groupby
# in the combiner again, and then apply again.
gbapply = sa((GroupBySplit(), Broadcast()), {}, DataFrameSplit())(gbapply)
gbsize = sa((GroupBySplit(), Broadcast()), {}, SizeSplit())(gbsize)
