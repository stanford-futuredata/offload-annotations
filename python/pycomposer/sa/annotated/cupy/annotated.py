
import numpy as np
import scipy.special as ss
import sharedmem
import cupy as cp

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *

class NdArraySplit(SplitType):

    def __init__(self):
        self.slice_col = False
        # self.merge = False
        self.merge = True
        self.supported_backends = [Backend.CPU, Backend.GPU]

    def combine(self, values, original=None):
        if self.merge and original is not None:
            assert isinstance(original, np.ndarray)
            original.data = np.concatenate(values)
            return original
        if self.merge:
            return np.concatenate(values)

    def split(self, start, end, value):
        if isinstance(value, np.ndarray) or isinstance(value, cp.ndarray):
            shape = value.shape
            ndims = len(value.shape)
            if ndims == 1:
                if start >= shape[0]:
                    return STOP_ITERATION
                return value[start:min(end, shape[0])]
            elif ndims == 2:
                if shape[1] == 1:
                    return value
                if self.slice_col:
                    return value[:,start:end]
                else:
                    return value[start:end,:]
            else:
                return NotImplementedError("ndarray with dim > 2 not supported")
        else:
            # Scalar.
            return value

    def elements(self, value):
        if isinstance(value, np.ndarray):
            if len(value.shape) == 2 and value.shape[1] == 1:
                return value.shape[0]
            return value.shape[-1]

    def backend(self, value):
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return Backend.SCALAR
        elif isinstance(value, np.ndarray):
            return Backend.CPU
        elif isinstance(value, cp.ndarray):
            return Backend.GPU
        else:
            raise Exception('unknown device: {}'.format(type(value)))

    def to(self, value, backend):
        current_backend = self.backend(value)
        if current_backend == Backend.SCALAR or current_backend == backend:
            return value
        elif current_backend == Backend.CPU and backend == Backend.GPU:
            self.merge = True
            return cp.array(value)
        elif current_backend == Backend.GPU and backend == Backend.CPU:
            return cp.asnumpy(value)
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return "NdArraySplit"

_args = (NdArraySplit(), NdArraySplit())
_kwargs = { 'out' : mut(NdArraySplit()), 'axis': Broadcast() }
_ret = NdArraySplit()


# Binary ops.
add         = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.add)(np.add)
subtract    = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.subtract)(np.subtract)
multiply    = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.multiply)(np.multiply)
divide      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.divide)(np.divide)
power       = sa(dc(_args), dc(_kwargs), dc(_ret))(np.power)

_args = (NdArraySplit(),)

# Unary ops.
log         = sa(dc(_args), dc(_kwargs), dc(_ret))(np.log)
log2        = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.log2)(np.log2)
exp         = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.exp)(np.exp)
sin         = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.sin)(np.sin)
arcsin      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.arcsin)(np.arcsin)
cos         = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.cos)(np.cos)
sqrt        = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=cp.sqrt)(np.sqrt)
erf         = sa(dc(_args), dc(_kwargs), dc(_ret))(ss.erf)

# addreduce = np.add.reduce
addreduce = sa(dc(_args), dc(_kwargs), dc(_ret))(np.add.reduce)

def ones(shape, dtype=None, order='C'):
    result = sharedmem.empty(shape)
    result[:] = np.ones(shape, dtype, order)[:]
    return result

def zeros(shape, dtype=None, order='C'):
    result = sharedmem.empty(shape)
    result[:] = np.zeros(shape, dtype, order)[:]
    return result

@alloc(NdArraySplit(), gpu=True, gpu_func=cp.empty)
def empty(shape, dtype=None):
    return sharedmem.empty(shape)
