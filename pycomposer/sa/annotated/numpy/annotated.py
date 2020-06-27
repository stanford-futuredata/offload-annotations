
import numpy as np
import scipy.special as ss
import sharedmem
import torch

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *

class NdArraySplit(OffloadSplitType):

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
        if isinstance(value, np.ndarray):
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
        elif isinstance(value, torch.Tensor) and value.device.type == 'cuda':
            return Backend.GPU
        else:
            raise Exception('unknown device: {}'.format(type(value)))

    def to(self, value, backend):
        current_backend = self.backend(value)
        if current_backend == Backend.SCALAR or current_backend == backend:
            return value
        elif current_backend == Backend.CPU and backend == Backend.GPU:
            self.merge = True
            return torch.from_numpy(value).to(torch.device('cuda'))
        elif current_backend == Backend.GPU and backend == Backend.CPU:
            return value.to(torch.device('cpu')).numpy()
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return "NdArraySplit"

_args = (NdArraySplit(), NdArraySplit())
_kwargs = { 'out' : mut(NdArraySplit()), 'axis': Broadcast() }
_ret = NdArraySplit()


# Binary ops.
add         = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.add)(np.add)
subtract    = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.sub)(np.subtract)
multiply    = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.mul)(np.multiply)
divide      = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.div)(np.divide)
power       = sa(dc(_args), dc(_kwargs), dc(_ret))(np.power)

_args = (NdArraySplit(),)

# Unary ops.
log         = sa(dc(_args), dc(_kwargs), dc(_ret))(np.log)
log2        = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.log2)(np.log2)
exp         = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.exp)(np.exp)
sin         = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.sin)(np.sin)
arcsin      = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.asin)(np.arcsin)
cos         = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.cos)(np.cos)
sqrt        = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.sqrt)(np.sqrt)
erf         = sa_gpu(dc(_args), dc(_kwargs), dc(_ret), func=torch.erf)(ss.erf)

# addreduce = np.add.reduce
addreduce = sa(dc(_args), dc(_kwargs), dc(_ret))(np.add.reduce)

def ones(shape, dtype=None, order='C'):
    result = sharedmem.empty(shape)
    result[:] = np.ones(shape, dtype, order)[:]
    return result
    # return np.ones(shape, dtype, order)

def zeros(shape, dtype=None, order='C'):
    result = sharedmem.empty(shape)
    result[:] = np.zeros(shape, dtype, order)[:]
    return result
    # return np.zeros(shape, dtype, order)

def _gpu_empty(shape, dtype=None):
    str_to_torch = {
        'float64': torch.float64,
        'float32': torch.float32,
        'int64': torch.int64,
        'int32': torch.int32,
    }
    return torch.empty(shape, dtype=str_to_torch[dtype], device=torch.device('cuda'))

@alloc_gpu(NdArraySplit(), func=_gpu_empty)
def empty(shape, dtype=None):
    return sharedmem.empty(shape)
    # return np.empty(shape, dtype)
