
import numpy as np
import scipy.special as ss
import sharedmem
import cupy as cp
import cudf

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.scheduling import *
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
            # if len(value.shape) == 2 and value.shape[1] == 1:
            #     return value.shape[0]
            # return value.shape[-1]
            return value.shape[0]

    def backend(self, value):
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return Backend.SCALAR
        elif isinstance(value, np.ndarray):
            return Backend.CPU
        elif isinstance(value, cp.ndarray):
            return Backend.GPU
        elif isinstance(value, cudf.Series) or isinstance(value, cudf.DataFrame):
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
            if isinstance(value, cp.ndarray) or isinstance(value, cudf.Series):
                return cp.asnumpy(value)
            elif isinstance(value, cudf.DataFrame):
                # Also convert back from cuDF objects
                return np.asarray(value.as_gpu_matrix())
        raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return "NdArraySplit"

_args = (NdArraySplit(), NdArraySplit())
_kwargs = { 'out' : mut(NdArraySplit()), 'axis': Broadcast() }
_ret = NdArraySplit()


# Binary ops.
add         = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.add)(np.add)
subtract    = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.subtract)(np.subtract)
multiply    = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.multiply)(np.multiply)
divide      = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.divide)(np.divide)
power       = sa(dc(_args), dc(_kwargs), dc(_ret))(np.power)

_args = (NdArraySplit(),)

# https://github.com/cupy/cupy/blob/master/cupyx/scipy/special/erf.py
cp_erf = cp.core.create_ufunc(
    'cupyx_scipy_erf', ('f->f', 'd->d'),
    'out0 = erf(in0)',
    doc='''Error function.
    .. seealso:: :meth:`scipy.special.erf`
    ''')

# Unary ops.
log         = sa(dc(_args), dc(_kwargs), dc(_ret))(np.log)
log2        = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.log2)(np.log2)
exp         = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.exp)(np.exp)
sin         = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.sin)(np.sin)
arcsin      = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.arcsin)(np.arcsin)
cos         = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.cos)(np.cos)
sqrt        = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.sqrt)(np.sqrt)
erf         = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp_erf)(ss.erf)
mean        = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.mean)(np.mean)
std         = oa(dc(_args), dc(_kwargs), dc(_ret), func=cp.std)(np.std)

# addreduce = np.add.reduce
addreduce = sa(dc(_args), dc(_kwargs), dc(_ret))(np.add.reduce)

def ones(shape, dtype=None, order='C'):
    # result = sharedmem.empty(shape)
    # result[:] = np.ones(shape, dtype, order)[:]
    # return result
    return np.ones(shape, dtype, order)

def zeros(shape, dtype=None, order='C'):
    # result = sharedmem.empty(shape)
    # result[:] = np.zeros(shape, dtype, order)[:]
    # return result
    return np.zeros(shape, dtype, order)

@oa_alloc(NdArraySplit(), func=cp.empty)
def empty(shape, dtype=None):
    # return sharedmem.empty(shape)
    return np.empty(shape, dtype=dtype)
