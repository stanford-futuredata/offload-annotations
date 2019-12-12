
import numpy as np
import torch

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *

float64 = torch.float64
cuda = torch.cuda

def ones(*args, **kwargs):
    return torch.ones(*args, **kwargs)

class TorchTensorSplit(SplitType):

    def __init__(self):
        self.slice_col = False
        # self.merge = False
        # TODO: hack because TorchTensorSplits that come out of allocations must be merged.
        # Assume all data must be merged given that we only materialize certain inputs
        # (otherwise we can set the materialize attribute to False). If it turns out a process's
        # merge time is longer than expected compared to, say, master, it might be because
        # we're merging here rather than assuming shared memory.
        self.merge = True
        self.supported_backends = [Backend.CPU, Backend.GPU]

    def combine(self, values, original=None):
        if self.merge and original is not None:
            assert isinstance(original, torch.Tensor)
            original[:] = torch.cat(values)
            return original
        if self.merge:
            return torch.cat(values)

    def split(self, start, end, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            shape = value.shape
            ndims = len(value.shape)
            if ndims == 1:
                if start >= shape[0]:
                    return STOP_ITERATION
                # return value[start:min(end, shape[0])]
                return value[start:end]
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
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if len(value.shape) == 2 and value.shape[1] == 1:
                return value.shape[0]
            return value.shape[-1]

    def backend(self, value):
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return Backend.SCALAR
        elif isinstance(value, torch.Tensor) and value.device.type == 'cpu':
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
            return value.to(torch.device('cuda'), non_blocking=True)
        elif current_backend == Backend.GPU and backend == Backend.CPU:
            self.merge = True
            return value.to(torch.device('cpu'), non_blocking=True)
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return "TorchTensorSplit"

def _gpu_empty(size, *args, **kwargs):
    return torch.empty(size, *args, **kwargs, device=torch.device('cuda'))

@alloc(TorchTensorSplit(), gpu=True, gpu_func=_gpu_empty)
def empty(*args, **kwargs):
    return torch.empty(*args, **kwargs, device=torch.device('cpu'))

_args = (TorchTensorSplit(), TorchTensorSplit())
_kwargs = { 'out' : mut(TorchTensorSplit()), 'axis': Broadcast() }
_ret = TorchTensorSplit()

# Binary ops.
add      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.add)
sub      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.sub)
mul      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.mul)
div      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.div)

_args = (TorchTensorSplit(),)

# Unary ops.
log2     = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.log2)
exp      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.exp)
sqrt     = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.sqrt)
erf      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.erf)
