import torch
import numpy as np
import sharedmem

from copy import deepcopy as dc
from sa.annotation import *
from sa.annotation.split_types import *

float64 = torch.float64

def ones(size, device=torch.device('cpu'), dtype=torch.float64):
    res = torch.ones(size, device=device, dtype=dtype)
    res.share_memory_()
    return res

class TorchTensorSplit(SplitType):

    def __init__(self):
        self.slice_col = False
        self.merge = False
        self.gpu = True

    def combine(self, values, original=None):
        if self.merge:
            return torch.cat(values)
        if original is not None:
            assert isinstance(original, torch.Tensor)
            original.data = torch.cat(values)
            return original

    def split(self, start, end, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            # import pdb; pdb.set_trace()
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
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            if len(value.shape) == 2 and value.shape[1] == 1:
                return value.shape[0]
            return value.shape[-1]

    def to_device(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            return value.to(torch.device('cuda'))
        else:
            return value

    def to_host(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            return value.to(torch.device('cpu'))
        else:
            return value

    def __str__(self):
        return "TorchTensorSplit"

_args = (TorchTensorSplit(), TorchTensorSplit())
_kwargs = { 'out' : mut(TorchTensorSplit()), 'axis': Broadcast() }
_ret = TorchTensorSplit()

def dont_call_me(*args, **kwargs):
    raise Exception('i said dont call me!!!!')

# Binary ops.
add      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True, gpu_func=torch.add)(dont_call_me)
sub      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.sub)
mul      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.mul)
div      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.div)

_args = (TorchTensorSplit(),)

# Unary ops.
log2     = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.log2)
exp      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.exp)
sqrt     = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.sqrt)
erf      = sa(dc(_args), dc(_kwargs), dc(_ret), gpu=True)(torch.erf)
