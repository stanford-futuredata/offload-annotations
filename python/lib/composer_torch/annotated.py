from pycomposer import *
import time

import sharedmem
import numpy as np
import torch

from copy import deepcopy as dc

float64 = torch.float64

def ones(size, device=torch.device('cpu'), dtype=torch.float64):
    res = torch.ones(size, device=device, dtype=dtype)
    res.share_memory_()
    return res

class TorchTensorSplit(SplitType):

    def __init__(self):
        self.slice_col = False
        self.merge = False

    def combine(self, values):
        if self.merge:
            return torch.cat(values)

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

    def __str__(self):
        return "TorchTensorSplit"

_args = (TorchTensorSplit(), TorchTensorSplit())
_kwargs = { 'out' : mut(TorchTensorSplit()), 'axis': Broadcast() }
_ret = TorchTensorSplit()


# Binary ops.
add      = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.add)
sub      = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.sub)
mul      = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.mul)
div      = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.div)

_args = (TorchTensorSplit(),)

# Unary ops.
log2     = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.log2)
exp      = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.exp)
sqrt     = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.sqrt)
erf      = sa(dc(_args), dc(_kwargs), dc(_ret))(torch.erf)
