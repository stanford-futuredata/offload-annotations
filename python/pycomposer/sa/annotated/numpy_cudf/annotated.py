
import numpy as np
import cudf

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
            return value.shape[0]

    def backend(self, value):
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return Backend.SCALAR
        elif isinstance(value, np.ndarray):
            return Backend.CPU
        elif isinstance(value, cudf.DataFrame) or isinstance(value, cudf.Series):
            return Backend.GPU
        else:
            raise Exception('unknown device: {}'.format(type(value)))

    def to(self, value, backend):
        current_backend = self.backend(value)
        if current_backend == Backend.SCALAR or current_backend == backend:
            return value
        elif current_backend == Backend.CPU and backend == Backend.GPU:
            self.merge = True
            if len(value.shape) == 1:
                return cudf.from_pandas(pd.Series(value))
            else:
                return cudf.from_pandas(pd.DataFrame(value))
        elif current_backend == Backend.GPU and backend == Backend.CPU:
            return value.to_pandas()
        else:
            raise Exception('cannot transfer from {} to {}'.format(current_backend, backend))

    def __str__(self):
        return "NdArraySplit"
