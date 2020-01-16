"""Simple cost model functions for scheduling algorithm.

The scheduler automatically suggests all operations that are eligible for the GPU to run on the
GPU, and remaining operations to run on the CPU. These operations are serialized into a list of
instructions. Before these instructions run, if estimators exist, they do a dynamic analysis of
the cost to transfer the inputs to a specific backend and run the function on that backend for
each backend: the CPU and the GPU. The instruction selects the backend with the lower cost and
transfers inputs as appropriate.

This class provides the most basic estimators for transfer and compute.

All estimators must satisfy the following interface, depending on whether they are
for estimating transfer or compute:

def transfer_estimate(value, ty, backend):
    # Estimates the transfer cost, returns a non-negative value.
    #
    # Parameters
    # ----------
    # value : object
    #     object to be transferred
    # ty : SplitType
    #     split type of the object
    # backend : Backend
    #     the backend the type needs to be transferred to
    pass

def compute_estimate(values, tys, backend):
    # Estimates the compute cost, returns a non-negative value.
    #
    # Parameters
    # ----------
    # values : List[object]
    #     function parameters
    # tys : List[SplitType]
    #     function parameter types
    # backend : Backend
    #     function backend
    pass
"""
import math
from sa.annotation import Backend


def gen_linear_transfer_estimator(a, b):
    """Generates a linear heuristic for estimating transfer cost.
    """
    def e(ty, value, backend):
        x = math.log2(ty.elements(value))
        return a * x + b
    return e

def gen_linear_compute_estimator(a_cpu, b_cpu, a_gpu, b_gpu):
    """Generates a linear heuristic for estimating compute cost.
    """
    def e(tys, values, backend):
        x = math.log2(tys[0].elements(values[0]))
        if backend == Backend.CPU:
            return a_cpu * x + b_cpu
        else:
            return a_gpu * x + b_gpu
    return e
