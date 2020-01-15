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
def transfer_estimator(value, ty, backend):
    if backend == ty.value(backend):
        return 0
    return 0.01 * ty.elements(value)

def compute_estimator(values, tys, backend):
    if backend == Backend.CPU:
        return 100
    else:
        return 0
