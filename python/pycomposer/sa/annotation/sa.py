from .annotation import Annotation, Allocation, mut
from .config import config
from .dag import LogicalPlan, evaluate_dag
from .split_types import *

import functools

import copy

# The task graph.
_DAG = LogicalPlan()

class sa(object):
    """ A split annotation.

    Split annotations are Python decorators over ordinary Python functions or
    methods. For each function, a split annotation uses *split types* to define
    how each argument in the function is split. By splitting function
    arguments, the underlying runtime can introduce parallelization and
    optimizations such as loop pipelining.

    """

    def __init__(self, types, kwtypes, return_type, gpu=False, gpu_func=None, splittable=True):
        """ Creates a split annotation (SA) for the provided function signature.

        The SA can either be used as a Python decorator or as a function that
        is called on another function. For the latter, users should
        ``deepcopy`` this class for each annotated function.

        Parameters
        ----------

        postypes : tuple of SplitType
            a tuple of split types for each positional argument. The number of
            elements in the tuple must match the number of positional arguments
            in the function.

        kwtypes : dict from str -> SplitType or None
            a dictionary of split types for each keyword argument. Providing
            split types for keyword arguments is optional. If a keyword
            argument does not have a split type, its split type will default to
            "broadcast."

        return_type : SplitType or None
            split type of the value returned by this function.

        gpu : boolean
            whether the annotated function can run on the gpu.

        gpu_func : callable function or None
            the function to call on the inputs if offloaded to the gpu in
            place if the original function, if they are different.

        splittable : boolean
            whether the annotated function can be split. defaults to true.

        """
        self.types = types
        self.kwtypes = kwtypes
        self.return_type = return_type
        self.gpu = gpu
        self.gpu_func = gpu_func
        self.splittable = splittable

    def __call__(self, func):
        annotation = Annotation(func,
                                self.types,
                                self.kwtypes,
                                self.return_type,
                                self.gpu,
                                self.gpu_func,
                                self.splittable)

        @functools.wraps(func)
        def _decorated(*args, **kwargs):
            return _DAG.register(func, args, kwargs, annotation)

        return _decorated

class alloc(object):
    """An allocation annotation.

    Python decorators over ordinary Python allocation functions for split types.
    By default, annotated on allocation functions on the CPU, with options to
    allocate memory on different backends.
    """

    def __init__(self, return_type, gpu=False, gpu_func=None):
        self.return_type = return_type
        self.gpu = gpu
        self.gpu_func = gpu_func

    def __call__(self, func):
        annotation = Allocation(func, self.return_type, self.gpu, self.gpu_func)

        @functools.wraps(func)
        def _decorated(*args, **kwargs):
            return _DAG.register(func, args, kwargs, annotation)

        return _decorated


class sa_gpu(object):
    """An annotation to indicate another function annotation is eligible for the GPU.

    Optionally supply a gpu function to run when executing on the gpu.
    """

    def __init__(self, types, kwtypes, return_type, func=None, splittable=True):
        self.types = types
        self.kwtypes = kwtypes
        self.return_type = return_type
        self.func = func
        self.splittable = splittable

    def __call__(self, func):
        annotation = Annotation(func,
                                self.types,
                                self.kwtypes,
                                self.return_type,
                                True,
                                self.func,
                                self.splittable)

        @functools.wraps(func)
        def _decorated(*args, **kwargs):
            return _DAG.register(func, args, kwargs, annotation)

        return _decorated


class alloc_gpu(object):
    """An annotation to indicate another allocation annotation is eligible for the GPU.

    Optionally supply a gpu function to run when executing on the gpu.
    """

    def __init__(self, return_type, func=None):
        self.return_type = return_type
        self.func = func

    def __call__(self, func):
        annotation = Allocation(func, self.return_type, True, self.func)

        @functools.wraps(func)
        def _decorated(*args, **kwargs):
            return _DAG.register(func, args, kwargs, annotation)

        return _decorated


def evaluate(workers=config["workers"],
             batch_size=config["batch_size"],
             profile=False,
             force_cpu=False,
             paging=False):
    evaluate_dag(_DAG, workers, batch_size, profile, force_cpu, paging)
