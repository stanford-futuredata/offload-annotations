import functools
import logging

from inspect import signature, Parameter, Signature
from .split_types import Broadcast

class Mut(object):
    """ Marker that marks values in an annotation as mutable. """

    __slots__ = [ "value" ]
    def __init__(self, value):
        self.value = value

def mut(x):
    """ A function that marks an argument as mutable.

    If a function mutates an argument (e.g., removing an item from a list), the
    SA should mark the split type for that argument with this function. This
    allows the runtime to track data dependencies among functions correctly.

    Parameters
    ----------
    x : SplitType
        Split type for argument that will be mutated within the annotation.

    Returns
    -------
    SplitType

    """
    return Mut(x)

class BaseAnnotation(object):

    __slots__ = [ "mutables", "arg_types", "return_type", "kwarg_types", "gpu", "gpu_func", "estimator" ]

    def types(self):
        """ Iterate over the split types in this annotation. """
        for ty in self.arg_types:
            yield ty
        for ty in self.kwarg_types.values():
            yield ty
        yield self.return_type

    def __str__(self):
        if len(self.arg_types) > 0:
            args = ", ".join([str(t) for t in self.arg_types])
        else:
            args = ", " if len(self.kwarg_types) > 0 else ""

        if len(self.kwarg_types) > 0:
            args += ", "
            args += ", ".join(["{}={}".format(k, v) for (k,v) in self.kwarg_types.items()])

        return "({}) -> {}".format(args, self.return_type)


class Allocation(BaseAnnotation):
    """ An annotation on an allocation function.
    """

    def __init__(self, func, return_type, gpu, gpu_func):
        """Initialize an annotation for an allocation with alternatives for different backends.

        Parameters
        __________

        func : the function that was invoked.
        return_type: the return type of the function.
        gpu: whether the function can run on the gpu.
        gpu_func : the gpu version of the function that was invoked, if it exists.

        """
        self.func = func
        self.return_type = return_type
        self.gpu = gpu
        self.gpu_func = gpu_func

        self.mutables = set()
        self.arg_types = []
        self.kwarg_types = {}
        self.estimator = None


class Annotation(BaseAnnotation):
    """ An annotation on a function.

    Annotations map arguments (by index for regular arguments and by name for
    keyword arguments) to their split type.

    """

    def __init__(self, func, types, kwtypes, return_type, gpu, gpu_func, splittable, estimator):
        """Initialize an annotation for a function invocation with the given
        arguments.

        Parameters
        __________

        func : the function that was invoked.
        types : the split types of the non-keyword arguments and return type.
        kwtypes : the split types of the keyword arguments.
        gpu: whether the function can run on the gpu.
        gpu_func : the gpu version of the function that was invoked, if it exists.
        splittable : whether the function is splittable.
        estimator: the compute estimator function

        """

        try:
            sig = signature(func)
            args = [(name, param) for (name, param) in sig.parameters.items()\
                    if param.kind == Parameter.POSITIONAL_OR_KEYWORD]

            num_required_types = 0
            for (name, param) in args:
                if param.default is Parameter.empty:
                    num_required_types += 1

            if len(types) != num_required_types:
                raise ValueError("invalid number of arguments in annotation (expected {}, got {})".format(len(args), len(types)))

            # Make sure there's no extraneous args.
            kwargs = set([name for (name, param) in args if param.default is not Parameter.empty])

            for name in kwargs:
                if name not in kwtypes:
                    kwtypes[name] = Broadcast()

            for name in kwtypes:
                assert(name in kwargs)

        except ValueError as e:
            logging.info("Continuing without verification of annotation")

        # The mutable values. These are indices for positionals and string
        # names for keyword args.
        self.mutables = set()

        # The argument types.
        self.arg_types = []

        for (i, ty) in enumerate(types):
            if isinstance(ty, Mut):
                self.arg_types.append(ty.value)
                self.mutables.add(i)
            else:
                self.arg_types.append(ty)

        # The return type. This can be None if the function doesn't return anything.
        self.return_type = return_type

        # Dictionary of kwarg types.
        self.kwarg_types = dict()
        for (key, value) in kwtypes.items():
            if isinstance(value, Mut):
                self.kwarg_types[key] = value.value
                self.mutables.add(key)
            else:
                self.kwarg_types[key] = value

        # GPU function version.
        self.gpu = gpu
        self.gpu_func = gpu_func

        # Whether the function is splittable.
        self.splittable = splittable

        # The compute estimator for this function.
        self.estimator = estimator

