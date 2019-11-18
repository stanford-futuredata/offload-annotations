
from abc import ABC, abstractmethod
import types

from .context import Context
from .driver import STOP_ITERATION
from ..backend import Backend

class Instruction(ABC):
    """
    An instruction that updates an operation in a lazy DAG.
    """

    @abstractmethod
    def evaluate(self, thread, start, end, values, context: Context):
        """
        Evaluates an instruction.

        Parameters
        ----------

        thread : the thread that is  currently executing
        start : the start index of the current split value.
        end : the end index of the current split value
        values : a global value map holding the inputs.
        context : context holding execution state.

        """
        pass

class Split(Instruction):
    """
    An instruction that splits the inputs to an operation.
    """

    def __init__(self, target, ty):
        """
        A Split instruction takes an argument and split type and applies
        the splitter on the argument.

        Parameters
        ----------

        target : the arg ID that will be split.
        ty : the split type.
        """
        self.target = target
        self.ty = ty
        self.splitter = None

    def __str__(self):
        return "v{} = split {}:{}".format(self.target, self.target, self.ty)

    def evaluate(self, thread, start, end, values, context):
        """ Returns values from the split. """

        if self.splitter is None:
            # First time - check if the splitter is actually a generator.
            result = self.ty.split(start, end, values[self.target])
            if isinstance(result, types.GeneratorType):
                self.splitter = result
                result = next(self.splitter)
            else:
                self.splitter = self.ty.split
        else:
            if isinstance(self.splitter, types.GeneratorType):
                result = next(self.splitter)
            else:
                result = self.splitter(start, end, values[self.target])

        if isinstance(result, str) and result == STOP_ITERATION:
            return STOP_ITERATION

        context.add(self.target, result)

class Merge(Instruction):
    """
    An instruction that merges the outputs of an operation.
    """

    def __init__(self, target, ty):
        """
        TODO(ygina)
        """
        self.target = target
        self.ty = ty

    def __str__(self):
        return "v{} = merge {}:{}".format(self.target, self.target, self.ty)

    def evaluate(self, _thread, _start, _end, _values, _context):
        pass

class Call(Instruction):
    """ An instruction that calls an SA-enabled function. """
    def __init__(self,  target, func, args, kwargs, ty, on_gpu):
        self.target = target
        # Function to call.
        self.func = func
        # Arguments: list of targets.
        self.args = args
        # Keyword arguments: Maps { name -> target }
        self.kwargs = kwargs
        # Return split type.
        self.ty = ty
        # Whether the call is executed on the GPU.
        self.on_gpu = on_gpu

    def __str__(self):
        args = ", ".join(map(lambda a: "v" + str(a), self.args))
        kwargs = list(map(lambda v: "{}=v{}".format(v[0], v[1]), self.kwargs.items()))
        arguments = ", ".join([args] + kwargs)
        prefix = ""
        if self.on_gpu:
            prefix += "(gpu) "
        if self.target is not None:
            prefix += "v{} = ".format(self.target)
        return prefix + "call {}({}):{}".format(self.func.__name__, arguments, str(self.ty))

    def get_args(self, context: Context):
        return [ context.get_last_value(target) for target in self.args ]

    def get_kwargs(self, context: Context):
        return dict([ (name, context.get_last_value(target)) for (name, target) in self.kwargs.items() ])

    def evaluate(self, _thread, _start, _end, _values, context):
        """
        Evaluates a function call by gathering arguments and calling the
        function.

        """
        args = self.get_args(context)
        kwargs = self.get_kwargs(context)
        result = self.func(*args, **kwargs)
        if self.target is not None:
            context.add(self.target, result)

    def remove_target(self):
        self.target = None

class To(Instruction):
    def __init__(self, target, ty, backend):
        self.target = target
        self.ty = ty
        self.backend = backend

    def __str__(self):
        prefix = "(gpu) " if self.backend == Backend.GPU else ""
        return "{}v{} = to_{}:{}".format(
            prefix, self.target, self.backend.value, str(self.ty))

    def evaluate(self, _thread, _start, _end, _values, context):
        old_value = context.get_last_value(self.target)
        new_value = self.ty.to(old_value, self.backend)
        context.set_last_value(self.target, new_value)
