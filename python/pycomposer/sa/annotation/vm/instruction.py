
from abc import ABC, abstractmethod
import types
import time

from .driver import STOP_ITERATION
from ..backend import Backend

class Instruction(ABC):
    """
    An instruction that updates an operation in a lazy DAG.
    """

    @abstractmethod
    def evaluate(self, thread, batch_index, values, context, last_batch):
        """
        Evaluates an instruction.

        Parameters
        ----------

        thread : the thread that is  currently executing
        batch_index : the index of the current split batch.
        values : a global value map holding the inputs.
        context : map holding execution state (arg ID -> value).

        """
        pass

class Split(Instruction):
    """
    An instruction that splits the inputs to an operation.
    """

    def __init__(self, target, ty, backend, batch_size):
        """
        A Split instruction takes an argument and split type and applies
        the splitter on the argument.

        Parameters
        ----------

        target : the arg ID that will be split.
        ty : the split type.
        backend : the backend the instruction is executed on.
        batch_size : the batch size of the instruction split.
        """
        self.target = target
        self.ty = ty
        self.splitter = None
        self.backend = backend
        self.batch_size = batch_size
        self.index_to_split = None

    def __str__(self):
        return "({}:{}) v{} = split {}:{}".format(
            self.backend.value, self.batch_size, self.target, self.target, self.ty)

    def get_index_to_split(self, context):
        # The index of the value to split in the target context is the last
        # value with more elements than the batch size.
        if self.index_to_split is None:
            self.index_to_split = 0
            for i in reversed(range(len(context[self.target]))):
                num_elements = self.ty.elements(context[self.target][i])
                if num_elements is None:
                    break  # scalar
                if num_elements > self.batch_size:
                    self.index_to_split = i
                    break
        return self.index_to_split

    def clear_index_to_split(self, context):
        context[self.target].pop(self.index_to_split)
        self.index_to_split = None

    def evaluate(self, thread, batch_index, values, context, last_batch):
        """ Returns values from the split. """
        start = self.batch_size * batch_index
        end = start + self.batch_size
        if len(context[self.target]) == 0:
            context[self.target].append(values[self.target])

        index_to_split = self.get_index_to_split(context)
        if self.splitter is None:
            # First time - check if the splitter is actually a generator.
            result = self.ty.split(start, end, context[self.target][index_to_split])
            if isinstance(result, types.GeneratorType):
                self.splitter = result
                result = next(self.splitter)
            else:
                self.splitter = self.ty.split
        else:
            if isinstance(self.splitter, types.GeneratorType):
                result = next(self.splitter)
            else:
                result = self.splitter(start, end, context[self.target][index_to_split])

        if last_batch:
            self.clear_index_to_split(context)
        if isinstance(result, str) and result == STOP_ITERATION:
            return STOP_ITERATION

        context[self.target].append(result)

class Merge(Instruction):
    """
    An instruction that merges the outputs of an operation.
    """

    def __init__(self, target, ty, backend, batch_size):
        """
        A merge instruction that merges all the values for the target in the
        context. Only inserted in a program prior to changing the batch size.

        Parameters
        ----------
        target : the target to merge
        ty : the split type of the target
        backend : the backend on which the merge is executed
        batch_size : the eventual batch size
        """
        self.target = target
        self.ty = ty
        self.backend = backend
        self.batch_size = batch_size

    def __str__(self):
        return "({}:{}) v{} = merge {}:{}".format(
            self.backend.value, self.batch_size, self.target, self.target, self.ty)

    def evaluate(self, _thread, _batch_index, _values, context, _last_batch):
        # The indices of the values to merge in the target context are the last
        # values that collectively have <= the number of elements in the batch size.
        index_to_merge = None
        total_num_elements = 0
        for i in reversed(range(len(context[self.target]))):
            num_elements = self.ty.elements(context[self.target][i])
            total_num_elements += self.ty.elements(context[self.target][i])
            if total_num_elements > self.batch_size:
                break
            index_to_merge = i
        values_to_merge = context[self.target][index_to_merge:]

        context[self.target] = context[self.target][:index_to_merge]
        context[self.target].append(self.ty.combine(values_to_merge))

class Call(Instruction):
    """ An instruction that calls an SA-enabled function. """
    def __init__(self,  target, func, args, kwargs, ty, backend, batch_size):
        self.target = target
        # Function to call.
        self.func = func
        # Arguments: list of targets.
        self.args = args
        # Keyword arguments: Maps { name -> target }
        self.kwargs = kwargs
        # Return split type.
        self.ty = ty
        # The backend the instruction is executed on.
        self.backend = backend
        # The batch size of the instruction split.
        self.batch_size = batch_size

    def __str__(self):
        args = ", ".join(map(lambda a: "v" + str(a), self.args))
        kwargs = list(map(lambda v: "{}=v{}".format(v[0], v[1]), self.kwargs.items()))
        arguments = ", ".join([args] + kwargs)
        return "({}:{}) {}call {}({}):{}".format(
            self.backend.value,
            self.batch_size,
            "" if self.target is None else "v{} = ".format(self.target),
            self.func.__name__,
            arguments,
            str(self.ty)
        )

    def get_args(self, context):
        return [ context[target][-1] for target in self.args ]

    def get_kwargs(self, context):
        return dict([ (name, context[target][-1]) for (name, target) in self.kwargs.items() ])

    def evaluate(self, _thread, _batch_index, _values, context, _last_batch):
        """
        Evaluates a function call by gathering arguments and calling the
        function.

        """
        args = self.get_args(context)
        kwargs = self.get_kwargs(context)
        result = self.func(*args, **kwargs)
        if self.target is not None:
            context[self.target].append(result)

    def remove_target(self):
        self.target = None

class To(Instruction):
    def __init__(self, target, ty, backend):
        self.target = target
        self.ty = ty
        self.backend = backend

    def __str__(self):
        return "({}) v{} = to_{}:{}".format(
            self.backend.value, self.target, self.backend.value, str(self.ty))

    def evaluate(self, _thread, _batch_index, _values, context, _last_batch):
        old_value = context[self.target][-1]
        new_value = self.ty.to(old_value, self.backend)
        context[self.target][-1] = new_value
