
from collections import defaultdict, deque
import copy
import logging

from .annotation import Annotation
from .config import config
from .split_types import *
from .unevaluated import UNEVALUATED
from .backend import Backend

from .vm.instruction import *
from .vm.vm import VM
from .vm import Program, Driver, STOP_ITERATION
from .vm.driver import DEFAULT_BATCH_SIZE

import functools

class Operation:
    """ A lazily evaluated computation in the DAG.

    Accessing any field of an operation will cause the DAG to execute. Operations
    keep a reference to their DAG to allow this.

    """

    def __init__(self, func, args, kwargs, annotation, owner_ref):
        """Initialize an operation.

        Parameters
        __________

        func : the function to evaluate
        args : non-keyword arguments
        kwargs : keyword arguments
        annotation : a _mutable_ annotation object for this function
        owner_ref : reference to the DAG.

        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.annotation = annotation

        # Reference to the computed output.
        self._output = UNEVALUATED

        # The pipeline this operator is a part of.
        self.pipeline = None
        # Disable sending results. Hack.
        self.dontsend = False

        # Reference to the DAG object.
        self._owner_ref = owner_ref
        # Signals whether this is a root expression with no parent.
        self.root = True
        # Children of this operation that must be evaluated first.
        self.children = []

        # Backends on which the operation supports execution. By default, all operations
        # support GPU execution. Operations support GPU execution based on the function
        # annotation, and input and return types.
        self.supported_backends = [Backend.CPU]
        supports_gpu = annotation.gpu
        supports_gpu &= Backend.GPU in annotation.return_type.supported_backends
        for (i, _) in enumerate(args):
            supports_gpu &= Backend.GPU in self.split_type_of(i).supported_backends
        for (key, _) in kwargs.items():
            supports_gpu &= Backend.GPU in self.split_type_of(key).supported_backends
        if supports_gpu:
            self.supported_backends.append(Backend.GPU)

    def all_args(self):
        """ Returns a list of all the args in this operation. """
        return tuple(self.args) + tuple(self.kwargs.values())

    def mutable_args(self):
        """ Returns a list of all the mutable args in this operation. """
        mutables = []
        for (i, arg) in enumerate(self.args):
            if self.is_mutable(i):
                mutables.append(arg)

        for (key, value) in self.kwargs.items():
            if self.is_mutable(key):
                mutables.append(value)

        return mutables

    def split_type_of(self, index):
        """ Returns the split type of the argument with the given index.

        index can be a number to access regular arguments or a name to access
        keyword arguments.

        """
        if isinstance(index, int):
            return self.annotation.arg_types[index]
        elif isinstance(index, str):
            return self.annotation.kwarg_types[index]
        else:
            raise ValueError("invalid index {}".format(index))

    def is_mutable(self, index):
        """ Returns whether the argument at the given index is mutable. """
        return index in self.annotation.mutables

    def dependency_of(self, other):
        """ Returns whether self is a dependency of other. """
        if self in other.args:
            return True
        elif self in other.kwargs.values():
            return True
        else:
            # Check if any of our mutable arguments appear in other.
            # NOTE: This will currenty create n^2 edges where n is the number
            # of nodes that mutate an argument...
            mutable_args = self.mutable_args()
            for arg in other.all_args():
                for arg2 in mutable_args:
                    if arg is arg2:
                        return True
        return False

    @property
    def value(self):
        """ Returns the value of the operation.

        Causes execution of the DAG this operation is owned by, if a value has
        not been computed yet.

        """
        if self._output is UNEVALUATED:
            evaluate_dag(self._owner_ref)
        return self._output

    def _str(self, depth):
        s = "{}@sa({}){}(...) (pipeline {})".format(
                "  " * depth,
                self.annotation,
                self.func.__name__,
                self.pipeline)
        deps = []
        for dep in self.children:
            deps.append(dep._str(depth+1))
        for dep in deps:
            s += "\n{}".format(dep)
        return s

    def pretty_print(self):
        return "\n" + self._str(0)

    ## Magic Methods
    # NOTE: Some of these are broken (e.g., == won't evaluate right now since
    # its needed by Operation)

    def __eq__(self, other):
        """ Override equality to always check by reference. """
        if self._output is UNEVALUATED:
            return id(self) == id(other)
        else:
            return self._output == other

    def __hash__(self):
        """ Override equality to always check by reference. """
        if self._output is UNEVALUATED:
            return id(self)
        else:
            return hash(self._output)

    def __getitem__(self, key):
        return self.value.__getitem__(key)

    def __setitem__(self, key, value):
        return self.value.__getitem__(key, value)

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return self.value.__iter__()

    def __reversed__(self):
        return self.value.__reversed__()

    def __contains__(self, item):
        return self.value.__contains__(item)

    def __missing__(self, key):
        return self.value.__missing__(key)


class Future:
    """
    A wrapper class that causes lazy values to evaluate when any attribute of this
    class is accessed.
    """

    __slots__ = [ "operation", "value" ]
    def __init__(self, operation):
        self.operation = operation
        self.value = None


class LogicalPlan:
    """ A logical plan representing dataflow.

    The plan is evaluated from leaves to the root.

    """

    def __init__(self):
        """Initialize a logical DAG.

        The DAG is meant to be used as a singleton for registering tasks.

        """
        # Roots in the DAG.
        self.roots = []

    def clear(self):
        """ Clear the operators in this DAG by removing its nodes. """
        self.roots = []

    def register(self, func, args, kwargs, annotation):
        """ Register a function invocation along with its annotation.

        This method will clone the annotation since its types will eventually be modified to reflect
        concrete split types, in the case where some of the annotations are generics.

        Parameters
        __________

        func : the function to call.
        args : the non-keyword args of the function.
        kwargs : the keyworkd args of the function.
        annotation : the split type annotation of the function.

        Returns
        _______

        A lazy object representing a computation. Accessing the lazy object
        will cause the full computation DAG to be evaluated.

        """

        annotation = copy.deepcopy(annotation)
        operation = Operation(func, args, kwargs, annotation, self)

        def wire(op, newop):
            """ Wire the new operation into the DAG.

            Existing operations are children of the new operation if:

            1. The new operation uses an existing one as an argument.
            2. The new operation uses a value that is mutated by the
            existing operation.

            """
            if op is newop:
                return
            if op.dependency_of(newop):
                newop.children.append(op)
                newop.root = False
                # Don't need to recurse -- we only want the "highest" dependency.
                return True

        self.walk(wire, operation)

        # Update the roots.
        self.roots = [root for root in self.roots if not root.dependency_of(operation)]
        self.roots.append(operation)
        return operation

    def _walk_bottomup(self, op, f, context, visited):
        """ Recursive bottom up DAG walk implementation. """
        if op in visited:
            return
        visited.add(op)
        for dep in op.children:
            self._walk_bottomup(dep, f, context, visited)
        f(op, context)

    def walk(self, f, context, mode="topdown"):
        """ Walk the DAG in the specified order.

        Each node in the DAG is visited exactly once.

        Parameters
        __________

        f : A function to apply to each record. The function takes an operation
        and an optional context (i.e., any object) as arguments.

        context : An initial context.

        mode : The order in which to process the DAG. "topdown" (the default)
        traverses each node as its visited in breadth-first order. "bottomup"
        traverses the graph depth-first, so the roots are visited after the
        leaves (i.e., nodes are represented in "execution order" where
        dependencies are processed first).

        """

        if mode == "bottomup":
            for root in self.roots:
                self._walk_bottomup(root, f, context, set())
            return

        assert mode == "topdown"

        visited = set()
        queue = deque(self.roots[:])
        while len(queue) != 0:
            cur = queue.popleft()
            if cur not in visited:
                should_break = f(cur, context)
                visited.add(cur)
                if should_break is None:
                    for child in cur.children:
                        queue.append(child)

    def infer_types(self):
        """ Infer concrete types for each argument in the DAG. """

        def uniquify_generics(op, ident):
            """ Give each generic type an annotation-local identifier. """
            for ty in op.annotation.types():
                if isinstance(ty, GenericType):
                    ty._id = ident[0]
            ident[0] += 1

        def infer_locally(op, changed):
            """ Sync the annotation type in the current op with the annotations
            in its children.
            """
            try:
                for (i, argument) in enumerate(op.args):
                    if isinstance(argument, Operation) and argument in op.children:
                        split_type = op.split_type_of(i)
                        changed[0] |= split_type._sync(argument.annotation.return_type)

                for (name, argument) in op.kwargs.items():
                    if isinstance(argument, Operation) and argument in op.children:
                        split_type = op.split_type_of(name)
                        changed[0] |= split_type._sync(argument.annotation.return_type)

                # Sync types within a single annotation so all generics have the same types.
                # This will also update the return type.
                for ty in op.annotation.types():
                    for other in op.annotation.types():
                        if ty is other: continue
                        if isinstance(ty, GenericType) and isinstance(other, GenericType) and\
                                ty.name == other.name and ty._id == other._id:
                            changed[0] |= ty._sync(other)
                op.pipeline = changed[1]
            except SplitTypeError as e:
                logging.debug("Pipeline break: {}".format(e))
                changed[1] += 1
                op.pipeline = changed[1]

        def finalize(op, _):
            """ Replace generics with concrete types. """
            op.annotation.arg_types = list(map(lambda ty: ty._finalized(),
                    op.annotation.arg_types))
            if op.annotation.return_type is not None:
                op.annotation.return_type = op.annotation.return_type._finalized()
            for key in op.annotation.kwarg_types:
                op.annotation.kwarg_types[key] = op.annotation.kwarg_types[key]._finalized()

        self.walk(uniquify_generics, [0])

        # Run type-inference to fix point.
        while True:
            changed = [False, 0]
            self.walk(infer_locally, changed, mode="bottomup")
            if not changed[0]:
                break

        self.walk(finalize, None)

    def to_vm(self):
        """
        Convert the graph to a sequence of VM instructions that can be executed
        on worker nodes. One VM program is constructed per pipeline.

        Returns a list of VMs, sorted by pipeline.
        """
        def mark_backends(op, vms):
            vm = vms[1][op.pipeline]
            added = vms[0]

            if op in added:
                return

            if Backend.GPU in op.supported_backends:
                vm.backends.add(Backend.GPU)
            added.add(op)

        def construct(op, vms):
            vm = vms[1][op.pipeline]
            added = vms[0]
            mutable = vms[2][op.pipeline]
            var_locs = vms[3][op.pipeline]

            # Already processed this op.
            if op in added:
                return

            args = []
            kwargs = {}

            # Split the arguments if it is our first encounter
            for (i, arg) in enumerate(op.args):
                valnum = vm.get(arg)
                if valnum is None:
                    ty = op.split_type_of(i)
                    valnum = vm.register_value(arg, ty)
                    setattr(ty, "mutable", op.is_mutable(i))
                args.append(valnum)

            for (key, value) in op.kwargs.items():
                valnum = vm.get(value)
                if valnum is None:
                    ty = op.split_type_of(key)
                    valnum = vm.register_value(value, ty)
                    setattr(ty, "mutable", op.is_mutable(key))
                kwargs[key] = valnum

            # Determine which backend to call the operation on.
            if Backend.GPU in op.supported_backends:
                inst_backend = Backend.GPU
            else:
                inst_backend = Backend.CPU

            # Assuming the vm backend includes both CPU and GPU, handle the
            # first time a value appears. If the value goes on the CPU, split
            # it. If the value goes on the GPU, transfer it without splitting.
            #
            # Also transfer arguments as necessary between backends.
            def transfer(valnum, var_locs, vm, op):
                ty = vm.split_type_of(valnum)
                assert ty is not None

                if valnum not in var_locs:
                    if inst_backend == Backend.GPU:
                        vm.program.insts.append(To(valnum, ty, Backend.GPU))
                        var_locs[valnum] = Backend.GPU
                    else:
                        vm.program.insts.append(Split(valnum, ty, Backend.CPU))
                        var_locs[valnum] = Backend.CPU
                else:
                    if var_locs[valnum] == Backend.CPU and inst_backend == Backend.GPU:
                        vm.program.insts.append(To(valnum, ty, Backend.GPU))
                        var_locs[valnum] = Backend.GPU
                        vm.program.insts.append(Merge(valnum, ty, Backend.GPU))
                    elif var_locs[valnum] == Backend.GPU and inst_backend == Backend.CPU:
                        vm.program.insts.append(To(valnum, ty, Backend.CPU))
                        vm.program.insts.append(Split(valnum, ty, Backend.CPU))
                        var_locs[valnum] = Backend.CPU
            for valnum in args:
                transfer(valnum, var_locs, vm, op)
            for _, valnum in kwargs.items():
                transfer(valnum, var_locs, vm, op)

            # Register the valnum of the return value and its backend location
            result = vm.register_value(op, op.annotation.return_type)
            var_locs[result] = inst_backend

            # In this context, mutability just means we need to merge objects.
            if op.annotation.return_type is not None:
                setattr(op.annotation.return_type, "mutable", not op.dontsend)
                if not op.dontsend:
                    mutable.add(result)
            # Choose which function to call based on whether the pipeline is on the gpu.
            if inst_backend == Backend.GPU and op.annotation.gpu_func is not None:
                func = op.annotation.gpu_func
            else:
                func = op.func
            vm.program.insts.append(Call(
                result, func, args, kwargs, op.annotation.return_type, inst_backend))
            added.add(op)

        # programs: Maps Pipeline IDs to VM Programs.
        # arg_id_to_ops: Maps Arguments to ops. Store separately so we don't serialize ops.
        vms = defaultdict(lambda: VM())
        mutables = defaultdict(lambda: set())
        var_locs = defaultdict(lambda: defaultdict(lambda: Backend.CPU))
        self.walk(mark_backends, (set(), vms), mode="bottomup")
        self.walk(construct, (set(), vms, mutables, var_locs), mode="bottomup")
        for pipeline in vms:
            # Move any variables still on the GPU back to the CPU,
            # and merge any variables on the CPU.
            for valnum, backend in var_locs[pipeline].items():
                ty = vms[pipeline].split_type_of(valnum)
                if ty is None:
                    continue
                if backend == Backend.GPU:
                    vms[pipeline].program.insts.append(To(valnum, ty, Backend.CPU))
                elif backend == Backend.CPU:
                    vms[pipeline].program.insts.append(Merge(valnum, ty, Backend.CPU))
            vms[pipeline].program.remove_unused_outputs(mutables[pipeline])
        return sorted(list(vms.items()))


    @staticmethod
    def commit(values, results):
        """
        Commit outputs into the DAG nodes so programs can access data.
        """
        for (arg_id, value) in values.items():
            if isinstance(value, Operation):
                value._output = results[arg_id]

    def __str__(self):
        roots = []
        for root in self.roots:
            roots.append(root.pretty_print())
        return "\n".join(roots)


def evaluate_dag(dag, workers=config["workers"], batch_size=config["batch_size"], profile=False):
    try:
        dag.infer_types()
    except (SplitTypeError) as e:
        logging.error(e)

    vms = dag.to_vm()
    for _, vm in vms:
        print(vm.program)
        print()
        driver = Driver(workers=workers, batch_size=batch_size, optimize_single=True, profile=profile)
        results = driver.run(vm.program, vm.values)

        dag.commit(vm.values, results)
        # TODO We need to update vm.values in the remaining programs to use the
        # materialized data in _DAG.operation.
        #
        # If in the future we see something like "object dag.Operation does not
        # have method <annotated method>" in a multi-stage program, that's why!
        pass

    dag.clear()
