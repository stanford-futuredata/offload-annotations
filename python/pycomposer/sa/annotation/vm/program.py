
from .driver import STOP_ITERATION
from .instruction import Call, Split

class Program:
    """
    A Composer Virtual Machine Program.

    A program stores a sequence of instructions to execute.

    """

    __slots__ = ["ssa_counter", "insts", "registered", "index"]

    def __init__(self):
        # Counter for registering instructions.
        self.ssa_counter = 0
        # Instruction list.
        self.insts = []
        # Registered values. Maps SSA value to real value.
        self.registered = {}

    def get(self, value):
        """
        Get the SSA value for a value, or None if the value is not registered.

        value : The value to lookup

        """
        for num, val in self.registered.items():
            if value is val:
                return num

    def set_range_end(self, range_end):
        for inst in self.insts:
            if isinstance(inst, Split):
                inst.ty.range_end = range_end

    def elements(self, values):
        """Returns the number of elements that this program will process.

        This quantity is retrieved by querying the Split instructions in the program.

        """
        elements = None
        from .. import dag
        for inst in self.insts:
            if isinstance(inst, Split):
                value = values[inst.target]
                if isinstance(value, dag.Operation):
                    value = value.value
                e = inst.ty.elements(value)
                if e is None:
                    continue
                if elements is not None:
                    assert(elements == e, inst)
                else:
                    elements = e
        return elements

    def remove_unused_outputs(self, mut_vals):
        visited = set()
        for i in range(len(self.insts) - 1, -1, -1):
            inst = self.insts[i]
            if isinstance(inst, Call):
                visited.update(inst.args)
                visited.update([valnum for (_, valnum) in inst.kwargs.items()])
            if inst.target in visited:
                continue

            # The instruction is mutable, merged, and later used.
            if inst.target in mut_vals:
                continue

            # The instruction result is not used in any following instructions.
            if isinstance(inst, Call):
                inst.remove_target()

    def __str__(self):
        return "\n".join([str(i) for i in self.insts])
