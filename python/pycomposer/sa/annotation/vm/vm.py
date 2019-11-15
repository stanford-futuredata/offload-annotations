
from .program import Program

class VM:
    """
    A Composer virtual machine, which holds a program and its associated data.
    """
    def __init__(self):
        # Counter for argument IDs
        self.ssa_counter = 0
        # Program
        self.program = Program()
        # Values, mapping argID -> values
        self.values = dict()
        # Split types, mapping argID -> split type
        self.types = dict()
        # Weather the VM can be computed on the GPU
        self.gpu = True

    def get(self, value):
        """
        Get the SSA value for a value, or None if the value is not registered.

        value : The value to lookup

        """
        for num, val in self.values.items():
            if value is val:
                return num

    def register_value(self, value, ty):
        """
        Register a counter to a value.
        """
        arg_id = self.ssa_counter
        self.ssa_counter += 1
        self.values[arg_id] = value
        self.types[arg_id] = ty
        return arg_id

    def split_type_of(self, arg_id):
        """
        Returns the split type of the value with the argument id.
        """
        if arg_id in self.types:
            return self.types[arg_id]
        else:
            return None
