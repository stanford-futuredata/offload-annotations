from enum import Enum

class Device(Enum):
	CPU = 'cpu'
	GPU = 'gpu'
	SCALAR = 'scalar'  # scalars are device agnostic
