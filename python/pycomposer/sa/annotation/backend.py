from enum import Enum

class Backend(Enum):
	CPU = 'cpu'
	GPU = 'gpu'
	SCALAR = 'scalar'  # scalars are backend agnostic
