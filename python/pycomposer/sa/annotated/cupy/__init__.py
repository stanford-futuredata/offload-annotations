
# Fall back to CuPy if we don't support something.
from cupy import *

from .annotated import *

# Provide an explicit evaluate function.
from sa.annotation import evaluate
