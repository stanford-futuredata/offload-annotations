
# Fall back to Torch if we don't support something.
from torch import *

from .annotated import *

# Provide an explicit evaluate function.
from sa.annotation import evaluate
