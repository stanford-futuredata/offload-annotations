
# Fall back to cuDF if we don't support something.
from cudf import *

from .annotated import *

# Provide an explicit evaluate function.
from sa.annotation import evaluate
