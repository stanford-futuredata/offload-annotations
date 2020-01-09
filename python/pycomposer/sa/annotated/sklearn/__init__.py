
# Fall back to scikit-learn if we don't support something.
from sklearn import *

from .annotated import *

# Provide an explicit evaluate function.
from sa.annotation import evaluate
