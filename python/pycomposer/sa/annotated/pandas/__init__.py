"""
Test
"""

# Fall back to Pandas if we don't support something.
from pandas import *

from .annotated import *
from sa.annotation import evaluate
import pandas as pd

class SeriesSplit(SplitType):
    def combine(self, values):
        return concat(values)

    def split(self, start, end, value):
        return value[start:end]

pd.Series.abs =  sa((SeriesSplit(),), {}, SeriesSplit())(pd.Series.abs)

pd.Series.add = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.add)
# pd.Series.all = sa((SeriesSplit(),), {}, AllSplit())(pd.Series.all)
# pd.Series.any = sa((SeriesSplit(),), {}, AnySplit())(pd.Series.any)
pd.Series.apply = sa((SeriesSplit(), Broadcast()), {}, SeriesSplit())(pd.Series.apply)

pd.Series.between = sa((SeriesSplit(), Broadcast(), Broadcast()), {}, SeriesSplit())(pd.Series.between)
pd.Series.between_time = sa((SeriesSplit(), Broadcast(), Broadcast()), {}, SeriesSplit())(pd.Series.between_time)

pd.Series.bfill = sa((SeriesSplit(), ), {}, SeriesSplit())(pd.Series.bfill)
pd.Series.clip = sa((SeriesSplit(),), {}, SeriesSplit())(pd.Series.clip)
pd.Series.combine = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.combine)
pd.Series.combine_first = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.combine_first)

pd.Series.div = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.div)
pd.Series.divide = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.divide)
pd.Series.divmod = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.divmod)

# pd.Series.dot = sa((SeriesSplit(), SeriesSplit()), {}, DotSplit())(pd.Series.dot)

# TODO(shoumik:): These should return an unknown split type since they change the output shape.
pd.Series.drop_duplicates = sa((SeriesSplit(),), {}, SeriesSplit())(pd.Series.drop_duplicates)
pd.Series.dropna = sa((SeriesSplit(),), {}, SeriesSplit())(pd.Series.dropna)

# pd.Series.eq = sa((SeriesSplit(), SeriesSplit()), {}, EqSplit())(pd.Series.eq)
# pd.Series.equals = sa((SeriesSplit(), SeriesSplit()), {}, EqualsSplit())(pd.Series.equals)

pd.Series.ffill = sa((SeriesSplit(), ), {}, SeriesSplit())(pd.Series.ffill)
pd.Series.floordiv = sa((SeriesSplit(), SeriesSplit()), {}, SeriesSplit())(pd.Series.floordiv)
