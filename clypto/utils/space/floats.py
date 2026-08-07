import numbers as nb
import typing

import numpy as np

from clypto.utils.space.base import BaseVar


class FloatVar(BaseVar):
    def __init__(self, lb=-10.0, ub=10.0, name="float"):
        if isinstance(lb, nb.Number) and isinstance(ub, nb.Number):
            lb, ub = np.array((lb,), dtype=float), np.array((ub,), dtype=float)
            n_vars = 1
        elif type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            if len(lb) == len(ub):
                lb, ub = np.array(lb, dtype=float), np.array(ub, dtype=float)
                n_vars = len(lb)
            else:
                raise ValueError(
                    f"Invalid lb or ub. Length of lb should equal to length of ub."
                )
        else:
            raise TypeError(
                f"Invalid lb or ub. It should be one of following: {self.SUPPORTED_ARRAY}"
            )

        super().__init__(n_vars, (lb, ub), name)

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=float)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.uniform(self.lb, self.ub)
