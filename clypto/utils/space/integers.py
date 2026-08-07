import numbers as nb
import typing

import numpy as np

from clypto.utils.space.base import BaseVar


class IntegerVar(BaseVar):
    eps: typing.Final[float] = 1e-4

    def __init__(self, lb=-10, ub=10, name="integer"):
        if isinstance(lb, nb.Number) and isinstance(ub, nb.Number):
            lb, ub = int(lb) - 0.5, int(ub) + 0.5 - self.eps
            lb, ub = np.array((lb,), dtype=float), np.array((ub,), dtype=float)
            return lb, ub, 1
        elif type(lb) in self.SUPPORTED_ARRAY and type(ub) in self.SUPPORTED_ARRAY:
            if len(lb) == len(ub):
                lb, ub = np.array(lb, dtype=float) - 0.5, np.array(ub, dtype=float) + (
                        0.5 - self.eps
                )
                return (lb, ub), len(lb)
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
        x = self.round(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.integers(self.lb + 0.5, self.ub + 0.5 + self.eps)
