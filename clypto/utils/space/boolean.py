import typing

import numpy as np
from clypto.utils import transfer

from clypto.utils.space.base import BaseVar


class BoolVar(BaseVar):
    eps: typing.Final[float] = 1e-4

    def __init__(self, n_vars=1, name="boolean"):
        super().__init__(
            n_vars, (np.zeros(self.n_vars), (2 - self.eps) * np.ones(self.n_vars)), name
        )

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        x = np.array(x, dtype=int)
        return x == 1

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.choice([True, False], self.n_vars, replace=True)


class TransferBoolVar(BaseVar):
    SUPPORTED_TF_FUNCS: typing.Final[tuple[str]] = (
        "vstf_01",
        "vstf_02",
        "vstf_03",
        "vstf_04",
        "sstf_01",
        "sstf_02",
        "sstf_03",
        "sstf_04",
    )

    def __init__(self, n_vars=1, name="boolean", tf_func="vstf_01", lb=-8.0, ub=8.0):
        super().__init__(
            n_vars, (lb * np.zeros(self.n_vars), ub * np.ones(self.n_vars)), name
        )

        if tf_func in self.SUPPORTED_TF_FUNCS:
            self.tf_name = tf_func
            self.tf_func = getattr(transfer, tf_func)
        else:
            raise ValueError(
                f"Invalid transfer function! The supported TF funcs are: {self.SUPPORTED_TF_FUNCS}"
            )

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        x = np.array(x, dtype=int)
        return x == 1

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        x = self.tf_func(x)

        cons = self.generator.random(len(x))

        return np.where(cons < x, 1, 0)

    def generate(self):
        return self.generator.choice([True, False], self.n_vars, replace=True)
