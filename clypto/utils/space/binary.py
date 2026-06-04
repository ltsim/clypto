import typing

import numpy as np

from clypto.utils import transfer
from clypto.utils.space.base import BaseVar


class BinaryVar(BaseVar):
    eps: typing.Final[float] = 1e-4

    def __init__(self, n_vars=1, name="binary"):
        super().__init__(
            n_vars, (np.zeros(n_vars), (2 - self.eps) * np.ones(n_vars)), name
        )

    def encode(self, x):
        return np.array(x, dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return np.array(x, dtype=int)

    def correct(self, x):
        return np.clip(x, self.lb, self.ub)

    def generate(self):
        return self.generator.integers(0, 2, self.n_vars)


class TransferBinaryVar(BinaryVar):
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

    def __init__(
            self,
            n_vars=1,
            name="tf-binary",
            tf_func="vstf_01",
            lb=-8.0,
            ub=8.0,
            all_zeros=True,
    ):
        super().__init__(n_vars, (lb * np.zeros(n_vars), ub * np.ones(n_vars)), name)

        if tf_func in self.SUPPORTED_TF_FUNCS:
            self.tf_name = tf_func
            self.tf_func = getattr(transfer, tf_func)
        else:
            raise ValueError(
                f"Invalid transfer function! The supported TF funcs are: {self.SUPPORTED_TF_FUNCS}"
            )

        self.all_zeros = all_zeros

    def __get_correct_x(self, x):
        if self.all_zeros:
            return x
        else:
            if np.sum(x) == 0:
                x[self.generator.integers(0, len(x))] = 1
            return x

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        x = self.tf_func(x)
        cons = self.generator.random(len(x))
        x = np.where(cons < x, 1, 0)
        return self.__get_correct_x(x)

    def generate(self):
        x = self.generator.integers(0, 2, self.n_vars)
        return self.__get_correct_x(x)
