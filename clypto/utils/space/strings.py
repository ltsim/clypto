import typing

import numpy as np

from clypto.utils.space.base import BaseVar
from clypto.utils.space.label import LabelEncoder


class StringVar(BaseVar):
    eps: typing.Final[float] = 1e-4

    def __init__(self, valid_sets=(("",),), name="string"):
        if type(valid_sets) in self.SUPPORTED_ARRAY:
            if type(valid_sets[0]) not in self.SUPPORTED_ARRAY:
                n_vars = 1
                valid_sets = (tuple(valid_sets),)
                le = LabelEncoder().fit(valid_sets)
                list_le = (le,)

                lb = np.array(
                    [
                        0.0,
                    ]
                )
                ub = np.array(
                    [
                        len(valid_sets) - self.eps,
                    ]
                )
            else:
                n_vars = len(valid_sets)

                if all(len(item) > 1 for item in valid_sets):
                    valid_sets = valid_sets
                    list_le = []
                    ub = []

                    for vl_set in valid_sets:
                        le = LabelEncoder().fit(vl_set)
                        list_le.append(le)
                        ub.append(len(vl_set) - self.eps)

                    lb = np.zeros(self.n_vars)
                    ub = np.array(ub)
                else:
                    raise ValueError(
                        f"Invalid valid_sets. All variables need to have at least 2 values."
                    )
        else:
            raise TypeError(f"Invalid valid_sets. It should be {self.SUPPORTED_ARRAY}.")

        super().__init__(n_vars, (lb, ub), name)

        self.__list_le = list_le
        self.__valid_sets = valid_sets

    @property
    def list_le(self):
        return self.__list_le

    @property
    def valid_sets(self):
        return self.__valid_sets

    def encode(self, x):
        return np.array(
            [le.transform(val)[0] for (le, val) in zip(self.list_le, x)], dtype=float
        )

    def decode(self, x):
        x = self.correct(x)
        return [le.inverse_transform(val)[0] for (le, val) in zip(self.list_le, x)]

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        return [
            self.generator.choice(np.array(vl_set, dtype=str))
            for vl_set in self.valid_sets
        ]
