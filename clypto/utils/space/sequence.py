import typing

import numpy as np

from clypto.utils.space.base import BaseVar


class SequenceVar(BaseVar):
    eps: typing.Final[float] = 1e-4

    def __init__(self, valid_sets, return_type=tuple, name="sequence"):
        super().__init__(
            1,
            (
                np.array(
                    [
                        0.0,
                    ]
                ),
                np.array(
                    [
                        len(valid_sets) - self.eps,
                    ]
                ),
            ),
            name,
        )

        self.__valid_sets = [
            tuple(v) for v in valid_sets
        ]  # Normalize to tuples for hashing
        self.return_type = return_type
        self.label_to_index = {val: i for i, val in enumerate(self.__valid_sets)}
        self.index_to_label = {i: val for i, val in enumerate(self.__valid_sets)}

    @property
    def valid_sets(self):
        return self.__valid_sets

    def encode(self, x):
        x_tuple = tuple(x)

        if x_tuple not in self.label_to_index:
            raise ValueError(f"Unknown sequence for encoding: {x}")

        return np.array([self.label_to_index[x_tuple]], dtype=float)

    def decode(self, x):
        x = self.correct(x)
        val = self.index_to_label[x[0]]

        return [self.return_type(val)]

    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        return np.array(x, dtype=int)

    def generate(self):
        idx = self.generator.integers(0, len(self.valid_sets))
        return self.valid_sets[idx]
