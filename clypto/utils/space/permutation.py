import typing

from clypto.utils.space.base import BaseVar


class PermutationVar(BaseVar):
    eps: typing.Final[float] = 1e-4

    def __init__(self, valid_set=(1, 2), name="permutation"):
        if type(valid_set) in self.SUPPORTED_ARRAY and len(valid_set) > 1:
            valid_set = np.array(valid_set)
            n_vars = len(valid_set)
            le = LabelEncoder().fit(valid_set)
            lb = np.zeros(self.n_vars)
            ub = (self.n_vars - self.eps) * np.ones(self.n_vars)
        else:
            raise TypeError(
                f"Invalid valid_set. It should be {self.SUPPORTED_ARRAY} and contains at least 2 variables"
            )

        super().__init__(n_vars, (lb, ub), name)

        self.__le = le

    @property
    def le(self):
        return self.__le

    def encode(self, x):
        return np.array(self.le.transform(x), dtype=float)

    def decode(self, x):
        x = self.correct(x)
        return self.le.inverse_transform(x)

    def correct(self, x):
        return np.argsort(x)

    def generate(self):
        return self.generator.permutation(self.valid_set)
