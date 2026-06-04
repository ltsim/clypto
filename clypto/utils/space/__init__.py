#!/usr/bin/env python
# Created by "Thieu" at 05:33, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import abc
import numbers as nb
import typing

import numpy as np

from clypto.utils import transfer


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    Especially, it can encode a list of mixed types include integer, float, and string. Better than scikit-learn module.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    @staticmethod
    def set_y(y):
        if type(y) not in (list, tuple, np.ndarray):
            y = (y,)

        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.
        """

        def safe_key(val):
            # Chuyển None -> 0, số -> 1, chuỗi -> 2, object khác -> 3
            if val is None:
                return (0, "")
            elif isinstance(val, nb.Number):
                return (1, val)
            elif isinstance(val, str):
                return (2, val)
            else:
                return (3, str(val))

        # self.unique_labels = sorted(set(y), key=lambda x: (isinstance(x, (int, float)), x))
        self.unique_labels = sorted(set(y), key=safe_key)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

        return self

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.

        Returns:
        --------
        encoded_labels : list
            Encoded integer labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")

        y = self.set_y(y)

        return [self.label_to_index[label] for label in y]

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : list, tuple
            Target values.

        Returns
        -------
        y : list
            Encoded labels.
        """
        y = self.set_y(y)

        self.fit(y)

        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : list, tuple
            Encoded integer labels.

        Returns:
        --------
        original_labels : list
            Original labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")

        y = self.set_y(y)

        return [
            self.unique_labels[i] if i in self.label_to_index.values() else "unknown"
            for i in y
        ]


class BaseVar(abc.ABC):
    SUPPORTED_ARRAY: typing.Final[tuple[type]] = tuple, list, np.ndarray

    def __init__(
        self, n_vars: int, bounds: tuple[typing.Any, typing.Any], name="variable"
    ):
        if type(n_vars) is int and n_vars > 0:
            self.__n_vars = n_vars
        else:
            raise ValueError(f"Invalid n_vars. It should be integer and > 0.")

        self.__name = name
        self.__seed = None
        self.__lb, self.__ub = bounds
        self.__n_vars = n_vars

        self.generator = np.random.default_rng()

    @property
    def lb(self):
        return self.__lb

    @property
    def ub(self):
        return self.__ub

    @property
    def name(self) -> str:
        return self.__name

    @property
    def n_vars(self):
        return self.__n_vars

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.__seed = value
        self.generator = np.random.default_rng(self.__seed)

    @abc.abstractmethod
    def encode(self, x):
        pass

    @abc.abstractmethod
    def decode(self, x):
        pass

    @abc.abstractmethod
    def correct(self, x):
        pass

    @abc.abstractmethod
    def generate(self):
        pass

    @staticmethod
    def round(x):
        frac = x - np.floor(x)
        t1 = np.floor(x)
        t2 = np.ceil(x)
        return np.where(frac < 0.5, t1, t2)


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


class CategoricalVar(StringVar):
    def __init__(self, valid_sets=(("",),), name="categorical"):
        super().__init__(valid_sets, name)

    def generate(self):
        return [
            self.generator.choice(np.array(vl_set, dtype=object))
            for vl_set in self.valid_sets
        ]


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
