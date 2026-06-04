import abc
import typing

import numpy as np


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
