#!/usr/bin/env python
# Created by "Thieu" at 17:28, 13/10/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numbers
import typing

import numpy as np

from mealpy.utils.space import (BaseVar, IntegerVar, FloatVar, StringVar, BinaryVar, BoolVar,
                                PermutationVar, CategoricalVar, SequenceVar, TransferBinaryVar, TransferBoolVar)
from mealpy.utils.target import Target


class Problem:
    SUPPORTED_VARS: typing.Final[tuple[
        type]] = IntegerVar, FloatVar, StringVar, BinaryVar, BoolVar, PermutationVar, CategoricalVar, SequenceVar, TransferBinaryVar, TransferBoolVar
    SUPPORTED_ARRAYS: typing.Final[tuple[type]] = list, tuple, np.ndarray

    def __init__(self, bounds: list | tuple | np.ndarray | BaseVar, minmax: str = "min", **kwargs) -> None:
        self.__obj_func = kwargs.get("obj_func", lambda _: 0)
        self.__name = kwargs.get("name", "Problem")
        self.__bounds = None
        self.__n_objs = None
        self.__lb = None
        self.__ub = None
        self.__n_dims = None
        self.__minmax = minmax
        self.__seed = None
        self.__obj_weights = None
        self.__save_population = None

        self.set_bounds(bounds)

    @property
    def lb(self):
        return self.__lb

    @property
    def ub(self):
        return self.__ub

    @property
    def n_dims(self):
        return self.__n_dims

    @property
    def minmax(self):
        return self.__minmax

    @property
    def bounds(self):
        return self.__bounds

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed

        for idx in range(len(self.__bounds)):
            self.__bounds[idx].seed = seed

    @property
    def save_population(self):
        return self.__save_population

    @property
    def n_objs(self):
        if self.__n_objs is None:
            x = self.generate_solution(encoded=True)

            result = self.obj_func(x)

            if isinstance(result, self.SUPPORTED_ARRAYS):
                self.__n_objs = len(np.asarray(result).ravel())
            elif isinstance(result, numbers.Number):
                self.__n_objs = 1
            else:
                raise ValueError("`obj_func` must return a number, list, tuple or numpy array.")

            if self.__obj_weights is None:
                if self.__n_objs > 1:
                    ...

                self.__obj_weights = np.ones(self.__n_objs)

            elif len(np.array(self.__obj_weights).ravel()) != self.__n_objs:
                raise ValueError(
                    f"`obj_weights` length {len(self.__obj_weights)} does not match number of objectives {self.__n_objs}.")

        return self.__n_objs

    def set_bounds(self, bounds):
        if isinstance(bounds, BaseVar):
            bounds.seed = self.seed

            self.__bounds = [bounds, ]
        elif type(bounds) in self.SUPPORTED_ARRAYS:
            self.__bounds = []

            for bound in bounds:
                if isinstance(bound, BaseVar):
                    bound.seed = self.seed
                else:
                    raise ValueError(
                        f"Invalid bounds. All variables in bounds should be an instance of {self.SUPPORTED_VARS}")
                self.__bounds.append(bound)
        else:
            raise TypeError(
                f"Invalid bounds. It should be type of {self.SUPPORTED_ARRAYS} or an instance of {self.SUPPORTED_VARS}")

        self.__lb = np.concatenate([bound.lb for bound in self.__bounds])
        self.__ub = np.concatenate([bound.ub for bound in self.__bounds])
        self.__n_dims = len(self.lb)

    def set_seed(self, seed: int | None = None) -> None:
        self.seed = seed

    def obj_func(self, x: np.ndarray) -> list | tuple | np.ndarray | int | float:
        """Objective function

        Args:
            x (numpy.ndarray): Solution.

        Returns:
            float: Function value of `x`.
        """
        return self.__obj_func(x)

    def get_name(self) -> str:
        """
        Returns:
            string: The name of the problem
        """
        return self.__name

    def get_class_name(self) -> str:
        """Get class name."""
        return self.__class__.__name__

    @staticmethod
    def encode_solution_with_bounds(x, bounds):
        x_new = []

        for idx, var in enumerate(bounds):
            x_new += list(var.encode(x[idx]))

        return np.array(x_new)

    @staticmethod
    def decode_solution_with_bounds(x, bounds):
        x_new, n_vars = {}, 0

        for idx, var in enumerate(bounds):
            temp = var.decode(x[n_vars:n_vars + var.n_vars])

            if var.n_vars == 1:
                x_new[var.__name] = temp[0]
            else:
                x_new[var.__name] = temp

            n_vars += var.n_vars

        return x_new

    @staticmethod
    def correct_solution_with_bounds(x: list | tuple | np.ndarray, bounds: list) -> np.ndarray:
        x_new, n_vars = [], 0

        for idx, var in enumerate(bounds):
            x_new += list(var.correct(x[n_vars:n_vars + var.n_vars]))
            n_vars += var.n_vars

        return np.array(x_new)

    @staticmethod
    def generate_solution_with_bounds(bounds: list | tuple | np.ndarray, encoded: bool = True) -> list | np.ndarray:
        x = [var.generate() for var in bounds]

        if encoded:
            return Problem.encode_solution_with_bounds(x, bounds)

        return x

    def encode_solution(self, x: list | tuple | np.ndarray) -> np.ndarray:
        """
        Encode the real-world solution to optimized solution (real-value solution)

        Args:
            x (Union[List, tuple, np.ndarray]): The real-world solution

        Returns:
            The real-value solution
        """
        return self.encode_solution_with_bounds(x, self.bounds)

    def decode_solution(self, x: np.ndarray) -> dict:
        """
        Decode the encoded solution to real-world solution

        Args:
            x (np.ndarray): The real-value solution

        Returns:
            The real-world (decoded) solution
        """
        return self.decode_solution_with_bounds(x, self.bounds)

    def correct_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Correct the solution to valid bounds

        Args:
            x (np.ndarray): The real-value solution

        Returns:
            The corrected solution
        """
        return self.correct_solution_with_bounds(x, self.bounds)

    def generate_solution(self, encoded: bool = True) -> list | np.ndarray:
        """
        Generate the solution.

        Args:
            encoded (bool): Encode the solution or not

        Returns:
            the encoded/non-encoded solution for the problem
        """
        return self.generate_solution_with_bounds(self.bounds, encoded)

    def get_target(self, solution: np.ndarray) -> Target:
        """
        Args:
            solution: The real-value solution

        Returns:
            The target object
        """
        return Target(objectives=self.obj_func(solution), weights=self.__obj_weights)
