#!/usr/bin/env python
# Created by "Thieu" at 15:51, 05/10/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numbers

import cython
import numpy as np


@cython.cclass
class Target:
    SUPPORTED_ARRAY = tuple, list, np.ndarray

    objectives: object = cython.declare(object, visibility="readonly")
    weights: object = cython.declare(object, visibility="readonly")
    fitness: cython.double = cython.declare(cython.double, visibility="readonly")

    def __init__(
        self,
        objectives: list | tuple | np.ndarray | int | float | None = None,
        weights: list | tuple | np.ndarray | None = None,
    ) -> None:
        """
        Initialize the Target with a list of objectives and a fitness value.

        Parameters:
            objectives: The list of objective values.
            weights: The weights for calculating fitness value
        """
        if objectives is None:
            raise ValueError(
                f"Invalid objectives. It should be a list, tuple, np.ndarray, int or float."
            )
        else:
            if type(objectives) not in self.SUPPORTED_ARRAY:
                if isinstance(objectives, numbers.Number):
                    objectives = [objectives]
                else:
                    raise ValueError(
                        f"Invalid objectives. It should be a list, tuple, np.ndarray, int or float."
                    )
            objectives = np.array(objectives).flatten()
        self.objectives = objectives

        if weights is None:
            weights = len(self.objectives)
        else:
            if type(weights) not in self.SUPPORTED_ARRAY:
                if isinstance(weights, numbers.Number):
                    weights = [
                        weights,
                    ] * len(self.objectives)
                else:
                    raise ValueError(
                        f"Invalid weights. It should be a list, tuple, np.ndarray."
                    )
            weights = np.array(weights).flatten()
        self.weights = weights

        fitness_weights = self.weights
        if not (
            type(fitness_weights) in self.SUPPORTED_ARRAY
            and len(fitness_weights) == len(self.objectives)
        ):
            fitness_weights = len(self.objectives) * (1.0,)
        self.fitness = np.dot(fitness_weights, self.objectives)

    def copy(self) -> "Target":
        return Target(self.objectives, self.weights)

    def __str__(self):
        return f"Objectives: {self.objectives}, Fitness: {self.fitness}"
