#!/usr/bin/env python
# Created by "Thieu" at 15:51, 05/10/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numbers

import numpy as np


class Target:
    SUPPORTED_ARRAY = tuple, list, np.ndarray

    def __init__(self, objectives: list | tuple | np.ndarray | int | float | None = None,
                 weights: list | tuple | np.ndarray | None = None) -> None:
        """
        Initialize the Target with a list of objectives and a fitness value.

        Parameters:
            objectives: The list of objective values.
            weights: The weights for calculating fitness value
        """

        def set_objectives(objs):
            if objs is None:
                raise ValueError(f"Invalid objectives. It should be a list, tuple, np.ndarray, int or float.")
            else:
                if type(objs) not in self.SUPPORTED_ARRAY:
                    if isinstance(objs, numbers.Number):
                        objs = [objs]
                    else:
                        raise ValueError(f"Invalid objectives. It should be a list, tuple, np.ndarray, int or float.")
                objs = np.array(objs).flatten()
            self.__objectives = objs

        def set_weights(weights):
            if weights is None:
                self.__weights = len(self.objectives)
            else:
                if type(weights) not in self.SUPPORTED_ARRAY:
                    if isinstance(weights, numbers.Number):
                        weights = [weights, ] * len(self.objectives)
                    else:
                        raise ValueError(f"Invalid weights. It should be a list, tuple, np.ndarray.")
                weights = np.array(weights).flatten()
            self.__weights = weights

        def calculate_fitness(weights):
            if not (type(weights) in self.SUPPORTED_ARRAY and len(weights) == len(self.objectives)):
                weights = len(self.objectives) * (1.,)
            self.__fitness = np.dot(weights, self.objectives)

        self.__objectives = None
        self.__weights = None
        self.__fitness = None

        set_objectives(objectives)
        set_weights(weights)
        calculate_fitness(self.weights)

    def copy(self) -> 'Target':
        return Target(self.objectives, self.weights)

    @property
    def objectives(self):
        """Returns the list of objective values."""
        return self.__objectives

    @property
    def weights(self):
        """Returns the list of weight values."""
        return self.__weights

    @property
    def fitness(self):
        """Returns the fitness value."""
        return self.__fitness

    def __str__(self):
        return f"Objectives: {self.objectives}, Fitness: {self.fitness}"
