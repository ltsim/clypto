#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from typing import Any

import numpy as np

from mealpy.agents.base import BaseAgent
from mealpy.utils.target import Target


class Agent(BaseAgent):
    def __init__(self, solution: np.ndarray | None = None, target: Target | None = None, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self.__solution = solution
        self.__target = target
        self.__kwargs = kwargs

    @property
    def target(self) -> Target | None:
        return self.__target

    @target.setter
    def target(self, target: Target | None) -> None:
        self.__target = target

    @property
    def solution(self) -> np.ndarray | None:
        return self.__solution

    @solution.setter
    def solution(self, solution: np.ndarray):
        self.__solution = solution

    def __getattr__(self, name: str) -> Any:
        return self.__dict__.get(name, None)

    def copy(self) -> 'BaseAgent':
        agent = Agent(self.solution, self.target.copy(), **self.__kwargs)

        for attr, value in vars(self).items():
            if attr not in ['target', 'solution', 'id', 'kwargs']:
                setattr(agent, attr, value)

        return agent

    def update_agent(self, solution: np.ndarray, target: Target) -> None:
        self.__solution = solution
        self.__target = target

    def update(self, **kwargs) -> None:
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def sync_if_duplicate(self, other: "Agent") -> bool:
        """
        Check if two agents are equal (using __eq__), and if so, synchronize the target from the other agent.

        Returns:
            bool: True if duplicate (and target updated), False otherwise.
        """
        is_eq = self == other

        if is_eq:  # use __eq__
            self.__target = other.target

        return is_eq

    def _compare_fitness(self, other: "Agent", minmax: str = "min") -> int:
        """
        Compare fitness between self and other.

        Returns:
            -1 if self is better
             0 if equal
             1 if other is better
        """
        if self.target.fitness == other.target.fitness:
            return 0
        elif minmax == "min":
            return -1 if self.target.fitness < other.target.fitness else 1
        else:
            return -1 if self.target.fitness > other.target.fitness else 1

    def get_better_solution(self, other: "Agent", minmax: str = "min") -> "Agent":
        """
        Return better solution

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self if self._compare_fitness(other, minmax) <= 0 else other

    def is_better_than(self, other: "Agent", minmax: str = "min") -> bool:
        """
        Compare the current agent with other agent. Return True if current agent is better and False otherwise

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self._compare_fitness(other, minmax) == -1

    def __repr__(self):
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other):
        """ Check if two agents are equal based on their solutions with a tolerance."""
        if not isinstance(other, Agent):
            return False

        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        """ Generate a hash based on the solution of the agent.
            This is useful for using agents in sets or as dictionary keys."""
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self) -> float:
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")

        return self.target.fitness
