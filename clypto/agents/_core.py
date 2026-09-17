#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import typing

import cython
import numpy as np
from clypto.hints.array import NDArrayType
from clypto.hints.sense import SenseType
from clypto.utils.target import Target


@cython.cclass
class LegacyAgent:
    def copy(self) -> "LegacyAgent":
        raise NotImplementedError

    def update_agent(self, solution: NDArrayType, target: Target) -> "LegacyAgent":
        raise NotImplementedError

    def update(self, *args: typing.Any, **kwargs: typing.Any) -> "LegacyAgent":
        raise NotImplementedError

    def sync_if_duplicate(self, other: "LegacyAgent") -> bool:
        raise NotImplementedError

    def _compare_fitness(self, other: "LegacyAgent", minmax: SenseType = "min") -> int:
        """
        Compare fitness between self and other.

        Returns:
            -1 if self is better
             0 if equal
             1 if other is better
        """
        raise NotImplementedError

    def get_better_solution(
        self, other: "LegacyAgent", minmax: SenseType = "min"
    ) -> "LegacyAgent":
        """
        Return better solution

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self if self._compare_fitness(other, minmax) <= 0 else other

    def is_better_than(self, other: "LegacyAgent", minmax: SenseType = "min") -> bool:
        """
        Compare the current agent with other agent. Return True if current agent is better and False otherwise

        Args:
            other: The compared agent
            minmax: The problem
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other: typing.Any) -> bool:
        """Check if two agents are equal based on their solutions with a tolerance."""
        raise NotImplementedError

    def __hash__(self) -> int:
        """Generate a hash based on the solution of the agent.
        This is useful for using agents in sets or as dictionary keys."""
        raise NotImplementedError

    def __float__(self) -> float:
        """Compute the fitness value from self and other."""
        raise NotImplementedError


@cython.cclass
class AgentStatic(LegacyAgent):
    solution: object = cython.declare(object, visibility="public")
    target: object = cython.declare(object, visibility="public")

    def __init__(
        self,
        solution: typing.Optional[NDArrayType] = None,
        target: typing.Optional[Target] = None,
    ) -> None:
        self.solution = solution
        self.target = target

    def __getattr__(self, name: str) -> typing.Any:
        return None

    def copy(self) -> "AgentStatic":
        return AgentStatic(self.solution, self.target.copy())

    def update_agent(self, solution: NDArrayType, target: Target) -> None:
        self.solution = solution
        self.target = target

    def update(self, *args, **kwargs) -> None:
        self.update_agent(
            solution=kwargs.get("solution", self.solution),
            target=kwargs.get("target", self.target),
        )

    def sync_if_duplicate(self, other: "AgentStatic") -> bool:
        """
        Check if two agents are equal (using __eq__), and if so, synchronize the target from the other agent.

        Returns:
            bool: True if duplicate (and target updated), False otherwise.
        """
        is_eq = self == other

        if is_eq:  # use __eq__
            self.target = other.target

        return is_eq

    def _compare_fitness(self, other: "AgentStatic", minmax: SenseType = "min") -> int:
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

    def get_better_solution(
        self, other: "AgentStatic", minmax: SenseType = "min"
    ) -> "AgentStatic":
        """
        Return better solution

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self if self._compare_fitness(other, minmax) <= 0 else other

    def is_better_than(self, other: "AgentStatic", minmax: SenseType = "min") -> bool:
        """
        Compare the current agent with other agent. Return True if current agent is better and False otherwise

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self._compare_fitness(other, minmax) == -1

    def __repr__(self):
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other: typing.Any):
        """Check if two agents are equal based on their solutions with a tolerance."""
        if not isinstance(other, AgentStatic):
            return False

        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        """Generate a hash based on the solution of the agent.
        This is useful for using agents in sets or as dictionary keys."""
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self) -> float:
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")

        return self.target.fitness


@cython.cclass
class AgentDynamic(LegacyAgent):
    solution: object = cython.declare(object, visibility="public")
    target: object = cython.declare(object, visibility="public")
    __dict__: dict

    def __init__(
        self,
        solution: typing.Optional[NDArrayType] = None,
        target: typing.Optional[Target] = None,
        **kwargs,
    ) -> None:
        self.solution = solution
        self.target = target
        self.__kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name: str) -> typing.Any:
        # cdef classes forbid `self.__dict__` access; cdef-class attribute lookup
        # already consults the instance dict before __getattr__ fires, so any
        # name reaching here is missing and maps to None (same contract as
        # AgentStatic.__getattr__ above).
        return None

    def copy(self) -> "LegacyAgent":
        agent = AgentDynamic(self.solution, self.target.copy(), **self.__kwargs)

        for attr, value in vars(self).items():
            if attr not in ["target", "solution", "id", "kwargs"]:
                setattr(agent, attr, value)

        return agent

    def update_agent(self, solution: NDArrayType, target: Target) -> None:
        self.solution = solution
        self.target = target

    def update(self, *args, **kwargs) -> None:
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def sync_if_duplicate(self, other: "LegacyAgent") -> bool:
        """
        Check if two agents are equal (using __eq__), and if so, synchronize the target from the other agent.

        Returns:
            bool: True if duplicate (and target updated), False otherwise.
        """
        is_eq = self == other

        if is_eq:  # use __eq__
            self.target = other.target

        return is_eq

    def _compare_fitness(self, other: "LegacyAgent", minmax: SenseType = "min") -> int:
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

    def get_better_solution(
        self, other: "LegacyAgent", minmax: SenseType = "min"
    ) -> "LegacyAgent":
        """
        Return better solution

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self if self._compare_fitness(other, minmax) <= 0 else other

    def is_better_than(self, other: "LegacyAgent", minmax: SenseType = "min") -> bool:
        """
        Compare the current agent with other agent. Return True if current agent is better and False otherwise

        Args:
            other: The compared agent
            minmax: The problem
        """
        return self._compare_fitness(other, minmax) == -1

    def __repr__(self):
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other: typing.Any):
        """Check if two agents are equal based on their solutions with a tolerance."""
        if not isinstance(other, AgentDynamic):
            return False

        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        """Generate a hash based on the solution of the agent.
        This is useful for using agents in sets or as dictionary keys."""
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self) -> float:
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")

        return self.target.fitness
