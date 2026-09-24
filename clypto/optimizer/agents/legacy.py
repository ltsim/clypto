#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Public classic agent used by the ``@cy.legacy`` decorator API.

This is intentionally a plain Python class, separate from the private compiled
``clypto.optimizer._native.agent._LegacyAgent`` used by the built-in algorithm collection.
"""
import typing

import numpy as np
from clypto.hints.array import NDArrayType
from clypto.hints.sense import SenseType
from clypto.optimizer.target import Target

__all__ = ["LegacyAgent"]


class LegacyAgent:
    """Minimal ``[solution, target]`` agent with the classic comparison API."""

    solution: typing.Optional[NDArrayType] = None
    target: typing.Optional[Target] = None

    def __init__(
        self,
        solution: typing.Optional[NDArrayType] = None,
        target: typing.Optional[Target] = None,
    ) -> None:
        self.solution = solution
        self.target = target

    def copy(self) -> "LegacyAgent":
        return LegacyAgent(
            self.solution,
            None if self.target is None else self.target.copy(),
        )

    def update_agent(self, solution: NDArrayType, target: Target) -> None:
        self.solution = solution
        self.target = target

    def update(self, **kwargs: typing.Any) -> None:
        if "solution" in kwargs:
            self.solution = kwargs.pop("solution")
        if "target" in kwargs:
            self.target = kwargs.pop("target")
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected attribute(s): {sorted(kwargs)}"
            )

    def sync_if_duplicate(self, other: "LegacyAgent") -> bool:
        if self == other:
            self.target = other.target
            return True
        return False

    def _compare_fitness(self, other: "LegacyAgent", minmax: SenseType = "min") -> int:
        if self.target.fitness == other.target.fitness:
            return 0

        if minmax == "min":
            return -1 if self.target.fitness < other.target.fitness else 1

        return -1 if self.target.fitness > other.target.fitness else 1

    def get_better_solution(
        self, other: "LegacyAgent", minmax: SenseType = "min"
    ) -> "LegacyAgent":
        return self if self._compare_fitness(other, minmax) <= 0 else other

    def is_better_than(self, other: "LegacyAgent", minmax: SenseType = "min") -> bool:
        return self._compare_fitness(other, minmax) == -1

    def __repr__(self) -> str:
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, LegacyAgent):
            return False

        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self) -> int:
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self) -> float:
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")

        return self.target.fitness
