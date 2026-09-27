#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The runtime agent used by the decorator API.

The decorator API gives every solution a *runtime agent*: an object whose
``solution`` is a plain NumPy vector and whose ``fitness`` is always derived
from the objective. Assigning ``agent.solution`` re-evaluates the objective
immediately, so ``fitness`` can never drift out of sync, and it is read-only.
"""

import typing

import numpy as np
from clypto.hints.array import NDArrayType
from clypto.optimizer.native.agent import Agent

__all__ = ["RuntimeAgent"]


class RuntimeAgent:
    """Solution-only agent used by ``@cy.optimizer`` when none is declared.

    This is intentionally a plain Python class (not a Cython ``cdef`` class) so
    that user agent classes created by ``@cy.agent`` can inject it as a base and
    carry arbitrary declared attributes.
    """

    _solution: typing.Optional[NDArrayType]
    _objectives: typing.Optional[NDArrayType]
    _fitness: typing.Optional[float]
    _evaluator: typing.Optional[typing.Callable[[NDArrayType], Agent]]

    def __init__(self, solution: typing.Optional[NDArrayType] = None, **attributes: typing.Any) -> None:
        object.__setattr__(self, "_solution", None)
        object.__setattr__(self, "_objectives", None)
        object.__setattr__(self, "_fitness", None)
        object.__setattr__(self, "_evaluator", None)

        for name, value in attributes.items():
            setattr(self, name, value)

        if solution is not None:
            self.solution = solution

    @property
    def solution(self) -> typing.Optional[NDArrayType]:
        return self._solution

    @solution.setter
    def solution(self, value: typing.Optional[NDArrayType]) -> None:
        # The only way to change fitness is through this setter: it stores the
        # new vector and immediately re-runs the evaluation callback, so the
        # cached objectives/fitness can never go stale.
        object.__setattr__(self, "_solution", None if value is None else np.asarray(value, dtype=float))

        evaluator = self._evaluator
        if evaluator is not None and self._solution is not None:
            evaluated = evaluator(self._solution)
            object.__setattr__(self, "_objectives", evaluated.objectives)
            object.__setattr__(self, "_fitness", evaluated.fitness)

    @property
    def objectives(self) -> typing.Optional[NDArrayType]:
        return self._objectives

    @property
    def fitness(self) -> typing.Optional[float]:
        return self._fitness

    def copy(self) -> "RuntimeAgent":
        new = type(self)(self._solution)
        object.__setattr__(new, "_objectives", self._objectives)
        object.__setattr__(new, "_fitness", self._fitness)
        object.__setattr__(new, "_evaluator", self._evaluator)

        for name in getattr(type(self), "_attributes", {}):
            setattr(new, name, getattr(self, name))

        return new

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fitness={self.fitness}, solution={self._solution})"
