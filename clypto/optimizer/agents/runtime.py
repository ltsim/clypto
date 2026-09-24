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
from clypto.optimizer.target import Target

__all__ = ["RuntimeAgent"]


class RuntimeAgent:
    """Solution-only agent used by ``@cy.optimizer`` when none is declared.

    This is intentionally a plain Python class (not a Cython ``cdef`` class) so
    that user agent classes created by ``@cy.agent`` can inject it as a base and
    carry arbitrary declared attributes.
    """

    id: typing.Optional[int]
    _solution: typing.Optional[NDArrayType]
    _target: typing.Optional[Target]
    _evaluator: typing.Optional[typing.Callable[[NDArrayType], Target]]

    def __init__(
        self,
        solution: typing.Optional[NDArrayType] = None,
        target: typing.Optional[Target] = None,
        **attributes: typing.Any,
    ) -> None:
        object.__setattr__(self, "id", None)
        object.__setattr__(self, "_solution", None)
        object.__setattr__(self, "_target", target)
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
        # cached target/fitness can never go stale.
        object.__setattr__(self, "_solution", None if value is None else np.asarray(value, dtype=float))

        evaluator = self._evaluator
        if evaluator is not None and self._solution is not None:
            object.__setattr__(self, "_target", evaluator(self._solution))

    @property
    def target(self) -> typing.Optional[Target]:
        return self._target

    @property
    def fitness(self) -> typing.Optional[float]:
        target = self._target
        return None if target is None else target.fitness

    def copy(self) -> "RuntimeAgent":
        new = type(self)(
            self._solution,
            self._target.copy() if self._target is not None else None,
        )
        new.id = self.id
        object.__setattr__(new, "_evaluator", self._evaluator)

        for name in getattr(type(self), "_attributes", {}):
            setattr(new, name, getattr(self, name))

        return new

    def __repr__(self) -> str:
        return f"{type(self).__name__}(fitness={self.fitness}, solution={self._solution})"
