#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Private legacy agent base used by the built-in algorithm collection.

The public classic agent lives in ``clypto.agents.legacy``; this module is the
compiled, field-private base that the ``.pyx`` collection subclasses. Algorithms
that need extra per-agent state declare their own ``cdef class`` subclass here.
"""
import numpy as np


cdef class _LegacyAgent:
    def __init__(self, solution=None, target=None):
        self.solution = solution
        self.target = target

    cpdef object copy(self):
        if type(self) is not _LegacyAgent:
            raise NotImplementedError(
                f"{type(self).__name__} must implement its own copy()"
            )
        cdef object target = None if self.target is None else self.target.copy()
        return _LegacyAgent(self.solution, target)

    cpdef void update_agent(self, solution, target):
        self.solution = solution
        self.target = target

    def update(self, **kwargs):
        if "solution" in kwargs:
            self.solution = kwargs.pop("solution")
        if "target" in kwargs:
            self.target = kwargs.pop("target")
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected attribute(s): {sorted(kwargs)}"
            )

    cpdef bint sync_if_duplicate(self, _LegacyAgent other):
        if self == other:
            self.target = other.target
            return True
        return False

    cdef int _compare_fitness(self, _LegacyAgent other, str minmax):
        cdef double f1 = self.target.fitness
        cdef double f2 = other.target.fitness
        if f1 == f2:
            return 0
        if minmax == "min":
            return -1 if f1 < f2 else 1
        return -1 if f1 > f2 else 1

    cpdef object get_better_solution(self, _LegacyAgent other, str minmax="min"):
        return self if self._compare_fitness(other, minmax) <= 0 else other

    cpdef bint is_better_than(self, _LegacyAgent other, str minmax="min"):
        return self._compare_fitness(other, minmax) == -1

    def __repr__(self):
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other):
        if not isinstance(other, _LegacyAgent):
            return False
        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self):
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")
        return self.target.fitness
