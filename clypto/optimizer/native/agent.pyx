#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""Agents and the functions that compare and copy them.

``LegacyAgent`` (legacy engine, ``@cy.legacy``) and ``LegacyNativeAgent``
(vectorize engine, typed ``NativeTarget``) only store data. Algorithms that need
extra per-agent state subclass one of them and declare ``cdef public`` fields;
``copy()`` and ``update(**fields)`` then work without being overridden.
"""
import numpy as np

cdef dict _FIELDS = {}


cdef tuple _fields_of(type cls):
    """Extra data fields of ``cls``: its ``cdef public`` attributes (cached per type)."""
    fields = _FIELDS.get(cls)
    if fields is None:
        fields = tuple(
            name
            for klass in cls.__mro__[:-1]
            for name, attr in vars(klass).items()
            if type(attr).__name__ == "getset_descriptor"
            and not name.startswith("__")
            and name not in ("solution", "target")
        )
        _FIELDS[cls] = fields
    return fields


cpdef object duplicate_agent(object agent):
    """Copy of ``agent``: every field is shared, only the ``target`` is copied."""
    cls = type(agent)
    new = cls.__new__(cls)
    for name in _fields_of(cls):
        setattr(new, name, getattr(agent, name))
    if hasattr(agent, "__dict__"):
        new.__dict__.update(agent.__dict__)
    new.solution = agent.solution
    new.target = None if agent.target is None else agent.target.copy()
    return new


cpdef bint sync_if_duplicate(object that, object other):
    """Give ``that`` the target of ``other`` when their solutions match."""
    if that == other:
        that.target = other.target
        return True
    return False


cpdef int compare_fitness(object that, object other, str sense="min"):
    """-1 if ``that`` is better, 1 if ``other`` is, 0 on a tie."""
    cdef double f1 = that.target.fitness
    cdef double f2 = other.target.fitness
    if f1 == f2:
        return 0
    if sense == "min":
        return -1 if f1 < f2 else 1
    return -1 if f1 > f2 else 1


cpdef object get_better_solution(object that, object other, str sense="min"):
    """The better agent; a tie keeps ``that``."""
    return that if compare_fitness(that, other, sense) <= 0 else other


cpdef bint is_better_than(object that, object other, str sense="min"):
    return compare_fitness(that, other, sense) == -1


cdef class LegacyAgent:
    def __init__(self, solution=None, target=None, **fields):
        self.solution = solution
        self.target = target
        for name, value in fields.items():
            setattr(self, name, value)

    def copy(self):
        return duplicate_agent(self)

    def update(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)

    def __repr__(self):
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other):
        if not isinstance(other, LegacyAgent):
            return False
        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self):
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")
        return self.target.fitness


cdef class LegacyNativeAgent:
    """Agent of the vectorize engine; its ``target`` is a ``NativeTarget``."""

    def __init__(self, solution=None, NativeTarget target=None, **fields):
        self.solution = solution
        self.target = target
        for name, value in fields.items():
            setattr(self, name, value)

    def copy(self):
        return duplicate_agent(self)

    def update(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)

    def __repr__(self):
        return f"Agent(target={self.target}, solution={self.solution})"

    def __eq__(self, other):
        if not isinstance(other, LegacyNativeAgent):
            return False
        return np.allclose(self.solution, other.solution, atol=1e-6)

    def __hash__(self):
        return hash(tuple(np.round(self.solution, 6)))

    def __float__(self):
        if self.target is None:
            raise ValueError("Agent cannot generate a value from fitness")
        return self.target.fitness
