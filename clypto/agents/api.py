#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""Agent declarations and the :func:`agent` decorator.

The decorator API gives every solution a *runtime agent*: an object whose
``solution`` is a plain NumPy vector and whose ``fitness`` is always derived
from the objective. Assigning ``agent.solution`` re-evaluates the objective
immediately, so ``fitness`` can never drift out of sync, and it is read-only.
"""

import typing

import numpy as np
from clypto.hints.array import NDArrayType
from clypto.utils.target import Target

__all__ = ["Attribute", "RuntimeAgent", "agent"]


class Attribute:
    """Declare a per-agent attribute on a class decorated with ``@cy.agent``.

    Example::

        @cy.agent
        class MyAgent:
            v: cy.Attribute(int, (1, 100), 5)

    Args:
        dtype: The attribute type (e.g. ``int``, ``float``, ``bool``).
        bound: Optional validation range, following the validator convention
            (a ``tuple`` bound is exclusive, a ``list`` bound inclusive).
        default: Value used when an agent is generated without an explicit one.
    """

    __slots__ = ("type", "bound", "default")

    def __init__(self, dtype: typing.Any = None, bound: typing.Any = None, default: typing.Any = None) -> None:
        self.type = dtype
        self.bound = bound
        self.default = default


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


def _declare_attributes(cls: typing.Any) -> dict:
    """Collect ``Attribute`` declarations from the class and its bases."""
    declarations: dict[str, Attribute] = {}

    for base in reversed(cls.__mro__):
        for name, declaration in (getattr(base, "__annotations__", {}) or {}).items():
            if isinstance(declaration, Attribute):
                declarations[name] = declaration
        for name, value in vars(base).items():
            if isinstance(value, Attribute):
                declarations[name] = value

    return declarations


def agent(cls: typing.Optional[typing.Any] = None, *, compile: bool = False):
    """Turn a plain class into an agent usable by ``@cy.optimizer``.

    The class may declare per-agent attributes with :class:`Attribute`; the
    decorated class gains ``solution``/``fitness``/``target`` from
    :class:`RuntimeAgent`. When ``compile`` is ``True`` the class is also
    Cython-compiled through the same JIT builder used by the legacy API.
    """

    def decorate(user_cls):
        from clypto.optimizer.api import _decorate_with_base, _maybe_compile

        decorated = _decorate_with_base(user_cls, RuntimeAgent)
        declarations = _declare_attributes(decorated)
        setattr(decorated, "_attributes", declarations)

        for name, declaration in declarations.items():
            # The compiled source materializes declarations as assignments, so
            # the class dict may still hold the Attribute placeholder itself.
            current = decorated.__dict__.get(name)
            if current is None or isinstance(current, Attribute):
                setattr(decorated, name, declaration.default)

        return _maybe_compile(
            decorated,
            compile,
            source_cls=user_cls,
            base_name="RuntimeAgent",
            import_line="from clypto.agents.api import RuntimeAgent",
        )

    return decorate(cls) if cls is not None else decorate
