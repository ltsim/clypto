#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``@cy.optimizer`` and ``@cy.legacy`` decorators.

* ``@cy.optimizer(agent=MyAgent)`` — a new-style optimizer. Its hyper-parameters
  are declared as :class:`Argument` attributes and it implements ``initialize``
  and ``evolve``. The base class supplies ``solve``, ``self.population``,
  ``self.rng``, ``self.bounds`` and ``generate_agent``.
* ``@cy.legacy(precompile=False)`` — the classic MEALPY-style API (``self.pop``,
  ``self.validator``, ``generate_empty_agent``, ...) without having to name the
  base class; ``precompile=True`` Cython-compiles it.

Both inject their base class instead of asking for inheritance, and both accept
a compile flag to run the same JIT Cython builder.
"""

import typing

from clypto.agents.template.runtime import RuntimeAgent
from clypto.agents.template.declaration import Attribute
from clypto.optimizer.classic import LegacyOptimizer
from clypto.optimizer.template.declaration import Argument
from clypto.optimizer.template.base import DecoratedOptimizer
from clypto.precompile.decoration import (
    collect_declarations,
    decorate_with_base,
    maybe_compile,
)

__all__ = ["optimizer", "legacy"]

_C = typing.TypeVar("_C")


@typing.overload
def optimizer(cls: _C) -> _C: ...


@typing.overload
def optimizer(
    *, agent: typing.Optional[typing.Any] = None, compile: bool = False
) -> typing.Callable[[_C], _C]: ...


def optimizer(cls=None, *, agent=None, compile=False):
    """Turn a plain class into a new-style optimizer (see module docstring)."""

    def decorate(user_cls):
        if "evolve" not in user_cls.__dict__:
            raise TypeError(f"{user_cls.__name__} must define an evolve(self, epoch) method.")

        decorated = decorate_with_base(user_cls, DecoratedOptimizer)

        if agent is not None:
            selected_agent = agent
            if not issubclass(selected_agent, RuntimeAgent):
                from clypto.agents.template.decorator import agent as agent_decorator

                selected_agent = agent_decorator(selected_agent)
            decorated.agent_class = selected_agent

        decorated.__clypto_arguments__ = collect_declarations(decorated, (Argument,))

        return maybe_compile(
            decorated,
            compile,
            source_cls=user_cls,
            base_name="DecoratedOptimizer",
            import_line="from clypto.optimizer.template.base import DecoratedOptimizer",
            declaration_types=(Argument, Attribute),
        )

    return decorate(cls) if cls is not None else decorate


@typing.overload
def legacy(cls: _C) -> _C: ...


@typing.overload
def legacy(*, precompile: bool = False) -> typing.Callable[[_C], _C]: ...


def legacy(cls=None, *, precompile=False):
    """Use the classic optimizer API without inheriting from it explicitly.

    ``precompile=True`` Cython-compiles the class through the JIT builder.
    """

    def decorate(user_cls):
        decorated = decorate_with_base(user_cls, LegacyOptimizer)
        return maybe_compile(
            decorated,
            precompile,
            source_cls=user_cls,
            base_name="LegacyOptimizer",
            import_line="from clypto import LegacyOptimizer",
        )

    return decorate(cls) if cls is not None else decorate
