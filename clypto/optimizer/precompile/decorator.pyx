#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``@cy.optimizer`` decorator.

``@cy.optimizer(agent=MyAgent)`` turns a plain class into an optimizer. Its
hyper-parameters are declared as :class:`Argument` attributes and it implements
``initialize`` and ``evolve``. The injected base supplies ``solve``,
``self.population``, ``self.rng``, ``self.bounds`` and ``generate_agent``, plus
the overridable ``generate(agent, solution)`` hook for seeding custom agent
state; ``compile=True`` runs the JIT Cython builder.
"""

import typing

from clypto.optimizer.agents.runtime import RuntimeAgent
from clypto.optimizer.agents.declaration import Attribute
from clypto.optimizer.precompile.declaration import Argument
from clypto.optimizer.precompile.base import DecoratedOptimizer
from clypto.optimizer.precompile.decoration import (
    collect_declarations,
    decorate_with_base,
    maybe_compile,
)

__all__ = ["optimizer"]

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
                from clypto.optimizer.agents.decorator import agent as agent_decorator

                selected_agent = agent_decorator(selected_agent)
            decorated.agent_class = selected_agent

        decorated.__clypto_arguments__ = collect_declarations(decorated, (Argument,))

        return maybe_compile(
            decorated,
            compile,
            source_cls=user_cls,
            base_name="DecoratedOptimizer",
            import_line="from clypto.optimizer.precompile.base import DecoratedOptimizer",
            declaration_types=(Argument, Attribute),
        )

    return decorate(cls) if cls is not None else decorate

