#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``@cy.agent`` decorator."""

import typing

from clypto.agents.template.runtime import RuntimeAgent
from clypto.agents.template.declaration import Attribute
from clypto.precompile.decoration import (
    collect_declarations,
    decorate_with_base,
    maybe_compile,
)

__all__ = ["agent"]

_C = typing.TypeVar("_C")


@typing.overload
def agent(cls: _C) -> _C: ...


@typing.overload
def agent(*, compile: bool = False) -> typing.Callable[[_C], _C]: ...


def agent(cls=None, *, compile=False):
    """Turn a plain class into an agent usable by ``@cy.optimizer``.

    The class may declare per-agent attributes with :class:`Attribute`; the
    decorated class gains ``solution``/``fitness``/``target`` from
    :class:`RuntimeAgent`. When ``compile`` is ``True`` the class is also
    Cython-compiled through the same JIT builder used by the legacy API.
    """

    def decorate(user_cls):
        decorated = decorate_with_base(user_cls, RuntimeAgent)
        declarations = collect_declarations(decorated, (Attribute,))
        decorated._attributes = declarations

        for name, declaration in declarations.items():
            # The compiled source materializes declarations as assignments, so
            # the class dict may still hold the Attribute placeholder itself.
            current = decorated.__dict__.get(name)
            if current is None or isinstance(current, Attribute):
                setattr(decorated, name, declaration.default)

        return maybe_compile(
            decorated,
            compile,
            source_cls=user_cls,
            base_name="RuntimeAgent",
            import_line="from clypto.agents.template.runtime import RuntimeAgent",
            declaration_types=(Attribute,),
        )

    return decorate(cls) if cls is not None else decorate
