#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``Attribute`` declaration used by ``@cy.agent`` classes."""

import typing

from clypto.precompile.decoration import parse_declaration_params

__all__ = ["Attribute"]


class Attribute:
    """Declare a per-agent attribute on a class decorated with ``@cy.agent``.

    Example::

        @cy.agent
        class MyAgent:
            v: cy.Attribute[int, (1, 100), 5]

    The bracket syntax follows the Python typing convention and accepts
    ``[dtype]``, ``[dtype, bound]`` and ``[dtype, bound, default]``; ``...`` in
    the bound position means "no bound".

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

    def __class_getitem__(cls, params: typing.Any) -> "Attribute":
        return cls(*parse_declaration_params(params))
