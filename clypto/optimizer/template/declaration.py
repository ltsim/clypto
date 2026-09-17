#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``Argument`` declaration used by ``@cy.optimizer`` classes."""

import typing

from clypto.precompile.decoration import parse_declaration_params

__all__ = ["Argument"]


class Argument:
    """Declare an optimizer hyper-parameter.

    Example::

        @cy.optimizer(agent=MyAgent)
        class MyOptimizer:
            a: cy.Argument[int, (1, 100), 5]
            c: cy.Argument[np.ndarray]

    The bracket syntax follows the Python typing convention and accepts
    ``[dtype]``, ``[dtype, bound]`` and ``[dtype, bound, default]``; ``...`` in
    the bound position means "no bound".

    Args:
        dtype: Expected type. ``np.array``/``np.ndarray`` accept array values.
        bound: Optional validation range (``tuple`` exclusive, ``list`` inclusive).
        default: Value used when the argument is not passed to the constructor.
    """

    __slots__ = ("type", "bound", "default")

    def __init__(self, dtype: typing.Any = None, bound: typing.Any = None, default: typing.Any = None) -> None:
        self.type = dtype
        self.bound = bound
        self.default = default

    def __class_getitem__(cls, params: typing.Any) -> "Argument":
        return cls(*parse_declaration_params(params))
