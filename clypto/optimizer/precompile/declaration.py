#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``Argument`` declaration used by ``@cy.optimizer`` classes."""

from clypto.precompile.decoration import Declaration

__all__ = ["Argument"]


class Argument(Declaration):
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

    __slots__ = ()
