#!/usr/bin/env python
# Created for clypto's decorator-based optimizer API.
# --------------------------------------------------%
"""The ``Attribute`` declaration used by ``@cy.agent`` classes."""

from clypto.precompile.decoration import Declaration

__all__ = ["Attribute"]


class Attribute(Declaration):
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

    __slots__ = ()
