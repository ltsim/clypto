#!/usr/bin/env python
# Created by "Thieu" at 16:19, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
"""IPython magic extension for Clypto (Jupyter Notebooks)."""

from __future__ import annotations

import inspect
from typing import Any

from clypto.compiler import compile_module

__all__ = ["ClyptoMagics", "load_ipython_extension"]

try:
    from IPython.core.magic import Magics, cell_magic, magics_class
except ImportError:
    def magics_class(cls: Any) -> Any:
        return cls

    def cell_magic(fn: Any) -> Any:
        return fn

    class Magics:  # type: ignore[no-redef]
        def __init__(self, shell: Any = None) -> None:
            self.shell = shell


@magics_class
class ClyptoMagics(Magics):
    """IPython magics for compiling Cython algorithms in notebooks."""

    @cell_magic
    def clypto(self, line: str, cell: str) -> Any:
        """Compile and load Cython code in the current cell.

        Automatically injects Clypto and NumPy include directories and compiler
        directives, then imports all defined classes, functions, and objects
        into the notebook's user namespace.

        Usage:
            %%clypto
            cimport clypto.core as cy
            import numpy as np

            cdef class MyAlgo(cy.Optimizer):
                ...
        """
        line_clean = line.strip()
        name: str | None = None
        if line_clean:
            parts = line_clean.split()
            if parts[0] in ("-n", "--name") and len(parts) > 1:
                name = parts[1]
            elif not parts[0].startswith("-"):
                name = parts[0]

        module = compile_module(cell, name=name)

        # Export all non-private symbols from the compiled module into ipython user_ns
        if self.shell is not None and hasattr(self.shell, "user_ns"):
            for attr_name, val in inspect.getmembers(module):
                if attr_name.startswith("__"):
                    continue
                if getattr(val, "__module__", None) == module.__name__ or not attr_name.startswith("_"):
                    self.shell.user_ns[attr_name] = val

        return None


def load_ipython_extension(ipython: Any) -> None:
    """Register the ``%%clypto`` cell magic with the active IPython session."""
    if hasattr(ipython, "register_magics"):
        ipython.register_magics(ClyptoMagics)
    elif hasattr(ipython, "register_magic_function"):
        magics = ClyptoMagics(shell=ipython)
        ipython.register_magic_function(magics.clypto, magic_kind="cell", magic_name="clypto")
