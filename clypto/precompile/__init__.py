#!/usr/bin/env python
"""Just-in-time Cython compilation for user-defined optimizers via pyximport."""
import inspect
import sys
import textwrap

from clypto.optimizer.classic import Optimizer
from clypto.precompile import _runtime, _builder

__all__ = ["precompile"]

_PRECOMPILING = False


def precompile(cls: type) -> type:
    """
    Compile a user-defined ``Optimizer`` subclass with Cython through pyximport.

    The decorator rebuilds the class from its source and returns the compiled
    class, so the name it is assigned to is the fast one. Requires Cython and a
    working C compiler; it raises ``ImportError`` if Cython is missing and
    ``RuntimeError`` if the source is unavailable or compilation fails.

    Args:
        cls: The optimizer class to compile.

    Returns:
        The Cython-compiled class (or ``cls`` itself when re-entered during
        compilation).
    """
    global _PRECOMPILING

    if _PRECOMPILING:
        return cls

    if not (inspect.isclass(cls) and issubclass(cls, Optimizer)):
        raise TypeError("precompile() only supports Optimizer subclasses.")

    version = _runtime.cython_version()

    caller_globals = sys._getframe(1).f_globals
    names = sorted(
        name
        for name in caller_globals
        if name.isidentifier() and not (name.startswith("__") and name.endswith("__"))
    )

    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "precompile() needs access to the class source; it is unavailable "
            "in a REPL, notebook, or dynamically created class."
        ) from exc

    module_name = _builder.build_module_name(
        cls.__name__, _runtime.source_digest(source, version)
    )

    _PRECOMPILING = True

    try:
        compiled_cls = _builder.compile_class(cls, source, module_name, names)
    finally:
        _PRECOMPILING = False

    compiled_cls.__clypto_precompiled__ = True  # type: ignore[attr-defined]

    return compiled_cls
