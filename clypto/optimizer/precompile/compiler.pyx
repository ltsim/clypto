#!/usr/bin/env python
"""Just-in-time Cython compilation for user-defined optimizers via pyximport."""
import inspect
import sys
import textwrap

from clypto.optimizer.precompile import _runtime, _builder

__all__ = ["compile_decorated", "is_precompiling"]

_PRECOMPILING = False


def is_precompiling() -> bool:
    """True while a compiled module is being imported and re-runs its decorators."""
    return _PRECOMPILING


def compile_decorated(
    cls: type,
    *,
    source: str | None = None,
    imports: list[str] | None = None,
) -> type:
    """Compile a class produced by the decorator API.

    The globals are read from ``cls.__module__``. ``source`` may override the class
    source (used when a decorator injects a base the source does not name), and
    ``imports`` are added to the generated preamble so that injected base names
    resolve in the compiled module.
    """
    global _PRECOMPILING

    if _PRECOMPILING:
        return cls

    version = _runtime.cython_version()

    if source is None:
        try:
            source = textwrap.dedent(inspect.getsource(cls))
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                "Compiling a decorated optimizer needs access to its class "
                "source; it is unavailable in a REPL, notebook, or dynamically "
                "created class."
            ) from exc

    module = sys.modules.get(cls.__module__)
    caller_globals = dict(vars(module)) if module is not None else {}
    names = sorted(
        name
        for name in caller_globals
        if name.isidentifier() and not (name.startswith("__") and name.endswith("__"))
    )

    module_name = _builder.build_module_name(
        cls.__name__, _runtime.source_digest(source, version)
    )

    _PRECOMPILING = True

    try:
        compiled_cls = _builder.compile_class(
            cls, source, module_name, names, imports=imports
        )
    finally:
        _PRECOMPILING = False

    compiled_cls.__clypto_precompiled__ = True  # type: ignore[attr-defined]

    return compiled_cls

