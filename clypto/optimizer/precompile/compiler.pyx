#!/usr/bin/env python
"""Just-in-time Cython compilation for user-defined optimizers via ``clypto.compiler``."""
import hashlib
import inspect
import sys
import textwrap

from clypto.compiler import compile_module

__all__ = ["compile_decorated", "is_precompiling"]

_PRECOMPILING = False


def is_precompiling() -> bool:
    """True while a compiled module is being imported and re-runs its decorators."""
    return _PRECOMPILING


def _build_preamble(module_name: str, names: list[str], imports: list[str] | None = None) -> str:
    # Each global is copied in with a real assignment so Cython sees it as a
    # declared module name; an undeclared name in a class base is a compile
    # error in .pyx mode, and `globals().update(...)` does not declare one.
    lines = [
        "import sys as __clypto_sys__",
        f"__clypto_mod__ = __clypto_sys__.modules.get({module_name!r})",
    ]

    # Injected base classes (the decorators add a base the source does not name)
    # are imported explicitly so the .pyx class header can reference them.
    for statement in imports or ():
        lines.append(statement)

    for name in names:
        lines.append(f"{name} = getattr(__clypto_mod__, {name!r}, None)")

    return "\n".join(lines) + "\n"


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

    module_source = (
        _build_preamble(cls.__module__, names, imports=imports) + "\n" + source + "\n"
    )
    # Digest includes the preamble so injected bases/global sets participate in
    # the cache key, not just the visible class body.
    digest = hashlib.sha1(module_source.encode("utf-8")).hexdigest()[:16]

    _PRECOMPILING = True

    try:
        module = compile_module(module_source, name=f"{cls.__name__}_{digest}")
        compiled_cls = getattr(module, cls.__name__)
    except Exception as exc:
        raise RuntimeError(f"Cython compilation of {cls.__name__} failed.") from exc
    finally:
        _PRECOMPILING = False

    compiled_cls.__clypto_precompiled__ = True  # type: ignore[attr-defined]

    return compiled_cls
