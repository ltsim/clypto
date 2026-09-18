#!/usr/bin/env python
"""Generate the ``.pyx`` source and build it into an extension module."""
import importlib
import typing

from clypto.precompile import _runtime

__all__ = ["build_module_name", "build_preamble", "compile_class"]

_HEADER = (
    "# cython: language_level=3, boundscheck=False, cdivision=True, "
    "always_allow_keywords=True"
)


def build_preamble(
    module_name: str, names: list[str], imports: typing.Optional[list[str]] = None
) -> str:
    # Each global is copied in with a real assignment so Cython sees it as a
    # declared module name; an undeclared name in a class base is a compile
    # error in .pyx mode, and `globals().update(...)` does not declare one.
    lines = [
        _HEADER,
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


def build_module_name(class_name: str, digest: str) -> str:
    return f"_clypto_{class_name}_{digest}"


def compile_class(
    cls: typing.Any,
    source: str,
    module_name: str,
    names: list[str],
    imports: typing.Optional[list[str]] = None,
) -> typing.Any:
    pyx_path = _runtime.pyx_path(module_name)

    with open(pyx_path, "w", encoding="utf-8") as fh:
        fh.write(build_preamble(cls.__module__, names, imports=imports))
        fh.write("\n")
        fh.write(source)
        fh.write("\n")

    _runtime.install_pyximport()

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Cython compilation of {cls.__name__} failed.") from exc

    return getattr(module, cls.__name__)
